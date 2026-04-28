# TurboQuant for llama.cpp KV cache

This directory documents the llama.cpp integration of TurboQuant
(arXiv:2504.19874) as a KV cache quantization type. The working reference is
`turbo_quant.py` in the repo root; the llama.cpp patch is in
`patches/turboquant-llama.cpp.patch`.

## What the algorithm does

The paper has two algorithms; both are implemented.

### Algorithm 1 — TurboQuant MSE (`GGML_TYPE_TQ_MSE_4`)

1. Draw a fixed random orthogonal matrix `Π` of size `d×d` (once, at startup).
2. Per input vector `x ∈ R^d`:
   - Normalize: `u = x / ‖x‖`.
   - Rotate: `y = Π u`.
   - For each coordinate `y[j]`, round to the nearest of `2^b` centroids from
     a Lloyd–Max quantizer designed for the asymptotic marginal distribution
     of rotated unit-sphere coordinates.
3. Store `(‖x‖, idx[0..d-1])` as the compressed block.
4. Dequantize by looking up centroids, forming `ỹ`, and computing
   `x̂ = ‖x‖ · Π^T ỹ`.

The random rotation makes the marginal coord distribution look Gaussian with
variance `1/d`, which is why a single pre-baked codebook works for every
vector regardless of the original basis the data was expressed in. This is the
whole reason the paper's MSE guarantee doesn't depend on the data
distribution.

### Algorithm 2 — TurboQuant Prod (`GGML_TYPE_TQ_PROD_4`)

Adds a JL sign-sketch correction on top of the Algorithm 1 reconstruction so
that the inner-product estimator `<y, x̂>` becomes *unbiased* (Algorithm 1's
estimator is biased because the codebook always rounds toward the nearest
centroid, never away from it).

Building on the MSE stage above, also draw a fixed `d×d` matrix `M` with iid
Gaussian entries (independent of `Π`, distinct PRNG seed). Per block:

1. Run Algorithm 1 to get `idx[]` and the rotated reconstruction
   `ỹ = codebook[idx]`. Form the residual *in rotated space*
   `r = y - ỹ` (where `y = Π u` from Algorithm 1).
2. Sketch the residual: `qjl[i] = sign((M r)[i])` — one bit per dim.
3. Store `γ = ‖r‖` as a second fp16 scale.

Block layout: `‖x‖` + `γ` + `idx[]` (4-bit) + `qjl[]` (1-bit) = 84 B / 128 elem
= **5.25 bpv**.

Dequantize:
```
ỹ_full = codebook[idx] + (√(π/2) / d) · γ · M^T · qjl_signed
x̂      = ‖x‖ · Π^T · ỹ_full
```

The coefficient `√(π/2)/d` is the inverse JL constant: for iid Gaussian `M`,
`E[M^T sign(M r)] = (d / √(π/2)) · r/‖r‖`, so the term recovers `r` in
expectation. That makes `E[x̂] = x` (over the randomness of `M`), hence
`E[<y, x̂>] = <y, x>`.

Note we sketch in *rotated* space rather than the prototype's unit space. The
two are equivalent in distribution — `M` plays the role of `S · Π^T` from
the prototype — and rotated-space sketching saves one matmul per quant and per
dequant.

## What the llama.cpp integration does

The patch adds two new ggml tensor types:

| type | bits per element on disk | block layout (per 128 elem) | algorithm |
|---|---|---|---|
| `GGML_TYPE_TQ_MSE_4`  | 4.125 | `fp16 ‖x‖` + `64 B` of 4-bit indices                                       | Algorithm 1 |
| `GGML_TYPE_TQ_PROD_4` | 5.250 | `fp16 ‖x‖` + `fp16 γ` + `64 B` 4-bit indices + `16 B` 1-bit JL sign bits | Algorithm 2 |

Both are reachable from the CLI as `--cache-type-k tq_mse_4` /
`tq_prod_4`. Block size is 128 (one head); CPU vec-dot type is F32 for both
(query stays float, no extra precision loss). Both can be used for K only (not
V) — see "Why K only" below.

Backend matrix:

| backend | TQ_MSE_4 KV write | TQ_MSE_4 KV read | TQ_PROD_4 KV write | TQ_PROD_4 KV read | attention matmul |
|---|---|---|---|---|---|
| CPU  | ref `quantize_row_tq_mse_4`  | ref `dequantize_row_tq_mse_4`  | ref `quantize_row_tq_prod_4`  | ref `dequantize_row_tq_prod_4` | scratch-dequant + plain F32 dot |
| CUDA | fused kernel, 128 threads / block | fused kernel, 128 threads / block | not yet | not yet | cuBLAS F32 fallback (K dequantized via `ggml_cuda_dequantize_row_tq_mse_4_fp32`) |

Both backends build the rotation matrix `Π` and codebook on the CPU from the
same deterministic xoshiro256** seed; the CUDA path then `cudaMemcpyToSymbol`s
the tables into `__device__` globals. As a result a CPU-quantized block and a
GPU-quantized block of the same input are byte-identical (up to a few
border-case rounding differences where a rotated coord lands exactly on a
codebook midpoint), and a dequantized vector from either backend matches to a
few ULPs.

### Files changed

The patch touches 19 files under `llama.cpp/`:

CPU path (covers both TQ_MSE_4 and TQ_PROD_4):

```
common/arg.cpp                   +2    # expose tq_mse_4, tq_prod_4 to --cache-type-k parser
ggml/include/ggml.h              +3    # GGML_TYPE_TQ_MSE_4 = 42, TQ_PROD_4 = 43
ggml/src/ggml-common.h           +28   # block_tq_mse_4 + block_tq_prod_4 structs
ggml/src/ggml-cpu/ggml-cpu.c     +12   # type_traits_cpu entries
ggml/src/ggml-cpu/ops.cpp        +2    # clamp switch cases (no-op)
ggml/src/ggml-cpu/quants.c       +78   # from_float forwarders + vec_dot for both
ggml/src/ggml-cpu/quants.h       +9    # declarations
ggml/src/ggml-quants.c           +437  # codebooks, Pi, M, quant/dequant, PRNG,
                                       #  init, host-side accessors for both types
ggml/src/ggml-quants.h           +25   # declarations
ggml/src/ggml.c                  +18   # ggml_type_traits entries + init dispatch
tests/CMakeLists.txt             +3    # register all three turboquant tests
tests/test-quantize-fns.cpp      +18   # buffer-size fix + per-type error bounds
tests/test-turboquant.cpp        +131  # MSE-4 round-trip + vec_dot sanity (CPU)
tests/test-turboquant-prod.cpp   +166  # Prod-4 round-trip + vec_dot + IP sanity (CPU)
```

CUDA path (TQ_MSE_4 only — TQ_PROD_4 CUDA support is a separate follow-up):

```
ggml/src/ggml-cuda/convert.cu   +5    # register fp32/fp16 dequant in the to-*_cuda tables
ggml/src/ggml-cuda/cpy.cu       +7    # dispatch F32↔TQ_MSE_4 on CUDA
ggml/src/ggml-cuda/ggml-cuda.cu +14   # supports_op: allow CPY + MUL_MAT;
                                      # exclude TQ from MMVQ / MMQ to force cuBLAS fallback
ggml/src/ggml-cuda/tq_mse_4.cu  +344  # device tables, init, quant/dequant kernels
ggml/src/ggml-cuda/tq_mse_4.cuh +45   # declarations
tests/test-turboquant-cuda.cpp  +206  # CPU ↔ GPU parity + GPU round-trip
```

## Design decisions

### Why block size 128, not "the head dim"?

ggml block types have a `blck_size` that is a compile-time constant per type.
A single type can't follow the head dim of whatever model you load. So the
type commits to `d = 128`, the most common head dim in modern LLMs
(Llama-3, Qwen-2.5, Mistral-7B, ...). On models where `n_embd_head_k` is a
multiple of 128 (Gemma-4: 512, Gemma-4 SWA: 256), the type still works: you
just end up with several TurboQuant blocks per head instead of exactly one
block per head. The per-128 rotation is still a valid quantizer, just not the
"one rotation per head" structure the paper envisions.

On models whose head dim isn't a multiple of 128 (e.g. older 64-dim Mistral
variants), the tensor allocation will fail a ggml alignment assertion. That
was judged acceptable for a proof of concept.

### Why a Gaussian codebook instead of the exact Beta codebook?

The Python prototype solves Lloyd–Max numerically against the exact Beta
distribution for the rotated-unit-sphere marginal. For `d = 128` that Beta is
already very close to a Gaussian with `σ = 1/√d`, and Lloyd–Max centroids for
a standard Gaussian (4 bits, 16 levels) are well-known tabulated constants
(Lloyd 1982). The C implementation uses those constants and scales them by
`σ`. This avoids shipping a numerical Lloyd–Max solver in C.

### Why a process-wide rotation matrix and a fixed seed?

Two reasons. First, determinism: KV-cache entries are quantized on write and
dequantized on read, potentially across independent threads and across
process restarts reading the same context. Both sides must agree on `Π`.
Second, correctness of the quantizer itself: the codebook is designed for the
distribution of `Π u`, which is only Gaussian if `Π` is orthogonal. A fresh
Gaussian matrix per call would violate that.

`Π` is built once with:

- splitmix64 → xoshiro256** PRNG seeded from a fixed constant (42);
- fill a `128×128` matrix with Gaussian samples via Box–Muller;
- orthonormalize rows with modified Gram–Schmidt.

This gives bit-identical `Π` on any platform with IEEE-754 fp32.

### Why vec-dot goes through dequant-to-scratch

The existing low-bit KV types fuse dequant with the inner product
(`ggml_vec_dot_q4_0_q8_0` etc.). For TurboQuant that fusion is much harder:
the dequantized vector is `‖x‖ · Π^T ỹ`, so every output element depends on
*all* `ỹ` entries, not just the one in the same column. A fused kernel would
have to either precompute `Π y_query` once per row (doable but invasive) or
do a full matvec per block. The proof-of-concept takes the simple route:
dequantize one 128-vector to a stack buffer, then plain-float dot with the
query. Slower than Q4_0; correct.

### Why vec-dot type is F32 and not F16/Q8_0

With `vec_dot_type = F32`, the query passes through unchanged — no extra
lossy conversion is stacked on top of the K-cache loss. F16 or Q8_0 would
both reduce the compute-side buffer by 2–4× but introduce an additional
error on the query side that isn't present in the paper's model. For a
first-cut evaluation of TurboQuant's standalone quality, F32 is the cleaner
baseline.

One pre-existing test (`tests/test-quantize-fns.cpp`) assumed the vec-dot
type was always a quantized type (1 byte per element), and its temporary
buffer was sized `2 × test_size` bytes. With F32 vec-dot type that buffer
overflows, corrupting the heap. The patch bumps those buffers to
`4 × test_size`.

### Thread safety of the lazy init

The tables (`Π`, `Π^T`, codebook) are initialized on first use. Callers can
come from two contexts:

1. `ggml_quantize_init(GGML_TYPE_TQ_MSE_4)`, which already holds the ggml
   critical section. Taking it again would deadlock (it is non-recursive).
2. `quantize_row_tq_mse_4_ref` / `dequantize_row_tq_mse_4` called directly,
   with no lock held.

Rather than conditionally locking, the init uses a small atomic state machine
(`<stdatomic.h>`):

```
0 (uninit) --CAS--> 1 (building) --store--> 2 (ready)
```

The CAS winner builds the tables; losers spin on the state word until it
reaches 2. Spinning is fine because the build runs once per process
(~16 ms on a modern laptop, dominated by the `sin`/`log` calls in Box–Muller).

## Why K only, not V

V is typically stored transposed in llama.cpp
(`[kv_size, n_embd_v_gqa, n_stream]`) to make the attention-value matmul
cache-friendly. In that layout, a contiguous 128-element block covers 128
consecutive *sequence positions* for one channel, not 128 channels of one
token. Applying a per-channel rotation along the sequence axis is
semantically wrong — those 128 values have no reason to be isotropic. Using
`tq_mse_4` for V would produce incoherent output.

Supporting V cleanly would require disabling the transpose for this type
specifically, which is a bigger change than the minimal PoC allows for.

## Verification

Build:

```bash
cd llama.cpp
cmake -B build -DGGML_CUDA=OFF
cmake --build build -j
```

Unit tests:

```bash
./build/bin/test-turboquant
# unit-norm round-trip MSE ~ 7e-5
# vec_dot matches dequant-then-dot to 0 ULP

./build/bin/test-quantize-fns
# all 30+ quant types pass, including tq_mse_4
```

End-to-end sanity:

```bash
./build/bin/llama-completion \
    -m gemma-4-E2B-it-Q8_0.gguf \
    -p "The capital of France is" \
    -n 20 \
    --cache-type-k tq_mse_4 --cache-type-v f16 \
    --jinja
# Generates "Paris"
```

## CUDA path specifics

### Tables

`Π`, `Πᵀ`, and the 16-entry codebook live in three `__device__` arrays in
`ggml-cuda/tq_mse_4.cu`. `Π` and `Πᵀ` are 128×128 floats each, i.e. 64 KB
apiece — too big for `__constant__` on most GPUs, hence ordinary global
memory. The tables are populated on first use by
`ggml_cuda_tq_mse_4_init`, which calls the CPU-side `tq_mse_4_init_impl`
(idempotent), reads the already-built host tables via the exported
`ggml_tq_mse_4_host_pi` / `piT` / `codebook` accessors, and
`cudaMemcpyToSymbolAsync`es them across. A `std::once_flag` guards against
repeat uploads.

### Kernels

Two kernels, both launched as one CUDA block (128 threads) per TQ output
block:

1. **`cpy_f32_to_tq_mse_4_kernel`** (KV write).
   Thread `t` owns rotated coord `y[t]`. Stages `x[]` in shared memory,
   reduces `‖x‖²`, computes `y[t] = Π[t,:]·x / ‖x‖`, argmin against 16
   centroids, writes the packed nibble at `qs[t/2]` (thread pairs
   cooperate on the byte).
2. **`cpy_tq_mse_4_to_f32_kernel`** (KV read / dequant).
   Thread `t` unpacks one nibble into `y_rot[]`, then computes
   `x[t] = scale · Σ_j Πᵀ[t,j]·y_rot[j]`. This access pattern *is*
   coalesced (`Πᵀ[t,*]` read by adjacent threads stride by 4 bytes).

### Matmul path

The CUDA backend claims to support `MUL_MAT` with TQ as `src0` (see
`supports_op` in `ggml-cuda.cu`), but forces `use_mul_mat_vec_q` and
`use_mul_mat_q` off for this type. That makes `ggml_cuda_mul_mat` fall
through to `ggml_cuda_op_mul_mat_cublas`, which dequantizes `K` to an
F32 (or F16) scratch buffer via `ggml_get_to_fp32_cuda(GGML_TYPE_TQ_MSE_4)`
and then runs a regular cuBLAS GEMM. All of this stays on GPU memory —
no CPU fallback round-trip.

Flash-attention is *not* enabled for this type (intentionally — the flash-attn
kernel template machinery would cost a lot of `.cu` instance files for what
is a first-cut PoC). Without flash-attn, llama.cpp falls back to explicit
`ggml_mul_mat(K, Q) + softmax + ggml_mul_mat(V, kq_soft)`, which is the
path that goes through the dequant above.

### Parity with CPU

Since both backends use literally the same `Π` and codebook, round-trips
are bit-identical up to matmul order: the `Πᵀ·y_rot` sum is accumulated
in a different order on the two backends (scalar vs. 128 threads), which
can differ by a few ULPs in the last place. The CUDA test
(`test-turboquant-cuda.cpp`) checks that:

- Packed bytes agree on ≥95% of blocks (the 5% slack absorbs the
  rare case where a rotated coord lands exactly on a codebook midpoint
  and the two backends round differently).
- Max element-wise dequant diff is below 5e-3.
- Round-trip MSE on GPU is within the CPU ballpark (~7e-5 per coord on
  unit vectors).

## Known limits / future work

- **TQ_PROD_4 is CPU-only so far.** Algorithm 2 / `GGML_TYPE_TQ_PROD_4`
  is implemented on CPU end to end; the matching CUDA kernels (cpy
  F32↔TQ_PROD_4, fp32 / fp16 dequant for the cuBLAS fallback) are a
  separate follow-up commit.
- **No fused vec-dot on CUDA.** The clever identity
  `<Πᵀỹ, q> = <ỹ, Πq>` lets you precompute `Πq` once per row and reduce
  each block's contribution to a plain 128-way dot. That's the real
  performance win and belongs in `fattn-common.cuh`, not in the cpy
  kernels. The current CUDA path uses the cuBLAS F32 fallback, which
  dequantizes K to F32 scratch every attention call — correct but not
  the intended speedup.
- **No flash-attention on CUDA for this type.** Adding it means ~14 new
  `fattn-vec-instance-*.cu` files (TQ-as-K × every V type, and vice-versa).
  Worth doing once the perplexity numbers confirm quality.
- **Pi stored in one orientation only on CUDA.** The quant kernel's Π
  reads are uncoalesced. A row+column-major mirror of Π (another 64 KB
  per GPU) would fix this. Trivial follow-up.
- **No Metal / Vulkan / ROCm ports.** CUDA only.
- **No perplexity numbers yet.** A baseline-vs-TQ perplexity comparison on a
  held-out corpus would be the right next step to validate the paper's
  guarantees end-to-end inside a real LLM.
- **Head dim tied to a multiple of 128.** A model with head dim 64 (some
  smaller Mistrals) will fail at ggml tensor allocation.
