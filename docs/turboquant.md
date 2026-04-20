# TurboQuant for llama.cpp KV cache

This directory documents the llama.cpp integration of TurboQuant
(arXiv:2504.19874) as a KV cache quantization type. The working reference is
`turbo_quant.py` in the repo root; the llama.cpp patch is in
`patches/turboquant-llama.cpp.patch`.

## What the algorithm does

TurboQuant MSE (Algorithm 1 of the paper):

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

## What the llama.cpp integration does

The patch adds a single new ggml tensor type, `GGML_TYPE_TQ_MSE_4`:

| field | value |
|---|---|
| block size (elements) | 128 (one head) |
| bits per element | 4 |
| block layout | `fp16 scale` + `64 B` of packed nibbles (code indices) |
| bits per value on disk | `66·8/128 = 4.125` |
| vec-dot type | `F32` (query stays float, no extra precision loss) |

It is reachable from the CLI as `--cache-type-k tq_mse_4`. The type can be
used for K only (not V) — see "Why K only" below.

### Files changed

The patch touches 13 files under `llama.cpp/`:

```
common/arg.cpp              +1   # expose tq_mse_4 to --cache-type-k parser
ggml/include/ggml.h         +2   # GGML_TYPE_TQ_MSE_4 = 42, COUNT = 43
ggml/src/ggml-common.h      +11  # block_tq_mse_4 struct + QK_TQ_MSE_4
ggml/src/ggml-cpu/ggml-cpu.c +6  # type_traits_cpu entry
ggml/src/ggml-cpu/ops.cpp   +1   # clamp switch case (no-op)
ggml/src/ggml-cpu/quants.c  +39  # from_float forwarder + vec_dot
ggml/src/ggml-cpu/quants.h  +5   # declarations
ggml/src/ggml-quants.c      +233 # codebook, rotation, quant/dequant, PRNG, init
ggml/src/ggml-quants.h      +9   # declarations
ggml/src/ggml.c             +9   # ggml_type_traits entry + init dispatch
tests/CMakeLists.txt        +1
tests/test-quantize-fns.cpp +15  # buffer-size fix + per-type error bound
tests/test-turboquant.cpp   +128 # new: round-trip + vec_dot sanity
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

## Known limits / future work

- **MSE only.** Algorithm 2 of the paper (TurboQuantProd: MSE stage +
  JL-sign residual) is not implemented; it doesn't fit one block cleanly
  because it emits a `(idx, sign_bits, γ)` triple.
- **CPU only.** No CUDA / Metal / Vulkan kernel. Those backends each ship
  their own quantized-matmul paths and would each need a port.
- **No perplexity numbers yet.** A baseline-vs-TQ perplexity comparison on a
  held-out corpus would be the right next step to validate the paper's
  guarantees end-to-end inside a real LLM.
- **Head dim tied to a multiple of 128.** A model with head dim 64 (some
  smaller Mistrals) will fail at ggml tensor allocation.
