# tejas-kv-compress

Experiments with KV-cache compression for LLM inference.

## Contents

- **`turbo_quant.py`** — numpy prototype of TurboQuant (arXiv:2504.19874),
  both the MSE quantizer (Algorithm 1) and the product/JL variant
  (Algorithm 2). Has a `run_tests()` entry point for quick round-trip and
  unbiasedness checks.
- **`patches/turboquant-llama.cpp.patch`** — integration of TurboQuant
  into llama.cpp. Adds two head-dim-128 KV-cache quant types, both with
  CPU + CUDA paths:
    - `GGML_TYPE_TQ_MSE_4`  (`--cache-type-k tq_mse_4`)  — Algorithm 1, 4.125 bpv.
    - `GGML_TYPE_TQ_PROD_4` (`--cache-type-k tq_prod_4`) — Algorithm 2 (MSE +
      1-bit JL sketch on the residual), 5.25 bpv. Unbiased inner-product
      estimator.
  Applies against upstream `ggml-org/llama.cpp@cf8b0dbda`.
- **`scripts/test-cuda-turboquant.sh`** — one-shot script for a GPU box:
  applies the patch, builds with CUDA, runs the CPU baseline test, the
  CPU↔CUDA parity test, and (optionally) an end-to-end generation with
  `--cache-type-k tq_mse_4` vs. the `f16` baseline.
- **`docs/turboquant.md`** — design notes for the llama.cpp integration:
  file-by-file summary of what changed, why block size 128, why the codebook
  is Gaussian instead of Beta, why K only, thread-safety of the lazy init,
  the CUDA kernel structure, and how to verify it end to end.

## Quick start — CPU

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout cf8b0dbda
git apply ../patches/turboquant-llama.cpp.patch

cmake -B build -DGGML_CUDA=OFF
cmake --build build -j

./build/bin/test-turboquant
./build/bin/test-turboquant-prod
./build/bin/llama-completion -m your-model.gguf -p "..." \
    --cache-type-k tq_mse_4 --cache-type-v f16 --jinja
./build/bin/llama-completion -m your-model.gguf -p "..." \
    --cache-type-k tq_prod_4 --cache-type-v f16 --jinja
```

## Quick start — CUDA (A100 / H100)

```bash
# Clones upstream at the right base, applies the patch, builds with CUDA,
# runs both tests, and optionally does end-to-end gen if -m is given.
./scripts/test-cuda-turboquant.sh -m path/to/model.gguf
```
