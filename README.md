# tejas-kv-compress

Experiments with KV-cache compression for LLM inference.

## Contents

- **`turbo_quant.py`** — numpy prototype of TurboQuant (arXiv:2504.19874),
  both the MSE quantizer (Algorithm 1) and the product/JL variant
  (Algorithm 2). Has a `run_tests()` entry point for quick round-trip and
  unbiasedness checks.
- **`patches/turboquant-llama.cpp.patch`** — CPU integration of
  TurboQuant MSE (4-bit, head-dim 128) into llama.cpp as a new KV-cache
  quantization type `GGML_TYPE_TQ_MSE_4` / `--cache-type-k tq_mse_4`.
  Applies against upstream `ggml-org/llama.cpp@cf8b0dbda`.
- **`docs/turboquant.md`** — design notes for the llama.cpp integration:
  file-by-file summary of what changed, why block size 128, why the codebook
  is Gaussian instead of Beta, why K only, thread-safety of the lazy init,
  and how to verify it end to end.

## Quick start (llama.cpp patch)

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout cf8b0dbda
git apply ../patches/turboquant-llama.cpp.patch

cmake -B build -DGGML_CUDA=OFF
cmake --build build -j

./build/bin/test-turboquant
./build/bin/llama-completion -m your-model.gguf -p "..." \
    --cache-type-k tq_mse_4 --cache-type-v f16 --jinja
```
