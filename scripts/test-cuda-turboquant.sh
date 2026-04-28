#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Single-shot CUDA PoC test for TurboQuant MSE-4 KV-cache quant in llama.cpp.
#
# What it does, in order:
#   1. Build llama.cpp with CUDA enabled (if not already built).
#   2. Run the CPU-only test (test-turboquant) as a sanity baseline.
#   3. Run the CPU↔CUDA parity test (test-turboquant-cuda).
#   4. If a model file is provided, run a short end-to-end generation with
#      --cache-type-k tq_mse_4 on GPU, and compare against f16 baseline.
#
# Usage:
#   scripts/test-cuda-turboquant.sh [-m path/to/model.gguf] [-a cc_list]
#
#   -m  optional path to a GGUF model. If omitted, step 4 is skipped. The
#       model's head dim must be a multiple of 128 (Llama-3, Qwen-2.5,
#       Gemma-4, ... all qualify).
#   -a  optional CMAKE_CUDA_ARCHITECTURES list (default: "native").
# -----------------------------------------------------------------------------
set -euo pipefail

# Locations are relative to the repo root (tejas-kv-compress/).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${REPO_ROOT}/llama.cpp"
BUILD_DIR="${LLAMA_DIR}/build-cuda"
PATCH="${REPO_ROOT}/patches/turboquant-llama.cpp.patch"

MODEL=""
ARCHS="native"
while getopts "m:a:" opt; do
    case "$opt" in
        m) MODEL="$OPTARG" ;;
        a) ARCHS="$OPTARG"  ;;
        *) echo "usage: $0 [-m model.gguf] [-a cc_list]"; exit 1 ;;
    esac
done

# ----- 0. Sanity: toolchain + patch -------------------------------------------
echo "== step 0: sanity =="
command -v nvcc         >/dev/null || { echo "nvcc not in PATH"; exit 1; }
command -v nvidia-smi   >/dev/null || { echo "nvidia-smi not in PATH"; exit 1; }
command -v cmake        >/dev/null || { echo "cmake not in PATH"; exit 1; }

nvidia-smi -L | head -4 || true
nvcc --version | tail -n 2

if [[ ! -d "$LLAMA_DIR" ]]; then
    echo "cloning upstream llama.cpp at the base commit the patch was made against..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    ( cd "$LLAMA_DIR" && git checkout cf8b0dbda )
fi

# Apply patch if the new files aren't already there.
if [[ ! -f "$LLAMA_DIR/ggml/src/ggml-cuda/tq_mse_4.cu" ]]; then
    echo "applying $PATCH ..."
    ( cd "$LLAMA_DIR" && git apply --check "$PATCH" && git apply "$PATCH" )
else
    echo "patch already applied (skipping)"
fi

# ----- 1. Build --------------------------------------------------------------
echo "== step 1: cmake build (CUDA=$ARCHS) =="
cmake -B "$BUILD_DIR" -S "$LLAMA_DIR" \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
      -DLLAMA_CURL=OFF \
      -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIR" -j "$(nproc)" \
      --target test-turboquant test-turboquant-prod \
              test-turboquant-cuda test-turboquant-prod-cuda \
              llama-completion

# ----- 2. CPU baselines ------------------------------------------------------
echo "== step 2a: CPU round-trip TQ_MSE_4 (test-turboquant) =="
"$BUILD_DIR/bin/test-turboquant"
echo "== step 2b: CPU round-trip TQ_PROD_4 (test-turboquant-prod) =="
"$BUILD_DIR/bin/test-turboquant-prod"

# ----- 3. CPU <-> CUDA parity ------------------------------------------------
echo "== step 3a: CPU<->CUDA parity TQ_MSE_4 (test-turboquant-cuda) =="
"$BUILD_DIR/bin/test-turboquant-cuda"
echo "== step 3b: CPU<->CUDA parity TQ_PROD_4 (test-turboquant-prod-cuda) =="
"$BUILD_DIR/bin/test-turboquant-prod-cuda"

# ----- 4. End-to-end generation (optional) -----------------------------------
if [[ -n "$MODEL" ]]; then
    if [[ ! -f "$MODEL" ]]; then
        echo "model not found at $MODEL — skipping step 4" >&2
        exit 0
    fi

    echo "== step 4: end-to-end generation on GPU =="
    echo "-- baseline (--cache-type-k f16) --"
    "$BUILD_DIR/bin/llama-completion" \
        -m "$MODEL" \
        -p "The capital of France is" \
        -n 20 \
        --cache-type-k f16 --cache-type-v f16 \
        --jinja -ngl 99 | tail -n 10

    echo "-- tq_mse_4 (--cache-type-k tq_mse_4) --"
    "$BUILD_DIR/bin/llama-completion" \
        -m "$MODEL" \
        -p "The capital of France is" \
        -n 20 \
        --cache-type-k tq_mse_4 --cache-type-v f16 \
        --jinja -ngl 99 | tail -n 10

    echo "-- tq_prod_4 (--cache-type-k tq_prod_4) --"
    "$BUILD_DIR/bin/llama-completion" \
        -m "$MODEL" \
        -p "The capital of France is" \
        -n 20 \
        --cache-type-k tq_prod_4 --cache-type-v f16 \
        --jinja -ngl 99 | tail -n 10

    echo
    echo "If all three generations produced reasonable tokens (e.g. \"Paris\"), the"
    echo "CUDA KV-cache write + cuBLAS-F32 fallback path is end-to-end correct"
    echo "for both TurboQuant variants."
fi

echo "== all done =="
