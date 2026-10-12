#!/usr/bin/env bash
# Build llama.cpp (pinned) with the Metal backend for Apple Silicon.
#
# The Intel build needed Vulkan (RADV) plus a caller-side SPIRV-Headers hack
# and a `g++-13` pin for CUDA 12.4. None of that exists here: Metal is a
# system framework, so the build needs only Xcode's toolchain and cmake.
#
#   ./scripts/build.sh              # Metal + Accelerate, Release
#
# Accelerate (vDSP/BLAS) is worthwhile even with `-ngl 99`: the token
# embedding, the SSM/conv step of this model's linear-attention layers and the
# final sampling stay on the CPU, and those are exactly the ops vDSP
# accelerates.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT/third_party/llama.cpp"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"

if [ ! -d "$LLAMA_DIR/.git" ]; then
  echo "build: $LLAMA_DIR is missing. Run:" >&2
  echo "build:   git clone https://github.com/ggml-org/llama.cpp $LLAMA_DIR" >&2
  exit 1
fi

have="$(git -C "$LLAMA_DIR" rev-parse HEAD)"
echo "build: llama.cpp at $have"

if command -v ninja >/dev/null 2>&1; then
  GEN=(-G Ninja)
else
  GEN=()
fi

# GGML_METAL is the whole GPU path on macOS. GGML_BLAS with the Apple vendor
# binds Accelerate's BLAS rather than requiring a separate OpenBLAS.
CMAKE_ARGS=(
  "${GEN[@]}"
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_METAL=ON
  -DGGML_METAL_EMBED_LIBRARY=ON
  -DGGML_ACCELERATE=ON
  -DGGML_BLAS=ON
  -DGGML_BLAS_VENDOR=Apple
  -DGGML_NATIVE=ON
  -DLLAMA_CURL=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_SERVER=ON
  -DGGML_BACKEND_DL=OFF
)

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}" \
  2>&1 | tee "$ROOT/logs/configure.log"

cmake --build "$BUILD_DIR" -j "$JOBS" 2>&1 | tee "$ROOT/logs/build.log"

echo "=== built binaries ==="
ls -la "$BUILD_DIR/bin/"
