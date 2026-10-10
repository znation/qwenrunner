#!/usr/bin/env bash
# Build llama.cpp (pinned) with the CPU + Vulkan backends.
#
# glslc/libshaderc are NOT installed system-wide (no sudo on this box); they were
# fetched with `apt-get download` and extracted into tools/local/. Both the
# configure step (extension probes run glslc) and the build step (shader
# generation) need PATH/LD_LIBRARY_PATH pointing there.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT/third_party/llama.cpp"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
PINNED_REV="aa5e0092fdb8980b9e89498dfb73049b4fe204c2"
JOBS="${JOBS:-$(nproc)}"

export PATH="$ROOT/tools/local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/tools/local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$ROOT/tools/local/usr:${CMAKE_PREFIX_PATH:-}"

if [ -d "$LLAMA_DIR/.git" ]; then
  have="$(git -C "$LLAMA_DIR" rev-parse HEAD)"
  [ "$have" = "$PINNED_REV" ] || echo "WARNING: llama.cpp at $have, expected $PINNED_REV" >&2
fi

# ggml-vulkan/CMakeLists.txt calls find_package(SPIRV-Headers CONFIG REQUIRED) but
# never links the imported target; upstream relies on SPIRV-Headers being installed
# system-wide (Debian: /usr/include/spirv/...). Ours lives under tools/local/, so add
# its include dir explicitly.
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-isystem $ROOT/tools/local/usr/include" \
  -DGGML_NATIVE=ON \
  -DGGML_VULKAN=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DGGML_BACKEND_DL=OFF \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  2>&1 | tee "$ROOT/logs/configure.log"

cmake --build "$BUILD_DIR" -j "$JOBS" 2>&1 | tee "$ROOT/logs/build.log"

echo "=== built binaries ==="
ls -la "$BUILD_DIR/bin/"
