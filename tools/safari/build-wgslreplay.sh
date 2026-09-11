#!/usr/bin/env bash
# Builds wgslreplay against one WebKit tag, the WGSL compiler a Safari release
# ships, and packages it with the libraries it needs into <out>.
#
#   tools/safari/build-wgslreplay.sh WebKit-7624.2.5.11.4 <work dir> <out dir>
#
# The tag is the Safari build in "About Safari" with its first digit dropped:
# 26.5 (21624.2.5.11.4) is WebKit-7624.2.5.11.4. Run it inside the conda
# environment tools/safari/environment.yml describes.
set -euo pipefail
TAG=${1:?usage: build-wgslreplay.sh <WebKit tag> <work dir> <out dir>}
WORK=$(mkdir -p "${2:?}" && cd "$2" && pwd)
OUT=$(mkdir -p "${3:?}" && cd "$3" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
PREFIX=${CONDA_PREFIX:?run inside the wgslreplay conda environment}
SRC=$WORK/src BUILD=$WORK/build

if [ ! -d "$SRC" ]; then
  git clone -q --filter=blob:none --no-checkout --depth=1 --branch "$TAG" \
    https://github.com/WebKit/WebKit.git "$SRC"
  git -C "$SRC" sparse-checkout set --cone Source/bmalloc Source/WTF Source/JavaScriptCore \
    Source/WebGPU Source/cmake Source/ThirdParty/unifdef Source/ThirdParty/capstone Tools/Scripts
  git -C "$SRC" checkout -q
  # The JSCOnly port builds WGSL once WebGPU is on. Its API tests are not in
  # the sparse checkout.
  sed -i 's/^set(ENABLE_WEBGPU OFF)$/set(ENABLE_WEBGPU ON)/; s/set(ENABLE_API_TESTS ON)/set(ENABLE_API_TESTS OFF)/' \
    "$SRC/Source/cmake/OptionsJSCOnly.cmake"
  cp "$HERE/wgslreplay.cpp" "$SRC/Source/WebGPU/WGSL/"
  if grep -q "add_library(WGSLCore" "$SRC/Source/WebGPU/WGSL/CMakeLists.txt"; then
    # WebKit main: wgslc links a WGSLCore library.
    cat >> "$SRC/Source/WebGPU/WGSL/CMakeLists.txt" <<'CM'
set(wgslreplay_SOURCES wgslreplay.cpp)
set(wgslreplay_LIBRARIES WGSLCore)
WEBKIT_EXECUTABLE_DECLARE(wgslreplay)
WEBKIT_EXECUTABLE(wgslreplay)
WEBKIT_REUSE_PREFIX_HEADER(wgslreplay WGSLCore WGSLPrefix.h PREFIX_LANGUAGES CXX)
CM
  else
    # Safari 26: wgslc is one executable, and the API is the 7624 one.
    cat >> "$SRC/Source/WebGPU/WGSL/CMakeLists.txt" <<'CM'
set(wgslreplay_SOURCES ${wgslc_SOURCES})
list(REMOVE_ITEM wgslreplay_SOURCES wgslc.cpp)
list(APPEND wgslreplay_SOURCES wgslreplay.cpp)
set(wgslreplay_FRAMEWORKS ${wgslc_FRAMEWORKS})
set(wgslreplay_PRIVATE_INCLUDE_DIRECTORIES ${wgslc_PRIVATE_INCLUDE_DIRECTORIES})
set(wgslreplay_DEPENDENCIES wgsl-types)
WEBKIT_EXECUTABLE_DECLARE(wgslreplay)
WEBKIT_EXECUTABLE(wgslreplay)
target_compile_definitions(wgslreplay PRIVATE WGSLREPLAY_WEBKIT_7624=1)
CM
  fi
fi

# conda's clang, pointed at conda's GCC for libstdc++ and at its sysroot.
GCC_DIR=$(dirname "$(ls "$PREFIX"/lib/gcc/x86_64-conda-linux-gnu/*/crtbegin.o | head -1)")
FLAGS="--gcc-install-dir=$GCC_DIR --sysroot=$PREFIX/x86_64-conda-linux-gnu/sysroot"
cmake -G Ninja -S "$SRC" -B "$BUILD" -DPORT=JSCOnly -DCMAKE_BUILD_TYPE=Release -DDEVELOPER_MODE=OFF \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ "-DCMAKE_C_FLAGS=$FLAGS" "-DCMAKE_CXX_FLAGS=$FLAGS" \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
  -DICU_ROOT="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" > "$WORK/configure.log" 2>&1 \
  || { tail -40 "$WORK/configure.log"; exit 1; }
ninja -C "$BUILD" wgslreplay > "$WORK/build.log" 2>&1 || { grep -m 20 " error" "$WORK/build.log"; exit 1; }

# The binary and the conda libraries it resolves, so it runs without conda.
mkdir -p "$OUT/bin" "$OUT/lib"
cp "$BUILD/bin/wgslreplay" "$OUT/bin/"
ldd "$BUILD/bin/wgslreplay" | awk -v prefix="$PREFIX" '$3 ~ "^" prefix { print $3 }' | xargs -r cp -L -t "$OUT/lib"
echo "$TAG" > "$OUT/webkit-tag"
echo "$OUT/bin/wgslreplay ($TAG)"
