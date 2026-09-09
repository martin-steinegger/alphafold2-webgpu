#!/usr/bin/env bash
# Builds the addon. AFWEBGPU_WGPU_PATH points the build at a wgpu checkout
# instead of the pinned release, which is the seam a fork is driven through.
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${AFWEBGPU_WGPU_PATH:-}" ]; then
  echo "building against the pinned wgpu release"
  cargo build --release
else
  echo "building against ${AFWEBGPU_WGPU_PATH}"
  # The lock file holds whichever release was resolved from the registry, and a
  # checkout is rarely that same version. Cargo will not apply a patch whose
  # version the resolved graph does not already carry, and says so in a warning
  # rather than an error, so the build silently keeps the release. Resolving
  # afresh lets the requirement in Cargo.toml, which is a range, take the
  # checkout. The lock is restored either way.
  trap 'mv -f Cargo.lock.release Cargo.lock 2>/dev/null || true' EXIT
  mv -f Cargo.lock Cargo.lock.release
  cargo build --release \
    --config "patch.crates-io.wgpu.path='${AFWEBGPU_WGPU_PATH}/wgpu'" \
    --config "patch.crates-io.naga.path='${AFWEBGPU_WGPU_PATH}/naga'"
fi
cp target/release/libafwebgpu_wgpu.so ../../afwebgpu-wgpu.node
echo "wrote afwebgpu-wgpu.node"
