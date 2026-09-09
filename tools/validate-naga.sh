#!/usr/bin/env bash
# Compiles every dumped shader through naga, which is the only check the wgpu
# dialect gets before a wgpu device exists.
set -uo pipefail
dir="${1:-/tmp/afwebgpu-shaders}"
# The dawn-prefixed sources use Dawn-only spellings on purpose; naga is asked
# only about what a wgpu device could be handed.
pattern="${2:-}"
naga="${NAGA:-naga}"
command -v "$naga" >/dev/null || { echo "naga not found; cargo install naga-cli"; exit 127; }
pass=0; fail=0; failed=()
for f in "$dir"/*.wgsl; do
  case "$(basename "$f")" in dawn-*) continue;; esac
  if out=$("$naga" --input-kind wgsl "$f" 2>&1); then
    pass=$((pass+1))
  else
    fail=$((fail+1)); failed+=("$(basename "$f")")
    printf '%s\n%s\n' "--- $(basename "$f") ---" "$out" | head -12
  fi
done
echo "naga: $pass accepted, $fail rejected"
[ "$fail" -eq 0 ] || printf '  %s\n' "${failed[@]}"
exit "$fail"
