#!/usr/bin/env bash
# Replays recordings from tools/record-safari-pipelines.ts through a packaged
# wgslreplay, and fails if Safari would reject any pipeline in them.
#
#   tools/safari/replay.sh <wgslreplay package> <recording dir> [more dirs]
set -uo pipefail
PACKAGE=${1:?usage: replay.sh <wgslreplay package> <recording dir>...}; shift
export LD_LIBRARY_PATH="$PACKAGE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "WebKit $(cat "$PACKAGE/webkit-tag" 2>/dev/null || echo "(unknown tag)")"
ok=0 rejected=0 modules=0
for dir in "$@"; do
  while IFS= read -r -d '' module; do
    pipelines=${module%.wgsl}.pipelines
    [ -f "$pipelines" ] || continue
    modules=$((modules + 1))
    out=$("$PACKAGE/bin/wgslreplay" --features=shader-f16 "$module" "$pipelines" 2>&1)
    ok=$((ok + $(grep -c '^ok' <<<"$out")))
    while IFS= read -r line; do
      rejected=$((rejected + 1))
      echo "$(basename "$dir") $(head -1 "$module" | cut -c4-): ${line#fail }"
    done < <(grep -v '^ok' <<<"$out")
  done < <(find "$dir" -name '*.wgsl' -print0)
done
echo "$modules modules, $ok pipelines accepted, $rejected rejected"
[ "$modules" -gt 0 ] && [ "$rejected" -eq 0 ]
