#!/usr/bin/env bash
# Prints the index of an idle card, or fails.
#
# This host is shared and three of its six cards usually carry someone else's
# work. A measurement on a busy one reads three times slow and moves 20% run to
# run, and two rounds of a comparison have been thrown away that way. Every
# timing run picks its card through this, and refuses to start if none is free.
#
#   CUDA_VISIBLE_DEVICES=$(tools/idle-gpu.sh) npm run ...
#
# AFWEBGPU_IDLE_MAX_UTILIZATION and AFWEBGPU_IDLE_MAX_MIB set what idle means;
# a card holding a little memory at 0% is another process that has finished.
set -euo pipefail
max_utilization=${AFWEBGPU_IDLE_MAX_UTILIZATION:-5}
max_mib=${AFWEBGPU_IDLE_MAX_MIB:-1024}
while IFS=, read -r index utilization memory; do
  if [ "$utilization" -le "$max_utilization" ] && [ "$memory" -le "$max_mib" ]; then
    echo "$index"
    exit 0
  fi
done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
  --format=csv,noheader,nounits | tr -d ' ')
echo "no idle GPU: every card is above ${max_utilization}% or holds more than ${max_mib} MiB" >&2
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >&2
exit 1
