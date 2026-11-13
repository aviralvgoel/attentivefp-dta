#!/usr/bin/env bash
set -euo pipefail
training_pid="$1"
output="$2"
while kill -0 "$training_pid" 2>/dev/null; do
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits >> "$output" || true
  sleep 5
done
