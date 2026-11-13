#!/usr/bin/env bash
# Start from an already activated attentivefp-pcba environment.
set -euo pipefail
addon_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${1:-.}" && pwd)"
if [ "$#" -gt 0 ]; then shift; fi
python_bin="${DTA_PYTHON:-python}"
mkdir -p "$project_dir/logs"
log_path="$(mktemp "$project_dir/logs/transfer_validation.XXXXXX.log")"
nohup "$python_bin" -u "$addon_dir/run.py" --project "$project_dir" "$@" \
    > "$log_path" 2>&1 < /dev/null &
pipeline_pid=$!
printf '%s\n' "$pipeline_pid" > "$log_path.pid"
echo "Pipeline PID: $pipeline_pid"
echo "Log: $log_path"
if command -v nvidia-smi >/dev/null 2>&1 && [ -f "$project_dir/scripts/monitor_gpu.sh" ]; then
    nohup bash "$project_dir/scripts/monitor_gpu.sh" "$pipeline_pid" "$log_path.gpu.csv" \
        > /dev/null 2>&1 < /dev/null &
    echo "GPU telemetry: $log_path.gpu.csv"
fi
echo "Ctrl+C stops monitoring only; the pipeline continues."
tail --pid="$pipeline_pid" -n 40 -F "$log_path" || true
