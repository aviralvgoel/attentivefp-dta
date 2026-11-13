#!/usr/bin/env bash
# Launch training with logs and GPU telemetry; then follow the training log.
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/run_training.sh pcba|dta CONFIG [training arguments...]" >&2
  exit 2
fi
task="$1"
config="$2"
shift 2
case "$task" in pcba|dta) ;; *) echo "Task must be pcba or dta" >&2; exit 2 ;; esac
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${DTA_PYTHON:-python}"
if [ ! -f "$config" ]; then echo "Missing configuration: $config" >&2; exit 2; fi
mkdir -p logs
log_path="$(mktemp "$project_dir/logs/${task}.XXXXXX.log")"
nohup "$python_bin" -u "train_${task}.py" --config "$config" "$@" > "$log_path" 2>&1 < /dev/null &
training_pid=$!
printf '%s\n' "$training_pid" > "$log_path.pid"
echo "Training PID: $training_pid"
echo "Training log: $log_path"
if command -v nvidia-smi >/dev/null 2>&1; then
  nohup bash scripts/monitor_gpu.sh "$training_pid" "$log_path.gpu.csv" > /dev/null 2>&1 < /dev/null &
  echo "GPU telemetry: $log_path.gpu.csv"
fi
echo "Ctrl+C stops log following; nohup training continues."
# GNU tail is available on the target Linux server. macOS falls back to tail -f.
if tail --help 2>/dev/null | grep -q -- '--pid'; then
  tail --pid="$training_pid" -n 40 -f "$log_path" || true
else
  tail -n 40 -f "$log_path" || true
fi
