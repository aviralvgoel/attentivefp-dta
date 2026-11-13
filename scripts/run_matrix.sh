#!/usr/bin/env bash
# Validation-only sequential experiments. Does not run test evaluation.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${DTA_PYTHON:-python}"
for split_mode in random cold_drug; do
  for variant in scratch frozen finetune; do
    for seed in 42 43 44; do
      "$python_bin" -u train_dta.py --config "configs/kiba_${variant}.yaml" \
        --set "seed=$seed" \
        --set "data.split_file=data/kiba/processed/splits/${split_mode}_seed42.json" \
        --set "training.output_dir=runs/kiba/${split_mode}/${variant}/seed${seed}"
    done
  done
done
