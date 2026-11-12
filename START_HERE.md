# First commands on the GPU server

1. Extract `attentivefp_dta_final.zip` into `/home/dmlab/Aviral/Attentive_FP/`.
   This creates a NEW sibling of the previous demo. Leave a running old job alone.
2. Run:

```bash
conda activate attentivefp-pcba
cd /home/dmlab/Aviral/Attentive_FP/attentivefp_dta_final
python -m pip install -e '.[esm,dev]'
python doctor.py --cuda
python smoke.py --device cuda --output runs/smoke_gpu
```

3. If those pass, audit the existing full PCBA data and begin a NEW training run:

```bash
python audit.py pcba \
  --processed-dir /home/dmlab/Aviral/Attentive_FP/data/processed \
  --split-file /home/dmlab/Aviral/Attentive_FP/data/splits/scaffold_seed42.json \
  --output reports/existing_pcba_audit.json

bash scripts/run_training.sh pcba configs/pcba_existing_server.yaml
```

The last command includes log following and background GPU telemetry. Ctrl+C
stops following, not training. It writes to a new run directory, not your old one.

If PCBA training is already underway in the old demo, let it finish and read
`docs/LEGACY_PCBA.md` before reusing its encoder. Do not silently transfer weights
between different molecular featurizers.

Then follow README sections 3–8 for KIBA/DAVIS data, ESM, the three transfer
variants and explicit test evaluation. No research data or trained weights are
bundled. Model-training commands run on your machine; nothing is launched
remotely by extracting this folder.
