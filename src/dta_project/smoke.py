"""Offline CPU/GPU integration test using explicitly synthetic assay/affinity labels.

No ESM downloads and no scientific performance claims. Exercises actual RDKit,
PyG and AttentiveFP training, not mocked model layers.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import atomic_save, load_checkpoint, load_config
from .evaluate import evaluate
from .prepare_dta import prepare as prepare_dta
from .prepare_pcba import prepare as prepare_pcba
from .train_dta import train as train_dta
from .train_pcba import train as train_pcba

SMILES = [
    "CCO",
    "CCCO",
    "CCCCO",
    "CCN",
    "CCCN",
    "CC(=O)O",
    "CC(=O)N",
    "[Na+]",
    "c1ccccc1",
    "Cc1ccccc1",
    "c1ccncc1",
    "c1ccoc1",
    "c1ccsc1",
    "C1CCCCC1",
    "C1CCCC1",
    "C1CCC1",
    "C1CC1",
    "C1COCCN1",
    "C1CCNCC1",
    "c1ccc2ccccc2c1",
]
SEQUENCES = [
    "MKTAYIAKQRQISFVKSHFSRQ",
    "ACDEFGHIKLMNPQRSTVWY",
    "GILGFVFTLTVPSER",
    "MKWVTFISLLLLFSSAYSR",
    "GAVLIPFYWSTCMNQDEKRH",
]


def run(output, device="cpu"):
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Smoke output not empty: {out}; choose a new --output")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    project = Path(__file__).resolve().parents[2]
    pcba_dir, dta_dir = out / "pcba", out / "dta"
    rng = np.random.default_rng(42)
    tasks = [f"PCBA-{i}" for i in range(4)]
    frame = pd.DataFrame({"smiles": SMILES})
    for task in tasks:
        frame[task] = rng.integers(0, 2, len(frame)).astype(float)
    frame.loc[2, tasks[1]] = np.nan
    prepare_pcba(frame, tasks, pcba_dir, shard_size=5)
    rows = [
        {"smiles": smi, "sequence": seq, "affinity": float(i / 5 + j / 3)}
        for i, smi in enumerate(SMILES)
        for j, seq in enumerate(SEQUENCES)
    ]
    prepare_dta(pd.DataFrame(rows), dta_dir, dataset="synthetic", mode="cold_drug")
    proteins = pd.read_parquet(dta_dir / "proteins.parquet")
    cache = out / "synthetic_proteins.pt"
    emb = {}
    for row in proteins.itertuples():
        # Deterministic dummy residue vectors; never confuse with real ESM features.
        generator = torch.Generator().manual_seed(
            int(hashlib.sha256(row.protein_sequence.encode()).hexdigest()[:8], 16)
        )
        emb[row.protein_id] = torch.randn((len(row.protein_sequence), 16), generator=generator)
    atomic_save(
        {
            "metadata": {"model": "synthetic_fixture", "pooling": "residues", "synthetic": True},
            "embeddings": emb,
            "sequence_sha256": {
                r.protein_id: hashlib.sha256(r.protein_sequence.encode()).hexdigest()
                for r in proteins.itertuples()
            },
        },
        cache,
    )
    pc = load_config(project / "configs/pcba.yaml")
    pc["data"].update(
        processed_dir=str(pcba_dir), split_file=str(pcba_dir / "splits/random_seed42.json")
    )
    pc["model"].update(hidden_dim=16, embedding_dim=16, num_layers=2, num_timesteps=2, dropout=0.0)
    pc["training"].update(
        output_dir=str(out / "pretrain"),
        device=device,
        precision="fp32",
        batch_size=5,
        workers=0,
        epochs=2,
        patience=5,
        progress=False,
        smoke_allow_bce_fallback=True,
    )
    pretrained = train_pcba(pc)
    # Exercise a clean epoch-boundary pause and resume with unchanged scientific settings.
    dc = load_config(project / "configs/kiba_scratch.yaml")
    dc["data"].update(
        processed_dir=str(dta_dir),
        split_file=str(dta_dir / "splits/cold_drug_seed42.json"),
        protein_cache=str(cache),
        allow_synthetic=True,
    )
    dc["model"]["drug"] = {k: v for k, v in pc["model"].items()}
    dc["model"].update(fusion_dim=16, dropout=0.0)
    dc["training"].update(
        output_dir=str(out / "scratch"),
        device=device,
        precision="fp32",
        batch_size=16,
        workers=0,
        epochs=3,
        patience=5,
        progress=False,
        stop_after_epoch=1,
    )
    first = train_dta(dc)
    dc["training"]["stop_after_epoch"] = None
    scratch = train_dta(dc, first.parent / "last.pt")
    assert load_checkpoint(scratch.parent / "last.pt")["epoch"] == 3
    for mode in ("frozen", "finetune"):
        cfg = copy.deepcopy(dc)
        cfg["transfer"].update(
            mode=mode, checkpoint=str(pretrained), warmup_epochs=1 if mode == "finetune" else 0
        )
        cfg["training"]["output_dir"] = str(out / mode)
        train_dta(cfg)
    cfg = copy.deepcopy(cfg)
    cfg["model"]["fusion"] = "cross_attention"
    cfg["training"]["output_dir"] = str(out / "attention")
    train_dta(cfg)
    evaluate(scratch, output=out / "heldout_test", device=device, batch_size=8)
    print(
        f"PASS: PCBA, scratch, resume, frozen, fine-tune, cross-attention and held-out prediction export: {out}"
    )
    print(
        "All labels/protein features in this smoke test are SYNTHETIC; scores are not research results."
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/smoke"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.output, args.device)


if __name__ == "__main__":
    main()
