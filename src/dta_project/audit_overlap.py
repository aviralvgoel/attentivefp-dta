"""Report PCBA pretraining exposure of downstream cold-start drugs/scaffolds."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .chemistry import canonicalize, scaffold
from .common import write_json
from .splits import PARTITIONS, audit_dta_split, check_indices


def overlap_report(pcba_dir, pcba_split, dta_dir, dta_split):
    index = pd.read_parquet(Path(pcba_dir) / "index.parquet")
    if "canonical_smiles" not in index:
        raise ValueError("Overlap audit requires canonical_smiles in PCBA index")
    source_split = json.loads(Path(pcba_split).read_text())
    check_indices(source_split, index.global_index)
    train = index[index.global_index.isin(source_split["train"])]
    seen_drugs = set(train.canonical_smiles.map(canonicalize))
    seen_scaffolds = {scaffold(s) for s in seen_drugs}
    pairs = pd.read_parquet(Path(dta_dir) / "pairs.parquet")
    target_split = json.loads(Path(dta_split).read_text())
    audit_dta_split(pairs, target_split)
    result = {
        "pcba_training_molecules": len(train),
        "downstream": {},
        "interpretation": "Downstream-cold is not necessarily pretraining-unseen. Compound exposure does not establish assay endpoint overlap.",
    }
    for part in PARTITIONS:
        table = pairs[pairs.pair_id.isin(target_split[part])]
        drugs = set(table.canonical_smiles)
        scaffolds = {scaffold(s) for s in drugs}
        result["downstream"][part] = {
            "unique_drugs": len(drugs),
            "drugs_in_pcba_train": len(drugs & seen_drugs),
            "unique_scaffolds": len(scaffolds),
            "scaffolds_in_pcba_train": len(scaffolds & seen_scaffolds),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcba-dir", type=Path, required=True)
    parser.add_argument("--pcba-split", type=Path, required=True)
    parser.add_argument("--dta-dir", type=Path, required=True)
    parser.add_argument("--dta-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = overlap_report(args.pcba_dir, args.pcba_split, args.dta_dir, args.dta_split)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
