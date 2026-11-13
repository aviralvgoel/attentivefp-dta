"""Add a split to an already curated DTA dataset without rebuilding graphs."""

import argparse
from pathlib import Path

import pandas as pd

from .common import sha256, write_json
from .splits import make_dta_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["random", "cold_drug", "scaffold", "cold_target", "cold_both"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pair_path = args.processed_dir / "pairs.parquet"
    output = args.processed_dir / "splits" / f"{args.mode}_seed{args.seed}.json"
    if output.exists():
        raise FileExistsError(f"Split already exists: {output}")
    split = make_dta_split(pd.read_parquet(pair_path), args.mode, args.seed)
    split["metadata"]["pairs_sha256"] = sha256(pair_path)
    write_json(output, split)
    print(output)


if __name__ == "__main__":
    main()
