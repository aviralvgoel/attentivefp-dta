"""Make a NEW encoder-transfer checkpoint only after full legacy feature verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chemistry import SCHEMA_HASH
from .common import atomic_save, load_checkpoint, sha256
from .models import PCBAClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--verified-audit", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Never overwrite the original checkpoint; choose a new output")
    report = json.loads(args.verified_audit.read_text())
    if (
        not report.get("all_graph_features_verified")
        or report.get("verified_schema_hash") != SCHEMA_HASH
    ):
        raise ValueError("Need a full --check-features --all-graphs audit, not a sample check")
    if report["index_sha256"] != sha256(args.processed_dir / "index.parquet"):
        raise ValueError("PCBA index changed after audit")
    for name, digest in report["shard_hashes"].items():
        if sha256(args.processed_dir / "graphs" / name) != digest:
            raise ValueError("Graph shards changed after full audit")
    checkpoint = load_checkpoint(args.checkpoint)
    # A checkpoint without schema metadata cannot cryptographically prove its
    # training-data provenance. Require its serialized architecture to match;
    # source run provenance remains a documented user responsibility.
    model = PCBAClassifier(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    checkpoint.update(
        task="pcba",
        schema_hash=SCHEMA_HASH,
        legacy_import={
            "source_checkpoint_sha256": sha256(args.checkpoint),
            "audit_sha256": sha256(args.verified_audit),
            "provenance_note": "User must confirm this checkpoint was trained on the audited data",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(checkpoint, args.output)
    print(
        f"Created transfer-only checkpoint {args.output}; legacy optimizer resume is not supported"
    )


if __name__ == "__main__":
    main()
