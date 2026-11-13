"""Cache frozen ESM-2 features once per exact protein sequence."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .common import atomic_save, choose_device, sha256


def residue_mean(hidden, attention_mask, special_tokens_mask):
    mask = attention_mask.bool() & ~special_tokens_mask.bool()
    if not mask.any(dim=1).all():
        raise ValueError("No amino-acid tokens after masking")
    return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)


def precompute(
    directory,
    output,
    model_name="facebook/esm2_t12_35M_UR50D",
    revision="main",
    pooling="mean",
    device="auto",
    max_residues=1000,
    local_only=False,
):
    # Lazy import: basic training/smoke tests do not require a model download.
    from transformers import AutoTokenizer, EsmModel

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Cache already exists: {output}; choose a new filename")
    proteins_path = Path(directory) / "proteins.parquet"
    proteins = pd.read_parquet(proteins_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, local_files_only=local_only
    )
    model = EsmModel.from_pretrained(model_name, revision=revision, local_files_only=local_only)
    model = model.to(choose_device(device)).eval()
    model.requires_grad_(False)
    if not 1 <= max_residues <= model.config.max_position_embeddings - 2:
        raise ValueError("max_residues exceeds the protein model's token context")
    if pooling not in ("mean", "residues"):
        raise ValueError("pooling must be mean or residues")
    embeddings, hashes, chunked = {}, {}, []
    for row in tqdm(proteins.itertuples(), total=len(proteins), desc="ESM-2"):
        sequence = row.protein_sequence
        chunks = [sequence[i : i + max_residues] for i in range(0, len(sequence), max_residues)]
        if len(chunks) > 1:
            chunked.append(row.protein_id)
        states = []
        for chunk in chunks:
            tokens = tokenizer(chunk, return_tensors="pt", return_special_tokens_mask=True)
            special = tokens.pop("special_tokens_mask").to(model.device)
            tokens = {k: v.to(model.device) for k, v in tokens.items()}
            with torch.inference_mode():
                hidden = model(**tokens).last_hidden_state
            valid = tokens["attention_mask"].bool() & ~special.bool()
            residues = hidden[0, valid[0]].float().cpu()
            if residues.shape[0] != len(chunk):
                raise ValueError("Tokenizer did not produce one token per residue")
            states.append(residues)
        states = torch.cat(states, dim=0)
        value = states.mean(0) if pooling == "mean" else states
        embeddings[row.protein_id] = value.to(torch.float16)
        hashes[row.protein_id] = hashlib.sha256(sequence.encode()).hexdigest()
    payload = {
        "metadata": {
            "model": model_name,
            "requested_revision": revision,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "pooling": pooling,
            "synthetic": False,
            "max_residues_per_chunk": max_residues,
            "chunked_proteins": chunked,
            "long_sequence_policy": "nonoverlapping independent chunks, no truncation",
            "proteins_sha256": sha256(proteins_path),
        },
        "embeddings": embeddings,
        "sequence_sha256": hashes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(payload, output)
    print(f"Cached {len(embeddings)} proteins to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument(
        "--revision", default="main", help="Pin a model commit for final experiments"
    )
    parser.add_argument("--pooling", choices=["mean", "residues"], default="mean")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-residues", type=int, default=1000)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    precompute(
        args.processed_dir,
        args.output,
        args.model,
        args.revision,
        args.pooling,
        args.device,
        args.max_residues,
        args.local_files_only,
    )


if __name__ == "__main__":
    main()
