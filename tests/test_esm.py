import hashlib

import pandas as pd
import pytest
import torch

from dta_project.precompute_esm import precompute, residue_mean


def test_special_tokens_excluded():
    hidden = torch.tensor([[[100.0, 100.0], [2.0, 4.0], [4.0, 8.0], [200.0, 200.0], [0.0, 0.0]]])
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1, 1]])
    assert torch.equal(residue_mean(hidden, attention, special), torch.tensor([[3.0, 6.0]]))


def test_local_esm_extraction_and_chunking(tmp_path):
    pytest.importorskip("transformers")
    from transformers import EsmConfig, EsmModel, EsmTokenizer

    local = tmp_path / "local_model"
    local.mkdir()
    vocab = [
        "<cls>",
        "<pad>",
        "<eos>",
        "<unk>",
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
        "<null_1>",
        "<mask>",
    ]
    (local / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = EsmTokenizer(str(local / "vocab.txt"))
    tokenizer.save_pretrained(local)
    model = EsmModel(
        EsmConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            pad_token_id=1,
            mask_token_id=32,
            max_position_embeddings=20,
        )
    )
    model.save_pretrained(local)
    data = tmp_path / "data"
    data.mkdir()
    sequence = "ACDEFGHIKLMNPQ"
    pd.DataFrame({"protein_id": ["test"], "protein_sequence": [sequence]}).to_parquet(
        data / "proteins.parquet"
    )
    precompute(
        data,
        tmp_path / "residues.pt",
        str(local),
        pooling="residues",
        device="cpu",
        max_residues=8,
        local_only=True,
    )
    cache = torch.load(tmp_path / "residues.pt", weights_only=True)
    assert cache["embeddings"]["test"].shape == (len(sequence), 16)
    assert cache["metadata"]["chunked_proteins"] == ["test"]
    assert cache["sequence_sha256"]["test"] == hashlib.sha256(sequence.encode()).hexdigest()
    precompute(
        data,
        tmp_path / "mean.pt",
        str(local),
        pooling="mean",
        device="cpu",
        max_residues=8,
        local_only=True,
    )
    mean = torch.load(tmp_path / "mean.pt", weights_only=True)["embeddings"]["test"].float()
    assert torch.allclose(mean, cache["embeddings"]["test"].float().mean(0), atol=0.002)
