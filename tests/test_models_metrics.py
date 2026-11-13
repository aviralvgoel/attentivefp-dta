import itertools

import numpy as np
import pytest
import torch

from dta_project.chemistry import SCHEMA_HASH, smiles_to_graph
from dta_project.common import load_checkpoint
from dta_project.dta import collate_pairs
from dta_project.metrics import concordance_index, masked_bce, regression_metrics
from dta_project.models import DTARegressor, transfer_encoder


def make_batch():
    return collate_pairs(
        [
            (smiles_to_graph("CCO"), torch.randn(5, 8), 1.0, 20),
            (smiles_to_graph("[Na+]"), torch.randn(2, 8), 2.0, 100),
        ]
    )


def test_masked_nan_loss_and_zero_gradients():
    logits = torch.zeros(2, 3, requires_grad=True)
    y = torch.tensor([[1.0, float("nan"), 0.0], [0.0, 1.0, 0.0]])
    mask = torch.tensor([[True, False, True], [False, True, False]])
    loss = masked_bce(logits, y, mask)
    loss.backward()
    assert loss.item() == pytest.approx(np.log(2))
    assert (logits.grad[~mask] == 0).all()


def test_no_observed_labels_still_differentiable():
    logits = torch.randn(2, 3, requires_grad=True)
    loss = masked_bce(
        logits, torch.full_like(logits, float("nan")), torch.zeros_like(logits, dtype=torch.bool)
    )
    loss.backward()
    assert loss.item() == 0 and (logits.grad == 0).all()


def test_loss_shape_guard():
    with pytest.raises(ValueError, match="Expected equal"):
        masked_bce(torch.zeros(2, 3), torch.zeros(6), torch.ones(6, dtype=torch.bool))


@pytest.mark.parametrize("fusion", ["concat", "cross_attention"])
def test_model_forward_backward(dta_config, fusion):
    dta_config["fusion"] = fusion
    model = DTARegressor(dta_config)
    prediction = model(make_batch())
    assert prediction.shape == (2,) and torch.isfinite(prediction).all()
    prediction.sum().backward()
    assert model.encoder.lin1.weight.grad is not None


def test_frozen_encoder_no_dropout_or_updates(dta_config):
    model = DTARegressor(dta_config)
    model.freeze_encoder(True)
    model.train()
    assert model.training and not model.encoder.training
    batch = make_batch()
    first = model.encoder.encode(batch["graph"])
    second = model.encoder.encode(batch["graph"])
    assert torch.equal(first, second)
    model(batch).sum().backward()
    assert all(p.grad is None for p in model.encoder.parameters())


def test_attention_padding_invariance(dta_config):
    dta_config["fusion"] = "cross_attention"
    model = DTARegressor(dta_config).eval()
    batch = make_batch()
    original, attention = model(batch, return_attention=True)
    batch["protein"][1, 2:] = 9999  # masked residues must not affect predictions
    changed = model(batch)
    assert torch.allclose(original, changed, atol=1e-6)
    assert (attention[1, :, 2:] == 0).all()


def test_transfer_and_schema_guard(smoke_root, drug_config):
    payload = load_checkpoint(smoke_root / "pretrain/best.pt")
    cfg = {
        "drug": drug_config,
        "protein_dim": 16,
        "fusion": "concat",
        "fusion_dim": 16,
        "dropout": 0.1,
        "attention_heads": 4,
    }
    model = DTARegressor(cfg)
    transfer_encoder(model, smoke_root / "pretrain/best.pt", SCHEMA_HASH)
    assert torch.equal(model.encoder.lin1.weight, payload["model_state"]["encoder.lin1.weight"])
    with pytest.raises(ValueError, match="schema"):
        transfer_encoder(model, smoke_root / "pretrain/best.pt", "wrong_schema")


@pytest.mark.parametrize("seed", range(10))
def test_fenwick_ci_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    y, p = rng.integers(0, 5, 40), rng.integers(0, 5, 40)
    num, den = 0.0, 0
    for a, b in itertools.combinations(range(len(y)), 2):
        if y[a] == y[b]:
            continue
        den += 1
        num += 0.5 if p[a] == p[b] else float((y[a] - y[b]) * (p[a] - p[b]) > 0)
    assert concordance_index(y, p) == pytest.approx(num / den)


def test_undefined_ci_and_correlations():
    metrics = regression_metrics([1, 1], [2, 2])
    assert metrics["ci"] is None and metrics["pearson"] is None
