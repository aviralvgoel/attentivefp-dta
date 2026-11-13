import pytest
import torch

from dta_project.smoke import run


@pytest.fixture(autouse=True)
def small_threadpool():
    torch.set_num_threads(2)


@pytest.fixture(scope="session")
def smoke_root(tmp_path_factory):
    return run(tmp_path_factory.mktemp("integration"))


@pytest.fixture
def drug_config():
    return {
        "node_dim": 45,
        "edge_dim": 14,
        "hidden_dim": 16,
        "embedding_dim": 16,
        "num_layers": 2,
        "num_timesteps": 2,
        "dropout": 0.3,
    }


@pytest.fixture
def dta_config(drug_config):
    return {
        "drug": drug_config,
        "protein_dim": 8,
        "fusion": "concat",
        "fusion_dim": 16,
        "dropout": 0.2,
        "attention_heads": 4,
    }
