"""Check dependencies, GPU kernels, and an actual AttentiveFP forward/backward."""

import argparse
import json

import torch
from torch_geometric.data import Batch

from .chemistry import EDGE_DIM, NODE_DIM, smiles_to_graph
from .common import choose_device, environment
from .models import PCBAClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Require and exercise CUDA")
    args = parser.parse_args()
    device = choose_device("cuda" if args.cuda else "auto")
    torch.set_num_threads(2)
    print(json.dumps(environment(), indent=2))
    if device.type == "cuda":
        print("GPU capability:", torch.cuda.get_device_capability(device))
        print("Compiled architectures:", torch.cuda.get_arch_list())
    cfg = {
        "node_dim": NODE_DIM,
        "edge_dim": EDGE_DIM,
        "num_tasks": 3,
        "hidden_dim": 16,
        "embedding_dim": 16,
        "num_layers": 2,
        "num_timesteps": 2,
        "dropout": 0.0,
    }
    model = PCBAClassifier(cfg).to(device)
    graph = Batch.from_data_list([smiles_to_graph("CCO"), smiles_to_graph("[Na+]")]).to(device)
    pred = model(graph)
    pred.square().mean().backward()
    assert pred.shape == (2, 3) and torch.isfinite(pred).all()
    print(f"PASS: real AttentiveFP forward/backward on {device}, including a zero-bond molecule")


if __name__ == "__main__":
    main()
