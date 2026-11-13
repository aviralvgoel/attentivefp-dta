"""Compatible PCBA encoder and controlled DTA fusion variants."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.utils import to_dense_batch

from .common import load_checkpoint


class DrugEncoder(AttentiveFP):
    def encode(self, batch):
        return super().forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    def with_atoms(self, batch):
        """Expose final atom states without changing PyG's encoder parameter names.

        The tested PyG AttentiveFP implementation applies ReLU after the final
        atom GRU. A temporary hook captures that state while the original global
        readout runs normally. Not intended for torch.compile or DDP.
        """
        states = []
        last_gru = self.atom_grus[-1] if len(self.atom_grus) else self.gru
        handle = last_gru.register_forward_hook(
            lambda module, inputs, output: states.append(output)
        )
        try:
            embedding = self.encode(batch)
        finally:
            handle.remove()
        atoms, atom_mask = to_dense_batch(states[0].relu(), batch.batch)
        return embedding, atoms, atom_mask


def encoder_from_config(config):
    if min(config["num_layers"], config["num_timesteps"]) < 1:
        raise ValueError("At least one message-passing layer and readout timestep required")
    return DrugEncoder(
        in_channels=config["node_dim"],
        edge_dim=config["edge_dim"],
        hidden_channels=config["hidden_dim"],
        out_channels=config["embedding_dim"],
        num_layers=config["num_layers"],
        num_timesteps=config["num_timesteps"],
        dropout=config["dropout"],
    )


class PCBAClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = encoder_from_config(config)
        e, h, drop = config["embedding_dim"], config["hidden_dim"], config["dropout"]
        # Same head parameter layout as the previous PCBA demo for legacy loading.
        self.head = nn.Sequential(
            nn.LayerNorm(e),
            nn.Linear(e, e),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(e, h),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(h, config["num_tasks"]),
        )

    def forward(self, graph):
        return self.head(self.encoder.encode(graph))


def masked_mean(features, mask):
    if features.shape[:2] != mask.shape or not mask.any(dim=1).all():
        raise ValueError("Empty or incorrectly shaped protein/atom mask")
    return (features * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)


class DTARegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = encoder_from_config(config["drug"])
        width, dropout = config["fusion_dim"], config["dropout"]
        self.fusion = config["fusion"]
        self.drug_projection = nn.Sequential(
            nn.LayerNorm(config["drug"]["embedding_dim"]),
            nn.Linear(config["drug"]["embedding_dim"], width),
        )
        self.protein_projection = nn.Sequential(
            nn.LayerNorm(config["protein_dim"]), nn.Linear(config["protein_dim"], width)
        )
        if self.fusion == "cross_attention":
            if width % config["attention_heads"]:
                raise ValueError("fusion_dim must be divisible by attention_heads")
            self.atom_projection = nn.Linear(config["drug"]["hidden_dim"], width)
            self.attention = nn.MultiheadAttention(
                width, config["attention_heads"], dropout=dropout, batch_first=True
            )
            self.interaction_norm = nn.LayerNorm(width)
        elif self.fusion != "concat":
            raise ValueError("fusion must be concat or cross_attention")
        self.head = nn.Sequential(
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, 1),
        )
        self.encoder_frozen = False

    def freeze_encoder(self, freeze):
        self.encoder_frozen = freeze
        self.encoder.requires_grad_(not freeze)
        if freeze:
            self.encoder.eval()  # frozen means deterministic features, including no dropout

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, batch, return_attention=False):
        protein = self.protein_projection(batch["protein"])
        p = masked_mean(protein, batch["protein_mask"])
        attention = None
        if self.fusion == "concat":
            d = self.drug_projection(self.encoder.encode(batch["graph"]))
        else:
            global_drug, atoms, atom_mask = self.encoder.with_atoms(batch["graph"])
            query = self.atom_projection(atoms)
            context, attention = self.attention(
                query,
                protein,
                protein,
                key_padding_mask=~batch["protein_mask"],
                need_weights=return_attention,
            )
            interaction = self.interaction_norm(query + context)
            d = self.drug_projection(global_drug) + masked_mean(interaction, atom_mask)
            if return_attention:
                attention = attention * atom_mask.unsqueeze(-1) * batch["protein_mask"].unsqueeze(1)
        fused = torch.cat((d, p, d * p, torch.abs(d - p)), dim=-1)
        prediction = self.head(fused).squeeze(-1)
        return (prediction, attention) if return_attention else prediction


def transfer_encoder(model, checkpoint_path, schema_hash):
    source = load_checkpoint(checkpoint_path)
    if source.get("task", "pcba") != "pcba":
        raise ValueError("Transfer source must be a PCBA checkpoint")
    if source.get("schema_hash") != schema_hash:
        raise ValueError(
            "PCBA and DTA feature schema is unknown or mismatched. See docs/LEGACY_PCBA.md."
        )
    state = {
        k.removeprefix("encoder."): v
        for k, v in source["model_state"].items()
        if k.startswith("encoder.")
    }
    if not state:
        raise ValueError("No encoder.* weights in checkpoint")
    # No strict=False escape hatch: architecture mismatches must be explicit.
    model.encoder.load_state_dict(state, strict=True)
    return source
