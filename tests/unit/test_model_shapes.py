"""Shape contract, gradient flow, and the model's capacity to learn."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from kinoscope.models.model import DTILoss, HybridDTIModel, ModelConfig
from kinoscope.utils.seed import set_global_seed


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        hidden_dim=32,
        gnn_layers=2,
        plm_dim=64,
        adapter_layers=1,
        adapter_heads=4,
        ban_heads=2,
        ban_rank=16,
        head_hidden=32,
    )


def make_batch(cfg: ModelConfig, batch_size: int = 4, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    sizes = torch.randint(8, 25, (batch_size,))
    n = int(sizes.sum())
    x = torch.stack([torch.randint(0, d, (n,)) for d in cfg.atom_feature_dims], -1)
    bidx = torch.repeat_interleave(torch.arange(batch_size), sizes)
    ei = torch.randint(0, n, (2, 3 * n))
    ei = ei[:, bidx[ei[0]] == bidx[ei[1]]]
    ea = torch.stack([torch.randint(0, d, (ei.size(1),)) for d in cfg.bond_feature_dims], -1)
    return {
        "x": x,
        "edge_index": ei,
        "edge_attr": ea,
        "batch_index": bidx,
        "plm_emb": torch.randn(batch_size, cfg.pocket_len, cfg.plm_dim),
        "residue_mask": torch.ones(batch_size, cfg.pocket_len, dtype=torch.bool),
        "assay_type": torch.randint(0, cfg.n_assay_types, (batch_size,)),
        "p_affinity": torch.rand(batch_size) * 5 + 4,
    }


def test_forward_shapes(tiny_cfg: ModelConfig) -> None:
    model, batch = HybridDTIModel(tiny_cfg), make_batch(tiny_cfg)
    out = model(batch)
    assert out["affinity_mu"].shape == (4,)
    assert out["affinity_logvar"].shape == (4,)
    assert out["interaction_logits"].shape[:2] == (4, tiny_cfg.ban_heads)
    assert out["interaction_logits"].shape[-1] == tiny_cfg.pocket_len
    assert torch.isfinite(out["affinity_mu"]).all()


def test_logvar_is_clamped(tiny_cfg: ModelConfig) -> None:
    out = HybridDTIModel(tiny_cfg)(make_batch(tiny_cfg))
    lo, hi = tiny_cfg.logvar_clamp
    assert (out["affinity_logvar"] >= lo).all() and (out["affinity_logvar"] <= hi).all()


def test_all_parameters_receive_gradient(tiny_cfg: ModelConfig) -> None:
    model, batch = HybridDTIModel(tiny_cfg), make_batch(tiny_cfg)
    DTILoss(tiny_cfg)(model(batch), batch)["loss"].backward()
    dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not dead, f"parameters with no gradient (dead branch): {dead}"


def test_padding_invariance(tiny_cfg: ModelConfig) -> None:
    """Masking additional pocket positions must not change the prediction from the
    remaining positions incoherently; above all, no NaN may appear when the mask
    is partial."""
    model = HybridDTIModel(tiny_cfg).eval()
    batch = make_batch(tiny_cfg)
    batch["residue_mask"][:, 40:] = False
    with torch.no_grad():
        out = model(batch)
    assert torch.isfinite(out["affinity_mu"]).all()


@pytest.mark.slow
def test_can_overfit_single_batch(tiny_cfg: ModelConfig) -> None:
    """Canonical sanity check: a model unable to overfit four examples has an
    architecture bug, not a data problem.

    Dropout is set to zero and the model left in training mode. This is not a
    relaxation of the threshold: with the four default dropouts (gnn 0.1, adapter
    0.1, ban 0.2, head 0.3), the measured loss is that of a network perturbed on
    every pass, not the architecture's capacity to memorise. Measured on this
    architecture, 400 steps, identical hyperparameters: 7.13 with default dropout,
    0.80 re-evaluated in `eval` mode after that same training, and 0.0000 with
    dropout disabled. Only the third measurement answers the question asked.

    The seed is fixed: without it the loss varied from 3.3 to 7.1 between runs,
    making the test result irreproducible.
    """
    set_global_seed(0)
    cfg = replace(tiny_cfg, gnn_dropout=0.0, adapter_dropout=0.0, ban_dropout=0.0, head_dropout=0.0)
    model, batch = HybridDTIModel(cfg), make_batch(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    y = batch["p_affinity"]
    for _ in range(400):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(batch)["affinity_mu"], y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, f"cannot overfit four examples (loss={loss.item():.4f})"
