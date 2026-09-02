"""Contrat de shapes, flux de gradient, et capacité d'apprentissage du modèle."""

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
    assert not dead, f"paramètres sans gradient (branche morte): {dead}"


def test_padding_invariance(tiny_cfg: ModelConfig) -> None:
    """Masquer des positions de poche supplémentaires ne doit pas modifier la
    prédiction issue des positions restantes de façon incohérente : on vérifie
    surtout qu'aucun NaN n'apparaît quand le masque est partiel."""
    model = HybridDTIModel(tiny_cfg).eval()
    batch = make_batch(tiny_cfg)
    batch["residue_mask"][:, 40:] = False
    with torch.no_grad():
        out = model(batch)
    assert torch.isfinite(out["affinity_mu"]).all()


@pytest.mark.slow
def test_can_overfit_single_batch(tiny_cfg: ModelConfig) -> None:
    """Sanity check canonique : un modèle incapable de sur-apprendre 4 exemples
    a un bug d'architecture, pas un problème de données.

    Le dropout est mis à zéro et le modèle laissé en mode entraînement. Ce n'est
    pas un aménagement du seuil : avec les quatre dropout par défaut (gnn 0.1,
    adapter 0.1, ban 0.2, head 0.3), la perte mesurée est celle d'un réseau
    bruité à chaque passe, pas la capacité de l'architecture à mémoriser. Mesuré
    sur cette architecture, 400 pas, mêmes hyperparamètres : 7.13 avec le dropout
    par défaut, 0.80 en réévaluant en mode `eval` après ce même entraînement, et
    0.0000 avec le dropout désactivé. Seule la troisième mesure répond à la
    question posée.

    La graine est fixée : sans elle, la perte variait de 3.3 à 7.1 d'une
    exécution à l'autre, ce qui rendait le résultat du test non reproductible.
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
    assert loss.item() < 0.05, f"impossible de sur-apprendre 4 exemples (loss={loss.item():.4f})"
