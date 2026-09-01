"""LightningModule : boucle d'entraînement, optimiseur, logging des métriques."""

from __future__ import annotations

from typing import Any

import lightning as L
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from kinoscope.eval.metrics import compute_metrics
from kinoscope.models.model import DTILoss, HybridDTIModel, ModelConfig


class KinoScopeModule(L.LightningModule):
    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        warmup_pct: float = 0.05,
        mse_warmup_steps: int = 500,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model_cfg"])
        self.cfg = model_cfg or ModelConfig()
        self.model = HybridDTIModel(self.cfg)
        self.criterion = DTILoss(self.cfg)
        self._val_buffer: list[tuple[np.ndarray, np.ndarray]] = []

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.model(batch)

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        out = self.model(batch)

        # Warmup : les premiers pas utilisent une MSE pure. Optimiser la
        # log-vraisemblance gaussienne dès le pas 0 laisse le modèle réduire la
        # perte en gonflant sigma sans jamais améliorer mu.
        if stage == "train" and self.global_step < self.hparams.mse_warmup_steps:
            loss = torch.nn.functional.mse_loss(out["affinity_mu"], batch["p_affinity"])
            losses = {"loss": loss, "loss_affinity": loss.detach()}
        else:
            losses = self.criterion(out, batch)

        bs = batch["p_affinity"].size(0)
        for k, v in losses.items():
            self.log(f"{stage}/{k}", v, on_step=(stage == "train"), on_epoch=True, batch_size=bs, prog_bar=(k == "loss"))

        if stage != "train":
            self._val_buffer.append(
                (batch["p_affinity"].detach().cpu().numpy(), out["affinity_mu"].detach().cpu().numpy())
            )
        return losses["loss"]

    def training_step(self, batch, batch_idx):  # noqa: ANN001, ANN201
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):  # noqa: ANN001, ANN201
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):  # noqa: ANN001, ANN201
        return self._step(batch, "test")

    def _flush(self, stage: str) -> None:
        if not self._val_buffer:
            return
        y = np.concatenate([a for a, _ in self._val_buffer])
        yhat = np.concatenate([b for _, b in self._val_buffer])
        self._val_buffer.clear()
        for k, v in compute_metrics(y, yhat, activity_threshold=self.cfg.activity_threshold).to_dict().items():
            self.log(f"{stage}/{k}", float(v), prog_bar=(k == "spearman_rho"))

    def on_validation_epoch_end(self) -> None:
        self._flush("val")

    def on_test_epoch_end(self) -> None:
        self._flush("test")

    def configure_optimizers(self):  # noqa: ANN201
        # Pas de weight decay sur les normalisations et les biais : convention
        # standard, mesurablement meilleure sur les architectures à LayerNorm.
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or "pos_embedding" in name else decay).append(p)

        opt = AdamW(
            [{"params": decay, "weight_decay": self.hparams.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.hparams.lr,
        )
        sched = OneCycleLR(
            opt,
            max_lr=self.hparams.lr,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=self.hparams.warmup_pct,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
