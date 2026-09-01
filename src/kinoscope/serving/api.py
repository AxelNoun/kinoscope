"""API d'inférence FastAPI.

Le modèle est chargé une fois au démarrage (lifespan), pas par requête. Le cache
des embeddings de poche est un dictionnaire en mémoire adossé à Redis : le kinome
tient en ~110 Mo en fp16, donc la totalité est chargeable en RAM.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status

from kinoscope.serving.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)
MODEL_VERSION = os.getenv("KINOSCOPE_MODEL_VERSION", "0.1.0-dev")
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201, ARG001
    from kinoscope.serving.predictor import Predictor  # noqa: PLC0415

    logger.info("chargement du prédicteur (version %s)", MODEL_VERSION)
    _state["predictor"] = Predictor.from_checkpoint(
        checkpoint=os.getenv("KINOSCOPE_CHECKPOINT", "artifacts/model.ckpt"),
        pocket_lmdb=os.getenv("KINOSCOPE_POCKETS", "data/processed/pockets.lmdb"),
        conformal=os.getenv("KINOSCOPE_CONFORMAL", "artifacts/conformal.json"),
    )
    yield
    _state.clear()


app = FastAPI(
    title="KinoScope",
    version=MODEL_VERSION,
    description="Prédiction d'affinité kinase-inhibiteur avec intervalles conformes calibrés.",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz() -> HealthResponse:
    p = _state.get("predictor")
    if p is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "modèle non chargé")
    return HealthResponse(
        status="ok", model_version=MODEL_VERSION, device=str(p.device), n_targets_cached=p.n_targets
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(req: PredictionRequest) -> PredictionResponse:
    p = _state["predictor"]
    try:
        return p.predict_one(req)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"cible inconnue: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


@app.post("/predict/batch", response_model=list[PredictionResponse], tags=["inference"])
async def predict_batch(req: BatchPredictionRequest) -> list[PredictionResponse]:
    return _state["predictor"].predict_many(req.items)


@app.post("/profile/kinome", tags=["inference"])
async def profile_kinome(smiles: str, top_k: int = 20) -> dict[str, Any]:
    """Profil de sélectivité : une molécule contre toutes les kinases couvertes.
    Retourne le classement et l'entropie de Shannon du profil (score de promiscuité)."""
    return _state["predictor"].profile_kinome(smiles, top_k=top_k)
