"""FastAPI inference service.

The model is loaded once at startup (lifespan), not per request. The pocket
embedding cache is an in-memory dictionary backed by Redis: the whole kinome fits
in ~110 MB at fp16, so all of it can be held in RAM.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

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
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    # `predictor.py` is delivered at step 7.2 of the MASTER_PLAN. This ignore
    # becomes unnecessary the moment it exists, and `warn_unused_ignores` will
    # then flag it.
    from kinoscope.serving.predictor import Predictor  # type: ignore[import-not-found]

    logger.info("loading predictor (version %s)", MODEL_VERSION)
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
    description="Kinase-inhibitor affinity prediction with calibrated conformal intervals.",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz() -> HealthResponse:
    p = _state.get("predictor")
    if p is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
    return HealthResponse(
        status="ok", model_version=MODEL_VERSION, device=str(p.device), n_targets_cached=p.n_targets
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(req: PredictionRequest) -> PredictionResponse:
    p = _state["predictor"]
    try:
        return cast(PredictionResponse, p.predict_one(req))
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown target: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


@app.post("/predict/batch", response_model=list[PredictionResponse], tags=["inference"])
async def predict_batch(req: BatchPredictionRequest) -> list[PredictionResponse]:
    return cast(list[PredictionResponse], _state["predictor"].predict_many(req.items))


@app.post("/profile/kinome", tags=["inference"])
async def profile_kinome(smiles: str, top_k: int = 20) -> dict[str, Any]:
    """Selectivity profile: one molecule against every covered kinase.
    Returns the ranking and the Shannon entropy of the profile (a promiscuity
    score)."""
    return cast(dict[str, Any], _state["predictor"].profile_kinome(smiles, top_k=top_k))
