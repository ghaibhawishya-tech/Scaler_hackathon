"""
RouterEnv-v1 — Core Server Implementation
==========================================
Standard OpenEnv server for remote agent evaluations.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from router_env.environment import RouterEnvironment
from router_env.models import RouterAction, RouterObservation, RouterState

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  │  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("router_env.server")

# ── App & Env ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RouterEnv-v1",
    version="2.0.0",
)

# Episode budget of $10.0, sequence of 5 tasks
env = RouterEnvironment(budget=10.0, sequence_length=5)


@app.get("/")
async def root():
    """Simple API status check."""
    return {"status": "online", "message": "RouterEnv-v1 API is operational. Visit /docs for Swagger."}


# ── Request / Response schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Body for ``POST /reset``."""
    seed: Optional[int] = Field(default=None)
    options: Optional[Dict[str, Any]] = Field(default=None)


class StepResponse(BaseModel):
    """Body returned by ``POST /step``."""
    observation: RouterObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResponse(BaseModel):
    """Body returned by ``POST /reset``."""
    observation: RouterObservation
    info: Dict[str, Any]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=ResetResponse)
def reset(body: Optional[ResetRequest] = None) -> ResetResponse:
    """Reset the environment and start a new workflow sequence."""
    seed = body.seed if body else None
    options = body.options if body else None
    try:
        obs, info = env.reset(seed=seed, options=options)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ResetResponse(observation=obs, info=info)


@app.post("/step", response_model=StepResponse)
def step(action: RouterAction) -> StepResponse:
    """Submit a routing action for the current task in the queue."""
    try:
        # Standard step returns 5 values: obs, reward, terminated, truncated, info
        obs, reward, done, truncated, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=RouterState)
def state() -> RouterState:
    """Return a snapshot of the internal metrics."""
    try:
        return env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy", "env": "RouterEnv-v1", "version": "2.0.0"}


@app.get("/info")
def info() -> Dict[str, Any]:
    return {
        "env_id": "RouterEnv-v1",
        "version": "2.0.0",
        "available_models": env.available_models(),
        "budget": 10.0,
        "sequence_length": 5
    }


def main():
    import uvicorn
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run the RouterEnv-v1 server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)), help="Port to bind to.")
    args = parser.parse_args()

    uvicorn.run("router_env.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
