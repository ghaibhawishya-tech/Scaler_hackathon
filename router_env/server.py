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


from fastapi.responses import HTMLResponse

# ── Request / Response schemas ───────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Beautiful dashboard for the environment."""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RouterEnv-v1 | Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; }}
            .glass {{ background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.1); }}
            .glow-purple {{ box-shadow: 0 0 20px rgba(139, 92, 246, 0.3); }}
            .accent-gradient {{ background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); }}
        </style>
    </head>
    <body class="min-h-screen py-12 px-4 md:px-12 bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-indigo-900/20 via-slate-900 to-slate-900">
        
        <!-- Header -->
        <div class="max-w-6xl mx-auto mb-12 text-center md:text-left flex flex-col md:flex-row justify-between items-end gap-6">
            <div>
                <span class="px-3 py-1 text-xs font-bold uppercase tracking-widest bg-indigo-500/20 text-indigo-400 rounded-full border border-indigo-500/30">OpenEnv v2.0 Compliant</span>
                <h1 class="text-5xl font-bold mt-4 tracking-tight">Router<span class="text-transparent bg-clip-text accent-gradient">Env</span>-v1</h1>
                <p class="text-slate-400 mt-3 text-lg font-light max-w-2xl">Intelligent LLM Orchestration Simulation. Solve the cost-performance paradox with agentic reasoning.</p>
            </div>
            <div class="flex gap-3">
                <a href="/docs" class="px-6 py-3 rounded-xl glass hover:bg-slate-800 transition-all font-semibold flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span> Swagger Docs
                </a>
                <a href="/info" class="px-6 py-3 rounded-xl accent-gradient glow-purple hover:scale-105 transition-all font-semibold">Environment Spec</a>
            </div>
        </div>

        <div class="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
            
            <!-- System Status -->
            <div class="md:col-span-2 space-y-8">
                
                <!-- Metrics Grid -->
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div class="glass p-6 rounded-3xl">
                        <p class="text-slate-400 text-sm font-semibold uppercase tracking-wider">Episode Budget</p>
                        <p class="text-4xl font-bold mt-1 text-emerald-400">$10.00</p>
                    </div>
                    <div class="glass p-6 rounded-3xl">
                        <p class="text-slate-400 text-sm font-semibold uppercase tracking-wider">Tasks / Seq</p>
                        <p class="text-4xl font-bold mt-1">5 Steps</p>
                    </div>
                    <div class="glass p-6 rounded-3xl">
                        <p class="text-slate-400 text-sm font-semibold uppercase tracking-wider">Logic Model</p>
                        <p class="text-xl font-mono mt-2 text-indigo-300">Stochastic</p>
                    </div>
                </div>

                <!-- Model Tiers -->
                <div class="glass p-8 rounded-[2.5rem] overflow-hidden relative">
                    <div class="absolute top-0 right-0 p-8 opacity-10">
                        <svg class="w-32 h-32" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path></svg>
                    </div>
                    <h3 class="text-2xl font-bold mb-6 flex items-center gap-3">
                        <svg class="w-6 h-6 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                        Model Roster
                    </h3>
                    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        <div class="bg-slate-800/50 p-6 rounded-2xl border border-white/5">
                            <p class="font-bold text-lg text-emerald-400">small-fast</p>
                            <div class="mt-4 space-y-2 text-sm">
                                <div class="flex justify-between"><span>Power:</span><span class="font-mono">0.3</span></div>
                                <div class="flex justify-between"><span>Cost:</span><span class="font-mono text-emerald-400">$0.005</span></div>
                                <div class="flex justify-between"><span>Latency:</span><span class="font-mono italic text-slate-500">0.1s</span></div>
                            </div>
                        </div>
                        <div class="bg-indigo-500/10 p-6 rounded-2xl border border-indigo-500/20 shadow-[0_0_15px_rgba(99,102,241,0.1)]">
                            <p class="font-bold text-lg text-indigo-300">medium-balanced</p>
                            <div class="mt-4 space-y-2 text-sm">
                                <div class="flex justify-between"><span>Power:</span><span class="font-mono">0.7</span></div>
                                <div class="flex justify-between"><span>Cost:</span><span class="font-mono text-emerald-400">$0.080</span></div>
                                <div class="flex justify-between"><span>Latency:</span><span class="font-mono italic text-slate-500">0.4s</span></div>
                            </div>
                        </div>
                        <div class="bg-purple-500/10 p-6 rounded-2xl border border-purple-500/20 shadow-[0_0_15px_rgba(168,85,247,0.1)]">
                            <p class="font-bold text-lg text-purple-400">large-reasoning</p>
                            <div class="mt-4 space-y-2 text-sm">
                                <div class="flex justify-between"><span>Power:</span><span class="font-mono">1.0</span></div>
                                <div class="flex justify-between"><span>Cost:</span><span class="font-mono text-emerald-400">$0.800</span></div>
                                <div class="flex justify-between"><span>Latency:</span><span class="font-mono italic text-slate-500">1.5s</span></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Reward Function -->
                <div class="glass p-8 rounded-[2.5rem]">
                    <h3 class="text-2xl font-bold mb-6 flex items-center gap-3">
                        <svg class="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        Reward Signal
                    </h3>
                    <div class="p-6 bg-slate-950/50 rounded-2xl font-mono text-indigo-400 text-lg border border-white/5">
                        Reward = (2.5 × α) - (0.4 × cost) - overkill_penalty
                    </div>
                    <ul class="mt-6 space-y-3 text-slate-400 text-sm italic">
                        <li>• α = Performance score (0.0 to 1.0) awarded by LLM Judge</li>
                        <li>• Overkill Penalty = -1.0 if 'large-reasoning' used for easy tasks</li>
                        <li>• Stochasticity ensures 15% failure rate even on perfect matches</li>
                    </ul>
                </div>
            </div>

            <!-- Side Cards -->
            <div class="space-y-8">
                <div class="glass p-8 rounded-[2.5rem] bg-indigo-500/5">
                    <h4 class="text-xl font-bold mb-4">Environment Setup</h4>
                    <p class="text-slate-400 text-sm leading-relaxed mb-6">Connect your agent via the standard OpenEnv SDK. Host defaults to 7860 on Hugging Face Spaces.</p>
                    <div class="p-4 bg-slate-950/80 rounded-xl font-mono text-xs text-indigo-300 leading-6 border border-white/5">
                        env = RouterEnv(budget=10.0)<br>
                        obs = env.reset()<br>
                        action = agent.decide(obs)<br>
                        obs, reward, done, info = env.step(action)
                    </div>
                </div>



                <div class="glass p-8 rounded-[2.5rem] flex flex-col items-center text-center">
                    <div class="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mb-4">
                         <svg class="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"></path></svg>
                    </div>
                    <h4 class="text-lg font-bold">Open Source</h4>
                    <p class="text-slate-400 text-xs mt-2 italic">Designed for Scaler Hackathon 2026. Built with FastAPI, Pydantic & OpenAI.</p>
                </div>
            </div>
        </div>

        <footer class="max-w-6xl mx-auto mt-20 pt-8 border-t border-white/5 text-center text-slate-500 text-xs uppercase tracking-widest">
            &copy; 2026 RouterEnv-v1 Team • Sprinting Snails Collaboration
        </footer>
    </body>
    </html>
    """


# ── Request / Response schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Body for ``POST /reset``."""
    seed: Optional[int] = Field(default=None)


class StepResponse(BaseModel):
    """Body returned by ``POST /step``."""
    observation: RouterObservation
    reward: float
    done: bool
    info: Dict[str, Any]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=RouterObservation)
def reset(body: ResetRequest) -> RouterObservation:
    """Reset the environment and start a new workflow sequence."""
    try:
        obs, info = env.reset(seed=body.seed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return obs


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
