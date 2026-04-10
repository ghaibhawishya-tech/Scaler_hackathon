"""
RouterEnv-v1 — Advanced LLM Routing Simulation
==============================================
This environment models the real-world challenge of building a persistent
orchestrator that routes a stream of varying-complexity tasks to the optimal 
LLM backend while balancing cost, quality, and latency.
"""

from __future__ import annotations
from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field


# ── Observation ──────────────────────────────────────────────────────────────

class RouterObservation(BaseModel):
    """What the agent perceives at each time-step."""
    task_description: str = Field(..., description="Human-readable description of the current task.")
    estimated_tokens: int = Field(..., description="Estimated token size of the payload.")
    budget_remaining: float = Field(..., description="Remaining episode budget in US dollars.")
    tasks_left: int = Field(..., description="Number of tasks remaining in the queue.")
    last_performance_score: float = Field(default=0.50, description="Score (strictly between 0 and 1) given by the Agent Grader for the previous choice.")
    last_success: bool = Field(default=False, description="Whether the previous task was considered a success.")
    task_id: str = Field(default="", description="Short identifier for the current task.")
    message: str = Field(default="", description="Environment feedback message.")


# ── Action ───────────────────────────────────────────────────────────────────

class RouterAction(BaseModel):
    """The agent's decision for the current task."""
    selected_model: Literal["small-fast", "medium-balanced", "large-reasoning"] = Field(
        ..., description="Which model tier to use."
    )


# ── Internal State ───────────────────────────────────────────────────────────

class RouterState(BaseModel):
    """Hidden state (not seen by the agent)."""
    current_task_index: int = Field(0)
    total_cost: float = Field(0.0)
    successes: int = Field(0)
    failures: int = Field(0)
    latency_total: float = Field(0.0)
    episode_budget: float = Field(10.0)
    task_queue: List[str] = Field(default_factory=list, description="IDs of tasks in this episode.")


# ── Helper Data Classes ──────────────────────────────────────────────────────

class TaskDefinition(BaseModel):
    """Static metadata for a task type."""
    task_id: str
    description: str
    complexity: float = Field(ge=0.0, le=1.0)
    tokens: int
    criticality: float = Field(1.0, description="Multiplier for failure penalties.")


class ModelSpec(BaseModel):
    """Static metadata for a model tier."""
    name: str
    power: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0)
    latency_base: float = Field(0.1, description="Base latency in seconds.")
