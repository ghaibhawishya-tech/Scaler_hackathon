"""
RouterEnv-v1 — High-Fidelity RL Environment (v2.3)
==================================================
Features a 15-task dynamic stream with Agent Graders.
Strictly calibrated for (0.1, 0.9) compliance.
"""

from __future__ import annotations
import os
import json
import random
import logging
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv

from .models import (
    ModelSpec,
    RouterAction,
    RouterObservation,
    RouterState,
    TaskDefinition,
)

logger = logging.getLogger("router_env.environment")
load_dotenv()

# ── Dynamic Agent Grader (LLM-as-a-Judge) ───────────────────────────────────

JUDGE_SYSTEM_PROMPT = """
You are a 'RouterEnv' Judge. Evaluate if the selected model can solve the task.
You must return only JSON.

SCORING RUBRIC (0.01 to 0.99) - STRICTLY EXCLUDE 0.0 AND 1.0:
0.99: Perfect choice. Frontier capability for hard tasks, or efficient for easy tasks.
0.65: Capable but risky or overkill.
0.01: Complete mismatch or total failure.

JSON SCHEMA:
{"performance_score": float, "reasoning": "string"}
"""

# ── The Hidden Task Catalogue (15 Tasks) ───────────────────────────────────

TASK_CATALOGUE: Dict[str, TaskDefinition] = {
    "sentiment": TaskDefinition(task_id="sentiment", description="Classify the sentiment of 5 customer reviews.", complexity=0.15, tokens=50),
    "spam_filter": TaskDefinition(task_id="spam_filter", description="Check if this email is spam: 'Win $1000 prize now!'", complexity=0.1, tokens=30),
    "regex_extract": TaskDefinition(task_id="regex_extract", description="Extract all dates in YYYY-MM-DD from plain text.", complexity=0.25, tokens=100),
    "markdown_gen": TaskDefinition(task_id="markdown_gen", description="Convert JSON object into a clean Markdown table.", complexity=0.2, tokens=80),
    "data_clean": TaskDefinition(task_id="data_clean", description="Remove duplicates and fix capitalization in list.", complexity=0.3, tokens=200),
    "refactor_mono": TaskDefinition(task_id="refactor_mono", description="Break down a 150-line monolithic function.", complexity=0.5, tokens=400),
    "sql_logic": TaskDefinition(task_id="sql_logic", description="Identify the logic flaw in this SQL query.", complexity=0.6, tokens=300),
    "unit_test_gen": TaskDefinition(task_id="unit_test_gen", description="Generate Pytest unit tests for a Python class.", complexity=0.55, tokens=500),
    "tech_summary": TaskDefinition(task_id="tech_summary", description="Summarize a technical whitepaper for business.", complexity=0.65, tokens=1500),
    "support_reply": TaskDefinition(task_id="support_reply", description="Draft a response to a frustrated customer.", complexity=0.45, tokens=250),
    "security_audit": TaskDefinition(task_id="security_audit", description="Analyze FastAPI endpoints for OWASP flaws.", complexity=0.9, tokens=1200, criticality=2.0),
    "legal_contract": TaskDefinition(task_id="legal_contract", description="Translate MSA contract EN to ZH.", complexity=1.0, tokens=3000, criticality=2.5),
    "arch_review": TaskDefinition(task_id="arch_review", description="Review AWS/Azure architecture for vulnerabilities.", complexity=0.85, tokens=2000, criticality=1.8),
    "api_design": TaskDefinition(task_id="api_design", description="Design RESTful API for complex fintech ledger.", complexity=0.95, tokens=1000, criticality=2.0),
    "pii_obfuscate": TaskDefinition(task_id="pii_obfuscate", description="Redact all PII from medical chat transcripts.", complexity=0.8, tokens=1800, criticality=2.2),
}

MODEL_ROSTER: Dict[str, ModelSpec] = {
    "small-fast": ModelSpec(name="small-fast", power=0.3, cost=0.005),
    "medium-balanced": ModelSpec(name="medium-balanced", power=0.7, cost=0.08),
    "large-reasoning": ModelSpec(name="large-reasoning", power=1.0, cost=0.80),
}


class RouterEnvironment:
    def __init__(self, budget: float = 10.0, sequence_length: int = 5) -> None:
        self._budget = budget
        self._sequence_length = sequence_length
        self._state: Optional[RouterState] = None
        self._done = True
        
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
        self._mock_mode = (not api_key or "your_token" in api_key)
        
        if not self._mock_mode:
            base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
            self._model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
            self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=10.0)
        else:
            self._client = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[RouterObservation, Dict[str, Any]]:
        self._seed = seed if seed is not None else random.randint(0, 1000000)
        random.seed(self._seed)
        
        all_ids = list(TASK_CATALOGUE.keys())
        task_queue = random.choices(all_ids, k=self._sequence_length)

        self._state = RouterState(
            current_task_index=0, total_cost=0.0, successes=0,
            failures=0, latency_total=0.0, episode_budget=self._budget,
            task_queue=task_queue
        )
        self._done = False
        self._last_perf = 0.0
        return self._get_current_obs(), {}

    def step(self, action: RouterAction) -> Tuple[RouterObservation, float, bool, bool, Dict[str, Any]]:
        if self._state is None or self._done:
            raise RuntimeError("Episode is over. Call reset().")

        task_id = self._state.task_queue[self._state.current_task_index]
        task = TASK_CATALOGUE[task_id]
        model = MODEL_ROSTER[action.selected_model]

        # 1. Evaluate
        if self._mock_mode:
            if model.power >= task.complexity: score = 0.99
            elif model.power >= task.complexity - 0.2: score = 0.65
            else: score = 0.01
            reasoning = f"Heuristic Score for {task_id}"
        else:
            try:
                score, reasoning = self._evaluate_with_agent(task.description, model)
            except:
                score, reasoning = 0.01, "Grader failed"

        # 2. Strict (0.01, 0.99) Constraint Enforcement
        score = max(0.01, min(0.99, score))
        
        # 3. Normalized Reward Calculation (0.01, 0.99)
        cost_efficiency = 1.0 - (model.cost / 0.81)
        reward = (0.7 * score) + (0.3 * cost_efficiency)
        reward = max(0.01, min(0.99, reward)) 

        self._state.current_task_index += 1
        terminated = (self._state.current_task_index >= len(self._state.task_queue))
        self._done = terminated
        
        obs = self._get_current_obs(f"Grader Verdict: {reasoning}")
        info = {
            "performance_score": round(score, 2),
            "task_score": round(score, 2),
            "reasoning": reasoning,
            "task_id": task_id
        }
        return obs, round(reward, 3), terminated, False, info

    def _evaluate_with_agent(self, description: str, model: ModelSpec) -> Tuple[float, str]:
        user_prompt = f"TASK: {description}\nMODEL: {model.name}\nPOWER: {model.power}\nScore this choice."
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content)
        return float(data.get("performance_score", 0.01)), str(data.get("reasoning", ""))

    def _get_current_obs(self, message: str = "") -> RouterObservation:
        idx = self._state.current_task_index
        if self._state is None or idx >= len(self._state.task_queue):
            return RouterObservation(
                task_description="EMPTY", estimated_tokens=0, budget_remaining=0.0, 
                tasks_left=0, last_performance_score=0.0, message="Done."
            )
        task_id = self._state.task_queue[idx]
        task = TASK_CATALOGUE[task_id]
        return RouterObservation(
            task_description=task.description, estimated_tokens=task.tokens,
            budget_remaining=max(0.0, self._state.episode_budget - self._state.total_cost),
            tasks_left=len(self._state.task_queue) - idx,
            last_performance_score=self._last_perf, task_id=task_id, message=message
        )

    def available_models(self) -> List[str]: return list(MODEL_ROSTER.keys())
    def state(self) -> RouterState: return self._state.model_copy(deep=True)
    def close(self): pass
