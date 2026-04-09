"""
RouterEnv-v1 — Advanced LLM Routing Simulation (v2.1)
======================================================
This version implements the 'LLM-as-a-Judge' agent grader and features 
a 15-task dynamic catalogue with strictly hidden complexity scores.
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

SCORING RUBRIC (0.0 to 1.0):
1.0: Perfect choice. Frontier capability for hard tasks, or efficient for easy tasks.
0.7: Capable but risky. The model might succeed but is under-powered for full reliability.
0.4: Weak choice. The model is likely to fail complex reasoning.
0.1: Complete mismatch (e.g., using a tiny model for a complex security audit).

JSON SCHEMA:
{"performance_score": float, "reasoning": "string"}
"""

# ── The Hidden Task Catalogue (15 Tasks) ───────────────────────────────────

TASK_CATALOGUE: Dict[str, TaskDefinition] = {
    # EASY (Complexity 0.1 - 0.3)
    "sentiment": TaskDefinition(
        task_id="sentiment",
        description="Classify the sentiment of 5 customer reviews as positive, neutral, or negative.",
        complexity=0.15,
        tokens=50,
    ),
    "spam_filter": TaskDefinition(
        task_id="spam_filter",
        description="Check if this email is spam: 'Win $1000 prize now! Click the link below!'",
        complexity=0.1,
        tokens=30,
    ),
    "regex_extract": TaskDefinition(
        task_id="regex_extract",
        description="Extract all dates in YYYY-MM-DD format from the provided plain text log.",
        complexity=0.25,
        tokens=100,
    ),
    "markdown_gen": TaskDefinition(
        task_id="markdown_gen",
        description="Convert the provided JSON object into a clean Markdown table for documentation.",
        complexity=0.2,
        tokens=80,
    ),
    "data_clean": TaskDefinition(
        task_id="data_clean",
        description="Remove duplicates and fix capitalization in a list of 50 user-provided names.",
        complexity=0.3,
        tokens=200,
    ),

    # MEDIUM (Complexity 0.4 - 0.7)
    "refactor_mono": TaskDefinition(
        task_id="refactor_mono",
        description="Break down a 150-line monolithic Python function into smaller, modular helper functions.",
        complexity=0.5,
        tokens=400,
    ),
    "sql_logic": TaskDefinition(
        task_id="sql_logic",
        description="Identify the logic flaw in this SQL query involving multiple LEFT JOINs and a subquery.",
        complexity=0.6,
        tokens=300,
    ),
    "unit_test_gen": TaskDefinition(
        task_id="unit_test_gen",
        description="Generate comprehensive Pytest unit tests for a Python class that manages a shopping cart.",
        complexity=0.55,
        tokens=500,
    ),
    "tech_summary": TaskDefinition(
        task_id="tech_summary",
        description="Summarize a technical whitepaper on 'Zero-Knowledge Proofs' for a business-level audience.",
        complexity=0.65,
        tokens=1500,
    ),
    "support_reply": TaskDefinition(
        task_id="support_reply",
        description="Draft a professional response to a frustrated customer whose refund was processed incorrectly.",
        complexity=0.45,
        tokens=250,
    ),

    # HARD (Complexity 0.8 - 1.0)
    "security_audit": TaskDefinition(
        task_id="security_audit",
        description="Analyze a FastAPI endpoint for OWASP vulnerabilities like SQLi, XSS, and SSRF. Suggest fixes.",
        complexity=0.9,
        tokens=1200,
        criticality=2.0,
    ),
    "legal_contract": TaskDefinition(
        task_id="legal_contract",
        description="Translate a 15-page Master Service Agreement (MSA) from EN to ZH with legal accuracy.",
        complexity=1.0,
        tokens=3000,
        criticality=2.5,
    ),
    "arch_review": TaskDefinition(
        task_id="arch_review",
        description="Review a multi-cloud AWS/Azure architecture diagram described in text for single-point-of-failures.",
        complexity=0.85,
        tokens=2000,
        criticality=1.8,
    ),
    "api_design": TaskDefinition(
        task_id="api_design",
        description="Design a RESTful API contract for a complex fintech ledger system with transaction atomicity.",
        complexity=0.95,
        tokens=1000,
        criticality=2.0,
    ),
    "pii_obfuscate": TaskDefinition(
        task_id="pii_obfuscate",
        description="Identify and redact all PII from a disorganized set of medical chat transcripts while preserving context.",
        complexity=0.8,
        tokens=1800,
        criticality=2.2,
    ),
}

MODEL_ROSTER: Dict[str, ModelSpec] = {
    "small-fast": ModelSpec(name="small-fast", power=0.3, cost=0.005),
    "medium-balanced": ModelSpec(name="medium-balanced", power=0.7, cost=0.08),
    "large-reasoning": ModelSpec(name="large-reasoning", power=1.0, cost=0.80),
}


class RouterEnvironment:
    """Advanced Environment featuring LLM-as-a-Judge for all transitions."""

    def __init__(self, budget: float = 10.0, sequence_length: int = 5) -> None:
        self._budget = budget
        self._sequence_length = sequence_length
        self._state: Optional[RouterState] = None
        self._done = True
        
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
        base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        self._model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
        
        self._mock_mode = False
        if not api_key or "your_token" in api_key:
            print("[WARN] No valid API key found. Enabling Rule-Based Fallback Grader (Mock Mode).")
            self._mock_mode = True
            self._client = None
        else:
            print(f"[DEBUG] Initializing OpenAI client to {base_url}")
            try:
                self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=10.0)
                print("[DEBUG] OpenAI client initialized successfully")
            except Exception as e:
                print(f"[WARN] Failed to initialize OpenAI client: {e}. Falling back to Mock Mode.")
                self._mock_mode = True
                self._client = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[RouterObservation, Dict[str, Any]]:
        self._seed = seed if seed is not None else random.randint(0, 1000000)
        random.seed(self._seed)
        
        all_ids = list(TASK_CATALOGUE.keys())
        task_queue = random.choices(all_ids, k=self._sequence_length)

        self._state = RouterState(
            current_task_index=0,
            total_cost=0.0,
            successes=0,
            failures=0,
            latency_total=0.0,
            episode_budget=self._budget,
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

        # Dynamic Grade (Wait for Judge Agent)
        try:
            score, reasoning = self._evaluate_with_agent(task.description, model)
        except Exception as e:
            score, reasoning = 0.0, f"Judge error: {str(e)}"

        # 🎲 Stochastic Outcome: 15% chance of failure even on high scores
        # This models real-world API flakiness and edge cases.
        stochastic_failure = random.random() < 0.15
        if score > 0.8 and stochastic_failure:
            score = 0.1
            reasoning = f"(Stochastic Failure) {reasoning}"

        success = score >= 0.7
        self._last_perf = score
        self._last_success = success
        
        if success: self._state.successes += 1
        else: self._state.failures += 1

        cost = model.cost
        self._state.total_cost += cost

        # Reward = PerformanceScore * Scale - CostWeight
        reward = (score * 2.5) - (cost * 0.4)
        
        # Hidden Overkill Penalty (Hardcoded in environment logic only)
        if model.name == "large-reasoning" and task.complexity < 0.4:
            reward -= 1.0 # Significant penalty for wasting the heavy model

        self._state.current_task_index += 1
        terminated = (self._state.current_task_index >= len(self._state.task_queue))
        if self._state.total_cost >= self._state.episode_budget:
            terminated = True
        self._done = terminated
        
        obs = self._get_current_obs(f"Grader Verdict: {reasoning}")
        info = {"score": score, "reasoning": reasoning, "task_id": task_id}
        return obs, round(reward, 4), terminated, False, info

    def _evaluate_with_agent(self, description: str, model: ModelSpec) -> Tuple[float, str]:
        """Invoke the LLM Judge Agent, or fallback to heuristics if unavailable."""
        if self._mock_mode:
            # Simple heuristic based on known task types
            # Find which task id this belongs to by description
            task_comp = 0.5
            for tid, tdef in TASK_CATALOGUE.items():
                if tdef.description == description:
                    task_comp = tdef.complexity
                    break
            
            # Simulated outcome: High chance of success if power >= complexity
            if model.power >= task_comp:
                return 1.0, f"Heuristic: {model.name} power ({model.power}) satisfies complexity ({task_comp})."
            elif model.power >= task_comp - 0.2:
                return 0.7, f"Heuristic: {model.name} power ({model.power}) is close to complexity ({task_comp})."
            else:
                return 0.1, f"Heuristic: {model.name} power ({model.power}) is too weak for complexity ({task_comp})."

        user_prompt = f"TASK: {description}\nMODEL: {model.name}\nPOWER: {model.power}\nScore this choice."
        
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content)
        perf = data.get("performance_score") or data.get("score") or 0.0
        reason = data.get("reasoning") or data.get("explanation") or "N/A"
        return float(perf), str(reason)

    def _get_current_obs(self, message: str = "") -> RouterObservation:
        idx = self._state.current_task_index
        if self._state is None or idx >= len(self._state.task_queue):
            return RouterObservation(
                task_description="EMPTY", estimated_tokens=0, budget_remaining=0.0, 
                tasks_left=0, last_performance_score=0.0, last_success=False,
                task_id="None", message="Complete."
            )
        task_id = self._state.task_queue[idx]
        task = TASK_CATALOGUE[task_id]
        return RouterObservation(
            task_description=task.description,
            estimated_tokens=task.tokens,
            budget_remaining=max(0.0, self._state.episode_budget - self._state.total_cost),
            tasks_left=len(self._state.task_queue) - idx,
            last_performance_score=self._last_perf,
            last_success=getattr(self, "_last_success", False),
            task_id=task_id,
            message=message
        )

    def available_models(self) -> List[str]: return list(MODEL_ROSTER.keys())
    def state(self) -> RouterState: return self._state.model_copy(deep=True)
    def close(self): pass
