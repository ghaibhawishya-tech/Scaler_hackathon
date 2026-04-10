"""
RouterEnv-v1 — Advanced Inference Agent (v2.1)
==============================================
Agent for evaluating Agent Grader-led environments.
"""

import os
import sys
import json
import time
import random
from typing import Dict, Any

from dotenv import load_dotenv
from router_env.environment import RouterEnvironment
from router_env.models import RouterAction
from router_env.graders import grade_episode
from router_env.environment import TASK_CATALOGUE

# ── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or "your_huggingface_token_here"
MOCK_AGENT_MODE = False

BENCHMARK = "RouterEnv-v1"

if not API_KEY or "your_token" in API_KEY:
    print("[WARN] No valid API key found. Enabling Heuristic Agent (Mock Mode).")
    MOCK_AGENT_MODE = True
else:
    os.environ["OPENAI_API_KEY"] = API_KEY

os.environ.setdefault("API_BASE_URL", "https://router.huggingface.co/v1")
os.environ.setdefault("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

# ── Prompting ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a high-performance AI Model Router.
Decide which LLM tier to route the current task to based on its description.

CORE TIERS:
- 'small-fast'      : Best for trivial yes/no classification only.
- 'medium-balanced' : Best for spam detection, sentiment analysis, code refactoring, creative rewriting, customer support.
- 'large-reasoning' : MANDATORY for security audits, legal analysis, PII redaction, or complex multi-step reasoning.

Strategy: Use the cheapest tier the RouterEnv Judge will accept as capable.
When in doubt, go one tier higher — an under-powered model is penalised heavily.

Respond ONLY with JSON:
{"selected_model": "small-fast" | "medium-balanced" | "large-reasoning"}
"""

FALLBACK_MODEL = "medium-balanced"
VALID_MODELS   = ["small-fast", "medium-balanced", "large-reasoning"]


# ── Routing decision ───────────────────────────────────────────────────────────
def get_routing_decision(task_description: str, budget_remaining: float, last_score: float) -> Dict[str, Any]:
    """Query the LLM to make a routing decision."""
    # Clamp score between 0.01 and 0.99
    last_score = max(0.01, min(0.99, last_score))

    if MOCK_AGENT_MODE:
        desc = task_description.lower()
        # Rule-based routing
        if any(w in desc for w in ["audit", "legal", "msa", "pii", "contract", "fintech"]):
            return {"selected_model": "large-reasoning"}
        elif any(w in desc for w in ["refactor", "unit test", "sql", "summary", "response"]):
            return {"selected_model": "medium-balanced"}
        else:
            return {"selected_model": "small-fast"}

    user_prompt = (
        f"TASK_DESCRIPTION: {task_description}\n"
        f"BUDGET_REMAINING: ${budget_remaining:.2f}\n"
        f"LAST_SCORE_GIVEN_BY_JUDGE: {last_score}"
    )

    for attempt in range(3):
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=API_KEY,
                base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
                timeout=30.0
            )
            response = client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                timeout=30.0
            )
            result = json.loads(response.choices[0].message.content)

            if "selected_model" in result and result["selected_model"] in VALID_MODELS:
                return result

            print(f"[WARN] Attempt {attempt+1}: Invalid value '{result.get('selected_model')}'. Retrying...")

        except json.JSONDecodeError as e:
            print(f"[WARN] Attempt {attempt+1}: JSON parse error: {e}. Retrying...")
        except Exception as e:
            print(f"[WARN] Attempt {attempt+1}: Inference error: {e}. Retrying...")
            if attempt < 2:
                time.sleep(2 ** attempt)

    print(f"[WARN] All attempts failed. Falling back to '{FALLBACK_MODEL}'.")
    return {"selected_model": FALLBACK_MODEL}


# ── Main agent loop ────────────────────────────────────────────────────────────
def run_task(task_id: str):
    """Run a full episode for one task and print in OpenEnv format."""
    env = RouterEnvironment(budget=10.0, sequence_length=5)
    all_rewards = []
    step_count = 0
    score = 0.0
    success = False

    print(f"[START] task={task_id} env={BENCHMARK} model={os.getenv('MODEL_NAME', 'MetaLlama3-8B')}", flush=True)

    try:
        obs, info = env.reset()

        task = TASK_CATALOGUE.get(task_id)
        if task:
            obs = env._get_current_obs(f"Task: {task.description}")

        actions_taken = []

        while True:
            step_count += 1

            decision = get_routing_decision(
                task_description=obs.task_description,
                budget_remaining=obs.budget_remaining,
                last_score=obs.last_performance_score
            )
            action = RouterAction(selected_model=decision["selected_model"])
            actions_taken.append(f"{obs.task_id}:{action.selected_model}")

            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                all_rewards.append(0.0)
                error_msg = str(e).replace("\n", " ")
                print(
                    f"[STEP] step={step_count} action=error "
                    f"reward=0.00 done=false error={error_msg}",
                    flush=True,
                )
                break

            all_rewards.append(reward)

            score = info.get('score', 0.0)
            clamped_score = max(0.01, min(0.99, score))

            done_str = "true" if terminated or truncated else "false"
            print(
                f"[STEP] step={step_count} model={action.selected_model} "
                f"task={info.get('task_id','N/A')} score={clamped_score:.2f} "
                f"reward={reward:.3f} done={done_str}",
                flush=True,
            )

            if terminated or truncated:
                break

        grade = grade_episode(task_id, {
            "actions_taken": actions_taken,
            "final_soc": 0.0,
        })

        score = grade["score"]
        success = grade["passed"]

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
        print(
            f"[END] success={str(success).lower()} steps={step_count} "
            f"score={score:.2f} rewards={rewards_str}",
            flush=True,
        )


def run_agent() -> int:
    """Run the agent across all tasks. Returns 0 on success, 1 on fatal error."""
    for task_id in TASK_CATALOGUE.keys():
        try:
            run_task(task_id)
        except Exception as e:
            print(f"[ERROR] Failed to run task {task_id}: {e}")
            continue
    return 0


if __name__ == "__main__":
    try:
        exit_code = run_agent()
        sys.exit(exit_code)
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}")
        sys.exit(1)
 