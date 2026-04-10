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

# ── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
MOCK_AGENT_MODE = False

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
def run_agent() -> int:
    """Run the agent. Returns 0 on success, 1 on fatal error."""
    try:
        env = RouterEnvironment(budget=10.0, sequence_length=5)
    except Exception as e:
        print(f"[ERROR] Failed to initialize RouterEnvironment: {e}")
        return 1

    try:
        obs, info = env.reset()
    except Exception as e:
        print(f"[ERROR] env.reset() failed: {e}")
        return 1

    print("[START] task=router-graders env=RouterEnv-v1 v=2.1 agent=MetaLlama3-8B")

    cumulative_reward = 0.0
    steps             = 0
    rewards_list      = []

    try:
        while True:
            steps += 1

            decision = get_routing_decision(
                task_description=obs.task_description,
                budget_remaining=obs.budget_remaining,
                last_score=obs.last_performance_score
            )
            action = RouterAction(selected_model=decision["selected_model"])

            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                print(f"[ERROR] env.step() failed at step {steps}: {e}")
                break

            cumulative_reward += reward
            rewards_list.append(reward)

            score = info.get('score', 0.0)
            clamped_score = max(0.01, min(0.99, score))

            done_bool = "true" if terminated else "false"
            print(f"[STEP] step={steps} model={action.selected_model} "
                  f"task={info.get('task_id','N/A')} score={clamped_score:.2f} "
                  f"reward={reward:.3f} done={done_bool}")
            print(f"       Reasoning: {info.get('reasoning','N/A')}")

            if terminated or truncated:
                break

    except Exception as e:
        print(f"[ERROR] Unexpected error in agent loop at step {steps}: {e}")
        return 1

    score_avg   = cumulative_reward / steps if steps > 0 else 0.0
    rewards_str = ",".join([f"{r:.2f}" for r in rewards_list])
    print(f"[END] avg_score={score_avg:.3f} steps={steps} rewards={rewards_str}")

    return 0


if __name__ == "__main__":
    try:
        exit_code = run_agent()
        sys.exit(exit_code)
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}")
        sys.exit(1)
 