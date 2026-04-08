
"""
RouterEnv-v1 — Advanced Inference Agent (v2.1)
==============================================
Agent for evaluating Agent Grader-led environments.
"""

import os
import json
import time
import socket
from typing import Dict, Any

from openai import OpenAI
from dotenv import load_dotenv
from router_env.environment import RouterEnvironment
from router_env.models import RouterAction

# ── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

if not API_KEY or "your_token" in API_KEY:
    print("[ERROR] Please set either OPENAI_API_KEY or HF_TOKEN in your .env file.")
    exit(1)

BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

# ✅ FIX: read host/port from env so it matches whatever the platform sets
ROUTER_ENV_HOST = os.getenv("ROUTER_ENV_HOST", "localhost")
ROUTER_ENV_PORT = int(os.getenv("ROUTER_ENV_PORT", "7860"))   # matches Dockerfile EXPOSE

print(f"[DEBUG] Initializing OpenAI client to {BASE_URL}")
try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=15.0)
    print("[DEBUG] OpenAI client initialized successfully")
except Exception as e:
    print(f"[ERROR] Failed to initialize OpenAI client: {e}")
    exit(1)

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


# ── Wait for RouterEnv server ──────────────────────────────────────────────────
def wait_for_env(host: str, port: int, timeout: int = 60) -> bool:
    """
    Block until the RouterEnvironment server is accepting TCP connections.
    This prevents the 'cannot join network of a non running container' crash.
    """
    print(f"[WAIT] Waiting for RouterEnv at {host}:{port} (timeout={timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"[WAIT] RouterEnv is reachable at {host}:{port} ✅")
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(2)
    print(f"[ERROR] RouterEnv not reachable after {timeout}s ❌")
    return False


# ── Routing decision ───────────────────────────────────────────────────────────
def get_routing_decision(observation: Any) -> Dict[str, Any]:
    """Query the LLM to make a routing decision."""
    try:
        obs_data = observation.model_dump()
    except Exception as e:
        print(f"[WARN] Failed to serialize observation: {e}. Using fallback.")
        return {"selected_model": FALLBACK_MODEL}

    user_prompt = (
        f"TASK_DESCRIPTION: {obs_data.get('task_description')}\n"
        f"BUDGET_REMAINING: ${obs_data.get('budget_remaining', 0.0):.2f}\n"
        f"LAST_SCORE_GIVEN_BY_JUDGE: {obs_data.get('last_performance_score', 0.0)}"
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
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
            print(f"[WARN] Attempt {attempt+1}: Inference error: {e}.")
            if attempt < 2:
                time.sleep(2 ** attempt)

    print(f"[WARN] All attempts failed. Falling back to '{FALLBACK_MODEL}'.")
    return {"selected_model": FALLBACK_MODEL}


# ── Main agent loop ────────────────────────────────────────────────────────────
def run_agent():
    # ✅ FIX: wait for the RouterEnv container to actually be up before connecting
    if not wait_for_env(ROUTER_ENV_HOST, ROUTER_ENV_PORT, timeout=60):
        print("[ERROR] Giving up — RouterEnv never came online.")
        return

    try:
        env = RouterEnvironment(budget=10.0, sequence_length=5)
    except Exception as e:
        print(f"[ERROR] Failed to initialize RouterEnvironment: {e}")
        print("[ERROR] Ensure the environment Docker container is running and reachable.")
        return

    try:
        obs, info = env.reset()
    except Exception as e:
        print(f"[ERROR] env.reset() failed: {e}")
        return

    print("[START] task=router-graders env=RouterEnv-v1 v=2.1 agent=MetaLlama3-8B")

    cumulative_reward = 0.0
    steps             = 0
    rewards_list      = []

    while True:
        steps += 1

        decision = get_routing_decision(obs)
        action   = RouterAction(selected_model=decision["selected_model"])

        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"[ERROR] env.step() failed at step {steps}: {e}")
            print("[ERROR] Environment container may have stopped. Ending run.")
            break

        cumulative_reward += reward
        rewards_list.append(reward)

        done_bool = "true" if terminated else "false"
        print(f"[STEP] step={steps} model={action.selected_model} "
              f"task={info.get('task_id','N/A')} score={info.get('score',0.0):.1f} "
              f"reward={reward:.3f} done={done_bool}")
        print(f"       Reasoning: {info.get('reasoning','N/A')}")

        if terminated or truncated:
            break

    score_avg   = cumulative_reward / steps if steps > 0 else 0.0
    rewards_str = ",".join([f"{r:.2f}" for r in rewards_list])
    print(f"[END] avg_score={score_avg:.3f} total_reward={cumulative_reward:.3f} "
          f"steps={steps} rewards={rewards_str}")


if __name__ == "__main__":
    try:
        run_agent()
        exit(0)
    except Exception as e:
        print(f"[FATAL] Unhandled exception in run_agent: {e}")
        exit(1)