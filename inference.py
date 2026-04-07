"""
RouterEnv-v1 — Advanced Inference Agent (v2.1)
==============================================
Agent for evaluating Agent Grader-led environments.
"""

import os
import json
import time
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv
from router_env.environment import RouterEnvironment
from router_env.models import RouterAction

# ── Configuration ────────────────────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

if not API_KEY or "your_token" in API_KEY:
    print("[ERROR] Please set either OPENAI_API_KEY or HF_TOKEN in your .env file.")
    exit(1)

BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

# Build the client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── Prompting ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a high-performance AI Model Router. 
Decide which LLM tier to route the current task to based on its description.

CORE TIERS:
- 'small-fast': Best for trivial classification, sentiment, spam.
- 'medium-balanced': Best for code refactoring and creative rewriting.
- 'large-reasoning': MANDATORY for security audits, legal transition, or complex reasoning.

Strategy: Use the cheapest possible model that the RouterEnv Judge will accept as capable.

Response ONLY with JSON:
{"selected_model": "small-fast" | "medium-balanced" | "large-reasoning"}
"""

def get_routing_decision(observation: Any) -> Dict[str, Any]:
    """Query the LLM to make a routing decision."""
    obs_data = observation.model_dump()
    
    user_prompt = (
        f"TASK_DESCRIPTION: {obs_data.get('task_description')}\n"
        f"BUDGET_REMAINING: ${obs_data.get('budget_remaining', 0.0):.2f}\n"
        f"LAST_SCORE_GIVEN_BY_JUDGE: {obs_data.get('last_performance_score', 0.0)}"
    )
    
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                timeout=30.0
            )
            result = json.loads(response.choices[0].message.content)
            
            if "selected_model" in result and result["selected_model"] in ["small-fast", "medium-balanced", "large-reasoning"]:
                return result
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
                continue
            raise RuntimeError(f"LLM_Inference_Failed: {str(e)}")
    
    raise RuntimeError("LLM_Validation_Failed: No valid choice returned from model")


def run_agent():
    env = RouterEnvironment(budget=10.0, sequence_length=5)
    obs, info = env.reset()
    
    print(f"[START] task=router-graders env=RouterEnv-v1 v=2.1 agent=MetaLlama3-8B")
    
    cumulative_reward = 0.0
    steps = 0
    rewards_list = []
    
    while True:
        steps += 1
        
        # 1. Decision
        decision = get_routing_decision(obs)
        action = RouterAction(selected_model=decision["selected_model"])
        
        # 2. Step with Judge
        obs, reward, terminated, truncated, info = env.step(action)
        
        cumulative_reward += reward
        rewards_list.append(reward)
        
        # [STEP] output
        done_bool = "true" if terminated else "false"
        print(f"[STEP] step={steps} model={action.selected_model} task={info['task_id']} score={info['score']:.1f} reward={reward:.3f} done={done_bool}")
        print(f"       Reasoning: {info['reasoning']}")

        if terminated:
            break
            
    # [END] output
    score_avg = cumulative_reward / steps if steps > 0 else 0.0
    rewards_str = ",".join([f"{r:.2f}" for r in rewards_list])
    print(f"[END] avg_score={score_avg:.3f} total_reward={cumulative_reward:.3f} steps={steps} rewards={rewards_str}")

if __name__ == "__main__":
    run_agent()
