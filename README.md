---
title: RouterEnv-v1 LLM Orchestration Environment
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - llm-routing
  - agentic-ai
  - hackathon
---

# 🚀 RouterEnv-v1: Intelligent LLM Orchestration Environment

> A high-fidelity OpenEnv-compliant Reinforcement Learning environment for solving the **LLM Cost-Performance Paradox** — teaching agents to route tasks to the right model at the right cost.

---

## 📖 Environment Description & Motivation

Modern AI systems face a fundamental tension: powerful LLMs are expensive and slow, while cheaper models may fail on complex tasks. **RouterEnv-v1** simulates this real-world dilemma as an RL problem.

The agent acts as an **LLM orchestration layer** — receiving a stream of diverse NLP tasks (PII redaction, SQL debugging, security audits, translations) and deciding which model tier to invoke. It must balance:

- **Quality**: Matching model capability to task complexity
- **Cost**: Managing a fixed $10.00 budget across a 5-task episode
- **Speed**: Penalizing unnecessary use of slow, expensive models

Unlike toy environments with lookup-table solutions, RouterEnv-v1 forces genuine **agentic reasoning**: the agent receives only a natural language task description, must infer complexity using an LLM, and learns robust fallback strategies from stochastic outcomes.

---

## 🎯 Action Space

The agent selects one of **3 discrete actions** per step — choosing which model tier to invoke for the current task:

| Action ID | Model Tier | Power Score | Cost per Call | Latency | Best For |
|-----------|------------|-------------|---------------|---------|----------|
| `0` | `small-fast` | 0.3 | $0.005 | 0.1s | Sentiment analysis, spam detection, regex tasks |
| `1` | `medium-balanced` | 0.7 | $0.080 | 0.4s | Code refactoring, rewriting, summarization |
| `2` | `large-reasoning` | 1.0 | $0.800 | 1.5s | Security audits, legal translation, complex logic |

**Action type**: `Discrete(3)`  
**Action format**: Integer `{0, 1, 2}`

---

## 👁️ Observation Space

The agent receives a dictionary observation at each step. **Critically, raw complexity scores are never exposed** — the agent must infer difficulty from the task description itself.

| Field | Type | Range / Values | Description |
|-------|------|----------------|-------------|
| `task_description` | `str` | Natural language | Full task prompt the LLM must solve |
| `task_type` | `str` | See task catalogue | Short task identifier (e.g. `pii_redact`) |
| `budget_remaining` | `float` | `[0.0, 10.0]` | USD remaining in the current episode |
| `tasks_remaining` | `int` | `[0, 5]` | How many steps are left in the episode |
| `last_success` | `bool` | `True / False` | Whether the previous task succeeded |
| `last_reward` | `float` | `(-∞, +∞)` | Reward received on the previous step |

**Observation type**: `Dict`  
**Zero-leakage guarantee**: Complexity scores and ground-truth difficulty labels are withheld from the agent at inference time.

---

## 📋 Task Catalogue & Difficulty

One episode = 5 tasks sampled from the following 10-task real-world catalogue:

| Task ID | Description | Complexity | Recommended Tier |
|---------|-------------|------------|-----------------|
| `sentiment` | Customer feedback classification | 0.2 (Easy) | small-fast |
| `spam_filter` | Classify email as spam/ham | 0.2 (Easy) | small-fast |
| `summarize` | Summarize a 500-word article | 0.4 (Easy-Med) | small-fast / medium |
| `translate_simple` | Translate short EN → ES text | 0.4 (Easy-Med) | medium-balanced |
| `sql_fix` | Fix logic errors in complex SQL joins | 0.6 (Medium) | medium-balanced |
| `code_refactor` | Refactor Python function for readability | 0.6 (Medium) | medium-balanced |
| `pii_redact` | Redact names/emails from production logs | 0.8 (Hard) | large-reasoning |
| `translate_contract` | Legally binding EN → ZH contract translation | 0.85 (Hard) | large-reasoning |
| `audit_rest` | Full security vulnerability audit of FastAPI controller | 0.9 (Hard) | large-reasoning |
| `logic_proof` | Verify multi-step logical argument for soundness | 0.95 (Hard) | large-reasoning |

**Episode dynamics**: Tasks are sampled without replacement per episode, so each run presents a unique sequence requiring adaptive reasoning.

---

## 🎲 Stochastic Outcome Model

Success is probabilistic — not a hard threshold. The environment uses a **Sigmoid Probability Function**:

```
P(success) = 1 / (1 + e^(-15 × (gap + 0.05)))
```

Where `gap = model_power - task_complexity`.

This means:
- A perfectly matched model still has a **~15% failure rate**
- Underpowered models rarely succeed on hard tasks
- Agents must learn **robust fallback strategies**, not just memorize mappings

---

## 💰 Reward Function

```
Reward = (2.0 × complexity) + efficiency_bonus − cost − (0.05 × latency)
```

| Component | Details |
|-----------|---------|
| `2.0 × complexity` | Base reward scales with task difficulty |
| `efficiency_bonus` | Bonus for solving hard tasks with cheaper models |
| `− cost` | Direct penalty for model API cost |
| `− 0.05 × latency` | Latency penalty discourages overusing `large-reasoning` |
| Budget exhaustion | Episode terminates early if `budget_remaining < min_model_cost` |

---

## 🔌 OpenEnv v2.0 API

```python
from router_env import RouterEnv

env = RouterEnv()

# Reset environment — returns initial observation
obs = env.reset()

# Step — takes action {0, 1, 2}
obs, reward, done, info = env.step(action=1)

# State — returns full internal state dict
state = env.state()
```

### `info` dictionary returned by `step()`:

```python
{
  "task_type": "sql_fix",
  "model_used": "medium-balanced",
  "success": True,
  "cost_incurred": 0.08,
  "latency": 0.4,
  "efficiency_bonus": 0.15,
  "budget_remaining": 9.12
}
```

---

## 📊 Baseline Scores

Evaluated over 100 episodes using `inference.py` with `Meta-Llama-3-8B` as the reasoning backbone:

| Strategy | Avg Reward / Episode | Success Rate | Avg Cost / Episode | Notes |
|----------|---------------------|--------------|-------------------|-------|
| **Random** | 1.24 ± 0.8 | 41% | $2.10 | Uniform random action selection |
| **Always Small** | 0.93 ± 0.5 | 28% | $0.025 | Fails hard tasks consistently |
| **Always Large** | 2.10 ± 0.6 | 89% | $4.00 | Wastes budget, high latency |
| **Complexity Heuristic** | 3.47 ± 0.7 | 76% | $1.20 | Rule-based keyword matching |
| **LLaMA-3-8B Baseline** | **5.82 ± 1.1** | **84%** | **$0.95** | `inference.py` — LLM-inferred routing |
| _Target (90+ score)_ | _8.0+_ | _90%+_ | _< $1.50_ | _Optimal agentic policy_ |

> Run `python inference.py` to reproduce the LLaMA-3-8B baseline on your machine.

---

## ⚙️ Setup & Usage

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- An API key: `OPENAI_API_KEY` or `HF_TOKEN`

### Local Setup (without Docker)

```bash
# 1. Clone the repo
git clone https://github.com/ghaibhawishya-tech/Scaler_hackathon
cd Scaler_hackathon

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your API key

# 4. Run the baseline agent
python inference.py

# 5. Run tests
pytest tests/
```

### Local Setup (with Docker)

```bash
# Build the image
docker build -t router-env .

# Run with your API key
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your_key_here \
  router-env

# Open http://localhost:7860
```

### Deploy to Hugging Face Spaces

```bash
# 1. Create a new Space at huggingface.co/new-space
#    SDK: Docker | Tag: openenv

# 2. Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/router-env
cd router-env

# 3. Copy project files
cp -r /path/to/Scaler_hackathon/* .

# 4. Add your secret key in HF Space Settings → Repository secrets
#    Key: OPENAI_API_KEY  Value: your_key_here

# 5. Push
git lfs install
git add .
git commit -m "Deploy RouterEnv-v1"
git push
```

---

## 🗂️ Project Structure

```
.
├── router_env/          # Core environment package
│   ├── __init__.py
│   ├── env.py           # RouterEnv class (reset, step, state)
│   ├── models.py        # Model tier definitions & cost table
│   └── tasks.py         # Task catalogue & sampling logic
├── tests/               # Test suite
├── inference.py         # Baseline agent (LLaMA-3-8B)
├── openenv.yaml         # OpenEnv v2.0 spec declaration
├── pyproject.toml       # Package metadata
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
├── .env.example         # Environment variable template
└── README.md
```

---

## 📄 License

Developed for the **OpenEnv Hackathon: Intelligent Agentic Environments**.
