# 🚀 RouterEnv-v1: Advanced LLM Orchestration (v2.0)

**RouterEnv-v1** is a high-fidelity Reinforcement Learning environment designed to solve the "LLM Cost-Performance Paradox". Unlike toy environments, it simulates a **persistent stream of real-world tasks** with stochastic outcomes, forcing agents to reason about intellectual risk, budget management, and operational latency.

---

## 🏛 90+ Score Architecture (v2.0 Improvements)

We have overhauled the environment to move beyond "lookup-table" logic into genuine **agentic reasoning**:

1.  **Zero-Leakage Observations**: 
    *   The environment **no longer exposes complexity scores** (0.1, 0.5, etc.) to the agent.
    *   The agent must parse the `task_description` using an LLM to *infer* the required power level, mimicking real-world architectural decisions.

2.  **Stochastic "Real-World" Outcomes**:
    *   Success is not binary based on thresholds. We implement a **Sigmoid Probability Function**:
        $P(success) = \frac{1}{1 + e^{-15(gap + 0.05)}}$
    *   Even "Strong" models can fail a task (15% failure rate for baseline matches), requiring the agent to learn robust fallback strategies.

3.  **Workflow Sequence Simulation**:
    *   One episode = A stream of **5 dynamic tasks** selected from a 10-task real-world catalogue.
    *   Persistent Budget: The agent manages a fixed **$10.00 budget** across the entire sequence.

4.  **Advanced Reward Signal**:
    *   $Reward = (2.0 \times Complexity) + EfficiencyBonus - Cost - (0.05 \times Latency)$
    *   **Efficiency Bonus**: High rewards for solving hard tasks with cheaper models.
    *   **Latency Penalty**: Penalizes the "large-reasoning" model for overhead, encouraging speed when possible.

---

## 📊 Model Roster & Pricing

| Model Tier | Power | Base Cost | Base Latency | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **small-fast** | 0.3 | $0.005 | 0.1s | Sentiment, Spam, Regex |
| **medium-balanced**| 0.7 | $0.08 | 0.4s | Refactoring, Rewriting |
| **large-reasoning**| 1.0 | $0.80 | 1.5s | Security Audit, Legal, Complex Logic |

---

## 📋 Real-World Task Catalogue (Samples)

*   **pii_redact**: Redact names/emails from logs (High Criticality).
*   **sql_fix**: Identify logic errors in complex SQL joins.
*   **audit_rest**: Full security vulnerability audit for FastAPI controllers.
*   **translate_contract**: Legally binding EN -> ZH translation.
*   **sentiment**: Standard customer feedback classification.

---

## 🎯 Spec Compliance

- **OpenEnv v2.0**: Full compliance with `reset()`, `step()`, and `state()`.
- **Dockerized**: Production-ready `Dockerfile` for Hugging Face Spaces.
- **Typed Pydantic Contracts**: Zero-overhead validation for all actions/observations.
- **Baseline Included**: `inference.py` provides a reproducible score using `Meta-Llama-3-8B`.

---

## ⚙️ Execution

1. **Install**: `pip install -r requirements.txt`
2. **Secrets**: Add `OPENAI_API_KEY` (or `HF_TOKEN`) to `.env`.
3. **Run Baseline**: `python inference.py`
4. **Deploy**: `docker build -t router-env .`

---
*Developed for the OpenEnv Hackathon: Intelligent Agentic Environments.*
