"""
Unit tests for RouterEnv-v1.

Covers:
  - reset() with valid and invalid task IDs
  - step() reward logic: success, partial, failure
  - Budget exhaustion mechanics
  - state() snapshot correctness
  - Guard-rail RuntimeError / ValueError paths
"""

from __future__ import annotations
import pytest
from router_env.environment import RouterEnvironment
from router_env.models import RouterAction, RouterObservation, RouterState

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def env() -> RouterEnvironment:
    """Fresh environment with default $5 budget."""
    return RouterEnvironment(budget=5.0)

# ── reset() tests ────────────────────────────────────────────────────────────

class TestReset:
    """Tests for RouterEnvironment.reset()."""

    def test_reset_easy(self, env: RouterEnvironment) -> None:
        obs, info = env.reset(options={"task_id": "easy"})
        assert isinstance(obs, RouterObservation)
        assert obs.complexity == pytest.approx(0.1)
        assert obs.estimated_tokens == 50
        assert obs.budget_remaining == pytest.approx(5.0)

    def test_reset_medium(self, env: RouterEnvironment) -> None:
        obs, info = env.reset(options={"task_id": "medium"})
        assert obs.complexity == pytest.approx(0.5)
        assert obs.estimated_tokens == 300

    def test_reset_hard(self, env: RouterEnvironment) -> None:
        obs, info = env.reset(options={"task_id": "hard"})
        assert obs.complexity == pytest.approx(0.9)
        assert obs.estimated_tokens == 1200

    def test_reset_invalid_task(self, env: RouterEnvironment) -> None:
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(options={"task_id": "impossible"})

# ── step() reward tests ─────────────────────────────────────────────────────

class TestStepRewards:
    """Tests for the dense reward function."""

    def test_success_reward_easy_small(self, env: RouterEnvironment) -> None:
        """Small model solves easy task → reward = 1.0 − 0.01 = 0.99."""
        env.reset(options={"task_id": "easy"})
        action = RouterAction(selected_model="small-fast", token_limit=100)
        obs, reward, term, trunc, info = env.step(action)

        assert info["outcome"] == "success"
        assert reward == pytest.approx(0.99)
        assert term is True

    def test_success_reward_medium_balanced(self, env: RouterEnvironment) -> None:
        """Medium model solves medium task → reward = 1.0 − 0.15 = 0.85."""
        env.reset(options={"task_id": "medium"})
        action = RouterAction(selected_model="medium-balanced", token_limit=500)
        obs, reward, term, trunc, info = env.step(action)

        assert info["outcome"] == "success"
        assert reward == pytest.approx(0.85)
        assert term is True

    def test_success_reward_large_reasoning(self, env: RouterEnvironment) -> None:
        """Large model solves hard task → reward = 1.0 − 1.50 = −0.50."""
        env.reset(options={"task_id": "hard"})
        action = RouterAction(selected_model="large-reasoning", token_limit=2000)
        obs, reward, term, trunc, info = env.step(action)

        assert info["outcome"] == "success"
        assert reward == pytest.approx(-0.50)
        assert term is True

    def test_partial_reward(self, env: RouterEnvironment) -> None:
        """Medium model on hard task (gap = 0.2) → partial reward +0.25."""
        env.reset(options={"task_id": "hard"})  # complexity 0.9
        action = RouterAction(selected_model="medium-balanced", token_limit=1500)
        obs, reward, term, trunc, info = env.step(action)

        assert info["outcome"] == "partial"
        assert reward == pytest.approx(0.25)

    def test_failure_penalty(self, env: RouterEnvironment) -> None:
        """Small model on hard task (gap = 0.6) → penalty −0.50."""
        env.reset(options={"task_id": "hard"})  # complexity 0.9
        action = RouterAction(selected_model="small-fast", token_limit=1500)
        obs, reward, term, trunc, info = env.step(action)

        assert info["outcome"] == "failure"
        assert reward == pytest.approx(-0.50)

    def test_overpowered_model_still_succeeds(self, env: RouterEnvironment) -> None:
        """Large model on easy task succeeds (reward = 1.0 - 1.5 = -0.5)."""
        env.reset(options={"task_id": "easy"})
        action = RouterAction(selected_model="large-reasoning", token_limit=100)
        obs, reward, term, trunc, info = env.step(action)

        assert info["outcome"] == "success"
        assert reward == pytest.approx(-0.50)

# ── Budget & episode lifecycle ───────────────────────────────────────────────

class TestEpisodeLifecycle:
    """Tests for budget tracking and episode termination."""

    def test_budget_decreases(self, env: RouterEnvironment) -> None:
        env.reset(options={"task_id": "hard"})
        action = RouterAction(selected_model="small-fast", token_limit=100)
        obs, reward, term, trunc, info = env.step(action)
        assert info["budget_remaining"] == pytest.approx(5.0 - 0.01)

    def test_step_after_done_raises(self, env: RouterEnvironment) -> None:
        env.reset(options={"task_id": "easy"})
        action = RouterAction(selected_model="small-fast", token_limit=100)
        env.step(action)  # succeeds → term=True
        with pytest.raises(RuntimeError, match="reset"):
            env.step(action)

# ── state() tests ────────────────────────────────────────────────────────────

class TestState:
    """Tests for the state() snapshot."""

    def test_state_after_reset(self, env: RouterEnvironment) -> None:
        env.reset(options={"task_id": "medium"})
        s = env.state()
        assert isinstance(s, RouterState)
        assert s.total_cost_spent == 0.0
        assert s.tasks_solved == 0
        assert s.current_difficulty == pytest.approx(0.5)

    def test_state_after_step(self, env: RouterEnvironment) -> None:
        env.reset(options={"task_id": "easy"})
        action = RouterAction(selected_model="small-fast", token_limit=100)
        env.step(action)
        s = env.state()
        assert s.tasks_solved == 1
        assert s.tasks_attempted == 1
        assert s.total_cost_spent == pytest.approx(0.01)
