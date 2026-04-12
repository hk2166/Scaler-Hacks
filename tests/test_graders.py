# tests/test_graders.py
"""
Smoke tests for graders — judges run these to verify scores aren't hardcoded.
"""
import pytest
from environment.env import CustomerServiceEnv
from environment.models import Action

GOOD_REPLY = "I'm happy to help resolve this for you! Your refund has been issued. We truly value you!"
BAD_REPLY  = "ok"

def make_env_at_end(task_id: str, do_tools: bool = True):
    """Helper: returns env after running all tools (or skipping them)."""
    env = CustomerServiceEnv(task_id)
    env.reset(seed=0)
    if do_tools and task_id in ("medium", "hard"):
        env.step(Action(tool_name="classify_ticket",     parameters={"category": "refund_request"}))
        env.step(Action(tool_name="lookup_order",        parameters={"order_id": "1042"}))
        env.step(Action(tool_name="validate_eligibility", parameters={}))
        env.step(Action(tool_name="issue_refund",        parameters={}))
    return env

# ── Reward range tests ──────────────────────────────────────────────────────

@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_reward_always_in_01_range(task_id):
    env = CustomerServiceEnv(task_id)
    env.reset(seed=0)
    _, reward, _, _ = env.step(
        Action(tool_name="send_reply", parameters={"reply_text": GOOD_REPLY})
    )
    assert 0.0 <= reward.score <= 1.0, f"{task_id}: reward {reward.score} out of [0,1]"

# ── Determinism tests ───────────────────────────────────────────────────────

@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_graders_are_deterministic(task_id):
    """Same seed + same action must produce identical reward."""
    def get_score():
        env = CustomerServiceEnv(task_id)
        env.reset(seed=7)
        _, reward, _, _ = env.step(
            Action(tool_name="classify_ticket", parameters={"category": "refund_request"})
        )
        return reward.score

    assert get_score() == get_score(), f"{task_id}: grader is non-deterministic"

# ── Graders produce different scores (not hardcoded) ───────────────────────

def test_good_reply_scores_higher_than_bad():
    env_good = CustomerServiceEnv("easy")
    env_bad  = CustomerServiceEnv("easy")
    env_good.reset(seed=0)
    env_bad.reset(seed=0)

    _, r_good, _, _ = env_good.step(Action(tool_name="send_reply", parameters={"reply_text": GOOD_REPLY}))
    _, r_bad,  _, _ = env_bad.step( Action(tool_name="send_reply", parameters={"reply_text": BAD_REPLY}))

    assert r_good.score > r_bad.score, "Good reply should outscore bad reply"

# ── Clean reset ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_reset_produces_clean_state(task_id):
    env = CustomerServiceEnv(task_id)
    env.reset(seed=0)
    env.step(Action(tool_name="classify_ticket", parameters={"category": "refund_request"}))

    # Reset and verify step counter is back to 0
    obs = env.reset(seed=0)
    assert obs.current_step == 0
    assert obs.tools_called == []
    assert obs.done is False

# ── Difficulty ordering ─────────────────────────────────────────────────────

def test_hard_task_is_harder_than_easy_for_random_agent():
    """A random/lazy agent should score worse on hard than easy."""
    lazy_action = Action(tool_name="send_reply", parameters={"reply_text": "ok"})

    scores = {}
    for task_id in ["easy", "hard"]:
        env = CustomerServiceEnv(task_id)
        env.reset(seed=0)
        _, reward, _, _ = env.step(lazy_action)
        scores[task_id] = reward.score

    assert scores["hard"] <= scores["easy"], \
        f"Hard ({scores['hard']}) should be ≤ easy ({scores['easy']}) for lazy agent"
