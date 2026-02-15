from __future__ import annotations
from environment.models import CustomerState


# How much anger changes per event
ANGER_DELTA = {
    "wrong_tool":        +1.5,
    "repeated_tool":     +2.0,
    "tool_failure":      +0.5,   # env failure, not agent fault — small bump
    "no_action":         +1.0,
    "good_lookup":       -0.5,
    "validated":         -0.3,
    "refund_issued":     -3.0,
    "escalated":         -2.0,
    "empathetic_reply":  -2.5,
    "cold_reply":        +1.0,
    "step_penalty":      +0.3,   # each step without resolution adds frustration
}


def update(state: CustomerState, event: str) -> CustomerState:
    """Return a new CustomerState after applying the given event."""
    delta = ANGER_DELTA.get(event, 0.0)
    new_anger = max(0.0, min(10.0, state.anger_level + delta))

    # Satisfaction is the inverse signal
    new_satisfaction = max(0.0, min(10.0, state.satisfaction - delta * 0.4))

    apologized = state.has_been_apologized_to
    if event == "empathetic_reply":
        apologized = True

    return CustomerState(
        anger_level=new_anger,
        satisfaction=new_satisfaction,
        language=state.language,
        has_been_apologized_to=apologized,
        wait_steps=state.wait_steps + 1,
    )


def anger_penalty(state: CustomerState) -> float:
    """Convert final anger level into a 0.0–0.3 score penalty."""
    # anger 0 → penalty 0.0 | anger 10 → penalty 0.3
    return round((state.anger_level / 10.0) * 0.3, 3)


def satisfaction_bonus(state: CustomerState) -> float:
    """Small bonus (0–0.1) for keeping satisfaction high."""
    return round((state.satisfaction / 10.0) * 0.1, 3)
