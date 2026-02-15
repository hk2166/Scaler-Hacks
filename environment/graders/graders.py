from __future__ import annotations
"""
Graders for all 3 tasks.
Each grader receives the full episode trajectory and returns a score 0.0–1.0.
"""
from typing import List, Dict, Any
from environment.models import Observation, Action, CustomerState
import environment.customer_state as cs_module


# ── helpers ──────────────────────────────────────────────────────────────────

def _sentiment_score(text: str) -> float:
    """
    Returns 0.0–1.0 tone score using TextBlob if available,
    falls back to keyword heuristics.
    """
    try:
        from textblob import TextBlob
        polarity = TextBlob(text).sentiment.polarity  # -1 to +1
        return round((polarity + 1) / 2, 3)           # normalise to 0–1
    except ImportError:
        pass

    # Keyword fallback
    text_lower = text.lower()
    positive = ["sorry", "apologize", "apologies", "understand", "happy to help",
                "resolve", "refund", "assist", "thank you", "please"]
    negative = ["unfortunately cannot", "not possible", "denied", "reject"]
    pos_hits = sum(1 for w in positive if w in text_lower)
    neg_hits = sum(1 for w in negative if w in text_lower)
    raw = (pos_hits * 0.15) - (neg_hits * 0.2)
    return max(0.0, min(1.0, 0.5 + raw))


def _efficiency_score(actual_steps: int, optimal_steps: int) -> float:
    """Penalise bloated trajectories. Returns 0.5–1.0."""
    if actual_steps <= optimal_steps:
        return 1.0
    ratio = optimal_steps / actual_steps
    return round(max(0.5, ratio), 3)


# ── Easy grader ──────────────────────────────────────────────────────────────

def grade_easy(
    trajectory: List[Dict[str, Any]],
    ground_truth_category: str,
    final_customer_state: CustomerState,
) -> Dict[str, Any]:
    """
    Task: Classify the ticket + send a short helpful reply.
    Max score: 1.0
      0.50 — correct category classification
      0.30 — reply sent with positive sentiment
      0.20 — anger stayed low (< 4)
    """
    breakdown: Dict[str, float] = {}

    # 1. Classification score
    classified_correctly = any(
        a["action"].tool_name == "classify_ticket"
        and a["result"].get("classified_as") == ground_truth_category
        for a in trajectory if not a.get("error")
    )
    breakdown["classification"] = 0.50 if classified_correctly else 0.0

    # 2. Reply quality
    reply_texts = [
        a["action"].parameters.get("reply_text", "")
        for a in trajectory
        if a["action"].tool_name == "send_reply" and not a.get("error")
    ]
    if reply_texts:
        best_reply = max(reply_texts, key=len)
        breakdown["reply_quality"] = round(_sentiment_score(best_reply) * 0.30, 3)
    else:
        breakdown["reply_quality"] = 0.0

    # 3. Anger control
    breakdown["anger_control"] = 0.20 if final_customer_state.anger_level < 4 else 0.0

    total = round(sum(breakdown.values()), 3)
    return {"score": min(total, 1.0), "breakdown": breakdown}


# ── Medium grader ─────────────────────────────────────────────────────────────

def grade_medium(
    trajectory: List[Dict[str, Any]],
    order_eligible: bool,
    final_customer_state: CustomerState,
) -> Dict[str, Any]:
    """
    Task: lookup → validate → (refund or escalate) → reply.
    Max score: 1.0
      0.20 — lookup called correctly
      0.20 — validate called after lookup (order matters)
      0.25 — correct resolution action (refund if eligible, escalate if not)
      0.20 — reply sent with good sentiment
      0.15 — anger bonus / penalty
    """
    breakdown: Dict[str, float] = {}
    tools_in_order = [
        a["action"].tool_name
        for a in trajectory if not a.get("error")
    ]

    # 1. Lookup
    breakdown["lookup"] = 0.20 if "lookup_order" in tools_in_order else 0.0

    # 2. Validate AFTER lookup
    lookup_idx    = next((i for i, t in enumerate(tools_in_order) if t == "lookup_order"), 999)
    validate_idx  = next((i for i, t in enumerate(tools_in_order) if t == "validate_eligibility"), 999)
    breakdown["validate_order"] = 0.20 if validate_idx > lookup_idx else 0.0

    # 3. Correct resolution
    issued   = "issue_refund"    in tools_in_order
    escalated= "escalate_to_human" in tools_in_order
    if order_eligible and issued:
        breakdown["resolution"] = 0.25
    elif not order_eligible and escalated:
        breakdown["resolution"] = 0.25
    elif order_eligible and escalated:
        breakdown["resolution"] = 0.10   # partial: escalated when refund was possible
    else:
        breakdown["resolution"] = 0.0

    # 4. Reply quality
    reply_texts = [
        a["action"].parameters.get("reply_text", "")
        for a in trajectory
        if a["action"].tool_name == "send_reply" and not a.get("error")
    ]
    if reply_texts:
        breakdown["reply_quality"] = round(_sentiment_score(max(reply_texts, key=len)) * 0.20, 3)
    else:
        breakdown["reply_quality"] = 0.0

    # 5. Anger
    anger_pen = cs_module.anger_penalty(final_customer_state)
    sat_bonus = cs_module.satisfaction_bonus(final_customer_state)
    breakdown["anger_penalty"]      = round(-anger_pen, 3)
    breakdown["satisfaction_bonus"] = round(sat_bonus, 3)

    total = round(sum(breakdown.values()), 3)
    return {"score": max(0.0, min(total, 1.0)), "breakdown": breakdown}


# ── Hard grader ───────────────────────────────────────────────────────────────

def grade_hard(
    trajectory: List[Dict[str, Any]],
    order_eligible: bool,
    final_customer_state: CustomerState,
    optimal_steps: int = 4,
) -> Dict[str, Any]:
    """
    Task: Full multi-step resolution with multilingual ticket.
    Max score: 1.0
      0.15 — lookup done
      0.15 — validate done after lookup
      0.20 — correct resolution (refund / escalate)
      0.25 — reply sentiment (NLP scored)
      0.15 — trajectory efficiency
      0.10 — anger / satisfaction balance
    """
    breakdown: Dict[str, float] = {}
    tools_in_order = [
        a["action"].tool_name
        for a in trajectory if not a.get("error")
    ]
    actual_steps = len(trajectory)

    # 1. Lookup
    breakdown["lookup"] = 0.15 if "lookup_order" in tools_in_order else 0.0

    # 2. Validate after lookup
    lookup_idx   = next((i for i, t in enumerate(tools_in_order) if t == "lookup_order"), 999)
    validate_idx = next((i for i, t in enumerate(tools_in_order) if t == "validate_eligibility"), 999)
    breakdown["validate_order"] = 0.15 if validate_idx > lookup_idx else 0.0

    # 3. Resolution
    issued    = "issue_refund"      in tools_in_order
    escalated = "escalate_to_human" in tools_in_order
    if order_eligible and issued:
        breakdown["resolution"] = 0.20
    elif not order_eligible and escalated:
        breakdown["resolution"] = 0.20
    elif issued or escalated:
        breakdown["resolution"] = 0.08   # wrong choice but at least acted
    else:
        breakdown["resolution"] = 0.0

    # 4. Reply quality (NLP)
    reply_texts = [
        a["action"].parameters.get("reply_text", "")
        for a in trajectory
        if a["action"].tool_name == "send_reply" and not a.get("error")
    ]
    if reply_texts:
        best = max(reply_texts, key=len)
        breakdown["reply_sentiment"] = round(_sentiment_score(best) * 0.25, 3)
    else:
        breakdown["reply_sentiment"] = 0.0

    # 5. Efficiency
    eff = _efficiency_score(actual_steps, optimal_steps)
    breakdown["efficiency"] = round(eff * 0.15, 3)

    # 6. Anger / satisfaction
    anger_pen = cs_module.anger_penalty(final_customer_state)
    sat_bonus = cs_module.satisfaction_bonus(final_customer_state)
    breakdown["anger_penalty"]      = round(-anger_pen * 0.10 / 0.30, 3)
    breakdown["satisfaction_bonus"] = round(sat_bonus, 3)

    total = round(sum(breakdown.values()), 3)
    return {"score": max(0.0, min(total, 1.0)), "breakdown": breakdown}
