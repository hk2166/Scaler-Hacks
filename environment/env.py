from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Tuple

from environment.models import (
    Observation, Action, Reward, CustomerState, OrderRecord, Ticket
)
from environment.tools import ToolRegistry, ToolError
from environment import customer_state as cs_module
from environment.graders.graders import grade_easy, grade_medium, grade_hard
import environment.ticket_generator as ticket_gen


TASK_MAX_STEPS = {"easy": 5, "medium": 8, "hard": 12}
TASK_OPTIMAL   = {"easy": 2, "medium": 4, "hard": 4}


class CustomerServiceEnv:
    """
    OpenEnv-compliant Customer Service Agent environment.

    Unique features:
    - LLM-generated tickets (infinite variety, multilingual for hard task)
    - Realistic tool failures (rate limits, dependency guards, flakiness)
    - Customer anger state machine (dynamic, non-sparse reward signal)
    - NLP sentiment grading (TextBlob) on agent replies
    - Trajectory efficiency penalty
    """

    def __init__(self, task_id: str = "easy"):
        assert task_id in ("easy", "medium", "hard"), \
            f"task_id must be 'easy', 'medium', or 'hard', got '{task_id}'"
        self.task_id = task_id
        self._max_steps = TASK_MAX_STEPS[task_id]

        # Episode state — initialised in reset()
        self._ticket:   Optional[Ticket]        = None
        self._order:    Optional[OrderRecord]   = None
        self._tools:    Optional[ToolRegistry]  = None
        self._cstate:   CustomerState           = CustomerState()
        self._step:     int                     = 0
        self._done:     bool                    = False
        self._trajectory: List[Dict[str, Any]] = []
        self._last_obs: Optional[Observation]   = None

    # ── OpenEnv interface ────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Generate a fresh ticket and reset all episode state."""
        self._ticket, self._order = ticket_gen.generate(self.task_id)
        self._tools    = ToolRegistry(self._order)
        self._cstate   = CustomerState(language=self._ticket.language)
        self._step     = 0
        self._done     = False
        self._trajectory = []

        obs = self._make_obs()
        self._last_obs = obs
        return obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        assert not self._done, "Episode is done. Call reset() first."
        assert self._ticket is not None, "Call reset() before step()."

        self._step += 1
        tool_result: Optional[Dict[str, Any]] = None
        tool_error:  Optional[str]            = None
        cs_event = "step_penalty"   # default: every step adds a tiny anger bump

        # ── Capture anger BEFORE update for intermediate reward ───────────────
        prev_anger = self._cstate.anger_level

        # ── Execute tool ────────────────────────────────────────────────────
        try:
            tool_result = self._tools.call(action.tool_name, action.parameters)

            # Map tool success to customer state events
            cs_event = {
                "lookup_order":         "good_lookup",
                "validate_eligibility": "validated",
                "issue_refund":         "refund_issued",
                "escalate_to_human":    "escalated",
                "send_reply":           self._classify_reply(action.parameters.get("reply_text", "")),
                "classify_ticket":      "step_penalty",
            }.get(action.tool_name, "step_penalty")

        except ToolError as e:
            tool_error = str(e)
            already_called = action.tool_name in [
                t["action"].tool_name for t in self._trajectory
            ]
            cs_event = "repeated_tool" if already_called else "tool_failure"

        # ── Update customer anger ────────────────────────────────────────────
        self._cstate = cs_module.update(self._cstate, cs_event)

        # ── Record trajectory step ───────────────────────────────────────────
        self._trajectory.append({
            "step":   self._step,
            "action": action,
            "result": tool_result or {},
            "error":  tool_error,
            "customer_anger": self._cstate.anger_level,
        })

        # ── Check done conditions ────────────────────────────────────────────
        reply_sent = any(
            t["action"].tool_name == "send_reply" and not t.get("error")
            for t in self._trajectory
        )
        self._done = reply_sent or self._step >= self._max_steps

        if self._done:
            reward = self._compute_reward()
        else:
            # ── Intermediate reward: non-sparse signal every step ────────────
            # Positive when anger drops (good action), near-zero when anger rises
            anger_delta = prev_anger - self._cstate.anger_level  # positive = improvement
            # Normalise: max single-step anger drop is ~3.0 (refund_issued)
            # Map to 0.0–0.05 range so it doesn't dominate the final grader score
            step_score = round(max(0.0, min(0.05, anger_delta / 60.0)), 4)
            reward = Reward(
                score=step_score,
                breakdown={
                    "anger_before":      round(prev_anger, 2),
                    "anger_after":       round(self._cstate.anger_level, 2),
                    "anger_improvement": round(anger_delta, 2),
                },
                reason=(
                    f"Step {self._step}: intermediate reward "
                    f"(anger {'↓' if anger_delta > 0 else '↑'} "
                    f"{abs(anger_delta):.2f})."
                ),
                done=False,
            )

        obs = self._make_obs(tool_result, tool_error)
        self._last_obs = obs
        return obs, reward, self._done, {}

    def state(self) -> Dict[str, Any]:
        return {
            "task_id":      self.task_id,
            "step":         self._step,
            "max_steps":    self._max_steps,
            "done":         self._done,
            "customer_anger": self._cstate.anger_level,
            "ticket_id":    self._ticket.ticket_id if self._ticket else None,
            "tools_called": [t["action"].tool_name for t in self._trajectory],
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _make_obs(
        self,
        last_result: Optional[Dict] = None,
        last_error:  Optional[str]  = None,
    ) -> Observation:
        return Observation(
            ticket=self._ticket,
            current_step=self._step,
            max_steps=self._max_steps,
            task_id=self.task_id,
            customer_state=self._cstate,
            last_tool_result=last_result,
            last_tool_error=last_error,
            tools_called=[t["action"].tool_name for t in self._trajectory],
            done=self._done,
        )

    def _classify_reply(self, text: str) -> str:
        """Decide the customer-state event for a send_reply action."""
        if not text:
            return "cold_reply"
        positive_words = ["sorry", "apologize", "understand", "happy to help",
                          "assist", "resolve", "refund", "escalate"]
        hits = sum(1 for w in positive_words if w in text.lower())
        return "empathetic_reply" if hits >= 2 else "cold_reply"

    def _compute_reward(self) -> Reward:
        assert self._order is not None
        grader_kwargs = dict(
            trajectory=self._trajectory,
            final_customer_state=self._cstate,
        )
        if self.task_id == "easy":
            result = grade_easy(
                ground_truth_category=self._ticket.category_hint or "general_inquiry",
                **grader_kwargs,
            )
        elif self.task_id == "medium":
            result = grade_medium(
                order_eligible=self._order.eligible_for_refund,
                **grader_kwargs,
            )
        else:
            result = grade_hard(
                order_eligible=self._order.eligible_for_refund,
                optimal_steps=TASK_OPTIMAL[self.task_id],
                **grader_kwargs,
            )

        return Reward(
            score=result["score"],
            breakdown=result["breakdown"],
            reason=f"Task '{self.task_id}' completed in {self._step} steps.",
            done=True,
        )
