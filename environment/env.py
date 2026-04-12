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

TICKET_BANK: dict[str, list[dict]] = {
    "easy": [
        {
            "ticket": {"ticket_id": "E1", "sender_name": "Ravi Sharma", "sender_email": "ravi@example.com", "subject": "Refund for order #1042", "body": "I want a refund for my order.", "order_id": "1042", "language": "en", "category_hint": "refund_request"},
            "order": {"order_id": "1042", "product": "wireless headphones", "status": "delivered", "amount": 150.0, "purchase_date": "2024-12-15", "eligible_for_refund": True}
        },
        {
            "ticket": {"ticket_id": "E2", "sender_name": "Priya Patel", "sender_email": "priya@example.com", "subject": "My package never arrived", "body": "Order #2091 never arrived.", "order_id": "2091", "language": "en", "category_hint": "delivery_issue"},
            "order": {"order_id": "2091", "product": "laptop stand", "status": "delayed", "amount": 45.0, "purchase_date": "2024-12-10", "eligible_for_refund": False}
        },
        {
            "ticket": {"ticket_id": "E3", "sender_name": "Amit Singh", "sender_email": "amit@example.com", "subject": "Wrong item received", "body": "I received a phone case instead of a keyboard.", "order_id": "3055", "language": "en", "category_hint": "wrong_item"},
            "order": {"order_id": "3055", "product": "mechanical keyboard", "status": "delivered", "amount": 120.0, "purchase_date": "2024-12-12", "eligible_for_refund": True}
        },
        {
            "ticket": {"ticket_id": "E4", "sender_name": "Sneha Reddy", "sender_email": "sneha@example.com", "subject": "Login issues", "body": "I cannot login to my account.", "order_id": "N/A", "language": "en", "category_hint": "account_problem"},
            "order": {"order_id": "N/A", "product": "N/A", "status": "N/A", "amount": 0.0, "purchase_date": "N/A", "eligible_for_refund": False}
        },
        {
            "ticket": {"ticket_id": "E5", "sender_name": "Rahul Verma", "sender_email": "rahul@example.com", "subject": "Product question", "body": "Does this hub support 4K?", "order_id": "4012", "language": "en", "category_hint": "general_inquiry"},
            "order": {"order_id": "4012", "product": "USB-C hub", "status": "delivered", "amount": 35.0, "purchase_date": "2024-12-14", "eligible_for_refund": False}
        },
        {
            "ticket": {"ticket_id": "E6", "sender_name": "Anjali Gupta", "sender_email": "anjali@example.com", "subject": "Damaged product", "body": "The webcam arrived with a cracked lens.", "order_id": "5067", "language": "en", "category_hint": "delivery_issue"},
            "order": {"order_id": "5067", "product": "webcam", "status": "delivered", "amount": 80.0, "purchase_date": "2024-12-11", "eligible_for_refund": True}
        },
        {
            "ticket": {"ticket_id": "E7", "sender_name": "Vikram Das", "sender_email": "vikram@example.com", "subject": "Cancel my order", "body": "I want to cancel order #6023.", "order_id": "6023", "language": "en", "category_hint": "refund_request"},
            "order": {"order_id": "6023", "product": "smartwatch", "status": "processing", "amount": 250.0, "purchase_date": "2024-12-16", "eligible_for_refund": True}
        },
        {
            "ticket": {"ticket_id": "E8", "sender_name": "Kavita Rao", "sender_email": "kavita@example.com", "subject": "Shipping update", "body": "When will my tablet ship?", "order_id": "7089", "language": "en", "category_hint": "general_inquiry"},
            "order": {"order_id": "7089", "product": "tablet", "status": "processing", "amount": 350.0, "purchase_date": "2024-12-16", "eligible_for_refund": False}
        },
        {
            "ticket": {"ticket_id": "E9", "sender_name": "Sanjay Mehra", "sender_email": "sanjay@example.com", "subject": "Billing error", "body": "I was charged twice.", "order_id": "8044", "language": "en", "category_hint": "account_problem"},
            "order": {"order_id": "8044", "product": "software sub", "status": "delivered", "amount": 20.0, "purchase_date": "2024-12-15", "eligible_for_refund": True}
        },
        {
            "ticket": {"ticket_id": "E10", "sender_name": "Deepa Nair", "sender_email": "deepa@example.com", "subject": "Return request", "body": "The laptop stand is too small.", "order_id": "9011", "language": "en", "category_hint": "refund_request"},
            "order": {"order_id": "9011", "product": "laptop stand", "status": "delivered", "amount": 40.0, "purchase_date": "2024-12-13", "eligible_for_refund": True}
        }
    ],
    "medium": [
        {
            "ticket": {"ticket_id": "M1", "sender_name": "Ravi Sharma", "sender_email": "ravi@example.com", "subject": "Refund for order #1042", "body": "I want a refund for my order.", "order_id": "1042", "language": "en", "category_hint": "refund_request"},
            "order": {"order_id": "1042", "product": "wireless headphones", "status": "delivered", "amount": 150.0, "purchase_date": "2024-12-15", "eligible_for_refund": True}
        },
        {
            "ticket": {"ticket_id": "M2", "sender_name": "Priya Patel", "sender_email": "priya@example.com", "subject": "My package never arrived", "body": "Order #2091 never arrived.", "order_id": "2091", "language": "en", "category_hint": "delivery_issue"},
            "order": {"order_id": "2091", "product": "laptop stand", "status": "delayed", "amount": 45.0, "purchase_date": "2024-12-10", "eligible_for_refund": False}
        },
        {
            "ticket": {"ticket_id": "M3", "sender_name": "Amit Singh", "sender_email": "amit@example.com", "subject": "Wrong item received", "body": "I received a phone case instead of a keyboard.", "order_id": "3055", "language": "en", "category_hint": "wrong_item"},
            "order": {"order_id": "3055", "product": "mechanical keyboard", "status": "delivered", "amount": 120.0, "purchase_date": "2024-12-12", "eligible_for_refund": True}
        },
        {
            "ticket": {"ticket_id": "M4", "sender_name": "Sneha Reddy", "sender_email": "sneha@example.com", "subject": "Login issues", "body": "I cannot login to my account.", "order_id": "N/A", "language": "en", "category_hint": "account_problem"},
            "order": {"order_id": "N/A", "product": "N/A", "status": "N/A", "amount": 0.0, "purchase_date": "N/A", "eligible_for_refund": False}
        },
        {
            "ticket": {"ticket_id": "M5", "sender_name": "Rahul Verma", "sender_email": "rahul@example.com", "subject": "Product question", "body": "Does this hub support 4K?", "order_id": "4012", "language": "en", "category_hint": "general_inquiry"},
            "order": {"order_id": "4012", "product": "USB-C hub", "status": "delivered", "amount": 35.0, "purchase_date": "2024-12-14", "eligible_for_refund": False}
        },
        {
            "ticket": {"ticket_id": "M6", "sender_name": "Anjali Gupta", "sender_email": "anjali@example.com", "subject": "Damaged product", "body": "The webcam arrived with a cracked lens.", "order_id": "5067", "language": "en", "category_hint": "delivery_issue"},
            "order": {"order_id": "5067", "product": "webcam", "status": "delivered", "amount": 80.0, "purchase_date": "2024-12-11", "eligible_for_refund": True}
        },
        {
            "ticket": {"ticket_id": "M7", "sender_name": "Vikram Das", "sender_email": "vikram@example.com", "subject": "Cancel my order", "body": "I want to cancel order #6023.", "order_id": "6023", "language": "en", "category_hint": "refund_request"},
            "order": {"order_id": "6023", "product": "smartwatch", "status": "processing", "amount": 250.0, "purchase_date": "2024-12-16", "eligible_for_refund": True}
        },
        {
            "ticket": {"ticket_id": "M8", "sender_name": "Kavita Rao", "sender_email": "kavita@example.com", "subject": "Shipping update", "body": "When will my tablet ship?", "order_id": "7089", "language": "en", "category_hint": "general_inquiry"},
            "order": {"order_id": "7089", "product": "tablet", "status": "processing", "amount": 350.0, "purchase_date": "2024-12-16", "eligible_for_refund": False}
        },
        {
            "ticket": {"ticket_id": "M9", "sender_name": "Sanjay Mehra", "sender_email": "sanjay@example.com", "subject": "Billing error", "body": "I was charged twice.", "order_id": "8044", "language": "en", "category_hint": "account_problem"},
            "order": {"order_id": "8044", "product": "software sub", "status": "delivered", "amount": 20.0, "purchase_date": "2024-12-15", "eligible_for_refund": True}
        },
        {
            "ticket": {"ticket_id": "M10", "sender_name": "Deepa Nair", "sender_email": "deepa@example.com", "subject": "Return request", "body": "The laptop stand is too small.", "order_id": "9011", "language": "en", "category_hint": "refund_request"},
            "order": {"order_id": "9011", "product": "laptop stand", "status": "delivered", "amount": 40.0, "purchase_date": "2024-12-13", "eligible_for_refund": True}
        }
    ],
    "hard": [
        {
            "ticket": {"ticket_id": "H1", "sender_name": "Ravi Sharma", "sender_email": "ravi@example.com", "subject": "Refund for order #1042", "body": "I want a refund for my order.", "order_id": "1042", "language": "en", "category_hint": "refund_request"},
            "order": {"order_id": "1042", "product": "wireless headphones", "status": "delivered", "amount": 150.0, "purchase_date": "2024-12-15", "eligible_for_refund": True}
        },
        {
            "ticket": {"ticket_id": "H2", "sender_name": "Priya Patel", "sender_email": "priya@example.com", "subject": "Package missing", "body": "Mi paquete nunca llegó.", "order_id": "2091", "language": "es", "category_hint": "delivery_issue"},
            "order": {"order_id": "2091", "product": "laptop stand", "status": "delayed", "amount": 45.0, "purchase_date": "2024-12-10", "eligible_for_refund": False}
        },
        {
            "ticket": {"ticket_id": "H3", "sender_name": "Amit Singh", "sender_email": "amit@example.com", "subject": "Mauvais article", "body": "J'ai reçu un article différent de celui commandé.", "order_id": "3055", "language": "fr", "category_hint": "wrong_item"},
            "order": {"order_id": "3055", "product": "mechanical keyboard", "status": "delivered", "amount": 120.0, "purchase_date": "2024-12-12", "eligible_for_refund": True}
        },
        {
            "ticket": {"ticket_id": "H4", "sender_name": "Sneha Reddy", "sender_email": "sneha@example.com", "subject": "लॉगिन समस्या", "body": "मैं अपने खाते में लॉगिन नहीं कर पा रहा हूँ।", "order_id": "N/A", "language": "hi", "category_hint": "account_problem"},
            "order": {"order_id": "N/A", "product": "N/A", "status": "N/A", "amount": 0.0, "purchase_date": "N/A", "eligible_for_refund": False}
        },
        {
            "ticket": {"ticket_id": "H5", "sender_name": "Rahul Verma", "sender_email": "rahul@example.com", "subject": "Product question", "body": "Does this hub support 4K?", "order_id": "4012", "language": "en", "category_hint": "general_inquiry"},
            "order": {"order_id": "4012", "product": "USB-C hub", "status": "delivered", "amount": 35.0, "purchase_date": "2024-12-14", "eligible_for_refund": False}
        },
        {
            "ticket": {"ticket_id": "H6", "sender_name": "Anjali Gupta", "sender_email": "anjali@example.com", "subject": "Damaged product", "body": "The webcam arrived with a cracked lens.", "order_id": "5067", "language": "en", "category_hint": "delivery_issue"},
            "order": {"order_id": "5067", "product": "webcam", "status": "delivered", "amount": 80.0, "purchase_date": "2024-12-11", "eligible_for_refund": True}
        },
        {
            "ticket": {"ticket_id": "H7", "sender_name": "Vikram Das", "sender_email": "vikram@example.com", "subject": "Cancel my order", "body": "I want to cancel order #6023.", "order_id": "6023", "language": "en", "category_hint": "refund_request"},
            "order": {"order_id": "6023", "product": "smartwatch", "status": "processing", "amount": 250.0, "purchase_date": "2024-12-16", "eligible_for_refund": True}
        },
        {
            "ticket": {"ticket_id": "H8", "sender_name": "Kavita Rao", "sender_email": "kavita@example.com", "subject": "Shipping update", "body": "When will my tablet ship?", "order_id": "7089", "language": "en", "category_hint": "general_inquiry"},
            "order": {"order_id": "7089", "product": "tablet", "status": "processing", "amount": 350.0, "purchase_date": "2024-12-16", "eligible_for_refund": False}
        },
        {
            "ticket": {"ticket_id": "H9", "sender_name": "Sanjay Mehra", "sender_email": "sanjay@example.com", "subject": "Billing error", "body": "I was charged twice.", "order_id": "8044", "language": "hi", "category_hint": "account_problem"},
            "order": {"order_id": "8044", "product": "software sub", "status": "delivered", "amount": 20.0, "purchase_date": "2024-12-15", "eligible_for_refund": True}
        },
        {
            "ticket": {"ticket_id": "H10", "sender_name": "Deepa Nair", "sender_email": "deepa@example.com", "subject": "Return request", "body": "The laptop stand is too small.", "order_id": "9011", "language": "es", "category_hint": "refund_request"},
            "order": {"order_id": "9011", "product": "laptop stand", "status": "delivered", "amount": 40.0, "purchase_date": "2024-12-13", "eligible_for_refund": True}
        }
    ]
}


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

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Generate a fresh ticket and reset all episode state."""
        if seed is not None:
            # Reproducible: pick from fixed bank
            bank = TICKET_BANK.get(self.task_id, [])
            if not bank:
                self._ticket, self._order = ticket_gen.generate(self.task_id)
            else:
                entry = bank[seed % len(bank)]
                self._ticket = Ticket(**entry["ticket"])
                self._order = OrderRecord(**entry["order"])
        else:
            # Dynamic: generate via LLM (existing behavior)
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
