from __future__ import annotations
import random
from typing import Any, Dict
from environment.models import OrderRecord


class ToolError(Exception):
    """Raised when a tool call fails due to a realistic error condition."""


class ToolRegistry:
    """
    Simulates a set of real customer-service backend tools.
    Tracks call history to enforce:
      - Rate limits (max calls per tool per episode)
      - Double-call guards (e.g. refund already issued)
      - Flaky network errors (random, recoverable)
      - Dependency enforcement (must lookup before validate, etc.)
    """

    RATE_LIMITS: Dict[str, int] = {
        "lookup_order":        3,
        "validate_eligibility":2,
        "issue_refund":        1,   # can only refund once
        "escalate_to_human":   1,
        "send_reply":          2,
        "classify_ticket":     2,
    }

    FLAKINESS: Dict[str, float] = {
        "lookup_order":        0.10,  # 10% chance of transient network error
        "validate_eligibility":0.05,
        "issue_refund":        0.08,
        "escalate_to_human":   0.03,
        "send_reply":          0.02,
        "classify_ticket":     0.0,
    }

    def __init__(self, order: OrderRecord):
        self.order = order
        self._call_counts: Dict[str, int] = {}
        self._order_fetched = False
        self._eligibility_checked = False
        self._refund_issued = False
        self._escalated = False
        self._reply_sent = False

    def call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool call. Returns result dict or raises ToolError."""
        self._check_rate_limit(tool_name)
        self._maybe_flake(tool_name)

        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            raise ToolError(f"Unknown tool: '{tool_name}'. Valid tools: {list(self.RATE_LIMITS)}")

        result = handler(parameters)
        self._call_counts[tool_name] = self._call_counts.get(tool_name, 0) + 1
        return result

    # ── private guards ──────────────────────────────────────────────────────

    def _check_rate_limit(self, tool_name: str) -> None:
        limit = self.RATE_LIMITS.get(tool_name, 99)
        used  = self._call_counts.get(tool_name, 0)
        if used >= limit:
            raise ToolError(
                f"Rate limit exceeded for '{tool_name}' "
                f"(called {used}/{limit} times this episode)."
            )

    def _maybe_flake(self, tool_name: str) -> None:
        prob = self.FLAKINESS.get(tool_name, 0.0)
        if random.random() < prob:
            raise ToolError(
                f"Transient network error calling '{tool_name}'. "
                "Please retry once."
            )

    # ── tool implementations ─────────────────────────────────────────────────

    def _tool_classify_ticket(self, params: Dict[str, Any]) -> Dict[str, Any]:
        category = params.get("category", "").strip().lower()
        valid = {"refund_request", "delivery_issue", "wrong_item",
                 "account_problem", "general_inquiry"}
        if category not in valid:
            raise ToolError(
                f"Invalid category '{category}'. Must be one of: {sorted(valid)}"
            )
        return {"status": "ok", "classified_as": category}

    def _tool_lookup_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        order_id = params.get("order_id", "").strip()
        if not order_id:
            raise ToolError("lookup_order requires 'order_id' parameter.")
        if order_id != self.order.order_id:
            raise ToolError(
                f"Order '{order_id}' not found. "
                "Check the order ID in the ticket."
            )
        self._order_fetched = True
        return {
            "status": "ok",
            "order_id":   self.order.order_id,
            "product":    self.order.product,
            "order_status": self.order.status,
            "amount":     self.order.amount,
            "purchase_date": self.order.purchase_date,
        }

    def _tool_validate_eligibility(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self._order_fetched:
            raise ToolError(
                "Cannot validate eligibility before looking up the order. "
                "Call lookup_order first."
            )
        self._eligibility_checked = True   # ← guard: refund/escalate now know validate ran
        return {
            "status": "ok",
            "eligible_for_refund": self.order.eligible_for_refund,
            "refund_window_days":  self.order.refund_window_days,
            "reason": (
                "Order is within refund window and meets policy criteria."
                if self.order.eligible_for_refund
                else "Order does not meet refund eligibility criteria."
            ),
        }

    def _tool_issue_refund(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self._order_fetched:
            raise ToolError("Must call lookup_order before issuing a refund.")
        if not self._eligibility_checked:
            raise ToolError(
                "Must call validate_eligibility before issuing a refund. "
                "Run validate_eligibility first."
            )
        if self._refund_issued:
            raise ToolError(
                "Refund already processed for this order. "
                "Cannot issue a second refund."
            )
        if not self.order.eligible_for_refund:
            raise ToolError(
                f"Order {self.order.order_id} is not eligible for a refund "
                "per current policy."
            )
        self._refund_issued = True
        return {
            "status": "ok",
            "refund_id": f"REF-{random.randint(100000, 999999)}",
            "amount_refunded": self.order.amount,
            "message": "Refund initiated. Customer will see it in 3–5 business days.",
        }

    def _tool_escalate_to_human(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._escalated:
            raise ToolError("Ticket already escalated. Cannot escalate twice.")
        self._escalated = True
        reason = params.get("reason", "No reason provided.")
        return {
            "status": "ok",
            "escalation_id": f"ESC-{random.randint(1000, 9999)}",
            "assigned_to": "Tier-2 Support",
            "reason_logged": reason,
            "eta_hours": random.randint(2, 24),
        }

    def _tool_send_reply(self, params: Dict[str, Any]) -> Dict[str, Any]:
        reply_text = params.get("reply_text", "").strip()
        if not reply_text:
            raise ToolError("send_reply requires a non-empty 'reply_text' parameter.")
        if len(reply_text) < 20:
            raise ToolError(
                "Reply is too short (< 20 chars). "
                "Please provide a complete, helpful response."
            )
        self._reply_sent = True
        return {
            "status": "ok",
            "message": "Reply sent to customer.",
            "preview": reply_text[:120] + ("..." if len(reply_text) > 120 else ""),
        }
