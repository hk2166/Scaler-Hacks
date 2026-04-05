from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class CustomerState(BaseModel):
    anger_level: float = Field(0.0, ge=0.0, le=10.0)
    satisfaction: float = Field(5.0, ge=0.0, le=10.0)
    language: str = "en"
    has_been_apologized_to: bool = False
    wait_steps: int = 0


class Ticket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    sender_name: str
    sender_email: str
    order_id: Optional[str] = None
    language: str = "en"
    category_hint: Optional[str] = None  # ground truth for grader


class OrderRecord(BaseModel):
    order_id: str
    product: str
    status: str           # "delivered", "delayed", "lost", "processing"
    amount: float
    purchase_date: str
    eligible_for_refund: bool
    refund_window_days: int = 30


class Observation(BaseModel):
    ticket: Ticket
    current_step: int
    max_steps: int
    task_id: str
    customer_state: CustomerState
    last_tool_result: Optional[Dict[str, Any]] = None
    last_tool_error: Optional[str] = None
    tools_called: List[str] = Field(default_factory=list)
    done: bool = False


class Action(BaseModel):
    tool_name: str = Field(
        description="One of: lookup_order, validate_eligibility, issue_refund, escalate_to_human, send_reply, classify_ticket"
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    reason: str
    done: bool
