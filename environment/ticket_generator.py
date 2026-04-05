from __future__ import annotations
import os, json, random, uuid
from openai import OpenAI
from environment.models import Ticket, OrderRecord
from dotenv import load_dotenv
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

LANGUAGES = {"easy": ["en"], "medium": ["en"], "hard": ["en", "hi", "es", "fr"]}

CATEGORIES = ["refund_request", "delivery_issue", "wrong_item", "account_problem", "general_inquiry"]

PRODUCTS = ["wireless headphones", "laptop stand", "mechanical keyboard",
            "USB-C hub", "webcam", "phone case", "smartwatch", "tablet"]

ORDER_STATUSES = {
    "refund_request":   ("delivered", True),
    "delivery_issue":   ("delayed",   False),
    "wrong_item":       ("delivered", True),
    "account_problem":  ("processing",False),
    "general_inquiry":  ("delivered", False),
}


def _make_order(category: str) -> OrderRecord:
    status, eligible = ORDER_STATUSES.get(category, ("delivered", False))
    return OrderRecord(
        order_id=f"ORD-{random.randint(10000,99999)}",
        product=random.choice(PRODUCTS),
        status=status,
        amount=round(random.uniform(20, 400), 2),
        purchase_date="2024-12-15",
        eligible_for_refund=eligible,
        refund_window_days=30,
    )


def _generate_ticket_llm(task_id: str, category: str, order: OrderRecord, lang: str) -> Ticket:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy-key-for-fallback")
    except Exception:
        client = None

    lang_instruction = {
        "en": "Write in English.",
        "hi": "Write entirely in Hindi (Devanagari script).",
        "es": "Write entirely in Spanish.",
        "fr": "Write entirely in French.",
    }.get(lang, "Write in English.")

    tone = random.choice(["frustrated", "polite but firm", "very angry", "confused", "desperate"])

    system = (
        "You are generating realistic customer support emails for a training dataset. "
        "Respond with ONLY a valid JSON object, no markdown, no explanation."
    )
    user = f"""Generate a customer support email with this context:
- Category: {category}
- Tone: {tone}
- Product: {order.product}
- Order ID: {order.order_id}
- Order status: {order.status}
- Amount paid: ${order.amount}
- {lang_instruction}

Return JSON with exactly these keys:
{{
  "subject": "...",
  "body": "...",
  "sender_name": "...",
  "sender_email": "..."
}}"""

    try:
        if client is None:
            raise RuntimeError("OpenAI client could not be created (missing credentials).")
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            max_tokens=500,
            temperature=0.9,
        )
        raw = resp.choices[0].message.content or "{}"
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(raw)
    except Exception:
        # Graceful fallback so env never crashes
        data = {
            "subject": f"Issue with my {order.product} order {order.order_id}",
            "body": f"Hi, I have a problem with order {order.order_id} for a {order.product}. Please help.",
            "sender_name": "Alex Johnson",
            "sender_email": "alex.johnson@example.com",
        }

    return Ticket(
        ticket_id=str(uuid.uuid4())[:8].upper(),
        subject=data.get("subject", "Support request"),
        body=data.get("body", "Please help me."),
        sender_name=data.get("sender_name", "Customer"),
        sender_email=data.get("sender_email", "customer@example.com"),
        order_id=order.order_id,
        language=lang,
        category_hint=category,
    )


def generate(task_id: str) -> tuple[Ticket, OrderRecord]:
    """Main entry point. Returns (ticket, order) for the given task difficulty."""
    category = random.choice(CATEGORIES)
    lang = random.choice(LANGUAGES.get(task_id, ["en"]))
    order = _make_order(category)
    ticket = _generate_ticket_llm(task_id, category, order, lang)
    return ticket, order
