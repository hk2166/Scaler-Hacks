"""
Baseline inference script — Scaler x Meta Hackathon
====================================================
Reads credentials from environment variables:
  API_BASE_URL      LLM endpoint (required, has default)
  MODEL_NAME        Model identifier (required, has default)
  HF_TOKEN          HuggingFace API key (required, no default)
  LOCAL_IMAGE_NAME  Docker image name for from_docker_image() (optional)
"""
from __future__ import annotations
import os, json, textwrap
from openai import OpenAI
from environment.env import CustomerServiceEnv
from environment.models import Action

# Environment variables — defaults only for API_BASE_URL and MODEL_NAME
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")          # no default — must be set
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional — used by from_docker_image()

TASKS       = ["easy", "medium", "hard"]
MAX_STEPS   = {"easy": 5, "medium": 8, "hard": 12}
TEMPERATURE = 0.2
MAX_TOKENS  = 800
FALLBACK    = Action(
    tool_name="send_reply",
    parameters={"reply_text": "Thank you for contacting us. We are looking into your issue and will get back to you shortly."}
)

SYSTEM_PROMPT = textwrap.dedent("""
    You are a customer service agent. Follow this EXACT sequence every episode.

    TOOLS — call them exactly as shown (parameter names matter):
      {"tool_name": "classify_ticket",      "parameters": {"category": "<one of the 5 below>"}}
      {"tool_name": "lookup_order",         "parameters": {"order_id": "<id from ticket>"}}
      {"tool_name": "validate_eligibility", "parameters": {}}
      {"tool_name": "issue_refund",         "parameters": {}}
      {"tool_name": "escalate_to_human",    "parameters": {"reason": "<why you cannot resolve>"}}
      {"tool_name": "send_reply",           "parameters": {"reply_text": "<your full reply>"}}

    CATEGORIES for classify_ticket (use key "category", NOT "classification"):
      refund_request   -> customer wants money back or return
      delivery_issue   -> package late, lost, damaged in transit
      wrong_item       -> received different product than ordered
      account_problem  -> login, billing, subscription issues
      general_inquiry  -> anything else

    IMPORTANT: The ticket includes a "Category hint" field — use it directly
    as your "category" value in classify_ticket. Do not guess.

    REQUIRED SEQUENCE (never skip steps):
      1. classify_ticket   — always first
      2. lookup_order      — use order_id from the ticket
      3. validate_eligibility — always after lookup
      4. issue_refund      — if eligible_for_refund is true
         OR escalate_to_human — if eligible_for_refund is false
      5. send_reply        — ALWAYS last. Your reply MUST:
         - Open with positive reassurance ("I'm happy to help resolve this for you!")
         - Acknowledge the issue briefly ("I understand this has been inconvenient")
         - Clearly state what action was taken (refund issued / escalated to our team)
         - Close with enthusiasm ("We truly value you and are pleased to assist!")
         - Use words like: happy, pleased, delighted, wonderful, excellent, resolved
         - Avoid starting with "sorry" or "apologize" — lead with solutions, not regret
         - Be at least 3 sentences long

    ERROR HANDLING:
      - "Transient network error" -> retry the SAME tool with SAME parameters
      - "Rate limit exceeded"     -> skip that tool, go to next step
      - "already processed"       -> go directly to send_reply

    NEVER use "classification" as a key — always "category".
    NEVER call send_reply before step 4.
    NEVER end without send_reply.

    Respond ONLY with a single JSON object. No explanation. No markdown.
""").strip()


def build_prompt(obs: dict) -> str:
    ticket = obs["ticket"]
    cstate = obs["customer_state"]
    last_error = obs.get("last_tool_error") or "None"
    steps_left = obs["max_steps"] - obs["current_step"]

    error_hint = ""
    if "Transient" in last_error:
        last_tool = obs["tools_called"][-1] if obs["tools_called"] else "unknown"
        error_hint = f"\n!! Retry '{last_tool}' with identical parameters now."
    elif "Rate limit" in last_error:
        error_hint = "\n!! Rate limit — skip to next step in sequence."
    elif "already processed" in last_error or "already escalated" in last_error:
        error_hint = "\n!! Already resolved — call send_reply now."
    elif "Invalid category" in last_error:
        error_hint = '\n!! Use parameter name "category" not "classification".'

    return textwrap.dedent(f"""
        TICKET
        ------
        From: {ticket['sender_name']} <{ticket['sender_email']}>
        Subject: {ticket['subject']}
        Body: {ticket['body']}
        Order ID: {ticket.get('order_id', 'N/A')}
        Language: {ticket.get('language', 'en')}
        Category hint: {ticket.get('category_hint', 'general_inquiry')}

        STATE
        -----
        Step: {obs['current_step']} / {obs['max_steps']}  ({steps_left} steps remaining)
        Customer anger: {cstate['anger_level']:.1f} / 10
        Tools already used: {obs['tools_called']}
        Last result: {json.dumps(obs.get('last_tool_result') or {})}
        Last error:  {last_error}{error_hint}

        What is your next action? Reply with JSON only.
    """).strip()


def parse_action(text: str) -> Action:
    try:
        # Strip all common markdown fence variants
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ``` etc)
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        data = json.loads(cleaned)
        return Action(
            tool_name=data["tool_name"],
            parameters=data.get("parameters", {}),
        )
    except Exception as e:
        print(f"           [parse_action FAILED] raw={text[:120]!r} err={e}")
        return FALLBACK


def run_task(client: OpenAI, task_id: str) -> float:
    env = CustomerServiceEnv(task_id)
    obs = env.reset()

    # START log — required structured format
    print(json.dumps({
        "event": "START",
        "task_id": task_id,
        "subject": obs.ticket.subject[:60],
        "order_id": obs.ticket.order_id,
        "language": obs.ticket.language,
    }))

    final_score = 0.0
    for step in range(1, MAX_STEPS[task_id] + 1):
        if obs.done:
            break

        prompt = build_prompt(obs.dict())
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = completion.choices[0].message.content or ""
        except Exception as e:
            raw = ""
            # STEP log for LLM error
            print(json.dumps({"event": "STEP", "task_id": task_id, "step": step,
                              "error": str(e)}))

        action = parse_action(raw)

        obs, reward, done, _ = env.step(action)

        # STEP log — required structured format
        print(json.dumps({
            "event": "STEP",
            "task_id": task_id,
            "step": step,
            "tool": action.tool_name,
            "params": action.parameters,
            "tool_error": obs.last_tool_error,
            "anger": round(obs.customer_state.anger_level, 2),
            "done": done,
        }))

        if done:
            final_score = reward.score
            break

    # END log — required structured format
    print(json.dumps({
        "event": "END",
        "task_id": task_id,
        "score": round(final_score, 4),
        "breakdown": getattr(reward, "breakdown", {}),
    }))

    return final_score


def main():
    # Fail fast with a clear message if HF_TOKEN is missing
    if not HF_TOKEN:
        raise EnvironmentError(
            "Missing HF_TOKEN environment variable. "
            "Set it to your HuggingFace API key before running."
        )

    # All LLM calls use the OpenAI client configured via API_BASE_URL, MODEL_NAME, HF_TOKEN
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    scores = {}
    for task_id in TASKS:
        scores[task_id] = run_task(client, task_id)

    avg = sum(scores.values()) / len(scores)
    print(json.dumps({"event": "SUMMARY", "scores": scores, "average": round(avg, 4)}))


if __name__ == "__main__":
    main()