#!/usr/bin/env python3
"""
Offline oracle baseline — no LLM or API key required.
Runs a hardcoded perfect-action policy against seed=0 tickets.
Writes results to baseline_scores.json.
"""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from environment.env import CustomerServiceEnv
from environment.models import Action

ORACLE_SEQUENCES = {
    "easy": [
        Action(tool_name="classify_ticket",    parameters={"category": "refund_request"}),
        Action(tool_name="send_reply",         parameters={"reply_text": "I'm happy to help resolve this for you! Your request has been received and we're looking into it right away. We truly value your business!"}),
    ],
    "medium": [
        Action(tool_name="classify_ticket",    parameters={"category": "refund_request"}),
        Action(tool_name="lookup_order",       parameters={"order_id": "1042"}),
        Action(tool_name="validate_eligibility", parameters={}),
        Action(tool_name="issue_refund",       parameters={}),
        Action(tool_name="send_reply",         parameters={"reply_text": "I'm pleased to confirm your refund has been processed! You'll see it in 3-5 business days. We appreciate your patience!"}),
    ],
    "hard": [
        Action(tool_name="classify_ticket",    parameters={"category": "refund_request"}),
        Action(tool_name="lookup_order",       parameters={"order_id": "1042"}),
        Action(tool_name="validate_eligibility", parameters={}),
        Action(tool_name="issue_refund",       parameters={}),
        Action(tool_name="send_reply",         parameters={"reply_text": "I'm delighted to confirm your refund is approved! This will be reflected within 5 business days. Thank you for your patience and we truly value you!"}),
    ],
}

def run_oracle(task_id: str) -> float:
    env = CustomerServiceEnv(task_id)
    obs = env.reset(seed=0)
    rewards = []
    for action in ORACLE_SEQUENCES[task_id]:
        if obs.done:
            break
        obs, reward, done, _ = env.step(action)
        rewards.append(reward.score)
    score = rewards[-1] if rewards else 0.0
    print(f"  {task_id}: score={score:.4f}  rewards={[round(r,4) for r in rewards]}")
    return score

if __name__ == "__main__":
    print("Running offline oracle baseline (no LLM required)...\n")
    results = {}
    for task_id in ["easy", "medium", "hard"]:
        results[task_id] = run_oracle(task_id)

    avg = sum(results.values()) / len(results)
    output = {"tasks": results, "average": round(avg, 4), "policy": "oracle", "seed": 0}
    
    out_path = pathlib.Path(__file__).parent.parent / "baseline_scores.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nAverage: {avg:.4f}")
    print(f"Written to {out_path}")
