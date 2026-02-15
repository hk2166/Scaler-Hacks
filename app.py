from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn

from environment.env import CustomerServiceEnv
from environment.models import Action

app = FastAPI(
    title="Customer Service Agent Environment",
    description="OpenEnv-compliant RL environment for customer service agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One env instance per task — stateful per process
_envs: Dict[str, CustomerServiceEnv] = {
    "easy":   CustomerServiceEnv("easy"),
    "medium": CustomerServiceEnv("medium"),
    "hard":   CustomerServiceEnv("hard"),
}


class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequest(BaseModel):
    task_id: str = "easy"
    tool_name: str
    parameters: Dict[str, Any] = {}


@app.get("/")
def root():
    return {
        "name": "Customer Service Agent Environment",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest):
    if req.task_id not in _envs:
        raise HTTPException(400, f"Unknown task_id '{req.task_id}'. Use: easy, medium, hard")
    obs = _envs[req.task_id].reset()
    return obs.dict()


@app.post("/step")
def step(req: StepRequest):
    if req.task_id not in _envs:
        raise HTTPException(400, f"Unknown task_id '{req.task_id}'.")
    action = Action(tool_name=req.tool_name, parameters=req.parameters)
    try:
        obs, reward, done, info = _envs[req.task_id].step(action)
    except AssertionError as e:
        raise HTTPException(400, str(e))
    return {
        "observation": obs.dict(),
        "reward":      reward.dict(),
        "done":        done,
        "info":        info,
    }


@app.get("/state")
def state(task_id: str = "easy"):
    if task_id not in _envs:
        raise HTTPException(400, f"Unknown task_id '{task_id}'.")
    return _envs[task_id].state()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)






