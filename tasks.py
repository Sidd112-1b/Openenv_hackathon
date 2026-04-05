from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from main import SupportEnv
from models import Action, Observation, Reward

app = FastAPI(title="Customer Support Triage OpenEnv")

# Global environment instance (for demo purposes, usually per-session)
env: Optional[SupportEnv] = None

class ResetRequest(BaseModel):
    task_id: Optional[str] = None

class StepRequest(BaseModel):
    action: Action

@app.post("/reset", response_model=Observation)
async def reset(request: Optional[ResetRequest] = None):
    global env
    task_id = request.task_id if request else None
    env = SupportEnv(task_id=task_id)
    return env.reset()

@app.post("/step")
async def step(request: StepRequest):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    obs, reward, done, info = env.step(request.action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
async def state():
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return env.state()

@app.get("/health")
async def health():
    return {"status": "healthy", "env": "customer-support-triage"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
