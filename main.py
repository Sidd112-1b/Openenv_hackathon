import os
import json
import textwrap
import time
from typing import List, Optional
from openai import OpenAI
from main import SupportEnv
from models import Action, Observation
from tasks import TASKS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME") or "gemini-3-flash-preview"
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "customer-support-triage")
MAX_STEPS = 10
TEMPERATURE = 0.0
MAX_TOKENS = 150

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, obs: Observation) -> Action:
    system_prompt = textwrap.dedent(
        """
        You are a customer support agent. Your goal is to resolve the user's ticket efficiently.
        Available tools:
        - search: {"query": "..."} - Search for billing or logs.
        - lookup_policy: {"query": "..."} - Look up company policies.
        - reply: {"message": "..."} - Reply to the user.
        - escalate: {"reason": "..."} - Escalate complex technical issues.

        Respond ONLY with a JSON object representing the action, e.g.:
        {"tool": "search", "args": {"query": "billing history"}}
        """
    ).strip()
    
    user_prompt = f"""Ticket ID: {obs.ticket_id}
Content: {obs.content}
History: {obs.history}
User Context: {obs.user_context}
Available Tools: {obs.available_tools}

What is your next action?"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        action_data = json.loads(completion.choices[0].message.content)
        return Action(tool=action_data["tool"], args=action_data.get("args", {}))
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return Action(tool="reply", args={"message": "I'm sorry, I'm having trouble processing your request."})

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    total_start_time = time.time()

    for task in TASKS:
        # Check runtime constraint (less than 20 min total)
        if time.time() - total_start_time > 1100: # 18.3 minutes
            break

        env = SupportEnv(task_id=task.id)
        rewards: List[float] = []
        steps_taken = 0
        success = False
        
        log_start(task=task.id, env=BENCHMARK, model=MODEL_NAME)
        
        obs = env.reset()
        done = False
        
        for step in range(1, MAX_STEPS + 1):
            action = get_model_action(client, obs)
            
            obs, reward_obj, done, info = env.step(action)
            reward = reward_obj.value
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action.tool, reward=reward, done=done, error=None)
            
            if done:
                break
        
        total_reward = sum(rewards)
        # Normalize score to [0, 1] based on task expectations
        score = min(max(total_reward, 0.0), 1.0)
        success = score >= 0.9
        
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
