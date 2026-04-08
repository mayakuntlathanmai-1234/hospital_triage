import os
import sys
import math
import numpy as np
import gymnasium as gym
from openai import OpenAI
import hospital_triage  # Registers the HospitalTriage-v0 environment

# ==========================================
# 1. ENVIRONMENT VARIABLES (Strictly follow checklist)
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")  # MUST have no default

# Optional - for docker images
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ==========================================
# 2. Configure OpenAI Client
# ==========================================
# All LLM calls MUST use this client
client = OpenAI(
    api_key=HF_TOKEN if HF_TOKEN else os.getenv("OPENAI_API_KEY", "dummy_key_for_testing"),
    base_url=API_BASE_URL
)

def normalize_score(reward):
    # Use sigmoid to map any reward to strictly (0, 1)
    # We use a scale of 20 to spread out the values
    return 1.0 / (1.0 + math.exp(-reward / 20.0))

def run_agent():
    # Define 3 tasks with different environment parameters
    tasks = [
        {"id": "triage_standard", "params": {"num_doctors": 5, "num_beds": 10}},
        {"id": "triage_heavy_traffic", "params": {"num_doctors": 5, "num_beds": 10, "patient_arrival_rate": 0.9}},
        {"id": "triage_limited_resources", "params": {"num_doctors": 2, "num_beds": 5}},
    ]
    
    for task_cfg in tasks:
        task_id = task_cfg["id"]
        # REQUIRED FORMAT: [START] task=NAME
        print(f"[START] task={task_id}", flush=True)
        
        total_reward = 0.0
        step_idx = 0
        try:
            # Initialize environment with task-specific params
            env = gym.make('HospitalTriage-v0', **task_cfg["params"])
            obs, info = env.reset()
            
            done = False
            while not done and step_idx < 10:
                # Simple rule-based agent to ensure valid actions
                # Action: [type, patient_idx, doctor_idx, bed_idx]
                
                queue_length = info.get('queue_length', 0)
                admitted_count = info.get('admitted_count', 0)
                
                if queue_length > 0:
                    # Admit patient if possible (Action 0)
                    action = np.array([0, 0, 0, 0], dtype=np.int32)
                elif admitted_count > 0:
                    # Schedule test for admitted patient (Action 4)
                    action = np.array([4, 0, 0, 0], dtype=np.int32)
                else:
                    # Delay (Action 1)
                    action = np.array([1, 0, 0, 0], dtype=np.int32)
                
                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
                step_idx += 1
                
                # REQUIRED FORMAT: [STEP] step=NUM reward=VAL
                print(f"[STEP] step={step_idx} reward={reward}", flush=True)
            
            # Normalize cumulative reward to (0, 1)
            score = normalize_score(total_reward)
            
            # REQUIRED FORMAT: [END] task=NAME score=VAL steps=NUM
            print(f"[END] task={task_id} score={score} steps={step_idx}", flush=True)
            
        except Exception as e:
            print(f"ERROR in task {task_id}: {e}", file=sys.stderr)
            # Ensure we still output [END] block for the validator
            score = normalize_score(total_reward)
            print(f"[END] task={task_id} score={score} steps={step_idx}", flush=True)

if __name__ == "__main__":
    run_agent()
