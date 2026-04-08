"""Demo script showcasing the Hospital Triage Environment."""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import hospital_triage


def demo_random_agent():
    """Demo with random agent."""
    print("\n" + "="*60)
    print("DEMO 1: Random Agent")
    print("="*60)
    
    env = gym.make('HospitalTriage-v0', render_mode='human')
    obs, info = env.reset()
    
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    env.render()
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Episode Stats: {info}")
    env.close()


def demo_rule_based_agent():
    """Demo with simple rule-based agent."""
    print("\n" + "="*60)
    print("DEMO 2: Rule-Based Agent")
    print("="*60)
    
    env = gym.make('HospitalTriage-v0', render_mode='human')
    obs, info = env.reset()
    
    total_reward = 0
    for step in range(100):
        # Simple heuristic: prioritize critical patients
        # Action format: [action_type, patient_id, doctor_id, bed_id]
        
        obs_tensor = obs
        queue_size = info.get('queue_length', 0)
        
        if queue_size > 0:
            # Try to admit the first patient in queue
            action = np.array([0, 0, step % len(env.state.doctors), 
                             step % len(env.state.beds)], dtype=np.int32)
        else:
            # If queue is empty, take a random action
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    env.render()
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Episode Stats: {info}")
    env.close()


def demo_trained_agent():
    """Demo with trained RL agent using PPO."""
    print("\n" + "="*60)
    print("DEMO 3: PPO Trained Agent")
    print("="*60)
    print("Training a PPO agent for 5000 timesteps...")
    
    # Create parallelized environment for faster training
    env = make_vec_env(
        lambda: gym.make('HospitalTriage-v0'),
        n_envs=4
    )
    
    # Train PPO agent
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=32,
        verbose=0,
    )
    
    model.learn(total_timesteps=5000)
    env.close()
    
    # Test trained agent
    print("\nTesting trained agent...")
    test_env = gym.make('HospitalTriage-v0', render_mode='human')
    obs, info = test_env.reset()
    
    total_reward = 0
    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    test_env.render()
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Episode Stats: {info}")
    test_env.close()


def demo_environment_features():
    """Demo environment features and API."""
    print("\n" + "="*60)
    print("DEMO 4: Environment Features")
    print("="*60)
    
    env = gym.make(
        'HospitalTriage-v0',
        num_doctors=15,
        num_beds=30,
        num_lab_tests=8,
        patient_arrival_rate=0.5,
        render_mode='ansi'
    )
    
    print(f"\nAction Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    obs, info = env.reset(seed=42)
    print(f"\nInitial Observation shape: {obs.shape}")
    print(f"Initial Info: {info}")
    
    # Take a few steps
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Info: {info}")
        print(env.render())
    
    env.close()


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("Hospital Triage Environment Demo")
    print("="*60)
    
    try:
        # Demo 1: Random agent
        demo_random_agent()
    except Exception as e:
        print(f"Random agent demo failed: {e}")
    
    try:
        # Demo 2: Rule-based agent
        demo_rule_based_agent()
    except Exception as e:
        print(f"Rule-based agent demo failed: {e}")
    
    try:
        # Demo 3: Trained agent
        demo_trained_agent()
    except Exception as e:
        print(f"Trained agent demo failed: {e}")
    
    try:
        # Demo 4: Environment features
        demo_environment_features()
    except Exception as e:
        print(f"Features demo failed: {e}")


if __name__ == "__main__":
    main()
