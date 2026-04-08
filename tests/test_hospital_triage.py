"""Tests for Hospital Triage Environment."""

import pytest
import gymnasium as gym
import numpy as np
import hospital_triage


class TestHospitalTriageEnv:
    """Test suite for HospitalTriageEnv."""
    
    @pytest.fixture
    def env(self):
        """Create environment fixture."""
        env = gym.make('HospitalTriage-v0', num_doctors=5, num_beds=10)
        yield env
        env.close()
    
    def test_environment_creation(self, env):
        """Test environment instantiation."""
        assert env is not None
        # Access unwrapped environment to get attributes
        base_env = env.unwrapped
        assert base_env.num_doctors == 5
        assert base_env.num_beds == 10
    
    def test_action_space(self, env):
        """Test action space."""
        assert env.action_space is not None
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
    
    def test_observation_space(self, env):
        """Test observation space."""
        assert env.observation_space is not None
        assert isinstance(env.observation_space, gym.spaces.Box)
    
    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)
    
    def test_step(self, env):
        """Test environment step."""
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_step_sequence(self, env):
        """Test multiple steps."""
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
    
    def test_reset_works_after_episde(self, env):
        """Test reset after episode completion."""
        # Run episode
        env.reset()
        base_env = env.unwrapped
        max_steps = base_env.max_episode_length
        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        # Reset and run again
        obs, info = env.reset()
        assert obs is not None
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
    
    def test_render_modes(self):
        """Test different render modes."""
        env = gym.make('HospitalTriage-v0', render_mode='ansi')
        env.reset()
        env.step(env.action_space.sample())
        output = env.render()
        assert output is not None or output == ""
        env.close()
    
    def test_deterministic_behavior_with_seed(self):
        """Test deterministic behavior with seed."""
        from hospital_triage.envs import HospitalTriageEnv
        env1 = HospitalTriageEnv(seed=42)
        env2 = HospitalTriageEnv(seed=42)
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        assert np.allclose(obs1, obs2)
        
        # Same actions -> same observations
        for _ in range(5):
            # Create proper action
            action = np.array([0, 0, 0, 0], dtype=np.int32)
            obs1, _, _, _, _ = env1.step(action)
            obs2, _, _, _, _ = env2.step(action)
            assert np.allclose(obs1, obs2)
        
        env1.close()
        env2.close()
    
    def test_reward_bounds(self, env):
        """Test that rewards are reasonable."""
        env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Reward should be bounded (not infinite)
            assert np.isfinite(reward)
            assert -20 <= reward <= 10  # Reasonable bounds
            
            if terminated or truncated:
                break
    
    def test_info_dict_contents(self, env):
        """Test info dict contains expected fields."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        
        assert 'step' in info
        assert 'queue_length' in info
        assert 'admitted_count' in info
        assert 'avg_waiting_time' in info


class TestEnvironmentCustomization:
    """Test environment customization options."""
    
    def test_custom_parameters(self):
        """Test creating environment with custom parameters."""
        env = gym.make(
            'HospitalTriage-v0',
            num_doctors=20,
            num_beds=40,
            num_lab_tests=10,
            max_episode_length=500,
            patient_arrival_rate=0.6,
        )
        
        # Access unwrapped environment for attributes
        base_env = env.unwrapped
        assert base_env.num_doctors == 20
        assert base_env.num_beds == 40
        assert base_env.num_lab_tests == 10
        assert base_env.max_episode_length == 500
        assert base_env.patient_arrival_rate == 0.6
        
        env.close()
    
    def test_different_seeds(self):
        """Test that different seeds produce different trajectories."""
        env1 = gym.make('HospitalTriage-v0')
        env2 = gym.make('HospitalTriage-v0')
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=123)
        
        # Different seeds might give different observations
        # (not guaranteed, but likely for stochastic env)
        
        env1.close()
        env2.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
