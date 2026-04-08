"""Hospital Triage and Resource Allocation Environment."""

from gymnasium.envs.registration import register

# Register the environment
register(
    id='HospitalTriage-v0',
    entry_point='hospital_triage.envs:HospitalTriageEnv',
    max_episode_steps=1000,
)

__version__ = '0.1.0'
