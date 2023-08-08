from gym.envs.registration import register
from .dataset_env import F1tenthDatasetEnv

# This will override the existing registration
register(
    id='f110_with_dataset-v0',
    entry_point='f110_orl_dataset.dataset_env:F1tenthDatasetEnv',
)