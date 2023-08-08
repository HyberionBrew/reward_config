import f110_gym
# print classes of f110_gym
print(dir(f110_gym))
from .dataset_env import F1tenthDatasetEnv

# Monkey-patching
f110_gym.F1tenthDatasetEnv = F1tenthDatasetEnv