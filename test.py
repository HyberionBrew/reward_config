import f110_gym
import f110_orl_dataset
import gymnasium as gym

# Assuming f110_orl_dataset has already updated F110Env with get_dataset
env = gym.make('f110_with_dataset-v0',
    # only terminals are available as of tight now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal", num_agents=1,
            params=dict(vmin=0.5, vmax=2.0)),
              render_mode="human")
    )#x = f110_orl_dataset.F1tenthDatasetEnv("dw", dict())
#x.get_dataset()
# This should now be possible
tr = env.get_dataset(zarr_path="/home/fabian/f110_rl/f110-sb3/trajectories4.zarr")#print(tr["observations"][0:10])
print(tr["observations"][0:2])
print("---------")
print(env.normalize_obs_batch(tr["observations"][0:2]))
#env.get_action_space()
#print(env.action_space)
#print(env.observation_space)
"""
print(env.observation_space.shape)
print(env.action_space.shape)
print(".........")
print(env.observation_spec().shape[0])
print(env.action_spec().shape[0])
print(tr["scans"].shape)
print(tr["observations"].shape)
print(tr["actions"].shape)
print(tr["actions"][10:20])
print(tr["observations"][10:20])
"""