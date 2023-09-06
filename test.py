import f110_gym
import f110_orl_dataset
import gymnasium as gym

# Assuming f110_orl_dataset has already updated F110Env with get_dataset
env = gym.make('f110_with_dataset-v0',
    # only terminals are available as of tight now 
    set_terminals=True,
    flatten_obs=True,
    flatten_acts=True,
    laser_obs=True,
    flatten_trajectories=False,
    subsample_laser=10,
    padd_trajectories=False,
    max_trajectories = None,
    progress_as_reward=False,
        **dict(name='f110_with_dataset-v0',
            map="Infsaal")
    )#x = f110_orl_dataset.F1tenthDatasetEnv("dw", dict())
#x.get_dataset()
# This should now be possible
tr = env.get_dataset(zarr_path="/mnt/hdd2/fabian/f1tenth_dope/ws_ope/f1tenth_orl_dataset/data/trajectories.zarr")#print(tr["observations"][0:10])
#env.get_action_space()
#print(env.action_space)
#print(env.observation_space)
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