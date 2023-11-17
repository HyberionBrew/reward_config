import numpy as np
from .config_new import Config
from f110_gym.envs.track import Track
import gymnasium as gym

class ProgressReward:
    def __init__(self, multiplier=100.0):
        self.multiplier = multiplier
    """
    (Batch, trajectory_length, obs_dim/act_dim) -> input
    """
    def __call__(self, obs, action, laser_scan):
        #if obs.shape[1] <= 1:
        #    return np.zeros((obs.shape[0], obs.shape[1]))
        assert len(obs.shape) == 3
        assert obs.shape[1] > 1
        progress = obs [..., -2:]
        # assert progress.shape[-1] == 2
        # sin and cos progress to progress
        progress = np.arctan2(progress[...,0], progress[...,1])
        # all where progress < 0 we add pi
        progress += np.pi
        progress = progress/ (2*np.pi)
        #print(progress.min())
        #print(progress.max())
        assert (progress.min() >= 0) and (progress.max() <= 1)
        #print(progress)
        #print(progress.shape)
        # along the second dimension we take the diff
        delta_progress = np.diff(progress, axis=1)
        # need to handle the case where we cross the 0/1 boundary
        # find indices where delta_progress < -0.9
        indices = np.where(delta_progress < -0.9)
        # at these indices we need to add 1 to delta_progress
        delta_progress[indices] += 1
        delta_progress *= self.multiplier
        # max between 0 and delta_progress
        delta_progress = np.maximum(0, delta_progress)
        reward = delta_progress
        # prepend a zero to the reward
        reward = np.concatenate([np.zeros_like(reward[...,0:1]), reward], axis=-1)
        return reward

class MinActReward:
    def __init__(self,low_steering, high_steering):
        self.low_steering = low_steering
        self.high_steering = high_steering
    """
    @param obs: observation with the shape (batch_size, trajectory_length, obs_dim).
    @returns reward with the shape (batch_size, trajectory_length)
    """
    def __call__(self, obs, action, laser_scan):
        delta_steering = np.abs(action[:,:,0])
        normalized_steering = (delta_steering / self.high_steering)**2
        inverse = 1 - normalized_steering
        reward = inverse
        assert reward.shape == (obs.shape[0], obs.shape[1])
        return reward


# harder to test right now, so implementing & testing later
class MinLidarReward:
    def __init__(self, high=0.15):
        self.high = high
    def __call__(self,obs):
        pass

class RacelineDeltaReward:
    def __init__(self, track:Track, max_delta=2.0):
        xs = track.raceline.xs
        ys = track.raceline.ys
        self.raceline = np.stack([xs,ys], axis=1)
        self.largest_delta_observed = max_delta

    def __call__(self, obs, action, laser_scan) -> float:
        pose = obs[...,:2]
        # batch_data shape becomes (batch_size, timesteps, 1, 2)
        # racing_line shape becomes (1, 1, points, 2)
        pose_exp = np.expand_dims(pose, axis=2)
        racing_line_exp = np.expand_dims(self.raceline, axis=0)
        racing_line_exp = np.expand_dims(racing_line_exp, axis=0)
        #distances = np.sum((racing_line_exp - np.array(pose_exp))**2, axis=1)
        #print(distances.shape)
        #min_distance_squared = np.min(distances,axis=1)
        squared_distances = np.sum((pose_exp - racing_line_exp) ** 2, axis=-1)
        # print(min_distance_squared.shape)
        min_distances = np.sqrt(np.min(squared_distances, axis=-1, keepdims=True))
        #clip reward to be between 0 and largest_delta
        min_distances = np.clip(min_distances, 0, self.largest_delta_observed)
        min_distance_norm = min_distances / self.largest_delta_observed
        reward = 1 - min_distance_norm
        reward = reward **2
        # print(reward.shape)
        # remove last dimension
        reward = reward[...,0]
        return reward

# CURRENTLY DOES NOT WORK WITH LIDAR!
#TODO! CAREFULL WHEN ADDING LIDAR TO NOT BREAK SOME REWARDS
class MixedReward:
    def __init__(self, env:gym.Env, config):
        # self.config = config
        self.env = env
        self.rewards = []
        self.config = config
        self.add_rewards_based_on_config(config)

    def add_rewards_based_on_config(self,config):
        self.rewards = []
        if config.has_progress_reward():
            self.rewards.append(ProgressReward())
        if config.has_min_action_reward():
            self.rewards.append(MinActReward(self.env.action_space.low[0][0],
                                            self.env.action_space.high[0][0]))
        if config.has_min_lidar_reward():
            self.rewards.append(MinLidarReward())
        if config.has_raceline_delta_reward():
            self.rewards.append(RacelineDeltaReward(self.env.track))
        if config.has_min_steering_reward():
            pass
            #self.rewards.append(MinSteeringReward())
    """
    @param obs: observation with the shape (batch_size, trajectory_length, obs_dim).
    Trajectory length needs to be at least > 1 for certain rewards.
    @param action: action with the shape (batch_size, trajectory_length, action_dim)
    """
    def __call__(self, obs, action, collision, done, laser_scan=None):
        assert obs.shape[:-1] == action.shape[:-1]
        assert len(obs.shape) == 3
        assert obs.shape[-1] == 11
        # need to handle laser scans somehow in the future

        # empty rewards array to collect the rewards
        rewards = np.zeros((obs.shape[0], obs.shape[1]))
        # now we need to handle each of the rewards
        #print("***", obs.shape)
        for reward in self.rewards:
            rewards += reward(obs, action, laser_scan)
        #print(rewards.shape)
        # where collision is true set the reward to -10
        rewards[collision] = self.config.collision_penalty
        return rewards, None
        

class StepMixedReward:
    def __init__(self, env, config):
        self.mixedReward = MixedReward(env, config)
        self.previous_obs = None
        self.previous_action = None
    def reset(self):
        self.previous_obs = None
        self.previous_action = None
    """
    obs need to have the following shape: (batch, 1, obs_dim) (since we only do stepwise)
    """
    def __call__(self, obs, action, collision, done):
        assert len(obs.shape) == 2
        assert len(action.shape) == 2
        assert action.shape[1] == 2
        assert obs.shape[1] == 11
        assert obs.shape[0] == action.shape[0]

        #print(collision.shape)
        #print(done.shape)
        # add a timestep dimension at axis 1
        obs = np.expand_dims(obs, axis=1)
        action = np.expand_dims(action, axis=1)
        collision = np.expand_dims(collision, axis=1)
        done = np.expand_dims(done, axis=1)

        if self.previous_obs is None:
            self.previous_obs = obs
            self.previous_action = action
        
        # now we join the previous obs and action with the current one along dim 1
        obs_t2 = np.concatenate([self.previous_obs, obs], axis=1)
        action_t2 = np.concatenate([self.previous_action, action], axis=1)
        #print(collision)
        collision = np.concatenate([collision, collision], axis=1)
        done = np.concatenate([done, done], axis=1)
        # now we can apply the mixed reward
        reward, _ = self.mixedReward(obs_t2, action_t2, collision, done)
        # now we discard the first timestep
        #print(reward)
        #print(reward.shape)
        reward = reward[:,1]
        self.previous_action = action
        self.previous_obs = obs
        assert reward.shape == (obs.shape[0],) # we only have a batch dimension
        return reward



import gymnasium as gym
import f110_orl_dataset


def calculate_reward(config, dataset, env, track):
    # init reward
    mixedReward = MixedReward(env, config)

    timesteps = dataset["observations"].shape[0]
    finished_trajectory = np.logical_or(dataset["terminals"],
                                        dataset["timeouts"])
    
    # find where the trajectory is finished
    finished = np.where(finished_trajectory)[0]

    timesteps = None # for debugging
    batch_obs = np.split(dataset["observations"][:timesteps], finished+1)
    batch_act = np.split(dataset["raw_actions"][:timesteps], finished+1)
    batch_col = np.split(dataset["terminals"][:timesteps], finished+1)
    batch_ter = np.split(dataset["terminals"][:timesteps], finished+1)
    
    all_rewards = np.zeros((1,0))
    
    for batch in zip(batch_obs, batch_act, batch_col, batch_ter):
        batch = list(batch)
        for i in range(len(batch)):
            batch[i] = np.expand_dims(batch[i], axis=0)

        if batch[0].shape[1] <= 1:
            break

        reward, _ = mixedReward(batch[0], batch[1], batch[2], batch[3])
        #print(reward.shape)
        all_rewards = np.concatenate([all_rewards, reward], axis=1)

    return all_rewards

def test_progress_reward():
    config = Config('reward_config.json')

    F110Env = gym.make('f110_with_dataset-v0',
    # only terminals are available as of tight now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal", num_agents=1),
            render_mode="human")
    )

    dataset =  F110Env.get_dataset(
                zarr_path= f"/home/fabian/msc/f110_dope/ws_ope/f1tenth_orl_dataset/data/trajectories.zarr", 
                alternate_reward=False,
                include_timesteps_in_obs=True,
                only_terminals=False,
                debug=True,
                #clip_trajectory_length=(0,timestep),
                )
    # apply the reward
    all_rewards = calculate_reward(config, dataset)
    rewards = dataset["rewards"]
    # set start of trajectory to zero, because this is 
    # different when computed offline
    finished_trajectory = np.logical_or(dataset["terminals"],
                                    dataset["timeouts"])
    finished = np.where(finished_trajectory)[0]
    rewards[0] = 0
    # add a zero at the end of rewards so we dont go ooB
    rewards = np.concatenate([rewards, np.zeros(1)])
    rewards[finished+1] = 0
    # remove the zero
    rewards = rewards[:-2]
    all_rewards = all_rewards[0,:rewards.shape[0]]
    indices = np.where(np.isclose(all_rewards, rewards, atol = 1e-5) == False)[0]
    # at these indices print the difference between the two
    #print(indices[:3])
    #print("Difference")
    #print(all_rewards[indices]- rewards[indices])
    assert len(indices) == 0
    print("[td_progress] Test passed")


def create_batch(obs_steps_t, action_steps_t, done_t, truncate_t, i):
    obs_steps = np.expand_dims(obs_steps_t[i], axis=[0,1])
    action_steps = np.expand_dims(action_steps_t[i], axis=[0,1])
    
    # make the batch size 2
    obs_steps = np.concatenate([obs_steps, obs_steps], axis=0)
    action_steps = np.concatenate([action_steps, action_steps], axis=0)
    # same for done truncate
    # expand done and truncate dimensions
    done = np.expand_dims(done_t[i], axis=[0,1])
    truncate = np.expand_dims(truncate_t[i], axis=[0,1])
    done = np.concatenate([done, done], axis=0)
    truncate = np.concatenate([truncate, truncate], axis=0)
    return obs_steps, action_steps, done, truncate

def test_online_batch_reward():
    config = Config('reward_config.json')
    stepMixedReward = StepMixedReward(config)

    F110Env = gym.make('f110_with_dataset-v0',
    # only terminals are available as of tight now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal", num_agents=1),
            render_mode="human")
    )

    dataset =  F110Env.get_dataset(
                zarr_path= f"/home/fabian/msc/f110_dope/ws_ope/f1tenth_orl_dataset/data/trajectories.zarr", 
                alternate_reward=False,
                include_timesteps_in_obs=True,
                only_terminals=False,
                #clip_trajectory_length=(0,timestep),
                )
    
    horizon = 1000000
    batch_size = 2
    # create a batch with batch_size trajectories taken from the dataset
    obs_steps_t = dataset["observations"][:horizon]
    action_steps_t = dataset["actions"][:horizon]
    
    done_steps_t = np.logical_or(dataset["terminals"][:horizon],
                                dataset["timeouts"][:horizon])
    collision_steps_t = dataset["terminals"][:horizon]

    rewards = np.zeros((2, horizon))
    for i in range(horizon):
        obs, act, done, coll = create_batch(obs_steps_t, 
                                            action_steps_t, 
                                            done_steps_t, 
                                            collision_steps_t,
                                            i)
        #print(done.shape)
        #print(coll.shape)
        # print(done.any())
        #print(obs.shape)
        reward = stepMixedReward(obs, act, coll, done)
        if done.any():
            print("Done", i)
            stepMixedReward.reset()
        #print(reward.shape)
        #print(reward)
        rewards[:,i] = reward
    
    precompted_rewards = dataset["rewards"][:horizon]

    indices = np.where(np.isclose(precompted_rewards, rewards[0], atol = 1e-5) == False)[0]
    print(indices)
    print()
    print(rewards[0].shape)
    print(precompted_rewards.shape)
    
    #print(indices)
    print(rewards[0])
    print(rewards[1])
    
    print(dataset["rewards"][:10])

    print(rewards[0][1000])
    print(dataset["rewards"][1000])
    print(precompted_rewards[indices])
    print(rewards[0][indices])
    # apply the reward in batch

def test_reward(config_file, dataset_folder):
    config = Config(config_file)
    
    # stepMixedReward = StepMixedReward(config)

    F110Env = gym.make('f110_with_dataset-v0',
    # only terminals are available as of tight now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal", num_agents=1),
            render_mode="human")
    )

    dataset =  F110Env.get_dataset(
                zarr_path= f"/home/fabian/msc/f110_dope/ws_ope/f1tenth_orl_dataset/data/{dataset_folder}", 
                alternate_reward=True,
                include_timesteps_in_obs=True,
                only_terminals=False,
                )
    print(config)
    all_rewards = calculate_reward(config, dataset, F110Env, F110Env.track)

    rewards = dataset["rewards"]
    # set start of trajectory to zero, because this is 
    # different when computed offline
    finished_trajectory = np.logical_or(dataset["terminals"],
                                    dataset["timeouts"])
    finished = np.where(finished_trajectory)[0]
    # rewards[0] = 0
    # add a zero at the end of rewards so we dont go ooB
    rewards = np.concatenate([rewards, np.zeros(1)])
    # rewards[finished+1] = 0
    # remove the zero
    rewards = rewards[:-2]
    all_rewards = all_rewards[0,:rewards.shape[0]]
    indices = np.where(np.isclose(all_rewards, rewards, atol = 1e-5) == False)[0]
    # at these indices print the difference between the two
    #print(indices[:3])
    #print("Difference")
    #print(all_rewards[indices]- rewards[indices])
    print(indices)
    print("Computed 10:", all_rewards[:10])
    print("Ground truth 10:", rewards[:10])
    print("Actions 10:", dataset["raw_actions"][:10])
    print(all_rewards[10063])
    print(rewards[10063])
    print(dataset["raw_actions"][10063])
    assert len(indices) == 0
    print(f"[{dataset_folder}] Test passed")

if __name__ == "__main__":
    #test_progress_reward()
    # test_online_batch_reward()
    #test_reward("reward_min_act.json", "trajectories_min_act.zarr")
    test_reward("reward_raceline.json", "trajectories_raceline.zarr")