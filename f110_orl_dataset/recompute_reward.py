import numpy as np
import zarr
from reward import MixedReward
import gymnasium as gym
import warnings
from tqdm import tqdm
from multiprocessing import Pool

class RecomputeReward:
    def __init__(self, zarr_path, env, 
                 decaying_crash=None, **reward_config):
        self.root = zarr.open(zarr_path, mode='wr')
        
        self.reward = MixedReward(env, env.track, **reward_config)
        self.decaying_crash = decaying_crash

    def process_chunk(self,chunk):
        # Process a chunk of data sequentially.
        rewards = []
        resetting = True
        for data in chunk:
            i, observations, done, truncated, collision = data
            obs = {}
            for key in observations.keys():
                obs[key] = [observations[key][i]]
            obs["scans"] = obs["lidar_occupancy"]
            
            if resetting:
                pose = (obs['poses_x'][0], obs['poses_y'][0])
                self.reset(pose)
            reward = self.step(np.array([[0,0]]), obs, collision, done)
            rewards.append(reward)
            resetting = truncated or done
        return rewards

    def apply(self, use_new_rewards = True):
        warnings.warn("Rewards using actions are not supported, check for yourself you are not using any. \
                    Most other rewards are slightly of, since reset obs is not recorded in the dataset \
                    Issues steam from the data available in the dataset.", UserWarning)
        if not(use_new_rewards):
            raise NotImplementedError
        
        if not(self.decaying_crash):
            observations = self.root["observations"]
            done_val = self.root["done"]
            truncated_val = self.root["truncated"]
            collision_val = self.root["collision"]
            # create new zarr group of size 
            # Split the data into chunks that can be processed sequentially.
            chunks = []
            current_chunk = []
            for i in tqdm(range(len(self.root["rewards"]))):
                current_chunk.append((i, observations, done_val[i], truncated_val[i], collision_val[i]))
                if truncated_val[i] or done_val[i]:
                    chunks.append(current_chunk)
                    current_chunk = []

            # Process each chunk in parallel
            with Pool(processes=4) as pool:
                results = list(tqdm(pool.imap(self.process_chunk, chunks), total=len(chunks)))

            # Flatten the results
            new_rewards = [reward for sublist in results for reward in sublist]
            
        else:
            finished = self.root["done"] or self.root["truncated"]
            
            new_rewards = self.decaying_reward(self.root["collision"], finished)
            #if True:
            #    new_rewards = np.zeros_like(self.root["collision"], dtype=float) # 
            #    collision_indices = np.where(self.root["collision"])[0]
            #    new_rewards[collision_indices] = -1
        self.root["new_rewards"] = new_rewards
        # print(self.root["new_rewards"][])

    def decaying_reward(self, collisions,
                        finished,
                        decay_factor=0.9, threshold=0.35): # standard means after 10 done so @20Hz 0.5 sec
        # Initialize rewards with zeros
        rewards = np.zeros_like(collisions, dtype=float)
        
        # Get indices where collision is True
        collision_indices = np.where(collisions)[0]
        # print(collision_indices)
        for index in tqdm(collision_indices):
            # Set reward at collision index to -1
            rewards[index] = -1
            decay = decay_factor
            
            # Iterate backwards from collision index to apply decay
            for i in range(index - 1, -1, -1):
                if finished[i] or decay < threshold:
                    # Stop decaying the rewards if done or truncated is True, or decay becomes very small
                    break
                
                rewards[i] = -decay
                decay *= decay_factor            
        return rewards * 100
    
    def step(self, action, observation, collision, done):
        reward, _ = self.reward(observation, action, 
                                      collision, done)
        return reward
    
    def reset(self, pose):
        self.reward.reset(pose)

standard_config = {
    "collision_penalty": 0.0,
    "progress_weight": 0.0,
    "raceline_delta_weight": 1.0,
    "velocity_weight": 0.0,
    "steering_change_weight": 0.0,
    "velocity_change_weight": 0.0,
    "pure_progress_weight": 0.0,
    "min_action_weight" : 0.0,
    "min_lidar_ray_weight" : 0.0,
    "inital_velocity": 1.5,
    "normalize": False,
}

if __name__ == "__main__":
    env = gym.make("f110_gym:f110-v0",
                config = dict(map="Infsaal",
                num_agents=1, 
                params=dict(vmin=0.5, vmax=2.0)),
                render_mode="human")
    rew_obj = RecomputeReward(zarr_path="/home/fabian/f110_rl/f110-sb3/trajectories4.zarr",
                              env = env,
                              decaying_crash=False, **standard_config)
    rew_obj.apply()
    # sanity check
    x = np.where(rew_obj.root["done"] or rew_obj.root["truncated"] == 1)[0]
    print(x[0])

    print(rew_obj.root["rewards"][int(x[0])-5: int(x[0])+5])
    print(rew_obj.root["new_rewards"][int(x[0]-5): int(x[0]+5)])
    y = 1131
    print(rew_obj.root["rewards"][int(y-20): int(y+5)])
    print(rew_obj.root["new_rewards"][int(y-20): int(y+5)])
    print(rew_obj.root["done"][int(y-20): int(y+5)])

