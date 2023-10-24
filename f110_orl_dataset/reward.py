import numpy as np
from f110_gym.envs.track import Track
# from code.wrappers import Progress
from typing import Tuple
import gymnasium as gym

from gymnasium.wrappers.normalize import RunningMeanStd, update_mean_var_count_from_moments


class Progress:
    def __init__(self, track: Track, lookahead: int = 20) -> None:
        # 
        xs = track.centerline.xs
        ys = track.centerline.ys
        self.centerline = np.stack((xs, ys), axis=-1)
        # append first point to end to make loop
        self.centerline = np.vstack((self.centerline, self.centerline[0]))

        self.segment_vectors = np.diff(self.centerline, axis=0)
        # print(segment_vectors.shape)
        self.segment_lengths = np.linalg.norm(self.segment_vectors, axis=1)
        
        # Extend segment lengths to compute cumulative distance
        self.cumulative_lengths = np.hstack(([0], np.cumsum(self.segment_lengths)))
        self.previous_closest_idx = 0
        self.max_lookahead = lookahead
        #print(self.centerline)
        #print(self.centerline.shape)
        #print("***********")

    def distance_along_centerline_np(self, pose_points):
        assert len(pose_points.shape) == 2 and pose_points.shape[1] == 2

        # centerpoints = np.array(centerpoints)
        #print(self.centerline.shape)
        #print(centerpoints[:-1])
        #print(pose_points)
        #print(".....")
        # assert pose points must be Nx2
        pose_points = np.array(pose_points)
        #print(pose_points.shape)

        def projected_distance(pose):
            rel_pose = pose - self.centerline[:-1]
            t = np.sum(rel_pose * self.segment_vectors, axis=1) / np.sum(self.segment_vectors**2, axis=1)
            t = np.clip(t, 0, 1)
            projections = self.centerline[:-1] + t[:, np.newaxis] * self.segment_vectors
            distances = np.linalg.norm(pose - projections, axis=1)
            points_len = self.centerline.shape[0]-1  # -1 because of last fake 
            lookahead_idx = (self.max_lookahead + self.previous_closest_idx) % points_len
            # wrap around
            if self.previous_closest_idx <= lookahead_idx:
                indices_to_check = list(range(self.previous_closest_idx, lookahead_idx + 1))
            else:
                # Otherwise, we need to check both the end and the start of the array
                indices_to_check = list(range(self.previous_closest_idx, points_len)) \
                    + list(range(0, lookahead_idx+1))
            # Extract the relevant distances using fancy indexing
            subset_distances = distances[indices_to_check]

            # Find the index of the minimum distance within this subset
            subset_idx = np.argmin(subset_distances)

            # Translate it back to the index in the original distances array
            closest_idx = indices_to_check[subset_idx]
            self.previous_closest_idx = closest_idx
            # print(closest_idx)
            return self.cumulative_lengths[closest_idx] + self.segment_lengths[closest_idx] * t[closest_idx]
        
        return np.array([projected_distance(pose) for pose in pose_points])
    
    def get_progress(self, pose: Tuple[float, float]):
        progress =  self.distance_along_centerline_np(pose)
        # print(self.cumulative_lengths.shape)
        # print(self.cumulative_lengths[-1])
        progress = progress / (self.cumulative_lengths[-1] + self.segment_lengths[-1])
        # clip between 0 and 1 (it can sometimes happen that its slightly above 1)
        # print("progress", progress)
        return np.clip(progress, 0, 1)
    
    def reset(self, pose):
        rel_pose = pose - self.centerline[:-1]
        t = np.sum(rel_pose * self.segment_vectors, axis=1) / np.sum(self.segment_vectors**2, axis=1)
        t = np.clip(t, 0, 1)
        projections = self.centerline[:-1] + t[:, np.newaxis] * self.segment_vectors
        distances = np.linalg.norm(pose - projections, axis=1)
        
        closest_idx = np.argmin(distances)
        self.previous_closest_idx = closest_idx


# adapted slightly from https://github.com/openai/gym/blob/master/gym/wrappers/normalize.py
class NormalizeReward():
        def __init__(self, env,
                gamma: float = 0.99,
                epsilon: float = 1e-8,):
            # TODO! Make this right
            # self.num_envs = getattr(env, "num_envs", 1)
            # self.is_vector_env = getattr(env, "is_vector_env", False)
            # print("self.num_envs", self.num_envs)
            # print("self.is_vector_env", self.is_vector_env)
            self.return_rms = RunningMeanStd(shape=())
            self.gamma = gamma
            self.epsilon = epsilon
            self.returns = np.zeros(1)
        def step(self,rews , done):
            self.returns = self.returns * self.gamma * (1 - done) + rews 
            rews = self.normalize(rews)
            return rews
        def normalize(self, rews):
            self.return_rms.update(self.returns)
            return rews / np.sqrt(self.return_rms.var + self.epsilon)

class PureProgressReward(object):
    def __init__(self, track:Track) -> None:
        self.inital_progress = None
        self.previous_progress = None
        self.progress_tracker = Progress(track)
        self.laps = 0
    def reset(self, new_pose: Tuple[float, float]):
        pose = np.array(new_pose)
        # print(pose)
        pose = pose[np.newaxis, :]
        self.progress_tracker.reset(pose)
        self.inital_progress = self.progress_tracker.get_progress(pose)[0] #always 0
        self.previous_progress = self.inital_progress
        self.laps = 0
        # print("Reset called")

    def __call__(self, pose: Tuple[float, float]):
        pose = np.array(pose)
        pose = pose[np.newaxis, :]
        new_progress = self.progress_tracker.get_progress(pose)[0]
        #print(".....")
        #print(new_progress)
        #print(self.laps)
        #print(self.previous_progress)
        #print(self.inital_progress)
        #print(self.previous_progress)
        if (new_progress - self.previous_progress) < -0.9 \
            and new_progress < 0.1 and self.previous_progress > 0.9:
            # crossed finish line
            self.laps += 1
            # exit(0)
        reward = new_progress + self.laps - self.inital_progress
        self.previous_progress = new_progress
        return reward


class ProgressReward(object):
    def __init__(self, track:Track) -> None:
        self._current_progress = None
        self.progress_tracker = Progress(track)
    
    def reset(self, new_pose: Tuple[float, float]):
        pose = np.array(new_pose)
        # print(pose)
        pose = pose[np.newaxis, :]
        self.progress_tracker.reset(pose)
        self.current_progress = self.progress_tracker.get_progress(pose)[0]

    def __call__(self, pose: Tuple[float, float]):
        pose = np.array(pose)
        pose = pose[np.newaxis, :]
        new_progress = self.progress_tracker.get_progress(pose)[0]
        delta_progress = 0
        # in this case we just crossed the finishing line!
        if (new_progress - self.current_progress) < -0.9:
            delta_progress = (new_progress + 1) - self.current_progress
        else:
            delta_progress = new_progress - self.current_progress
        # delta_progress = delta_progress **2
        delta_progress = max(delta_progress, 0)
        delta_progress *= 100
        self.current_progress = new_progress
        reward = delta_progress
        return reward
"""
@brief Minimum Lidar Ray Reward
"""
class MinLidarRayReward(object):
    def __init__(self, high=0.15): # should be 2
        self.high = high
    def __call__(self, laser_scan) -> float:
        # find minimum lidar ray
        min_ray = np.min(laser_scan)
        # normalize the min_distance
        # clip it to the high value
        min_ray = np.clip(min_ray, 0, self.high)
        reward = min_ray / self.high
        min_ray = min_ray ** 4 
        return reward

"""
@brief     Reward function based on the distance from the raceline
"""
class RacelineDeltaReward(object):
    def __init__(self, track: Track) -> None:
        xs = track.raceline.xs
        ys = track.raceline.ys
        self.centerline = np.stack((xs, ys), axis=-1)
        self.largest_delta_observed = 2.0

    def __call__(self, pose: Tuple[float, float]) -> float:
        # calculate the squared distances to all points on the raceline
        distances = np.sum((self.centerline - np.array(pose))**2, axis=1)
        # get the minimum squared distance
        min_distance_squared = np.min(distances)
        # take the square root to get the actual minimum distance
        min_distance = np.sqrt(min_distance_squared)
        if min_distance > self.largest_delta_observed:
            self.largest_delta_observed = min_distance

        # normalize the min_distance
        reward = min_distance / self.largest_delta_observed
        # the negative of the distance, meaning closer to the line is better
        reward = 1.0 - reward
        reward = reward ** 2
        return reward
"""
@brief    Reward function based on the change in action
"""
class MinActionReward(object):    
    def __init__(self, low_steering, high_steering):
        self.low_steering = low_steering
        self.high_steering = high_steering
    
    def __call__(self, action: np.ndarray) -> float:
        # penalize the change in steering angle
        assert(len(action) == 2)
        delta_steering = abs((action[0]))
        normalized_steering = (delta_steering / self.high_steering)**2 # prefer smaller shifts
        inverse_steering = (1.0 - normalized_steering)
        reward = inverse_steering
        return reward
    
"""
@brief    Reward function based on the change in steering angle
"""
class MinSteeringChangeReward(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self._last_action = 0.0
    def __call__(self, action: np.ndarray) -> float:
        # penalize the change in steering angle
        assert(len(action) == 1 or (len(action) == 2))
        delta = abs((action[0] - self._last_action))
        #print ("delta", delta)

        normalized = delta / (self.high - self.low)
        assert(normalized<=1.0)
        #print(self.high, self.low)
        inverse = 1.0 - normalized
        delta = (inverse) **2 # prefer smaller shifts
        # delta = np.clip(delta, self.low, self.high)
        reward = delta
        self._last_action = action[0]
        #print(reward)
        return reward

    def reset(self):
        self._last_action = 0.0

"""
@brief    Reward function based on the change in velocity
"""
class MinVelocityChangeReward(object):
    def __init__(self, low, high, inital_velocity=1.5): #TODO! fix intial_velocity
        self.low = low
        self.high = high
        self.inital_velocity = inital_velocity
        self._last_velocity = inital_velocity

    def __call__(self, vel_x, vel_y) -> float:
        # penalize the change in velocity angle
        velocity = np.sqrt(vel_x**2 + vel_y**2)
        delta = abs((velocity - self._last_velocity))
        normalized = delta / (self.high - self.low)
        inverse = 1.0 - normalized
        reward = (inverse) **2 # prefer smaller shifts
        self._last_velocity = velocity
        # print("vel")
        # print(delta)
        # print(reward)
        return reward
    def reset(self, velocity):
        self._last_velocity = velocity

"""
@brief    Reward function based on the speed of the car
"""
class VelocityReward(object):
    def __init__(self, track: Track) -> None:
        pass

    def __call__(self, velocity_x: float, velocity_y) -> float:
        # faster is better
        velocity = np.sqrt(velocity_x**2 + velocity_y**2)
        return velocity


"""
@brief Sparse laptime reward function
"""
class LaptimeReward(object):
    pass

class MixedReward(object):
    def __init__(self, env: gym.Env,
                 track: Track, 
                 collision_penalty: float = -10.0, 
                 progress_weight: float = 1.0, 
                 raceline_delta_weight: float = 0.0, 
                 velocity_weight: float = 0.0, 
                 steering_change_weight: float = 0.0, 
                 velocity_change_weight: float= 0.0,
                 min_action_weight: float = 0.0,
                 min_lidar_ray_weight: float = 0.0,
                 pure_progress_weight: float = 0.0,
                 inital_velocity: float = 1.5,
                 save_all_rewards: bool = False,
                 normalize: bool = True ,
                 ):
        # print the args used
        print("Reward function args:")
        print("collision_penalty: ", collision_penalty)
        print("progress_weight: ", progress_weight)
        print("raceline_delta_weight: ", raceline_delta_weight)
        print("velocity_weight: ", velocity_weight)
        print("steering_change_weight: ", steering_change_weight)
        print("velocity_change_weight: ", velocity_change_weight)
        print("min_action_weight", min_action_weight)
        print("min_lidar_ray_weight", min_lidar_ray_weight)
        print("inital_velocity: ", inital_velocity)
        print("normalize: ", normalize)
        self.weights = [progress_weight, raceline_delta_weight, velocity_weight, steering_change_weight, velocity_change_weight, 
                        pure_progress_weight, min_action_weight, min_lidar_ray_weight]
        self.rewards = [ProgressReward, RacelineDeltaReward, VelocityReward, 
                        MinSteeringChangeReward, MinVelocityChangeReward, 
                        PureProgressReward, MinActionReward, MinLidarRayReward]
        self.rewards_name = ["ProgressReward", "RacelineDeltaReward", 
                             "VelocityReward", "MinSteeringChangeReward", 
                             "MinVelocityChangeReward", "PureProgressReward", 
                             "MinActionReward", "MinLidarRayReward"]
        self.normalize = normalize
        self.collision_penalty = collision_penalty
        self.progress_weight = progress_weight
        self.raceline_delta_weight = raceline_delta_weight
        self.velocity_weight = velocity_weight
        self.steering_change_weight = steering_change_weight
        self.velocity_change_weight = velocity_change_weight
        self.pure_progress_weight = pure_progress_weight
        self.min_action_weight = min_action_weight
        self.min_lidar_ray_weight = min_lidar_ray_weight

        self.progress_reward = ProgressReward(track)
        self.raceline_reward = RacelineDeltaReward(track)
        self.velocity_reward = VelocityReward(track)
        self.pure_progress_reward = PureProgressReward(track)
        # print(env.action_space.low)
        self.steering_change_reward = MinSteeringChangeReward(env.action_space.low[0][0], 
                                                              env.action_space.high[0][0])
        self.velocity_change_reward = MinVelocityChangeReward(env.action_space.low[0][1], 
                                                              env.action_space.high[0][1],
                                                              inital_velocity=inital_velocity)
        self.min_action_reward = MinActionReward(env.action_space.low[0][0],
                                                    env.action_space.high[0][0])
        
        self.min_lidar_ray_reward = MinLidarRayReward(high=0.3)

        # for each reward instantiate a normalization object
        self.normalizers = []
        self.save_all_rewards = save_all_rewards
        for reward in self.rewards:
            self.normalizers.append(NormalizeReward(env))

    def __call__(self, obs, action, collision, done):
        # get pose_x and pose_y from obs
        # assert action is a numpy array of length 2 if velocity change is not zero
        assert (len(action[0]) == 2 or self.velocity_change_reward == 0)
        assert ('poses_x' in obs and
                'poses_y' in obs and 
                'linear_vels_x' in obs and 
                'linear_vels_y' in obs and
                'scans' in obs), "Some keys are missing from the obs dictionary"

        
        
        pose = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        rewards = np.zeros(len(self.rewards))
        
        #print(rewards)
        #print(self.weights)
        if self.progress_weight > 0.0001 or self.save_all_rewards:
            progress_reward = self.progress_reward(pose)
            # print("progress_reward", progress_reward)
            rewards[0] = progress_reward
        if self.raceline_delta_weight > 0.0001 or self.save_all_rewards:
            raceline_reward = self.raceline_reward(pose)
            rewards[1] = raceline_reward
        if self.velocity_weight > 0.0001 or self.save_all_rewards:
            velocity_reward = self.velocity_reward(obs['linear_vels_x'][0], obs['linear_vels_y'][0])
            rewards[2] = velocity_reward
        if self.steering_change_weight > 0.0001 or self.save_all_rewards:
            steering_change_reward = self.steering_change_reward(action[0])
            rewards[3] = steering_change_reward
        if self.velocity_change_weight > 0.0001 or self.save_all_rewards:
            velocity_change_reward = self.velocity_change_reward(obs['linear_vels_x'][0], obs['linear_vels_y'][0]) 
            rewards[4] = velocity_change_reward
        if self.pure_progress_weight > 0.0001 or self.save_all_rewards:
            pure_progress_reward = self.pure_progress_reward(pose)
            rewards[5] = pure_progress_reward
        if self.min_action_weight > 0.0001 or self.save_all_rewards:
            min_action_reward = self.min_action_reward(action[0])
            rewards[6] = min_action_reward
        if self.min_lidar_ray_weight > 0.0001 or self.save_all_rewards:
            min_lidar_ray_reward = self.min_lidar_ray_reward(obs['lidar_occupancy'][0])
            rewards[7] = min_lidar_ray_reward
        #print(" [ProgressReward, RacelineDeltaReward, VelocityReward, MinSteeringChangeReward, MinVelocityChangeReward]")
        #print("before normalization: ", rewards)
        assert(len(self.weights) == len(self.rewards))
        if self.normalize:
            for i in range(len(self.rewards)):
                if (abs(self.weights[i]) < 0.0001):
                    continue
                else:
                    # normalize the reward
                    rewards[i] = self.normalizers[i].step(rewards[i], done) 
        
        rewards_dict = {}
        for i, rew_name in enumerate(self.rewards_name):
            rewards_dict[str(rew_name)] = rewards[i]
        #print("after norm", rewards) # now normalized
        rewards = rewards* self.weights
        #print("after weighting", rewards)
        # sum rewards
        reward = np.sum(rewards)
        # create dict from rewards


        #print("rewards", reward)
        # now apply penalty for collision
        if collision:
            # print("Collision Penalty")
            reward = self.collision_penalty
        # print(rewards_dict)
        return reward , rewards_dict

    def reset(self, pose: Tuple[float, float], velocity :float):
        self.velocity_change_reward.reset(velocity)
        self.steering_change_reward.reset()
        # print(pose)
        self.progress_reward.reset(new_pose=pose)
        self.pure_progress_reward.reset(new_pose=pose)

    


        
