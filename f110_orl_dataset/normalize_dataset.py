import numpy as np
from .config import *
from gymnasium.spaces import Box

from collections import OrderedDict
import gymnasium as gym

def normalize(value, low, high):
    """Normalize value between -1 and 1."""
    return 2 * ((value - low) / (high - low)) - 1

def normalize_zero(value, low, high):
    """Normalize value between 0 and 1."""
    return ((value - low) / (high - low))

def clip(value, low, high):
    """Clip a value between low and high."""
    return np.clip(value, low, high)
    
class Normalize:
    def __init__(self):
        state_dict = OrderedDict()
        # TODO more elegant below
        s_min = -0.4189
        s_max = 0.4189
        # Append each box in desired order
        state_dict['poses_x'] = Box(POSE_LOW, POSE_HIGH, (1,), np.float32)
        state_dict['poses_y'] = Box(POSE_LOW, POSE_HIGH, (1,), np.float32)
        state_dict['theta_sin'] = Box(-1.0, 1.0, (1,), np.float32)
        state_dict['theta_cos'] = Box(-1.0, 1.0, (1,), np.float32)
        state_dict['ang_vels_z'] = Box(VEL_LOW, VEL_HIGH, (1,), np.float32)
        state_dict['linear_vels_x'] = Box(VEL_LOW, VEL_HIGH, (1,), np.float32)
        state_dict['linear_vels_y'] = Box(VEL_LOW, VEL_HIGH, (1,), np.float32)
        state_dict['previous_action'] = Box(low=np.asarray([[s_min, MIN_VEL]]), 
                                        high=np.asarray([[s_max, MAX_VEL]]), 
                                        shape=(1, 2), dtype=np.float32)
        state_dict['progress_sin'] = Box(-1.0, 1.0, (1,), np.float32)
        state_dict['progress_cos'] = Box(-1.0, 1.0, (1,), np.float32)
            # Convert the ordered dictionary to a gym space dict
        self.state_space = gym.spaces.Dict(state_dict)

        new_state_dict = OrderedDict()
        new_state_dict['poses_x'] = Box(-1.0, 1.0, (1,), np.float32)
        new_state_dict['poses_y'] = Box(-1.0, 1.0, (1,), np.float32)
        new_state_dict['theta_sin'] = Box(-1.0, 1.0, (1,), np.float32)
        new_state_dict['theta_cos'] = Box(-1.0, 1.0, (1,), np.float32)
        new_state_dict['ang_vels_z'] = Box(-1.0, 1.0, (1,), np.float32)
        new_state_dict['linear_vels_x'] = Box(-1.0, 1.0, (1,), np.float32)
        new_state_dict['linear_vels_y'] = Box(-1.0, 1.0, (1,), np.float32)
        new_state_dict['previous_action'] = Box(low=np.asarray([[-1.0, -1.0]]), 
                                        high=np.asarray([[1.0, 1.0]]), 
                                        shape=(1, 2), dtype=np.float32)
        new_state_dict['progress_sin'] = Box(-1.0, 1.0, (1,), np.float32)
        new_state_dict['progress_cos'] = Box(-1.0, 1.0, (1,), np.float32)
        self.new_state_space = gym.spaces.Dict(new_state_dict)

    def unflatten_batch(self, batch):
        batch = np.asarray(batch)

        assert len(batch.shape) == 2, "Batch should be 2D"

        batch_dict = {}
        
        start_idx = 0
        for key, space in self.state_space.spaces.items():
            # Calculate how many columns this part of the observation takes up
            space_shape = np.prod(space.shape)
            
            # Slice the appropriate columns from the batch
            batch_slice = batch[:, start_idx:start_idx+space_shape]
            
            # If the space has multi-dimensions, reshape it accordingly
            if len(space.shape) > 1:
                batch_slice = batch_slice.reshape((-1,) + space.shape)
            
            batch_dict[key] = batch_slice
            start_idx += space_shape

        assert start_idx == batch.shape[1], "Mismatch in the number of columns"
        return batch_dict
    
    def flatten_batch(self, batch_dict):
        batch = np.zeros((len(batch_dict[list(batch_dict.keys())[0]]), self.state_space.shape[0]))
        print(batch.shape)
        for key, obs in self.state_space.spaces.items():
            # Calculate how many columns this part of the observation takes up
            space_shape = np.prod(obs.shape)
            # Slice the appropriate columns from the batch
            batch_slice = batch_dict[key]
            # If the space has multi-dimensions, reshape it accordingly
            if len(obs.shape) > 1:
                batch_slice = batch_slice.reshape((-1, space_shape))
            batch[:, :space_shape] = batch_slice
        return batch
    
    def normalize_obs_batch(self, batch_dict):
        # Unflatten the batch observations
        for key, obs in batch_dict.items():
            # Skip specific keys that you don't want to normalize
            if key not in ['progress_sin', 'progress_cos', 'theta_cos','poses_theta', 'theta_sin','lidar_occupancy']:
                low = self.state_space.spaces[key].low
                high = self.state_space.spaces[key].high
                
                # If the observation space is multi-dimensional, reshape to 2D for vectorized operations
                original_shape = obs.shape
                if len(original_shape) > 2:
                    obs = obs.reshape(original_shape[0], -1)

                obs = clip(obs, low, high)
                obs = normalize(obs, low, high)

                # If the observation space is multi-dimensional, reshape back to original shape
                if len(original_shape) > 2:
                    obs = obs.reshape(original_shape)

                batch_dict[key] = obs

        return batch_dict
    
    def normalize_laser_scan(self, batch_laserscan):
        batch_laserscan = np.asarray(batch_laserscan)
        assert len(batch_laserscan.shape) == 2, "Batch should be 2D"
        batch_laserscan = clip(batch_laserscan, 0, 10)
        batch_laserscan = batch_laserscan / 10
        return batch_laserscan