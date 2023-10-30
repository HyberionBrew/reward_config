import gymnasium as gym
# import so PAth
from pathlib import Path
from f110_gym.envs.f110_env import F110Env
# import spaces
from gymnasium.spaces import Box, Discrete
import numpy as np

import os 
import sys
import pickle
import zarr
from typing import Union, Tuple, Dict, Optional, List, Any
from f110_gym.envs import F110Env

from .config import *

# import ordered dict
from collections import OrderedDict
obs_dictionary_keys = [
    "poses_x",
    "poses_y",
    "theta_sin",
    "theta_cos",
    "ang_vels_z",
    "linear_vels_x",
    "linear_vels_y",
    "previous_action",
    "progress_sin",
    "progress_cos",
]
def normalize(value, low, high):
    """Normalize value between -1 and 1."""
    return 2 * ((value - low) / (high - low)) - 1

def clip(value, low, high):
    """Clip a value between low and high."""
    return np.clip(value, low, high)

class F1tenthDatasetEnv(F110Env):
    def __init__(
        self,
        name,
        # f110_gym_kwarg,
        root_dir=None,
        dataset_url=None,
        flatten_obs=True,
        flatten_trajectories = False,
        laser_obs = True,
        subsample_laser = 10, # 1 is no subsampling
        padd_trajectories = True,
        trajectory_max_length = 500,
        as_trajectories = True,
        **kwargs
        ):
        print(kwargs)
        super(F1tenthDatasetEnv, self).__init__(**kwargs)

        print("hi")
        if flatten_trajectories:
            raise NotImplementedError("TODO! Implement flatten_trajectories")
        self.name = name
        self.root_dir = root_dir
        self.dataset_url = dataset_url
        self.padd_trajectories = padd_trajectories
        self.trajectory_max_length = trajectory_max_length
        
        self.flatten_obs = flatten_obs
        self.subsample_laser = subsample_laser
        self.as_trajectories = as_trajectories
        self.laser_obs = laser_obs
        # open with zarr
        # self.dataset = zarr.open(self.root_dir, mode='r')
        # list the subdirs
        # print(self.dataset.tree())
        super(F1tenthDatasetEnv, self).__init__(**kwargs)

        # now we define our action and observation spaces
        #action_space_low = np.array([-1.0,-1])
        #action_space_high = np.array([1.0, 1.0])
        #self.action_space = gym.spaces.Box(action_space_low, action_space_high)
        print("===")
        print(self.action_space)
        # observation space is a dict with keys:
        # the first one is the scan
        # the second one is the odometry
        print("low")
        print(SUBSAMPLE)
        rays = int(1080/SUBSAMPLE)
        # laser_scan_space = gym.spaces.Box(low=np.array([0]*rays), high=np.array([30]*rays))
        # Dict('ang_vels_z': Box(-20.0, 20.0, (1,), float32), 
        # 'linear_vels_x': Box(-20.0, 20.0, (1,), float32), 
        # 'linear_vels_y': Box(-20.0, 20.0, (1,), float32), 
        # 'poses_theta': Box(-1e+30, 1e+30, (1,), float32), 
        # 'poses_x': Box(-30.0, 30.0, (1,), float32), 
        # 'poses_y': Box(-30.0, 30.0, (1,), float32), 
        # 'progress': Box(0.0, 1.0, (1,), float32), 
        # 'lidar_occupancy': Box(0, 255, (1, 80, 80), uint8), 
        # 'previous_action': Box(-1.0, 1.0, (1, 2), float32))
        #state_space = gym.spaces.Box(low=np.array([-100, -100, 0,-10.0, -10.0, -10.0]), high=np.array([100, 100,2*np.pi, 10,10,10]))
        # Initialize an empty dictionary
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


        #print(state_space)
        
        self.observation_space = self.state_space #gym.spaces.Dict({
            # 'laser_scan': Box(0, 255, (1, 80, 80), np.uint8),
            #'state': state_space
        #})
        self.observation_space_orig = self.observation_space
        self.laser_obs_space = gym.spaces.Box(0, 1, (rays,), np.float32)
        # print(self.observation_space)
        self._orig_flat_obs_space = gym.spaces.flatten_space(self.observation_space)
        
        if self.flatten_obs:
            self.observation_space = self._orig_flat_obs_space
           
            print(self.observation_space)
            print("***********")
        self.dataset = dict(
            actions=[],
            observations=[],
            rewards=[],
            terminals=[],
            infos=[],)
        
    def shorten_trajectories(self, dataset):
        # where are the terminals
        terminals = np.where(dataset['terminals'])[0]
        print(terminals)
    def normalize_actions(self, actions):
        if self.flatten_obs:
            actions = clip(actions, self.action_space.low, self.action_space.high)
            actions = normalize(actions, self.action_space.low, self.action_space.high)
        else:
            raise NotImplementedError("TODO! Implement normalize_actions")
        
    
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
    
    
    """
    def unflatten_batch(self, batch):
        batch = np.asarray(batch)
        if self.flatten_obs:
            print("........")
            assert(len(batch.shape) <= 2)
            if (len(batch.shape)==1):
                return gym.spaces.unflatten(self.observation_space_orig, batch)
            else:
                print(self.observation_space_orig.spaces.keys())
                batch_dict = {f"{key}" : [] for key in self.observation_space_orig.spaces.keys()}
                print(batch.shape[1])
                print(batch)
                print(len(obs_dictionary_keys))
                assert(batch.shape[1]==len(obs_dictionary_keys))
                for i, key in enumerate(self.observation_space_orig.spaces.keys()):
                    for 
                    batch_dict[key] = batch[:,i]
                return batch
        else:
            return batch
    """
    """
    def normalize_obs_batch(self, batch_obs):
        # deprecated raise error
        assert(False)
        if self.flatten_obs:
            print(batch_obs)
            print(self.observation_space)
            print(batch_obs.shape)
            print(self.observation_space.shape[0])
            for i in range(self.observation_space.shape[0]):
                # if key != 'lidar_occupancy' and key != 'progress':
                low = self.observation_space.low[i]
                high = self.observation_space.high[i]
                if np.isclose(low,0) and np.isclose(high,1):
                    #print("called")
                    #print(batch_obs[:,i])
                    continue
                batch_obs[:,i] = clip(batch_obs[:,i], low, high)
                batch_obs[:,i] = normalize(batch_obs[:,i], low, high)
        else:
            raise NotImplementedError("TODO! Implement normalize_obs_batch")
        return batch_obs
    """
    def normalize_obs_batch(self, batch_dict):
        if self.flatten_obs:
            # Unflatten the batch observations
            for key, obs in batch_dict.items():
                # Skip specific keys that you don't want to normalize
                if key not in ['lidar_occupancy', 'progress_sin', 'progress_cos', 'theta_cos', 'theta_sin']:
                    low = self.observation_space_orig.spaces[key].low
                    high = self.observation_space_orig.spaces[key].high
                    
                    # If the range is [0, 1], then skip normalization
                    print(low)
                    if np.isclose(low, 0) and np.isclose(high, 1):
                        continue

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
        else:
            raise NotImplementedError("TODO! Implement normalize_obs_batch for non-flattened observations")
    # def normalize_act_batch(self, batch_act):
    def pad_trajectories(self, dataset, trajectory_max_length):
        # split dataset into trajectories at done signal = 1
        trajectories = np.split(dataset['actions'], np.where(dataset['terminals'])[0]+1)[0:-1]
        # padd to max_length
        # for trajectories in 
        print([len(tra) for tra in trajectories])
        
        return dataset
    
    def get_change_indices(self, model_names):
        change_indices = []
        current_name = model_names[0]
        for idx, name in enumerate(model_names):
            if name != current_name:
                change_indices.append(idx)
                current_name = name
        return change_indices + [len(model_names)]

    def to_dict(self, batch):
        if self.flatten_obs:
            return gym.spaces.unflatten(self._orig_flat_obs_space, batch)
        else:
            return batch
    
    def get_model_names(self, dataset):
        pass
    
    #def get_scan(self, indices):
    #    root = zarr.open(self.zarr_path, mode='r')
    #    scan = root['observations']['lidar_occupancy'][indices]
    #    return scan

    def get_dataset(
        self,
        zarr_path: Union[str, os.PathLike] = None,
        clip: bool = True,
        # rng: Optional[Tuple[int, int]] = None,
        skip_inital : int = 0,
        split_trajectories : int = 0,
        without_agents: Optional[np.ndarray] = [],
        only_agents: Optional[np.ndarray] = [],
        alternate_reward: bool = False,
        remove_short_trajectories: bool = False,
        min_trajectory_length: int = 0,
        skip_inital_random_min: int = 0,
        skip_inital_random_max: int = 0,
        max_trajectory_length:int = 0,
        only_terminals: bool = False,
        include_timesteps_in_obs: bool = False,
    ) -> Dict[str, Any]:
        """ 
        TODO! this is copied from https://github.com/rr-learning/trifinger_rl_datasets/blob/master/trifinger_rl_datasets/dataset_env.py
        Get the dataset.

        When called for the first time, the dataset is automatically downloaded and
        saved to ``~/.trifinger_rl_datasets``.

        Args:
            zarr_path:  Optional path to a Zarr directory containing the dataset, which will be
                used instead of the default.
            clip:  If True, observations are clipped to be within the environment's
                observation space.
            rng:  Optional range to return. rng=(m,n) means that observations, actions
                and rewards m to n-1 are returned. If not specified, the entire
                dataset is returned.
            indices: Optional array of timestep indices for which to load data. rng
                and indices are mutually exclusive, only one of them can be set.
            n_threads: Number of threads to use for processing the images. If None,
                the number of threads is set to the number of CPUs available to the
                process.
        Returns:
            A dictionary containing the following keys

                - observations: Either an array or a list of dictionaries
                  containing the observations depending on whether
                  `flatten_obs` is True or False.
                - actions: Array containing the actions.
                - rewards: Array containing the rewards.
                - timeouts: Array containing the timeouts (True only at
                  the end of an episode by default. Always False if
                  `set_terminals` is True).
                - terminals: Array containing the terminals (Always
                  False by default. If `set_terminals` is True, only
                  True at the last timestep of an episode).
                - images (only if present in dataset): Array of the
                  shape (n_control_timesteps, n_cameras, n_channels,
                  height, width) containing the image data. The cannels
                  are ordered as RGB.
        """
        assert len(only_agents)==0 or len(without_agents)==0, "Cannot specify both only_agents and without_agents"
        # The ORL Dataset is loaded from Zarr arrays that look as follows:
        # TODO!
        if zarr_path is None:
            raise NotImplementedError("TODO! Download the dataset from the web")
            # zarr_path = self._download_dataset()
        
        #store = zarr.LMDBStore(zarr_path, readonly=True)
        print(zarr_path)
        self.zarr_path = zarr_path
        root = zarr.open(zarr_path, mode='r') #(store=store)

        #print(root.tree())
        model_names = root['model_name'][:]
        # print all unique model names
        print(np.unique(model_names))
        data_dict = {}
        print("len(model_names)", len(model_names))
        indices = np.where(~np.isin(model_names, without_agents))[0]
        if len(without_agents) > 0:
            indices = np.where(~np.isin(model_names, without_agents))[0]
        if len(only_agents) > 0:
            indices = np.where(np.isin(model_names, only_agents))[0]
        print("Indices:", len(indices))
        if alternate_reward:
            print("Using alternate reward")
            data_dict['rewards'] = root['new_rewards'][indices]
        else:
            data_dict['rewards'] = root['rewards'][indices]
        # print("hi")
        data_dict['terminals'] = root['done'][indices]
        data_dict['timeouts'] = root['truncated'][indices]
        # print("hi")
        data_dict['actions'] = root['actions'][indices]
        data_dict['log_probs'] = root['log_prob'][indices]
        data_dict['raw_actions'] = root['raw_actions'][indices]
        # print("hi")
        data_dict['index'] = indices #root['timestep'][indices]
        # loop over observation keys
        data_dict['observations'] = dict()
        for key in root['observations'].array_keys():
            print(key)
            if key != 'lidar_occupancy':
                data_dict['observations'][key] = root['observations'][key][indices]
            else:
                data_dict['scans'] = root['observations']['lidar_occupancy'][indices]
                # print(root['observations']['lidar_occupancy'].shape)
                continue
                # data_dict['scan'] = root['observations'][key][indices]
        # data_dict = root['observations'][indices]
        # New code for flattening
        if self.flatten_obs:
            # Extract all the filtered arrays and flatten them
            # Expand dimensions and prepare arrays for concatenation
            # print([arr.shape for arr in data_dict['observations'][key]])
            arrays_to_concat = [data_dict['observations'][key].reshape([data_dict['observations'][key].shape[0], -1]) for key in obs_dictionary_keys]
    
            # print the shapes of all arrays to concatenate
            # print([arr.shape for arr in arrays_to_concat])
            # Concatenate all arrays along the last dimension
            concatenated_obs = np.concatenate(arrays_to_concat, axis=-1)
            
            # Reshape the concatenated array: [timesteps, -1] means keeping the number of timesteps the same, but flattening all other dimensions.
            data_dict['observations'] = concatenated_obs.reshape([concatenated_obs.shape[0], -1])
            
        data_dict['infos'] = dict()
        # print("hi")
        data_dict['infos']['model_name']= root['model_name'][indices]

        if skip_inital != 0:
            data_dict = self.skip_inital_values(data_dict, skip_inital)
        if skip_inital_random_max != 0:
            data_dict = self.skip_inital_values_random(data_dict, 
                                                skip_inital_random_min,
                                                skip_inital_random_max)
        if max_trajectory_length != 0:
            data_dict = self.clip_trajectories(data_dict, 0, max_trajectory_length)

        if split_trajectories != 0:
            data_dict = self.split_trajectories(data_dict, split_trajectories, remove_short_trajectories)
        # print the number of all timesteps
        if min_trajectory_length != 0:
            data_dict = self.filter_trajectories_by_length(data_dict, min_trajectory_length)
        print("Number of timesteps:", len(data_dict['rewards']))

        if only_terminals:
            # or truncates and terminals
            truncated_or_terminals = np.logical_or(data_dict['terminals'], data_dict['timeouts'])
            data_dict['terminals'] = truncated_or_terminals
            data_dict['timeouts'] = truncated_or_terminals
        
        if include_timesteps_in_obs:
            data_dict = self.timesteps_to_obs(data_dict)

        return data_dict
    
    def timesteps_to_obs(self, data_dict):
        # Ensure that 'observations' is in the data_dict
        if 'observations' not in data_dict:
            print("Error: 'observations' key not found in the data dictionary.")
            return data_dict

        # Find the end of each trajectory using 'terminals' and 'timeouts'
        terminals = np.logical_or(data_dict['terminals'], data_dict['timeouts'])
        end_indices = np.where(terminals)[0] + 1

        # Initialize the list to store the modified observations
        modified_observations = []

        # Iterate through the trajectories and append timesteps
        start_idx = 0
        for end_idx in end_indices:
            trajectory_observations = data_dict['observations'][start_idx:end_idx]
            timesteps = np.arange(end_idx - start_idx)[:, None]
            modified_trajectory_obs = np.hstack((trajectory_observations, timesteps))
            modified_observations.append(modified_trajectory_obs)
            start_idx = end_idx

        # Update the observations in data_dict
        data_dict['observations'] = np.vstack(modified_observations)

        return data_dict



    def skip_inital_values_random(self, data_dict, skip_inital_min,skip_initial_max):
        terminals = np.logical_or(data_dict['terminals'],  data_dict['timeouts'])
        start_indices = np.where(terminals[:-1] & ~terminals[1:])[0] + 1
        start_indices = np.concatenate(([0], start_indices))
        # Prepare new data dict to store modified trajectories
        new_data_dict = {
            'rewards': [],
            'terminals': [],
            'timeouts': [],
            'actions': [],
            'raw_actions': [],
            'log_probs': [],
            'index': [],
            'observations': [],
            'scans' : [],
            'infos': {
                'model_name': []
            }
        }

        for i in range(len(start_indices) - 1):
            start, end = start_indices[i], start_indices[i + 1]
            skip_initial = np.random.randint(skip_inital_min, skip_initial_max)
            if (end - start) > skip_initial:
                new_data_dict['rewards'].extend(data_dict['rewards'][start + skip_initial:end])
                new_data_dict['terminals'].extend(data_dict['terminals'][start + skip_initial:end])
                new_data_dict['timeouts'].extend(data_dict['timeouts'][start + skip_initial:end])
                new_data_dict['actions'].extend(data_dict['actions'][start + skip_initial:end])
                new_data_dict['raw_actions'].extend(data_dict['raw_actions'][start + skip_initial:end])
                new_data_dict['log_probs'].extend(data_dict['log_probs'][start + skip_initial:end])
                new_data_dict['index'].extend(data_dict['index'][start + skip_initial:end])

                new_data_dict['observations'].extend(data_dict['observations'][start + skip_initial:end,:])
                new_data_dict['scans'].extend(data_dict['scans'][start + skip_initial:end,:])

                new_data_dict['infos']['model_name'].extend(data_dict['infos']['model_name'][start + skip_initial:end])
        
        # Convert lists back to numpy arrays
        new_data_dict['rewards'] = np.array(new_data_dict['rewards'])
        new_data_dict['terminals'] = np.array(new_data_dict['terminals'])
        new_data_dict['timeouts'] = np.array(new_data_dict['timeouts'])
        new_data_dict['actions'] = np.array(new_data_dict['actions'])
        new_data_dict['raw_actions'] = np.array(new_data_dict['raw_actions'])
        new_data_dict['log_probs'] = np.array(new_data_dict['log_probs'])
        new_data_dict['index'] = np.array(new_data_dict['index'])
        # for key in new_data_dict['observations'].keys():
        new_data_dict['observations'] = np.array(new_data_dict['observations'])
        new_data_dict['scans'] = np.array(new_data_dict['scans'])
        new_data_dict['infos']['model_name'] = np.array(new_data_dict['infos']['model_name'])
        return new_data_dict
    

    def clip_trajectories(self, data_dict: dict(), 
                          min_len: int= 0, 
                          max_len:int =100):

        terminals = np.logical_or(data_dict['terminals'],  data_dict['timeouts'])
        start_indices = np.where(terminals[:-1] & ~terminals[1:])[0] + 1
        start_indices = np.concatenate(([0], start_indices))
        
        # Prepare new data dict to store modified trajectories
        new_data_dict = {key: [] for key in data_dict.keys()}
        if 'infos' in data_dict:
            new_data_dict['infos'] = {key: [] for key in data_dict['infos'].keys()}
        
        for i in range(len(start_indices) - 1):
            start, end = start_indices[i], start_indices[i + 1]
            
            # Clip the trajectory based on min_len and max_len
            clipped_start = max(start + min_len, start)
            clipped_end = min(start + max_len, end)

            if clipped_end > clipped_start:
                for key, value in data_dict.items():
                    if key == 'infos':
                        for info_key in data_dict['infos'].keys():
                            new_data_dict['infos'][info_key].extend(data_dict['infos'][info_key][clipped_start:clipped_end])
                    else:
                        new_data_dict[key].extend(value[clipped_start:clipped_end])
        
        # Convert lists back to numpy arrays
        for key, value in new_data_dict.items():
            if key == 'infos':
                for info_key in new_data_dict['infos'].keys():
                    new_data_dict['infos'][info_key] = np.array(new_data_dict['infos'][info_key])
            else:
                new_data_dict[key] = np.array(value)
        
        return new_data_dict


    def skip_inital_values(self, data_dict, skip_initial):
        terminals = np.logical_or(data_dict['terminals'],  data_dict['timeouts'])
        start_indices = np.where(terminals[:-1] & ~terminals[1:])[0] + 1
        start_indices = np.concatenate(([0], start_indices))
        # Prepare new data dict to store modified trajectories
        new_data_dict = {
            'rewards': [],
            'terminals': [],
            'timeouts': [],
            'actions': [],
            'raw_actions': [],
            'log_probs': [],
            'index': [],
            'observations': [],
            'scans' : [],
            'infos': {
                'model_name': []
            }
        }

        for i in range(len(start_indices) - 1):
            start, end = start_indices[i], start_indices[i + 1]
            if (end - start) > skip_initial:
                new_data_dict['rewards'].extend(data_dict['rewards'][start + skip_initial:end])
                new_data_dict['terminals'].extend(data_dict['terminals'][start + skip_initial:end])
                new_data_dict['timeouts'].extend(data_dict['timeouts'][start + skip_initial:end])
                new_data_dict['actions'].extend(data_dict['actions'][start + skip_initial:end])
                new_data_dict['raw_actions'].extend(data_dict['raw_actions'][start + skip_initial:end])
                new_data_dict['log_probs'].extend(data_dict['log_probs'][start + skip_initial:end])
                new_data_dict['index'].extend(data_dict['index'][start + skip_initial:end])

                new_data_dict['observations'].extend(data_dict['observations'][start + skip_initial:end,:])
                new_data_dict['scans'].extend(data_dict['scans'][start + skip_initial:end,:])

                new_data_dict['infos']['model_name'].extend(data_dict['infos']['model_name'][start + skip_initial:end])
        
        # Convert lists back to numpy arrays
        new_data_dict['rewards'] = np.array(new_data_dict['rewards'])
        new_data_dict['terminals'] = np.array(new_data_dict['terminals'])
        new_data_dict['timeouts'] = np.array(new_data_dict['timeouts'])
        new_data_dict['actions'] = np.array(new_data_dict['actions'])
        new_data_dict['raw_actions'] = np.array(new_data_dict['raw_actions'])
        new_data_dict['log_probs'] = np.array(new_data_dict['log_probs'])
        new_data_dict['index'] = np.array(new_data_dict['index'])
        # for key in new_data_dict['observations'].keys():
        new_data_dict['observations'] = np.array(new_data_dict['observations'])
        new_data_dict['scans'] = np.array(new_data_dict['scans'])
        new_data_dict['infos']['model_name'] = np.array(new_data_dict['infos']['model_name'])
        return new_data_dict

    import numpy as np

    def filter_trajectories_by_length(self, data_dict, min_length):
        terminals = np.logical_or(data_dict['terminals'], data_dict['timeouts'])
        start_indices = np.where(terminals[:-1] & ~terminals[1:])[0] + 1
        start_indices = np.concatenate(([0], start_indices))

        # Prepare new data dict to store modified trajectories
        new_data_dict = {
            'rewards': [],
            'terminals': [],
            'timeouts': [],
            'actions': [],
            'raw_actions': [],
            'log_probs': [],
            'index': [],
            'observations': [],
            'scans': [],
            'infos': {
                'model_name': []
            }
        }

        for i in range(len(start_indices) - 1):
            start, end = start_indices[i], start_indices[i + 1]
            if (end - start) >= min_length:
                new_data_dict['rewards'].extend(data_dict['rewards'][start:end])
                new_data_dict['terminals'].extend(data_dict['terminals'][start:end])
                new_data_dict['timeouts'].extend(data_dict['timeouts'][start:end])
                new_data_dict['actions'].extend(data_dict['actions'][start:end])
                new_data_dict['raw_actions'].extend(data_dict['raw_actions'][start:end])
                new_data_dict['log_probs'].extend(data_dict['log_probs'][start:end])
                new_data_dict['index'].extend(data_dict['index'][start:end])
                new_data_dict['observations'].extend(data_dict['observations'][start:end,:])
                new_data_dict['scans'].extend(data_dict['scans'][start:end,:])
                new_data_dict['infos']['model_name'].extend(data_dict['infos']['model_name'][start:end])

        # Convert lists back to numpy arrays
        new_data_dict['rewards'] = np.array(new_data_dict['rewards'])
        new_data_dict['terminals'] = np.array(new_data_dict['terminals'])
        new_data_dict['timeouts'] = np.array(new_data_dict['timeouts'])
        new_data_dict['actions'] = np.array(new_data_dict['actions'])
        new_data_dict['raw_actions'] = np.array(new_data_dict['raw_actions'])
        new_data_dict['log_probs'] = np.array(new_data_dict['log_probs'])
        new_data_dict['index'] = np.array(new_data_dict['index'])
        new_data_dict['observations'] = np.array(new_data_dict['observations'])
        new_data_dict['scans'] = np.array(new_data_dict['scans'])
        new_data_dict['infos']['model_name'] = np.array(new_data_dict['infos']['model_name'])
        
        return new_data_dict


    def observation_spec(self):
        return self.observation_space
    
    def action_spec(self):
        return self.action_space
    

    def get_laser_scan(self, states, subsample_laser):
        xy = states[:, :2]
        theta = states[:, 2]
        # Expand the dimensions of theta
        theta = np.expand_dims(theta, axis=-1)
        joined = np.concatenate([xy, theta], axis=-1)
        
        all_scans = []
        for pose in joined:
            # print("sampling at pose:", pose)
            # Assuming F110Env.sim.agents[0].scan_simulator.scan(pose, None) returns numpy array
            scan = self.sim.agents[0].scan_simulator.scan(pose, None)[::subsample_laser]
            scan = scan.astype(np.float32)
            all_scans.append(scan)
        # normalize the laser scan
        all_scans = np.array(all_scans)
        return all_scans

    def split_trajectories(self, data_dict, Y, remove_short_trajectories):
        terminals = data_dict['terminals']
        timeouts = data_dict['timeouts']

        # Identify start of each trajectory
        start_indices = np.where(terminals[:-1] & ~terminals[1:])[0] + 1
        # Add the first and last index for completeness
        start_indices = np.concatenate(([0], start_indices, [len(terminals)]))

        # Split each trajectory into sub-trajectories of length Y
        new_data_dict = {
            'rewards': [],
            'terminals': [],
            'timeouts': [],
            'actions': [],
            'raw_actions': [],
            'log_probs': [],
            'index': [],
            'observations': [],
            'scans' : [],
            'infos': {
                'model_name': []
            }
        }


        for i in range(len(start_indices) - 1):
            start, end = start_indices[i], start_indices[i + 1]

            num_subtrajs = (end - start) // Y
            last_subtraj_size = (end - start) % Y

            # For all complete sub-trajectories
            for j in range(num_subtrajs):
                slice_start, slice_end = start + j * Y, start + (j + 1) * Y

                new_data_dict['rewards'].extend(data_dict['rewards'][slice_start:slice_end])
                new_data_dict['terminals'].extend(data_dict['terminals'][slice_start:slice_end])
                new_data_dict['timeouts'].extend(data_dict['timeouts'][slice_start:slice_end])
                new_data_dict['actions'].extend(data_dict['actions'][slice_start:slice_end])
                new_data_dict['raw_actions'].extend(data_dict['raw_actions'][slice_start:slice_end])
                new_data_dict['log_probs'].extend(data_dict['log_probs'][slice_start:slice_end])
                new_data_dict['index'].extend(data_dict['index'][slice_start:slice_end])
                new_data_dict['observations'].extend(data_dict['observations'][slice_start :slice_end,:]) 
                new_data_dict['scans'].extend(data_dict['scans'][slice_start :slice_end,:])
                new_data_dict['infos']['model_name'].extend(data_dict['infos']['model_name'][slice_start:slice_end])
                new_data_dict['terminals'][-1] = False
                new_data_dict['timeouts'][-1] = True
            # For the last sub-trajectory (if exists)
            if last_subtraj_size > 0 and (not timeouts[end-1]) and not remove_short_trajectories:
                slice_start, slice_end = end - last_subtraj_size, end

                new_data_dict['rewards'].extend(data_dict['rewards'][slice_start:slice_end])
                new_data_dict['terminals'].extend(data_dict['terminals'][slice_start:slice_end])
                new_data_dict['timeouts'].extend(data_dict['timeouts'][slice_start:slice_end])
                new_data_dict['actions'].extend(data_dict['actions'][slice_start:slice_end])
                new_data_dict['raw_actions'].extend(data_dict['raw_actions'][slice_start:slice_end])
                new_data_dict['log_probs'].extend(data_dict['log_probs'][slice_start:slice_end])
                new_data_dict['index'].extend(data_dict['index'][slice_start:slice_end])
                new_data_dict['observations'].extend(data_dict['observations'][slice_start :slice_end,:])
                new_data_dict['scans'].extend(data_dict['scans'][slice_start :slice_end,:])
                new_data_dict['infos']['model_name'].extend(data_dict['infos']['model_name'][slice_start:slice_end])
            #elif last_subtraj_size > 0:
            #    new_data_dict['timeouts'][-1] = True
            #    new_data_dict['terminals'][-1] = True
        #
        # Convert lists back to numpy arrays
        new_data_dict['rewards'] = np.array(new_data_dict['rewards'])
        new_data_dict['terminals'] = np.array(new_data_dict['terminals'])
        new_data_dict['timeouts'] = np.array(new_data_dict['timeouts'])
        new_data_dict['actions'] = np.array(new_data_dict['actions'])
        new_data_dict['raw_actions'] = np.array(new_data_dict['raw_actions'])
        new_data_dict['log_probs'] = np.array(new_data_dict['log_probs'])
        new_data_dict['index'] = np.array(new_data_dict['index'])
        
        new_data_dict['observations'] = np.array(new_data_dict['observations'])
        new_data_dict['scans'] = np.array(new_data_dict['scans'])
        new_data_dict['infos']['model_name'] = np.array(new_data_dict['infos']['model_name'])

        return new_data_dict
