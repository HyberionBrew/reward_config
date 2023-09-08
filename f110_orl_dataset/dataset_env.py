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
    "poses_theta",
    "ang_vels_z",
    "linear_vels_x",
    "linear_vels_y",
    "previous_action",
    "progress"
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

        # Append each box in desired order
        state_dict['poses_x'] = Box(POSE_LOW, POSE_HIGH, (1,), np.float32)
        state_dict['poses_y'] = Box(POSE_LOW, POSE_HIGH, (1,), np.float32)
        state_dict['poses_theta'] = Box(POSE_THETA_LOW, POSE_THETA_HIGH, (1,), np.float32)
        state_dict['ang_vels_z'] = Box(VEL_LOW, VEL_HIGH, (1,), np.float32)
        state_dict['linear_vels_x'] = Box(VEL_LOW, VEL_HIGH, (1,), np.float32)
        state_dict['linear_vels_y'] = Box(VEL_LOW, VEL_HIGH, (1,), np.float32)
        state_dict['previous_action'] = Box(low=np.asarray([[self.action_space.low[0][0], MIN_VEL]]), 
                                        high=np.asarray([[self.action_space.high[0][0], MAX_VEL]]), 
                                        shape=(1, 2), dtype=np.float32)
        state_dict['progress'] = Box(0.0, 1.0, (1,), np.float32)
        # Convert the ordered dictionary to a gym space dict
        state_space = gym.spaces.Dict(state_dict)

        print(state_space)
        
        self.observation_space = state_space #gym.spaces.Dict({
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
                if key not in ['lidar_occupancy', 'progress']:
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
        print("hi")
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
        without_agents: Optional[np.ndarray] = [],
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
        indices = np.where(~np.isin(model_names, without_agents))[0]
        # print("Indices:", indices)
        data_dict['rewards'] = root['rewards'][indices]
        # print("hi")
        data_dict['terminals'] = root['done'][indices]
        data_dict['timeouts'] = root['truncated'][indices]
        # print("hi")
        data_dict['actions'] = root['actions'][indices]
        data_dict['log_probs'] = root['log_prob'][indices]
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
        return data_dict
        
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
            print("sampling at pose:", pose)
            # Assuming F110Env.sim.agents[0].scan_simulator.scan(pose, None) returns numpy array
            scan = self.sim.agents[0].scan_simulator.scan(pose, None)[::subsample_laser]
            scan = scan.astype(np.float32)
            all_scans.append(scan)
        # normalize the laser scan
        all_scans = np.array(all_scans)
        return all_scans