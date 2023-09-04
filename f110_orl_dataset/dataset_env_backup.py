# import gymnasium as gym
import gym
# import so PAth
from pathlib import Path
from f110_gym.envs.f110_env import F110Env
# import spaces
from gym.spaces import Box, Discrete
import numpy as np

import os 
import sys
import pickle
import tensorflow as tf
import zarr
from typing import Union, Tuple, Dict, Optional, List, Any
from f110_gym.envs import F110Env


class F1tenthDatasetEnv(F110Env):
    def __init__(
        self,
        name,
        # f110_gym_kwarg,
        dataset_url=None,
        flatten_obs=True,
        flatten_acts=True,
        set_terminals=True,
        flatten_trajectories = False,
        laser_obs = True,
        subsample_laser = 10, # 1 is no subsampling
        padd_trajectories = True,
        trajectory_max_length = None,
        max_trajectories = None,
        progress_as_reward = False,
        **kwargs
        ):
        # Call the parent class's init
        print(kwargs)
        super(F1tenthDatasetEnv, self).__init__(**kwargs)


        if not(set_terminals):
            # not implemented yet
            raise NotImplementedError("TODO! Implement non-terminal trajectories")
        self.set_terminals = set_terminals
        self.name = name
        self.dataset_url = dataset_url
        self.max_trajectories = max_trajectories
        self.padd_trajectories = padd_trajectories
        self.trajectory_max_length = trajectory_max_length
        # self.sim_env = F110Env(**f110_gym_kwarg)
        self.flatten_obs = flatten_obs
        self.subsample_laser = subsample_laser
        self.laser_obs = laser_obs
        self.flatten_trajectories = flatten_trajectories
        self.flatten_acts = flatten_acts
        self.progress_as_reward = progress_as_reward
        # open with zarr
        # self.dataset = zarr.open(self.root_dir, mode='r')
        # list the subdirs
        # print(self.dataset.tree())


        # now we define our action and observation spaces
        action_space_low = np.array([-np.pi/2,0])
        action_space_high = np.array([np.pi/2, 10.0])
        self.action_space = gym.spaces.Box(action_space_low, action_space_high)
        
        # observation space is a dict with keys:
        # the first one is the scan
        # the second one is the odometry

        state_space = gym.spaces.Box(low=np.array([-100, -100, 0,-10.0, -10.0, -10.0]), high=np.array([100, 100,2*np.pi, 10,10,10]))
        obs_space = {'state': state_space}
        if self.laser_obs:
            rays = int(1080/subsample_laser)
            laser_scan_space = gym.spaces.Box(low=np.array([0]*rays), high=np.array([30]*rays))
            obs_space['scan'] = laser_scan_space
        
        # print(state_space)
        self.observation_space = gym.spaces.Dict(obs_space)
        # print(self.observation_space)
        self._orig_flat_obs_space = gym.spaces.flatten_space(self.observation_space)
        
        if self.flatten_obs:
            self.observation_space = self._orig_flat_obs_space
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
        
    def pad_trajectories(self, dataset, trajectory_max_length):
        # split dataset into trajectories at done signal = 1
        trajectories = np.split(dataset['actions'], np.where(dataset['terminals'])[0]+1)[0:-1]
        # padd to max_length
        # for trajectories in 
        print([len(tra) for tra in trajectories])
        return dataset

    def get_dataset(
        self,
        zarr_path: Union[str, os.PathLike] = None,
        clip: bool = True,
        rng: Optional[Tuple[int, int]] = None,
        indices: Optional[np.ndarray] = None,
        agents: dict = None,
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
        if rng or indices:
            raise NotImplementedError("TODO! Implement rng and indices")
        if rng is not None and indices is not None:
            raise ValueError("rng and indices cannot be specified at the same time.")
        
        if zarr_path is None:
            raise NotImplementedError("TODO! Download the dataset from the web")
            # zarr_path = self._download_dataset()
        
        #store = zarr.LMDBStore(zarr_path, readonly=True)
        print(zarr_path)
        root = zarr.open(zarr_path, mode='r') #(store=store)

        #print(root.tree())
        trajectories = []
        # find max key in root.group_keys()
        # in order to have ordered trajectories
        max_traj = max([int(i) for i in root.group_keys()]) 
        if self.max_trajectories is None:
            max_traj = max_traj + 1
        else:
            max_traj = min(max_traj + 1, self.max_trajectories)

        for trajectory in range(max_traj):
            if agents is not None:
                #print(agents)
                #print( root[trajectory]['infos']['agent'][0])
                if root[trajectory]['infos']['agent'][0] not in agents["agents"] or \
                    root[trajectory]['infos']['target_speed'][0] not in agents["target_speed"]:
                    #print("continued")
                    continue
                print("added")
            # print(trajectory)
            trajectory = str(trajectory)
            data_dict = {}
            data_dict['rewards'] = np.array(root[trajectory]['rewards'][:])

            data_dict['terminals'] = np.array(root[trajectory]['terminals'][:])
            # array like terminals always false
            if self.set_terminals:
                data_dict['timeouts'] = np.zeros_like(data_dict['terminals'])
            
            data_dict['progress'] = np.array(root[trajectory]['progress'][:])
            if self.progress_as_reward:
                data_dict['rewards'] = data_dict['progress']
        
            data_dict['infos'] = dict()
            data_dict['infos']['agent'] = np.array(root[trajectory]['infos']['agent'][:])
            data_dict['infos']['target_speed'] = np.array(root[trajectory]['infos']['target_speed'][:])
            data_dict['infos']['index'] = np.array([trajectory] * len(data_dict['rewards']))
            if self.flatten_acts:
                data_dict['actions'] = np.vstack((root[trajectory]['actions']['velocity'][:], root[trajectory]['actions']['steering'][:]))
            else:
                data_dict['actions']['steering'] = np.array(root[trajectory]['actions']['steering'][:])
                data_dict['actions']['velocity'] = np.array(root[trajectory]['actions']['velocity'][:])
                #root[trajectory]['infos']
            if self.flatten_obs:
                # just concatenate the state values (each in an array in observations)
                # exepct the laser scan
                observation_keys = root[trajectory]['observations'].array_keys()
                # numpy array of size rewards
                obs = np.empty((0,len(data_dict['rewards'])))
                data_dict['obs_keys'] = []
                for key in observation_keys:
                    if key != 'scan':
                        data_dict['obs_keys'].append(key)
                        #print(obs.shape)
                        #print(key)
                        #print(root[trajectory]['observations'][key][:].shape)
                        obs = np.vstack((obs, root[trajectory]['observations'][key][:]))
                data_dict['obs_keys'] = np.array(data_dict['obs_keys'])
                data_dict['observations'] = obs
            else:
                raise NotImplementedError("TODO! Implement non-flattened observations")
                        
            if self.laser_obs:
                data_dict['scan'] = root[trajectory]['observations']['scan'][:,::self.subsample_laser].T

            trajectories.append(data_dict)
        
        if self.flatten_trajectories:
            assert(self.flatten_obs)
            traj_dict = dict()
            # loop over all trajectories
            for i, trajectory in enumerate(trajectories):
                # loop over all keys in trajectory
                for key in trajectory.keys():
                    # if the key is not in traj_dict, add it
                    if key not in traj_dict.keys():
                        assert(i==0)
                        if isinstance(trajectory[key], dict):
                            traj_dict[key] = dict()
                            for sub_key in trajectory[key].keys():
                                traj_dict[key][sub_key] = np.array([])
                        else:
                            # like the array assoicated with the key
                            if isinstance(trajectory[key], np.ndarray):
                                if trajectory[key].ndim == 1:
                                    traj_dict[key] = np.array([])
                                else:
                                    traj_dict[key] = np.empty((trajectory[key].shape[0],0))
                            else:
                                traj_dict[key] = np.array([])
                    # concatenate the key

                    # if trajectory[key] is a dict unroll
                    if isinstance(trajectory[key], dict):
                        for sub_key in trajectory[key].keys():
                            #print(f"{key}{sub_key}")
                            #print(traj_dict[key][sub_key].shape)
                            #print(trajectory[key][sub_key])
                            traj_dict[key][sub_key] = np.concatenate((traj_dict[key][sub_key], trajectory[key][sub_key])) 
                    else:
                        #print(f"{key}")
                        # concatenate along last dimension
                        #print(traj_dict[key].shape)
                        #print(trajectory[key].shape)
                        traj_dict[key] = np.concatenate((traj_dict[key], trajectory[key]), axis=-1)
            # print(traj_dict.keys())
            #print(traj_dict['observations'].shape)
            traj_dict['observations'] = traj_dict['observations'].T
            traj_dict['actions'] = traj_dict['actions'].T
            if self.laser_obs:
                traj_dict['scan'] = traj_dict['scan'].T
            #print(traj_dict['observations'].shape)
            # print("hi")
            trajectories = traj_dict
            #print(trajectories.keys())
        self.dataset = trajectories
        return trajectories
    
    def observation_spec(self):
        return self.observation_space
    
    def action_spec(self):
        return self.action_space