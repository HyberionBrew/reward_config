from stable_baselines3 import PPO
import os
import inspect
import torch
import numpy as np

class F110Actor(object):
    def __init__(self, name="td_progress", deterministic=False):
        self.name = name
        current_file_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        model_path = os.path.join(current_file_path, f"data/{name}")
        print(model_path)
        self.model = PPO.load(model_path, env=None)
        self.deterministic = deterministic
        self.obs_keys = [
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
                "lidar_occupancy"
            ]

    def __call__(self, obs, std=0.0, actions=None):
        # make sure obs is a dict and contains all the keys
        if std!=0.0:
            # not implemented error
            raise NotImplementedError
        assert isinstance(obs, dict)
        assert all([key in obs for key in self.obs_keys])
        log_prob = None
        if actions is None:
            #print("hi")
            with torch.no_grad():
                tensor_obs = self.model.policy.obs_to_tensor(obs)[0]

                actions, _, log_prob = self.model.policy.forward(tensor_obs, deterministic=self.deterministic)# [0]
                actions = actions.squeeze(1).cpu().numpy()
                # print("wwwwww")
            #    print("actions", actions)
        else:
            # assert()
            with torch.no_grad():
                # actions to tensor
                # check if it is a torch tensor already
                if not isinstance(actions, torch.Tensor):
                    # print("crash here", actions)
                    # tf tensor to numpy
                    actions = actions.numpy()
                    # numpy to torch tensor
                    # print(self.model.policy.device)
                    actions = torch.tensor(actions, device=self.model.policy.device) #.unsqueeze(0)
                # TODO! check what the actions shape is and should be
                # print(actions.shape)
                
                obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
                # print(obs_tensor)
                _, log_prob, _ = self.model.policy.evaluate_actions(obs_tensor, actions)
                # flatten log_prob
                log_prob = log_prob.flatten()
                # print(log_prob)
                log_prob = log_prob.cpu().numpy()
                assert(log_prob.shape[0] == actions.shape[0])
                assert(log_prob.shape[0] == obs_tensor["poses_x"].shape[0])
            # return log_prob
            # keep same actions
        return None, actions, log_prob
    
class F110Stupid(object):
    def __init__(self):
        pass
    def __call__(self, obs, std=0.0, actions=None):
        # return action [0.0, 1.0] in batch shape of obs
        # print(obs["previous_action"].shape)
        actions = np.zeros_like(obs["previous_action"])
        actions = actions.squeeze(1)
        actions[:, 1] = 1.0
        log_prob = - np.ones((actions.shape[0], 1), dtype=np.float32)
        # print(actions.shape)
        return None, actions, log_prob