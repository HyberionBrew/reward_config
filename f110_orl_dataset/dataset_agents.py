from stable_baselines3 import PPO
import os
import inspect
import torch

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
                "poses_theta",
                "ang_vels_z",
                "linear_vels_x",
                "linear_vels_y",
                "previous_action",
                "progress",
                "lidar_occupancy"
            ]

    def __call__(self, obs, actions=None):
        # make sure obs is a dict and contains all the keys
        assert isinstance(obs, dict)
        assert all([key in obs for key in self.obs_keys])
        log_prob = None
        if actions is None:
            #print("hi")
            with torch.no_grad():
                actions = self.model.predict(obs, deterministic=self.deterministic)[0].squeeze(1)
            #    print("actions", actions)
        else:
            # assert()
            with torch.no_grad():
                # actions to tensor
                # check if it is a torch tensor already
                if not isinstance(actions, torch.Tensor):
                    actions = torch.tensor(actions) #.unsqueeze(0)
                # TODO! check what the actions shape is and should be
                #print(actions.shape)
                
                obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
                print("pns")
                print(obs_tensor)
                # assert(False)
                _, log_prob, _ = self.model.policy.evaluate_actions(obs_tensor, actions)
                # flatten log_prob
                log_prob = log_prob.flatten()
                print(log_prob)
                assert(log_prob.shape[0] == actions.shape[0])
                assert(log_prob.shape[0] == obs_tensor["poses_x"].shape[0])
            # return log_prob
            # keep same actions
        return None, actions, log_prob