import os
import sys
import json
import numpy as np
from src.base_models.base_model import BaseModel
from src.base_networks.base_network import Network
from collections import OrderedDict
import torch
nn = torch.nn

WORK_DIR = os.environ['ROOT_DIR']
sys.path.append(WORK_DIR)


class Model(BaseModel):
    def __init__(self, model_config, hyperparam_config, env_config):
        super(Model, self).__init__(model_config=model_config,
                                    hyperparam_config=hyperparam_config,
                                    env_config=env_config)

        self.network = Network(nb_features=env_config['nb_observations'],
                               nb_actions=env_config['nb_actions'],
                               params=self.params,
                               seed=self.hyperparams['random_seed'])
        self.optimizer = torch.optim.Adam(
            params=self.network.network.parameters(),
            lr=self.hyperparams['init_learning_rate'])

        # self.memory = ExperienceBuffer()

    def terminate_training_status(self, episode_counts, **kwargs):
        return np.mean(episode_counts) >= self.hyperparams['max_episodes']

    @staticmethod
    def terminate_episode_status(max_reached_statuses, local_done_statuses):
        max_reached_status = np.any(max_reached_statuses)
        local_done_status = np.any(local_done_statuses)
        # NOTE: This environment should always return the same value for each
        # agent in both the max_reached and local_done arrays, be it all False
        # or all True.  This may not be the case for other parallel
        # environments.
        return np.any([max_reached_status, local_done_status])

    def progress_bar(self, step_counts, **kwargs):
        return OrderedDict([('step_count', int(np.mean(step_counts))),
                            ('max episode', self.best_episode_score),
                            ('mean episode', self.mean_episode_score)])

    def get_next_actions(self, states):
        state_tensor = torch.from_numpy(states).float()
        actions = self.network.network.forward(state_tensor).data.numpy()

        # TODO: add noise?

        return actions

    def store_experience(self, states, actions, rewards, next_states,
                         episode_statuses):
        pass

    def check_training_status(self):
        """
        Check state of experience buffer or episode status and determine whether
        a training step should be run.

        Returns:
            bool (True if training step should be run)
        """

        return True

    def execute_training_step(self):
        """
        Coordinates the gradient estimation and backpropagation of the object
        network
        Returns:

        """
        states = torch.from_numpy(np.random.randn(20, 33)).float()
        actions = self.network.network.forward(input=states)
        self.optimizer.zero_grad()
        actions.backward(torch.randn(20, 4))
        self.optimizer.step()


    @property
    def mean_episode_score(self):
        try:
            arr = np.load(self.dir_util.results_filename)
        except FileNotFoundError:
            return 0

        if len(arr) < 100:
            return np.round(np.mean(arr), 3)

        return np.round(np.mean(arr[-100:]), 3)

    @property
    def best_episode_score(self):
        try:
            arr = np.load(self.dir_util.results_filename)
        except FileNotFoundError:
            return 0
        return np.round(np.max(arr), 3)


# class ExperienceBuffer(object):
#     """Buffer to store experience tuples."""
#
#     def __init__(self, nb_agents, nb_state_features, nb_actions):
#         self.states = np.empty()