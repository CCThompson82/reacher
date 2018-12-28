import os
import sys
import json
import numpy as np
from src.base_models.base_model import BaseModel
from src.base_networks.actor import Network as Actor
from src.base_networks.critic import Network as Critic
from src.buffers.experience_buffer import ExperienceBuffer
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

        self.actor = Actor(nb_features=env_config['nb_observations'],
                           nb_actions=env_config['nb_actions'],
                           params=self.params,
                           seed=self.hyperparams['random_seed'])
        self.actor_target = Actor(nb_features=env_config['nb_observations'],
                                  nb_actions=env_config['nb_actions'],
                                  params=self.params,
                                  seed=self.hyperparams['random_seed'])

        self.critic = Critic(nb_features=env_config['nb_observations'],
                             nb_actions=env_config['nb_actions'],
                             params=self.params,
                             seed=self.hyperparams['random_seed'])
        self.critic_target = Critic(nb_features=env_config['nb_observations'],
                                    nb_actions=env_config['nb_actions'],
                                    params=self.params,
                                    seed=self.hyperparams['random_seed'])

        self.soft_update(
            src_model=self.actor, dst_model=self.actor_target, tau=1.0)
        self.soft_update(
            src_model=self.critic, dst_model=self.critic_target, tau=1.0)

        self.memory = ExperienceBuffer(
            action_size=env_config['nb_actions'],
            max_buffer_size=self.params['experience_buffer']['max_tuples'],
            batch_size=self.hyperparams['batch_size'],
            seed=self.hyperparams['random_seed'])



        # self.optimizer = torch.optim.Adam(
        #     params=self.network.network.parameters(),
        #     lr=self.hyperparams['init_learning_rate'])

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
                            ('buffer_size', self.buffer_size),
                            ('max episode', self.best_episode_score),
                            ('mean episode', self.mean_episode_score)])

    def get_next_actions(self, states):
        state_tensor = torch.from_numpy(states).float()
        actions = self.actor.network.forward(state_tensor).data.numpy()

        # TODO: add noise?

        return actions

    def store_experience(self, states, actions, rewards, next_states,
                         episode_statuses):

        for idx in range(self.env_config['nb_agents']):
            self.memory.add(state=states[idx],
                            action=actions[idx],
                            reward=rewards[idx],
                            next_state=next_states[idx],
                            done=episode_statuses[idx])

    def check_training_status(self):
        """
        Check state of experience buffer or episode status and determine whether
        a training step should be run.

        Returns:
            bool (True if training step should be run)
        """
        status = (self.params['experience_buffer']['min_for_training'] <=
                  self.memory.__len__())
        return status

    def execute_training_step(self):
        """
        Coordinates the gradient estimation and backpropagation of the object
        network
        Returns:

        """
        states, actions, rewards, next_states, dones = self.memory.sample()
        states_tensor = torch.from_numpy(states).float()
        actions_tensor = torch.from_numpy(actions).float()
        next_states_tensor = torch.from_numpy(next_states).float()
        rewards_tensor = torch.from_numpy(rewards).float()

        # train the critic
        Q_expected = self.critic.forward(states_tensor, actions_tensor)

        Q_target = rewards_tensor + (
                self.hyperparams['gamma'] * self.critic_target.forward(
            next_states_tensor, self.actor_target.forward(next_states_tensor)))


        # train the actor



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

    @property
    def buffer_size(self):
        return self.memory.__len__()

    @staticmethod
    def soft_update(src_model, dst_model, tau):
        for dst_param, src_param in zip(dst_model.parameters(),
                                        src_model.parameters()):
            updated_param = tau*src_param.data + (1.0-tau)*dst_param.data
            dst_param.data.copy_(updated_param)

