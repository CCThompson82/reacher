"""
Client that
* abstracts dynamic model loading
* provides a consistent API for training steps
"""

import os
import sys
import json
from pydoc import locate
import numpy as np

WORK_DIR = os.environ['ROOT_DIR']
sys.path.append(WORK_DIR)
MODEL_CONFIG_PATH = os.path.join(WORK_DIR, 'config', 'model.json')
HYPERPARAMS_CONFIG_PATH = os.path.join(WORK_DIR, 'config', 'hyperparams.json')


class ModelClient(object):
    def __init__(self, env_config):

        with open(MODEL_CONFIG_PATH, 'r') as handle:
            model_config = json.load(handle)

        self.model = self.load_model(model_config=model_config,
                                     env_config=env_config)

        # TODO: turn this into an object
        self.metrics = {'step_counts': np.zeros([env_config['nb_agents']]),
                        'episode_counts': np.zeros([env_config['nb_agents']]),
                        'episode_scores': np.zeros([env_config['nb_agents']])}

    @staticmethod
    def load_model(model_config, env_config):
        model_name = model_config['model_name']
        Model = locate('model.{}.model.Model'.format(model_name))
        if Model is None:
            raise FileNotFoundError('{} does not exist'.format(model_name))

        model = Model(model_config=model_config,
                      env_config=env_config)
        return model

    def training_finished(self):
        return self.model.terminate_training_status(**self.metrics)

    def terminate_episode(self, max_reached_statuses, local_done_statuses):
        return self.model.terminate_episode_status(
            max_reached_statuses, local_done_statuses)

    @property
    def progress_bar(self):
        return self.model.progress_bar(**self.metrics)

    def get_next_actions(self, states):
        actions = self.model.get_next_actions(
            states=states, episode=np.mean(self.metrics['episode_counts']))

        if np.any(np.fabs(actions) > 1.0):
            raise ValueError('Continuous actions cannot exceed absolute of 1')
        return actions

    def store_experience(self, states, actions, rewards, next_states,
                         episode_statuses):
        self.model.store_experience(states, actions, rewards, next_states,
                                    episode_statuses)

    def training_status(self):
        return self.model.check_training_status(
            step=self.metrics['step_counts'][0])

    def train_model(self):
        self.model.train_model()

    def update_metrics(self, rewards):
        self.metrics['step_counts'] += 1
        self.metrics['episode_scores'] += rewards

    def record_episode_scores(self):
        try:
            arr = np.load(self.model.dir_util.results_filename)
            arr = np.concatenate(
                [arr, np.array([self.metrics['episode_scores']])], axis=0)
        except FileNotFoundError:
            arr = np.array([self.metrics['episode_scores']])

        np.save(self.model.dir_util.results_filename, arr)
        self.reset_episode()

    def reset_episode(self):
        self.metrics['episode_scores'][:] = 0
        self.metrics['episode_counts'] += 1

    def checkpoint_step(self):
        return (self.metrics['episode_counts'][0] %
                self.model.hyperparams['checkpoint_freq'] == 0)

    def create_checkpoint(self):
        self.model.checkpoint_model(
            episode_count=self.metrics['episode_counts'][0])

    @property
    def checkpoints_list(self):
        return os.listdir(self.model.dir_util.checkpoint_dir)

    def evaluate_checkpoint(self, index):
        """
        Evaluates a model checkpoint for peak performance, i.e. not using
        action noise, collecting experience, or training model networks.

        Args:
            index:

        Returns:
            None (writes results to disk)
        """


