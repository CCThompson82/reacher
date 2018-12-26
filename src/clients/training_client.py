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
MODEL_CONFIG_PATH = os.path.join(WORK_DIR, 'config', 'train.json')
HYPERPARAMS_CONFIG_PATH = os.path.join(WORK_DIR, 'config', 'hyperparams.json')


class ModelClient(object):
    def __init__(self, env_config):

        with open(MODEL_CONFIG_PATH, 'r') as handle:
            model_config = json.load(handle)

        with open(HYPERPARAMS_CONFIG_PATH, 'r') as handle:
            hyperparams_config = json.load(handle)

        self.model = self.load_model(model_config=model_config,
                                     hyperparams_config=hyperparams_config,
                                     env_config=env_config)

        # TODO: turn this into an object
        self.state = {'step_count': np.zeros([env_config['nb_agents']]),
                      'episode_count': np.zeros([env_config['nb_agents']]),
                      'episode_score': np.zeros([env_config['nb_agents']])}

    @staticmethod
    def load_model(model_config, hyperparams_config, env_config):
        model_name = model_config['model_name']
        Model = locate('model.{}.model.Model'.format(model_name))
        if Model is None:
            raise FileNotFoundError('{} does not exist'.format(model_name))

        model = Model(model_config=model_config,
                      hyperparam_config=hyperparams_config,
                      env_config=env_config)
        return model
