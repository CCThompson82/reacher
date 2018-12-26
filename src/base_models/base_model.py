#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base object that abstracts filenames, paths, and provides the generic attr or methods that all top
level model objects will utilise.
"""

import os
import sys
import json
import shutil

ROOT_DIR = os.environ['ROOT_DIR']
sys.path.append(ROOT_DIR)


class BaseModel(object):
    def __init__(self, model_config, hyperparam_config, env_config):
        """

        Args:
            train_config:
            hyperparam_config:
            env_config:
        """
        self.model_config = model_config
        self.hyperparams = hyperparam_config
        self.env_config = env_config

        # directory management
        self.dir_util = FileManager(model_config=model_config)
        self.dir_util.dump_experiment_info(hyperparams=hyperparam_config)


class FileManager(object):
    def __init__(self, model_config):
        self.model_name = model_config['model_name']
        self.experiment_id = model_config['experiment_id']
        self.overwrite_experiment = model_config['overwrite_experiment']
        self.mode = model_config['mode']

        # store paths and filenames
        self.model_dir = os.path.join(
            ROOT_DIR, 'data', self.model_name, self.experiment_id)
        self.results_dir = os.path.join(self.model_dir, 'results')
        self.results_filename = os.path.join(self.results_dir,
                                             'episode_scores.npy')
        self.evaluation_dir = os.path.join(self.model_dir, 'evaluation')
        self.checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
        self.experiment_info_dir = os.path.join(
            self.model_dir, 'experiment_info')

        if self.mode == 'eval':
            if not os.path.isdir(self.model_dir):
                raise FileNotFoundError(
                    'This model experiment has not been trained previously.')
            else:
                try:
                    os.mkdir(self.evaluation_dir)
                except FileExistsError:
                    pass

        elif self.mode == 'train':
            if not os.path.isdir(self.model_dir):
                self.create_directory_structure()
            elif self.overwrite_experiment:
                shutil.rmtree(self.model_dir)
                self.create_directory_structure()
            else:
                raise IOError(
                    'An experiment for {}: {} already exists.  Set overwrite to '
                    'True in  `config/train.json` if you wish to '
                    'overwrite the previous experiment.'.format(
                        self.model_name, self.experiment_id))
        else:
            raise AttributeError('`mode` must be in ["train", "eval"]')

    def create_directory_structure(self):
        os.makedirs(self.model_dir)
        os.mkdir(os.path.join(self.model_dir, 'results'))
        os.mkdir(os.path.join(self.model_dir, 'checkpoints'))
        os.mkdir(os.path.join(self.model_dir, 'experiment_info'))

    def dump_experiment_info(self, hyperparams):
        hyperparams['model_name'] = self.model_name
        hyperparams['experiment_id'] = self.experiment_id

        filename = os.path.join(self.experiment_info_dir, 'params.json')
        with open(filename, 'w') as out:
            json.dump(hyperparams, out)


