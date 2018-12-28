#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic script that dynamically loads the named model from `config/model.json`,
and proceeds to train the agent.  Data regarding training
performance and model checkpoints will be written output regularly to
`data/<model name>/<experiment id>/` based on the parameters set in
`config/hyperparameters.json`.
"""
import os
import sys

import json
from tqdm import tqdm
from collections import OrderedDict
from unityagents import UnityEnvironment

WORK_DIR = os.environ['ROOT_DIR']
sys.path.append(WORK_DIR)

from src.clients.training_client import ModelClient

UNITY_ENV_PATH = os.environ['UNITY_ENV_PATH']

if __name__ == '__main__':

    env = UnityEnvironment(file_name=UNITY_ENV_PATH)
    brain = env.brains[env.brain_names[0]]

    env_config = {'nb_actions': brain.vector_action_space_size,
                  'actions_type': brain.vector_action_space_type,
                  'nb_observations': brain.vector_observation_space_size,
                  'observations_type': brain.vector_observation_space_type,
                  'nb_agents': env.reset(train_mode=False)[
                      env.brain_names[0]].vector_observations.shape[0]}

    client = ModelClient(env_config=env_config)

    # build buffer with by running episodes
    pbar = tqdm(total=client.model.hyperparams['max_episodes'])

    while not client.training_finished():
        pbar.set_postfix(
            ordered_dict=client.progress_bar)
        pbar.update()

        # reset for new episodes
        env_info = env.reset(train_mode=True)[brain.brain_name]
        states = env_info.vector_observations

        while not client.terminate_episode(
                max_reached_statuses=env_info.max_reached,
                local_done_statuses=env_info.local_done):

            actions = client.get_next_actions(states=states)

            env_info = env.step(actions)[brain.brain_name]

            rewards = env_info.rewards
            next_states = env_info.vector_observations
            episode_statuses = env_info.local_done

            client.store_experience(
                states, actions, rewards, next_states, episode_statuses)
            client.update_metrics(rewards=rewards)

            if client.training_status():
                client.train_model()

        client.record_episode_scores()

        # if client.checkpoint_step(model_config['checkpoint_frequency']):
        #     client.checkpoint_model()






