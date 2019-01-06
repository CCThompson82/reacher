#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic script that dynamically loads the named model from `config/eval.json`,
and proceeds to evaluate checkpoints that were made during the training of the 
named model.
"""
import os
import sys
from tqdm import tqdm
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

    if client.model.dir_util.mode != 'eval':
        raise AttributeError('Cannot run the evaluation script if the mode is'
                             'not set to `eval` in `model.json`.')

    checkpoint_index = set(
        [ckpt_name[:ckpt_name.find('.pth')] for ckpt_name in
         [fname.split('_')[-1] for fname in client.checkpoints_list]])


    # run baseline with random weights
    for episode_nb in tqdm(range(100)):
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

            client.update_metrics(rewards=rewards)
            states = next_states
        client.record_eval_episode_score(ckpt_index=0)

    for ckpt_index in checkpoint_index:
        client.restore_checkpoint(ckpt_index)

        for episode_nb in tqdm(range(100)):
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

                client.update_metrics(rewards=rewards)
                states = next_states
            client.record_eval_episode_score(ckpt_index=ckpt_index)

