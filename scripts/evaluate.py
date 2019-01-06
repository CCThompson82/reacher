#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic script that dynamically loads the named model from `config/eval.json`,
and proceeds to evaluate checkpoints that were made during the training of the 
named model.
"""import os
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

    print(client.checkpoints_list)
    raise ValueError()

    checkpoint_index = [fname.split('_')[-1][:fname.find('.pth')] for
                        fname in client.checkpoints_list]

    print(checkpoint_index)
    raise ValueError()

    # run baseline with random weights
    for episode_nb in range(100):
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
        eval_filename = os.path.join(client.model.dir_util.evalulation_dir,
                                     'ckpt_0_evaluation.npy')
        client.record_episode_scores(
            filename=eval_filename)

    for idx in checkpoint_index:
        client.restore_checkpoint(index=idx)
        client.evalulate_checkpoint(index=idx)


    # for checkpoint in checkpoint_set:


    #
    #
    #
    #
    #
    #
    #
    # checkpoint_dir = os.path.join(
    #     WORK_DIR, 'data', eval_config['model_name'],
    #     eval_config['experiment_id'], 'checkpoints')
    # checkpoint_set = os.listdir(checkpoint_dir)
    # checkpoint_set = ['ckpt_0.pth'] + checkpoint_set
    #
    # pbar = tqdm(total=len(checkpoint_set)*int(eval_config['nb_evaluations']))
    # for checkpoint in checkpoint_set:
    #
    #     if checkpoint != 'ckpt_0.pth':
    #         client.restore_checkpoint(checkpoint)
    #
    #     trial = eval_config['evaluation_id']
    #     for episode in range(eval_config['nb_evaluations']):
    #         pbar.set_postfix(
    #             ordered_dict=OrderedDict(
    #                 [('checkpoint', checkpoint.split('.')[0]),
    #                  ('trial episode', episode),
    #                  ('mean episode score', client.mean_eval_score(
    #                      checkpoint, str(trial)))]))
    #         pbar.update()
    #
    #         env_info = env.reset(train_mode=True)[brain.brain_name]
    #         state = env_info.vector_observations[0]
    #
    #         while not (env_info.local_done[0] or env_info.max_reached[0]):
    #             action = client.get_next_action(state=state)
    #             env_info = env.step(action)[brain.brain_name]
    #             reward = env_info.rewards[0]
    #             next_state = env_info.vector_observations[0]
    #
    #             client.store_reward(reward)
    #             state = next_state
    #         client.record_eval_episode_score(str(trial), checkpoint)
    #
