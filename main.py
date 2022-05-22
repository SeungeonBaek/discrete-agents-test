import os
from datetime import datetime
from typing import Dict

import numpy as np

import pandas as pd

from tensorboardX import SummaryWriter
import wandb

from agent_env_config import env_agent_config
from utils.rl_logger import RLLogger
from utils.rl_loader import RLLoader


def main(env_config: Dict, agent_config: Dict, rl_confing: Dict, data_save_path: str, rl_logger: RLLogger, rl_loader: RLLoader):
    # Env
    env, env_obs_space, env_act_space = rl_loader.env_loader()
    print(f"env_name : {env_config['env_name']}, obs_space : {env_obs_space}, act_space : {env_act_space}")

    if len(env_obs_space) > 1:
        obs_space = 1
        for space in env_obs_space:
            obs_space *= space
    else:
        obs_space = env_obs_space[0]

    act_space = env_act_space

    # Agent
    RLAgent = rl_loader.agent_loader()
    Agent = RLAgent(agent_config, obs_space, act_space)
    print('agent_name: {}'.format(agent_config['agent_name']))

    # csv logging
    if rl_confing['csv_logging']:
        episode_data = dict()
        episode_data['episode_score'] = np.zeros(env_config['max_episode'], dtype=np.float32)
        episode_data['mean_reward']   = np.zeros(env_config['max_episode'], dtype=np.float32)
        episode_data['episode_step']  = np.zeros(env_config['max_episode'], dtype=np.float32)

    for episode_num in range(1, env_config['max_episode']):
        episode_score = 0
        episode_step = 0
        done = False

        prev_obs = None
        prev_action = None
        episode_rewards = []

        obs = env.reset()
        obs = np.array(obs)
        obs = obs.reshape(-1)

        action = None

        while not done:
            if env_config['render']:
                env.render()
            episode_step += 1

            action = Agent.action(obs)
            
            obs, reward, done, _ = env.step(action)
            obs = np.array(obs)
            obs = obs.reshape(-1)

            action = np.array(action)

            episode_score += reward
            episode_rewards.append(reward)

            # Save_xp
            reward_int = 0
            if episode_step > 2:
                if Agent.extension_name == 'ICM' or Agent.extension_name == 'RND' or Agent.extension_name == 'NGU':
                    reward_int = Agent.get_intrinsic_reward(prev_obs, obs, prev_action)
                Agent.save_xp(prev_obs, obs, reward+reward_int, prev_action, done)

            prev_obs = obs
            prev_action = action

            if episode_step >= env_config['max_step']:
                done = True
                continue
            
            
            if Agent.extension_name == 'ICM' or Agent.extension_name == 'RND' or Agent.extension_name == 'NGU':
                rl_logger.step_logging(Agent, reward_int)
            else:
                rl_logger.step_logging(Agent)

        env.close()

        rl_logger.episode_logging(Agent, episode_score, episode_step, episode_num, episode_rewards)

        if rl_confing['csv_logging']:
            episode_data['episode_score'][episode_num-1] = episode_score
            episode_data['mean_reward'][episode_num-1]   = episode_score/episode_step
            episode_data['episode_step'][episode_num-1]  = episode_step

            if episode_num % 10 == 0:
                episode_data_df = pd.DataFrame(episode_data)
                episode_data_df.to_csv(data_save_path+'episode_data.csv', mode='w',encoding='UTF-8' ,compression=None)

        print('epi_num : {episode}, epi_step : {step}, score : {score}, mean_reward : {mean_reward}'.format(episode= episode_num, step= episode_step, score = episode_score, mean_reward=episode_score/episode_step))
        
    env.close()

if __name__ == '__main__':
    """
    Env
    1: LunarLander-v2, 2: procgen, 3: high-way

    Agent
     1: DQN,     2: ICM_DQN,   3: RND_DQN,   4: NGU_DQN
     5: PPO,     6: MEPPO
     7: SAC,     8: TQC_SAC
     9: QR_DQN, 10: IQN,      11: QUOTA,    12: IDAC
    13: RAINBOW 14: ICM_RAINBOW, 15: RND_RAINBOW, 16: NGU_RAINBOW
    17: Agent-57
    18: REDQ,   19: ICM_REDQ, 20: RND_REDQ, 21: NGU_REDQ
    """

    env_switch = 1
    agent_switch = 3

    env_config, agent_config = env_agent_config(env_switch, agent_switch)

    rl_config = {'csv_logging': False, 'wandb': False, 'tensorboard': True}

    parent_path = str(os.path.abspath(''))
    time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    result_path = parent_path + '/results/{env}/{agent}_{extension}_result/'.format(env=env_config['env_name'], agent=agent_config['agent_name'], extension=agent_config['extension']['name']) + time_string
    data_save_path = parent_path + '\\results\\{env}\\{agent}_{extension}_result\\'.format(env=env_config['env_name'], agent=agent_config['agent_name'], extension=agent_config['extension']['name']) + time_string + '\\'

    summary_writer = SummaryWriter(result_path+'/tensorboard/')
    if rl_config["wandb"]:
        wandb_session = wandb.init(project="RL-test-2", job_type="train", name=time_string)
    else:
        wandb_session = None

    rl_logger = RLLogger(agent_config, rl_config, summary_writer, wandb_session)
    rl_loader = RLLoader(env_config, agent_config)

    main(env_config, agent_config, rl_config, data_save_path, rl_logger, rl_loader)
