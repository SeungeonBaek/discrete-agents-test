import os, sys
from datetime import datetime
from typing import Dict

import numpy as np

from tensorboardX import SummaryWriter

if __name__ == "__main__":
	sys.path.append(os.getcwd())

from utils.rl_logger import RLLogger
from utils.rl_loader import RLLoader
from utils.state_logger import StateLogger


MAX_SCORE = 0

def main(env_config: Dict,
         agent_config: Dict,
         rl_config: Dict,
         learned_model_path: str,
         result_path: str,
         rl_logger: RLLogger,
         rl_loader: RLLoader,
         state_logger: StateLogger):
    # Env
    env, env_obs_space, env_act_space = rl_loader.env_loader()
    env_name = env_config['env_name']
    print(f"env_name : {env_name}, obs_space : {env_obs_space}, act_space : {env_act_space}")

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
    if rl_config['use_learned_model']:
        Agent.load_models(path=learned_model_path + "score_" + str(rl_config['learned_model_score']) + "_model")
    else:
        pass

    agent_name = agent_config['agent_name']
    extension_name = Agent.extension_name
    print(f"agent_name: {agent_name}, extension_name: {extension_name}")

    # define max step
    if 'highway-v0' in env_name: # vanilla highway and custom highway
        max_step = env.config['duration'] * env.config['policy_frequency']
        feature_range_x = env.config['observation']['features_range']['x']
        feature_range_y = env.config['observation']['features_range']['y']
        feature_range_vx = env.config['observation']['features_range']['vx']
        feature_range_vy = env.config['observation']['features_range']['vy']

    else:
        max_step = env_config['max_step']

    # csv logging
    if rl_config['csv_logging']:
        state_logger.initialize_memory(env_config['max_episode'], max_step, act_space)

    total_step = 0

    for episode_num in range(1, env_config['max_episode']):
        episode_score = 0
        episode_step = 0
        done = False

        prev_obs = None
        prev_action = None
        episode_rewards = []

        obs = env.reset()
        if env_name == 'custom_highway-v0':
            obs = np.array(obs[0])
        else:
            obs = np.array(obs)

        obs = obs.reshape(-1)
        if rl_config['use_prev_obs']:
            enlonged_obs = np.concatenate((obs, obs))

        action = None

        while not done:
            if env_config['render']:
                env.render()
            episode_step += 1
            total_step += 1

            if rl_config['use_prev_obs']:
                action, action_values = Agent.action(enlonged_obs, rl_config['use_learned_model'])
            else:
                action, action_values = Agent.action(obs, rl_config['use_learned_model'])

            # obs parsing per env
            if env_name == 'LunarLander-v2' or env_name == 'highway-v0':
                obs, reward, done, _ = env.step(action)
                origin_obs = np.array([])

            elif env_name == 'custom_highway-v0':
                obs, reward, done, _ = env.step(action)
                obs, origin_obs = obs[0], obs[1]

            elif env_name == None: # Todo: Consider the various environment
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            obs = np.array(obs)
            obs = obs.reshape(-1)

            if rl_config['use_prev_obs']:
                if episode_step >= 2:
                    enlonged_obs = np.concatenate((prev_obs, obs))
                else:
                    enlonged_obs = np.concatenate((obs, obs))

            action = np.array(action)

            episode_score += reward
            episode_rewards.append(reward)

            # Save_xp
            reward_int = 0
            if rl_config['use_learned_model']:
                pass # does not save the transition
            else:
                if episode_step >= 2:
                    if rl_config['use_prev_obs']:
                        if Agent.extension_name == 'ICM' or Agent.extension_name == 'RND' or Agent.extension_name == 'NGU':
                            reward_int = Agent.get_intrinsic_reward(prev_enlonged_obs, enlonged_obs, prev_action)
                        Agent.save_xp(prev_enlonged_obs, enlonged_obs, reward+reward_int, prev_action, done)
                    else:
                        if Agent.extension_name == 'ICM' or Agent.extension_name == 'RND' or Agent.extension_name == 'NGU':
                            reward_int = Agent.get_intrinsic_reward(prev_obs, obs, prev_action)
                        Agent.save_xp(prev_obs, obs, reward+reward_int, prev_action, done)

            if rl_config['use_prev_obs']:
                prev_enlonged_obs = enlonged_obs

            prev_obs = obs
            prev_action = action

            if episode_step >= max_step:
                done = True
                continue
            
            if Agent.extension_name == 'ICM' or Agent.extension_name == 'RND' or Agent.extension_name == 'NGU':
                rl_logger.step_logging(Agent, reward_int, rl_config['use_learned_model'])
            else:
                rl_logger.step_logging(Agent, rl_config['use_learned_model'])

            if rl_config['csv_logging']:
                state_logger.step_logger(episode_num, episode_step, origin_obs, obs, action_values, action)

        env.close()

        rl_logger.episode_logging(Agent, episode_score, episode_step, episode_num, episode_rewards, inference_mode=rl_config['use_learned_model'])
        
        if rl_config['csv_logging']:
            state_logger.episode_logger(episode_num, episode_score, episode_step)
            state_logger.save_data(episode_num)

        print('epi_num : {episode}, epi_step : {step}, score : {score}, mean_reward : {mean_reward}'.format(episode= episode_num, step= episode_step, score = episode_score, mean_reward=episode_score/episode_step))

        # Evaluation
        if rl_config['evaluation'] == True and rl_config['use_learned_model'] == False:
            if episode_num % rl_config['eval_freq'] == 0:
                eval_num = int(episode_num // rl_config['eval_freq']) 
                eval_main(eval_num, env, env_config, Agent, rl_config, rl_logger)

    env.close()


def eval_main(eval_num:int,
             env:object,
             env_config:Dict,
             agent:object,
             rl_config:Dict,
             rl_logger:RLLogger):
    global MAX_SCORE

    env_name = env_config['env_name']

    # define max step
    if 'highway-v0' in env_name: # vanilla highway and custom highway
        max_step = env.config['duration'] * env.config['policy_frequency']
        feature_range_x = env.config['observation']['features_range']['x']
        feature_range_y = env.config['observation']['features_range']['y']
        feature_range_vx = env.config['observation']['features_range']['vx']
        feature_range_vy = env.config['observation']['features_range']['vy']

    else:
        max_step = env_config['max_step']

    eval_score = 0
    eval_step = 0
    eval_done = False

    prev_obs = None
    prev_action = None
    eval_rewards = []

    obs = env.reset()
    if env_name == 'custom_highway-v0':
        obs = np.array(obs[0])
    else:
        obs = np.array(obs)

    obs = obs.reshape(-1)
    if rl_config['use_prev_obs']:
        enlonged_obs = np.concatenate((obs, obs))

    action = None

    while not eval_done:
        if env_config['render']:
            env.render()
        eval_step += 1

        if rl_config['use_prev_obs']:
            action, action_values = agent.action(enlonged_obs, True)
        else:
            action, action_values = agent.action(obs, True)

        # obs parsing per env
        if env_name == 'LunarLander-v2' or env_name == 'highway-v0':
            obs, reward, eval_done, _ = env.step(action)
            origin_obs = np.array([])

        elif env_name == 'custom_highway-v0':
            obs, reward, eval_done, _ = env.step(action)
            obs, origin_obs = obs[0], obs[1]

        elif env_name == None: # Todo: Consider the various environment
            obs, reward, terminated, truncated, _ = env.step(action)
            eval_done = terminated or truncated

        obs = np.array(obs)
        obs = obs.reshape(-1)

        if rl_config['use_prev_obs']:
            if eval_step >= 2:
                enlonged_obs = np.concatenate((prev_obs, obs))
            else:
                enlonged_obs = np.concatenate((obs, obs))

        action = np.array(action)

        eval_score += reward
        eval_rewards.append(reward)

        prev_obs = obs
        prev_action = action

        if eval_step >= max_step:
            eval_done = True
            continue
        
    env.close()

    rl_logger.eval_logging(agent, eval_score, eval_step, eval_num)
    
    # if rl_config['csv_logging']: # Todo: Develop the evaluation csv logger
    #     state_logger.eval_logger(eval_num, eval_score, eval_step)
    #     state_logger.save_data(eval_num, eval=True)

    if eval_score > MAX_SCORE:
        if os.name == 'nt':
            print(f"result_path: {result_path}")
            agent.save_models(path=result_path + "\\", score=round(eval_score, 3))
        elif os.name == 'posix':
            agent.save_models(path=result_path + "/", score=round(eval_score, 3))
        MAX_SCORE = eval_score

    print(f'eval_num : {eval_num}, eval_step : {eval_step}, score : {eval_score}, mean_reward : {eval_score/eval_step}')


if __name__ == '__main__':
    from agent_env_config import env_agent_config, agent_network_config

    """
    Env
    1: LunarLander-v2, 2: procgen, 3: highway, 4: custom-highway

    Agent
     0: Blank DQN,
     1: DQN,      2: PPO        3: SAC
     4: C51,      5: QR_DQN,    6: QUOTA
     7: IQN,      8: FQF,       9: MMDQN
     10: C2D,    11: Agent-57, 12: REDQ

    Extension/Auxiliary
    1: Vanilla,   2: ICM,            3: RND,
    4: NGU,       5: Model Ensemble, 6: TQC

    Extractor
    1: None, name = None,
    2: MLP, name = {'Flatten', 'MLP', 'AutoEncoder1D', 'Inception1D', 'UNet1D'},
    3: Convolutional, name = {#Todo: Writing the name of Extractors},
    4: Recurrent, name = {'RNN', 'LSTM', 'GRU', 'CNN1D'},
    5: Attention, name = {#Todo: Writing the name of Extractors},
    6: Graph, name = {#Todo: Writing the name of Extractors},
    7: Custom, name = {'SimpleMLP', 'SimpleInception', 'Residual', 'AE', 'UNet', 'SimpleGRU', 'LSTM', 'CNN1D', \
                       'BiLSTM', 'Attention', 'TransductiveGNN', 'InductiveGNN', 'Transformer'}
    """

    env_switch = 1
    agent_switch = 7
    aux_switch = 5
    extractor_switch = 4
    extractor_name = 'CNN1D'

    fcn_config={'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005, 'network_architecture': [256],\
                'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}

    env_config, agent_config = env_agent_config(env_switch, agent_switch, aux_switch)
    agent_config = agent_network_config(agent_config=agent_config, extractor_switch=extractor_switch, extractor_name=extractor_name, fcn_config=fcn_config)

    rl_config = {'csv_logging': False, 'wandb': False, 'tensorboard': True, 'use_prev_obs': False, 'evaluation': True, 'eval_freq': 10,
                 'use_learned_model': False, 'learned_time': '2022-11-29_14-58-22', 'learned_model_score': 61.283}

    parent_path = str(os.path.abspath(''))
    time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    result_path = parent_path + f"/results/{env_config['env_name']}/{agent_config['agent_name']}_{agent_config['extension']['name']}_result/" + time_string

    if os.name == 'nt':
        learned_model_path = parent_path + f"\\results\\{env_config['env_name']}\\{agent_config['agent_name']}_{agent_config['extension']['name']}_result\\" + rl_config['learned_time'] + "\\"
        data_save_path = parent_path + f"\\results\\{env_config['env_name']}\\{agent_config['agent_name']}_{agent_config['extension']['name']}_result\\" + time_string + '\\'
    elif os.name == 'posix':
        learned_model_path = parent_path + f"/results/{env_config['env_name']}/{agent_config['agent_name']}_{agent_config['extension']['name']}_result/" + rl_config['learned_time'] + "/"
        data_save_path = parent_path + f"/results/{env_config['env_name']}/{agent_config['agent_name']}_{agent_config['extension']['name']}_result/" + time_string + '/'

    summary_writer = SummaryWriter(result_path+'/tensorboard/')
    if rl_config['wandb'] == True:
        import wandb
        wandb_session = wandb.init(project="RL-test-2", job_type="train", name=time_string)
    else:
        wandb_session = None

    rl_logger = RLLogger(agent_config, rl_config, summary_writer, wandb_session)
    rl_loader = RLLoader(env_config, agent_config)

    state_logger = StateLogger(env_config, agent_config, rl_config, data_save_path)

    main(env_config, agent_config, rl_config, learned_model_path, result_path, rl_logger, rl_loader, state_logger)