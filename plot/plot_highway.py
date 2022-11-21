import os, sys
from typing import Tuple, List
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def loading_data(data_save_path: str, episode_num: int)-> Tuple[pd.DataFrame]:
    episode_data = pd.read_csv(data_save_path + "episode_data.csv", index_col=0)
    if os.name == 'nt':
        step_data = pd.read_csv(data_save_path + f"\\step_data\\episode_{episode_num}_data.csv", index_col=0)
    elif os.name == 'posix':
        step_data = pd.read_csv(data_save_path + f"/step_data/episode_{episode_num}_data.csv", index_col=0)

    return episode_data, step_data


def plot_highway(env_name: str,
                 agent_name: str,
                 episode_data: pd.DataFrame,
                 step_data: pd.DataFrame,
                 avg_num: int,
                 end_of_episode: int,
                 end_of_step: int,
                 action_num: int,
                 quantile_num: int,
                 steel_shot_list:List[int])-> None:
    sns.set_theme(style = "darkgrid")

    spec_0 = [['Score', 'MeanReward']]

    spec_1 = [['Pos_xy', 'Pos_xy'   ],
              ['Pos_xy', 'Pos_xy'   ],
              ['Pos_x', 'Pos_y'     ],
              ['Vel_x', 'Vel_y'     ]]

    spec_2 = [['THW', 'TTCi'        ],
              ['THW_vs_TTCi', 'LC'  ]]

    spec_3 = [[f"Pos_xy_{steel_shot_idx}" for steel_shot_idx in steel_shot_list]]

    if agent_name == 'QR_DQN' or agent_name == 'QUOTA' or agent_name == 'IQN':
        for action_idx in range(action_num):
            spec_3.append([f"Value_{steel_shot_idx}_{action_idx}" for steel_shot_idx in steel_shot_list])

    elif agent_name == 'DQN' or agent_name == 'PPO' or agent_name == 'SAC':
        spec_3.append([f"Value_{steel_shot_idx}" for steel_shot_idx in steel_shot_list])

    fig_0, axes_0 = plt.subplot_mosaic(spec_0, figsize=(12, 6))
    fig_0.suptitle(f'{env_name}, {agent_name} - RL metric')

    fig_1, axes_1 = plt.subplot_mosaic(spec_1, figsize=(12, 6))
    fig_1.suptitle(f'{env_name}, {agent_name} - Vehicle states')

    fig_2, axes_2 = plt.subplot_mosaic(spec_2, figsize=(12, 6))
    fig_2.suptitle(f'{env_name}, {agent_name}')

    fig_3, axes_3 = plt.subplot_mosaic(spec_3, figsize=(12, 6))
    fig_3.suptitle(f'{env_name}, {agent_name} - Q values')

    palette = sns.cubehelix_palette(light=.5, n_colors=30, gamma=0.5)
    # palette1 = sns.color_palette("mako_r", 6)
    colors = [[0.2, 0.2, 0.8], [0.8, 0.5, 0.2], [0.3, 0.8, 0.2]]

    # Episode data
    sns.lineplot(ax=axes_0['Score'], x=range(len(episode_data['episode_score'][:end_of_episode])), y=episode_data['episode_score'][:end_of_episode].rolling(window=avg_num).mean())
    # axes_0['Score'].set_ylim((-400, 400))

    sns.lineplot(ax=axes_0['MeanReward'], x=range(len(episode_data['mean_reward'][:end_of_episode])), y=episode_data['mean_reward'][:end_of_episode].rolling(window=avg_num).mean())
    # axes_0['MeanReward'].set_ylim((-100, -10))

    # Step data - 1: line plot
    sns.lineplot(ax=axes_1['Pos_xy'], x=step_data['position_x'][:end_of_step], y=step_data['position_y'][:end_of_step], \
        marker='s', dashes=True)

    sns.scatterplot(ax=axes_1['Pos_xy'], x=step_data['other_1_pos_x'][:end_of_step], y=step_data['other_1_pos_y'][:end_of_step], \
        marker='s', color=[0.75, 0.75, 0.75])
    sns.scatterplot(ax=axes_1['Pos_xy'], x=step_data['other_2_pos_x'][:end_of_step], y=step_data['other_2_pos_y'][:end_of_step], \
        marker='s', color=[0.75, 0.75, 0.75])
    sns.scatterplot(ax=axes_1['Pos_xy'], x=step_data['other_3_pos_x'][:end_of_step], y=step_data['other_3_pos_y'][:end_of_step], \
        marker='s', color=[0.75, 0.75, 0.75])
    sns.scatterplot(ax=axes_1['Pos_xy'], x=step_data['other_4_pos_x'][:end_of_step], y=step_data['other_4_pos_y'][:end_of_step], \
        marker='s', color=[0.75, 0.75, 0.75])
    axes_1['Pos_xy'].set_xlim((0, 5000))

    sns.lineplot(ax=axes_1['Pos_x'], x=range(len(step_data['position_x'][:end_of_step])), y=step_data['position_x'][:end_of_step].rolling(window=avg_num).mean())
    # axes_1['MeanReward'].set_ylim((-100, -10))

    sns.lineplot(ax=axes_1['Pos_y'], x=range(len(step_data['position_y'][:end_of_step])), y=step_data['position_y'][:end_of_step].rolling(window=avg_num).mean())
    # axes_1['MeanReward'].set_ylim((-100, -10))

    sns.lineplot(ax=axes_1['Vel_x'], x=range(len(step_data['velocity_x'][:end_of_step])), y=step_data['velocity_x'][:end_of_step].rolling(window=avg_num).mean())
    # axes_1['MeanReward'].set_ylim((-100, -10))

    sns.lineplot(ax=axes_1['Vel_y'], x=range(len(step_data['velocity_y'][:end_of_step])), y=step_data['velocity_y'][:end_of_step].rolling(window=avg_num).mean())
    # axes_1['MeanReward'].set_ylim((-100, -10))

    # Step data - 2: LC histogram
    sns.histplot(ax=axes_2['THW'], data=step_data['time_headway'][:end_of_step], stat="count", color=colors[0], kde=True)
    # axes_2['MeanReward'].set_xlim((-100, -10))
    # axes_2['MeanReward'].set_ylim((-100, -10))

    sns.histplot(ax=axes_2['TTCi'], data=step_data['inverse_of_ttc'][:end_of_step], stat="count", color=colors[0], kde=True)
    # axes_2['MeanReward'].set_xlim((-100, -10))
    # axes_2['MeanReward'].set_ylim((-100, -10))

    sns.scatterplot(ax=axes_2['THW_vs_TTCi'], x=step_data['time_headway'][:end_of_step], y=step_data['inverse_of_ttc'][:end_of_step], s=5, color=".15")
    sns.histplot(ax=axes_2['THW_vs_TTCi'], x=step_data['time_headway'][:end_of_step], y=step_data['inverse_of_ttc'][:end_of_step], bins=50, pthresh=.1, cmap="mako")
    sns.kdeplot(ax=axes_2['THW_vs_TTCi'], x=step_data['time_headway'][:end_of_step], y=step_data['inverse_of_ttc'][:end_of_step], levels=5, color="w", linewidths=1)
    # axes_2['MeanReward'].set_xlim((-100, -10))
    # axes_2['MeanReward'].set_ylim((-100, -10))

    sns.histplot(ax=axes_2['LC'], data=step_data['lane_change_flag'][:end_of_step], stat="count", color=colors[0], kde=True)
    # axes_2['MeanReward'].set_xlim((-100, -10))
    # axes_2['MeanReward'].set_ylim((-100, -10))

    # Step data - 3: LC histogram
    for steel_shot_num in steel_shot_list:
        sns.lineplot(ax=axes_3[f"Pos_xy_{steel_shot_num}"], x=[step_data['position_x'][steel_shot_num]], y=[step_data['position_y'][steel_shot_num]], \
            marker='s', dashes=True)

        sns.scatterplot(ax=axes_3[f"Pos_xy_{steel_shot_num}"], x=[step_data['other_1_pos_x'][steel_shot_num]], y=[step_data['other_1_pos_y'][steel_shot_num]], \
            marker='s', color=[0.75, 0.75, 0.75])
        sns.scatterplot(ax=axes_3[f"Pos_xy_{steel_shot_num}"], x=[step_data['other_2_pos_x'][steel_shot_num]], y=[step_data['other_2_pos_y'][steel_shot_num]], \
            marker='s', color=[0.75, 0.75, 0.75])
        sns.scatterplot(ax=axes_3[f"Pos_xy_{steel_shot_num}"], x=[step_data['other_3_pos_x'][steel_shot_num]], y=[step_data['other_3_pos_y'][steel_shot_num]], \
            marker='s', color=[0.75, 0.75, 0.75])
        sns.scatterplot(ax=axes_3[f"Pos_xy_{steel_shot_num}"], x=[step_data['other_4_pos_x'][steel_shot_num]], y=[step_data['other_4_pos_y'][steel_shot_num]], \
            marker='s', color=[0.75, 0.75, 0.75])

        if agent_name == 'QR_DQN' or agent_name == 'QUOTA' or agent_name == 'IQN':
            for steel_shot_idx in steel_shot_list:
                per_action_q_value = defaultdict(lambda:[])
                for action_idx in range(action_num):
                    for quantile_idx in range(quantile_num):
                        per_action_q_value[action_idx].append(step_data[f"value_{action_idx}_{quantile_idx}"][steel_shot_num])
                    # Todo:
                    sns.histplot(ax=axes_3[f"Value_{steel_shot_idx}_{action_idx}"], data=pd.DataFrame(per_action_q_value[action_idx]), stat="count", color=colors[0], kde=True)

        elif agent_name == 'DQN' or agent_name == 'PPO' or agent_name == 'SAC':
            for steel_shot_idx in steel_shot_list:
                sns.histplot(ax=axes_3[f"Value_{steel_shot_idx}"], x=range(action_num), y=step_data['lane_change_flag'][steel_shot_num], stat="count", color=colors[0], kde=True)
                # spec_3.append([f"Value_{steel_shot_idx}" for steel_shot_idx in range(steel_shot_list)])

    plt.show()


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    from agent_env_config import env_agent_config
    """
    Env
     1: LunarLander-v2, 2: procgen, 3: highway, 4: custom-highway

    Agent
     1: DQN,     2: ICM_DQN,      3: RND_DQN,      4: NGU_DQN,
     5: PPO,     6: MEPPO,
     7: SAC,     8: TQC_SAC,
     9: QR_DQN, 10: ICM_QR_DQN   11: RND_QR_DQN,  12: NGU_QR_DQN,
    13: IQN,    14: QUOTA,
    15: RAINBOW 16: ICM_RAINBOW, 17: RND_RAINBOW, 18: NGU_RAINBOW,
    19: Agent-57,
    20: REDQ,   21: ICM_REDQ,    22: RND_REDQ,    23: NGU_REDQ,
    """

    env_switch = 4
    agent_switch = 9

    env_config, agent_config = env_agent_config(env_switch, agent_switch)

    # episode information
    avg_window = 5 # time window of moving average of socre ,mean reward data
    end_of_episode = 32 # clip the rl episode number

    # step_information
    episode_num = 19 # which step data of episode you wanna observe?
    end_of_step = 79 # clip the total step number

    # value inforation
    action_num = 5
    quantile_num = 51
    steel_shot_list = [3, 22, 38, 57]

    parent_path = str(os.path.abspath(''))

    if os.name == 'nt':
        data_save_path = parent_path + f"\\results\\{env_config['env_name']}\\{agent_config['agent_name']}_{agent_config['extension']['name']}_result\\"
        # Notice: you should change the time frame of data
        data_save_path = data_save_path + '2022-11-21_10-51-25\\'
    elif os.name == 'posix':
        data_save_path = parent_path + f"/results/{env_config['env_name']}/{agent_config['agent_name']}_{agent_config['extension']['name']}_result/"
        # Notice: you should change the time frame of data
        data_save_path = data_save_path + '2022-11-18_10-23-01/'

    episode_data, step_data = loading_data(data_save_path, episode_num)
    plot_highway(env_config['env_name'], agent_config['agent_name'], episode_data, step_data, avg_window, end_of_episode, end_of_step, action_num, quantile_num, steel_shot_list)
