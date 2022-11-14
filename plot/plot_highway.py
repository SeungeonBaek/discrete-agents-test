import os, sys
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def loading_data(data_save_path: str, episode_num: int)-> Tuple[pd.DataFrame]:
    episode_data = pd.read_csv(data_save_path + "episode_data.csv", index_col=0)
    step_data = pd.read_csv(data_save_path + f"\\step_data\\episode_{episode_num}_data.csv", index_col=0)

    return episode_data, step_data


def plot_highway(env_name: str, agent_name: str, episode_data: pd.DataFrame, step_data: pd.DataFrame, avg_num: int, episode_num: int, step_num: int)-> None:
    sns.set_theme(style = "darkgrid")

    spec_1 = [['Score', 'MeanReward'],
              ['Pos_xy', 'Pos_xy'   ],
              ['Pos_xy', 'Pos_xy'   ],
              ['Pos_x', 'Pos_y'     ],
              ['Vel_x', 'Vel_y'     ]]

    spec_2 = [['THW', 'TTCi'        ],
              ['THW_vs_TTCi', 'LC'  ]]

    fig_1, axes_1 = plt.subplot_mosaic(spec_1, figsize=(12, 6))
    fig_1.suptitle(f'{env_name}, {agent_name}')

    fig_2, axes_2 = plt.subplot_mosaic(spec_2, figsize=(12, 6))
    fig_2.suptitle(f'{env_name}, {agent_name}')

    palette = sns.cubehelix_palette(light=.5, n_colors=30, gamma=0.5)
    # palette1 = sns.color_palette("mako_r", 6)
    colors = [[0.2, 0.2, 0.8], [0.8, 0.5, 0.2], [0.3, 0.8, 0.2]]

    # Episode data
    sns.lineplot(ax=axes_1['Score'], x=range(len(episode_data['episode_score'][:episode_num])), y=episode_data['episode_score'][:episode_num].rolling(window=avg_num).mean(), palette = palette)
    # axes_1['Score'].set_ylim((-400, 400))

    sns.lineplot(ax=axes_1['MeanReward'], x=range(len(episode_data['mean_reward'][:episode_num])), y=episode_data['mean_reward'][:episode_num].rolling(window=avg_num).mean(), palette = palette)
    # axes_1['MeanReward'].set_ylim((-100, -10))

    # Step data - 1: line plot
    sns.lineplot(ax=axes_1['Pos_xy'], x=step_data['position_x'][:step_num], y=step_data['position_y'][:step_num], palette = palette)
    # axes_1['MeanReward'].set_ylim((-100, -10))

    sns.lineplot(ax=axes_1['Pos_x'], x=range(len(step_data['position_x'][:step_num])), y=step_data['position_x'][:step_num].rolling(window=avg_num).mean(), palette = palette)
    # axes_1['MeanReward'].set_ylim((-100, -10))

    sns.lineplot(ax=axes_1['Pos_y'], x=range(len(step_data['position_y'][:step_num])), y=step_data['position_y'][:step_num].rolling(window=avg_num).mean(), palette = palette)
    # axes_1['MeanReward'].set_ylim((-100, -10))

    sns.lineplot(ax=axes_1['Vel_x'], x=range(len(step_data['velocity_x'][:step_num])), y=step_data['velocity_x'][:step_num].rolling(window=avg_num).mean(), palette = palette)
    # axes_1['MeanReward'].set_ylim((-100, -10))

    sns.lineplot(ax=axes_1['Vel_y'], x=range(len(step_data['velocity_y'][:step_num])), y=step_data['velocity_y'][:step_num].rolling(window=avg_num).mean(), palette = palette)
    # axes_1['MeanReward'].set_ylim((-100, -10))

    # Step data - 2: histogram
    sns.histplot(ax=axes_2['THW'], data=step_data['time_headway'][:step_num], stat="count", color=colors[0], kde=True)
    # axes_2['MeanReward'].set_xlim((-100, -10))
    # axes_2['MeanReward'].set_ylim((-100, -10))

    sns.histplot(ax=axes_2['TTCi'], data=step_data['inverse_of_ttc'][:step_num], stat="count", color=colors[0], kde=True)
    # axes_2['MeanReward'].set_xlim((-100, -10))
    # axes_2['MeanReward'].set_ylim((-100, -10))

    sns.scatterplot(ax=axes_2['THW_vs_TTCi'], x=step_data['time_headway'][:step_num], y=step_data['inverse_of_ttc'][:step_num], s=5, color=".15")
    sns.histplot(ax=axes_2['THW_vs_TTCi'], x=step_data['time_headway'][:step_num], y=step_data['inverse_of_ttc'][:step_num], bins=50, pthresh=.1, cmap="mako")
    sns.kdeplot(ax=axes_2['THW_vs_TTCi'], x=step_data['time_headway'][:step_num], y=step_data['inverse_of_ttc'][:step_num], levels=5, color="w", linewidths=1)
    # axes_2['MeanReward'].set_xlim((-100, -10))
    # axes_2['MeanReward'].set_ylim((-100, -10))

    sns.histplot(ax=axes_2['LC'], data=step_data['lane_change_flag'][:step_num], stat="count", color=colors[0], kde=True)
    # axes_2['MeanReward'].set_xlim((-100, -10))
    # axes_2['MeanReward'].set_ylim((-100, -10))

    # Legend
    # axes_2[0].legend([])

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
    agent_switch = 1

    env_config, agent_config = env_agent_config(env_switch, agent_switch)

    episode_num = 19 # which step data of episode you wanna observe?
    avg_window = 5 # time window of moving average of socre ,mean reward data
    end_of_episode = 20 # clip the rl episode number
    end_of_step = 79 # clip the total step number

    parent_path = str(os.path.abspath(''))

    if os.name == 'nt':
        data_save_path = parent_path + f"\\results\\{env_config['env_name']}\\{agent_config['agent_name']}_{agent_config['extension']['name']}_result\\"
    elif os.name == 'linux':
        data_save_path = parent_path + f"/results/{env_config['env_name']}/{agent_config['agent_name']}_{agent_config['extension']['name']}_result/"

    # Notice: you should change the time frame of data
    data_save_path = data_save_path + '2022-11-13_00-08-05\\'

    episode_data, step_data = loading_data(data_save_path, episode_num)
    plot_highway(env_config['env_name'], agent_config['agent_name'], episode_data, step_data, avg_window, end_of_episode, end_of_step)
