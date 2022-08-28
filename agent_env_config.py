
def env_agent_config(env_switch, agent_switch):
    # Env
    if env_switch == 1:
        env_config = {'env_name': 'LunarLander-v2', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 3000}
    elif env_switch == 2: # Todo
        env_config = {'env_name': 'coin-run', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 501}
    elif env_switch == 3: # Todo
        env_config = {'env_name': 'highway-v0', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 501}
    elif env_switch == 4: # Todo
        env_config = {'env_name': 'custom_highway-v0', 'seed': 777, 'render': True, 'max_step': 1000, 'max_episode': 501, 'render_mode': 'human'}
        env_config['config'] = {"observation": { "type": "Kinematics"},
                                "action": { "type": "DiscreteMetaAction",},
                                "lanes_count": 5,
                                "vehicles_count": 50,
                                "duration": 40,  # [s]
                                "initial_spacing": 2,
                                "simulation_frequency": 15,  # [Hz]
                                "policy_frequency": 1,  # [Hz]
                                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                                "screen_width": 600,  # [px]
                                "screen_height": 150,  # [px]
                                "centering_position": [0.3, 0.5],
                                "scaling": 5.5,
                                "show_trajectories": True,
                                "render_agent": True,
                                "offscreen_rendering": False,
                                "vehicles_density": 1,
                                "collision_reward": -1,    # The reward received when colliding with a vehicle.
                                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                                        # zero for other lanes.
                                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                                        # lower speeds according to config["reward_speed_range"].
                                "lane_change_reward": 0,   # The reward received at each lane change action.
                                "reward_speed_range": [20, 30],
                                "normalize_reward": True,
                                "offroad_terminal": False
                                }
    else:
        raise ValueError('Please try to correct env_switch')

    # DQN
    if agent_switch == 1:
        from agent_config import DQN_Vanilla_agent_config
        agent_config = DQN_Vanilla_agent_config
    elif agent_switch == 2:
        from agent_config import DQN_ICM_agent_config
        agent_config = DQN_ICM_agent_config
    elif agent_switch == 3:
        from agent_config import DQN_RND_agent_config
        agent_config = DQN_RND_agent_config
    elif agent_switch == 4:
        from agent_config import DQN_NGU_agent_config
        agent_config = DQN_NGU_agent_config

    # PPO
    elif agent_switch == 5:
        from agent_config import PPO_Vanilla_ageng_config
        agent_config = PPO_Vanilla_ageng_config
    elif agent_switch == 6:
        from agent_config import MEPPO_agent_config
        agent_config = MEPPO_agent_config

    # SAC
    elif agent_switch == 7:
        from agent_config import SAC_Vanilla_agent_config
        agent_config = SAC_Vanilla_agent_config
    elif agent_switch == 8:
        from agent_config import SAC_TQC_Vanilla_agent_config
        agent_config = SAC_TQC_Vanilla_agent_config

    # Distributional RL
    elif agent_switch == 9:
        from agent_config import QR_DQN_Vanilla_agent_config
        agent_config = QR_DQN_Vanilla_agent_config
    elif agent_switch == 10:
        from agent_config import IQN_Vanilla_agent_config
        agent_config = IQN_Vanilla_agent_config
    elif agent_switch == 11:
        from agent_config import QUOTA_Vanilla_agent_config
        agent_config = QUOTA_Vanilla_agent_config
    elif agent_switch == 12:
        from agent_config import IDAC_Vanilla_agent_config
        agent_config = IDAC_Vanilla_agent_config

    # RAINBOW DQN
    elif agent_switch == 13:
        from agent_config import RAINBOW_DQN_Vanilla_agent_config
        agent_config = RAINBOW_DQN_Vanilla_agent_config
    elif agent_switch == 14:
        from agent_config import RAINBOW_DQN_ICM_agent_config
        agent_config = RAINBOW_DQN_ICM_agent_config
    elif agent_switch == 15:
        from agent_config import RAINBOW_DQN_RND_agent_config
        agent_config = RAINBOW_DQN_RND_agent_config
    elif agent_switch == 16:
        from agent_config import RAINBOW_DQN_NGU_agent_config
        agent_config = RAINBOW_DQN_NGU_agent_config

    # Agent-57
    elif agent_switch == 17:
        from agent_config import Agent57_agent_config
        agent_config = Agent57_agent_config

    # REDQ
    elif agent_switch == 18:
        from agent_config import REDQ_Vanilla_agent_config
        agent_config = REDQ_Vanilla_agent_config
    elif agent_switch == 19:
        from agent_config import REDQ_ICM_agent_config
        agent_config = REDQ_ICM_agent_config
    elif agent_switch == 20:
        from agent_config import REDQ_RND_agent_config
        agent_config = REDQ_RND_agent_config
    elif agent_switch == 21:
        from agent_config import REDQ_NGU_agent_config
        agent_config = REDQ_NGU_agent_config

    else:
        raise ValueError('Please try to correct agent_switch')
    
    return env_config, agent_config