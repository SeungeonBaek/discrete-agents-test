
def env_agent_config(env_switch, agent_switch):
    # Env
    if env_switch == 1:
        env_config = {'env_name': 'LunarLander-v2', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 20000}
    elif env_switch == 2: # Todo
        env_config = {'env_name': 'coin-run', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 501}
    elif env_switch == 3: # Todo
        env_config = {'env_name': 'highway-v0', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 501}
    elif env_switch == 4: # Todo
        env_config = {'env_name': 'custom_highway-v0', 'seed': 777, 'render': True, 'max_step': 1000, 'max_episode': 50000, 'render_mode': 'human'}
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

    # QR-DQN
    elif agent_switch == 9:
        from agent_config import QR_DQN_Vanilla_agent_config
        agent_config = QR_DQN_Vanilla_agent_config
    elif agent_switch == 10:
        from agent_config import QR_DQN_ICM_agent_config
        agent_config = QR_DQN_ICM_agent_config
    elif agent_switch == 11:
        from agent_config import QR_DQN_RND_agent_config
        agent_config = QR_DQN_RND_agent_config
    elif agent_switch == 12:
        from agent_config import QR_DQN_NGU_agent_config
        agent_config = QR_DQN_NGU_agent_config

    # IQN
    elif agent_switch == 13:
        from agent_config import IQN_Vanilla_agent_config
        agent_config = IQN_Vanilla_agent_config

    # QUOTA
    elif agent_switch == 14:
        from agent_config import QUOTA_Vanilla_agent_config
        agent_config = QUOTA_Vanilla_agent_config

    # RAINBOW DQN
    elif agent_switch == 15:
        from agent_config import RAINBOW_DQN_Vanilla_agent_config
        agent_config = RAINBOW_DQN_Vanilla_agent_config
    elif agent_switch == 16:
        from agent_config import RAINBOW_DQN_ICM_agent_config
        agent_config = RAINBOW_DQN_ICM_agent_config
    elif agent_switch == 17:
        from agent_config import RAINBOW_DQN_RND_agent_config
        agent_config = RAINBOW_DQN_RND_agent_config
    elif agent_switch == 18:
        from agent_config import RAINBOW_DQN_NGU_agent_config
        agent_config = RAINBOW_DQN_NGU_agent_config

    # Agent-57
    elif agent_switch == 19:
        from agent_config import Agent57_agent_config
        agent_config = Agent57_agent_config

    # REDQ
    elif agent_switch == 20:
        from agent_config import REDQ_Vanilla_agent_config
        agent_config = REDQ_Vanilla_agent_config
    elif agent_switch == 21:
        from agent_config import REDQ_ICM_agent_config
        agent_config = REDQ_ICM_agent_config
    elif agent_switch == 22:
        from agent_config import REDQ_RND_agent_config
        agent_config = REDQ_RND_agent_config
    elif agent_switch == 23:
        from agent_config import REDQ_NGU_agent_config
        agent_config = REDQ_NGU_agent_config

    else:
        raise ValueError('Please try to correct agent_switch')
    
    return env_config, agent_config