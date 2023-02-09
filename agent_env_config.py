
def env_agent_config(env_switch: int, agent_switch: int, ext_switch: int):
    # Env
    if env_switch == 1:
        env_config = {'env_name': 'LunarLander-v2', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 20000}
    elif env_switch == 2: # Todo
        env_config = {'env_name': 'coin-run', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 501}
    elif env_switch == 3: # Todo
        env_config = {'env_name': 'highway-v0', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 501}
    elif env_switch == 4: # Todo
        env_config = {'env_name': 'custom_highway-v0', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 50000, 'render_mode': 'human'}
    else:
        raise ValueError('Please try to correct env_switch')

    # Agent algorithm - Auxiliary algorithm
    ## DQN
    if agent_switch == 1:
        if ext_switch == 1:
            from agent_config import DQN_Vanilla_agent_config
            agent_config = DQN_Vanilla_agent_config
        elif ext_switch == 2:
            from agent_config import DQN_ICM_agent_config
            agent_config = DQN_ICM_agent_config
        elif ext_switch == 3:
            from agent_config import DQN_RND_agent_config
            agent_config = DQN_RND_agent_config
        elif ext_switch == 4:
            from agent_config import DQN_NGU_agent_config
            agent_config = DQN_NGU_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for DQN")

    ## PPO
    elif agent_switch == 2:
        if ext_switch == 1:
            from agent_config import PPO_Vanilla_ageng_config
            agent_config = PPO_Vanilla_ageng_config
        elif ext_switch == 2:
            from agent_config import PPO_ICM_ageng_config
            agent_config = PPO_ICM_ageng_config
        elif ext_switch == 3:
            from agent_config import PPO_RND_ageng_config
            agent_config = PPO_RND_ageng_config
        elif ext_switch == 4:
            from agent_config import PPO_NGU_ageng_config
            agent_config = PPO_NGU_ageng_config
        elif ext_switch == 5:
            from agent_config import MEPPO_agent_config
            agent_config = MEPPO_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4, 5] for PPO")

    ## SAC
    elif agent_switch == 3:
        if ext_switch == 1:
            from agent_config import SAC_Vanilla_agent_config
            agent_config = SAC_Vanilla_agent_config
        elif ext_switch == 2:
            from agent_config import SAC_ICM_agent_config
            agent_config = SAC_ICM_agent_config
        elif ext_switch == 3:
            from agent_config import SAC_RND_agent_config
            agent_config = SAC_RND_agent_config
        elif ext_switch == 4:
            from agent_config import SAC_NGU_agent_config
            agent_config = SAC_NGU_agent_config
        elif ext_switch == 6:
            from agent_config import SAC_TQC_agent_config
            agent_config = SAC_TQC_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4, 6] for SAC")

    ## C51
    elif agent_switch == 4:
        if ext_switch == 1:
            from agent_config import C51_Vanilla_agent_config
            agent_config = C51_Vanilla_agent_config
        elif ext_switch == 2:
            from agent_config import C51_ICM_agent_config
            agent_config = C51_ICM_agent_config
        elif ext_switch == 3:
            from agent_config import C51_RND_agent_config
            agent_config = C51_RND_agent_config
        elif ext_switch == 4:
            from agent_config import C51_NGU_agent_config
            agent_config = C51_NGU_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for C51")

    ## QR-DQN
    elif agent_switch == 5:
        if ext_switch == 1:
            from agent_config import QR_DQN_Vanilla_agent_config
            agent_config = QR_DQN_Vanilla_agent_config
        elif ext_switch == 2:
            from agent_config import QR_DQN_ICM_agent_config
            agent_config = QR_DQN_ICM_agent_config
        elif ext_switch == 3:
            from agent_config import QR_DQN_RND_agent_config
            agent_config = QR_DQN_RND_agent_config
        elif ext_switch == 4:
            from agent_config import QR_DQN_NGU_agent_config
            agent_config = QR_DQN_NGU_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for QR-DQN")

    ## QUOTA
    elif agent_switch == 6:
        if ext_switch == 1:
            from agent_config import QUOTA_Vanilla_agent_config
            agent_config = QUOTA_Vanilla_agent_config
        elif ext_switch == 2:
            from agent_config import QUOTA_ICM_agent_config
            agent_config = QUOTA_ICM_agent_config
        elif ext_switch == 3:
            from agent_config import QUOTA_RND_agent_config
            agent_config = QUOTA_RND_agent_config
        elif ext_switch == 4:
            from agent_config import QUOTA_NGU_agent_config
            agent_config = QUOTA_NGU_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for QUOTA")

    ## IQN
    elif agent_switch == 7:
        if ext_switch == 1:
            from agent_config import IQN_Vanilla_agent_config
            agent_config = IQN_Vanilla_agent_config
        elif ext_switch == 2:
            from agent_config import IQN_ICM_agent_config
            agent_config = IQN_ICM_agent_config
        elif ext_switch == 3:
            from agent_config import IQN_RND_agent_config
            agent_config = IQN_RND_agent_config
        elif ext_switch == 4:
            from agent_config import IQN_NGU_agent_config
            agent_config = IQN_NGU_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for IQN")

    ## FQF
    elif agent_switch == 8:
        if ext_switch == 1:
            from agent_config import RAINBOW_DQN_Vanilla_agent_config
            agent_config = RAINBOW_DQN_Vanilla_agent_config

    # MMDQN
    elif agent_switch == 9:
        if ext_switch == 1:
            from agent_config import MMDQN_Vanilla_agent_config
            agent_config = MMDQN_Vanilla_agent_config
        elif ext_switch == 2:
            from agent_config import MMDQN_ICM_agent_config
            agent_config = MMDQN_ICM_agent_config
        elif ext_switch == 3:
            from agent_config import MMDQN_RND_agent_config
            agent_config = MMDQN_RND_agent_config
        elif ext_switch == 4:
            from agent_config import MMDQN_NGU_agent_config
            agent_config = MMDQN_NGU_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for MMDQN")
    ## C2D
    elif agent_switch == 10:
        if ext_switch == 1:
            from agent_config import C2D_Vanilla_agent_config
            agent_config = C2D_Vanilla_agent_config
        elif ext_switch == 2:
            from agent_config import C2D_ICM_agent_config
            agent_config = C2D_ICM_agent_config
        elif ext_switch == 3:
            from agent_config import C2D_RND_agent_config
            agent_config = C2D_RND_agent_config
        elif ext_switch == 4:
            from agent_config import C2D_NGU_agent_config
            agent_config = C2D_NGU_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for C2D")

    ## Agent-57
    elif agent_switch == 11:
        if ext_switch == 1:
            from agent_config import Agent57_agent_config
            agent_config = Agent57_agent_config
        else:
            raise ValueError("Please correct aux switch in [1] for Agent-57")

    ## REDQ
    elif agent_switch == 12:
        if ext_switch == 1:
            from agent_config import REDQ_Vanilla_agent_config
            agent_config = REDQ_Vanilla_agent_config
        elif ext_switch == 2:
            from agent_config import REDQ_ICM_agent_config
            agent_config = REDQ_ICM_agent_config
        elif ext_switch == 3:
            from agent_config import REDQ_RND_agent_config
            agent_config = REDQ_RND_agent_config
        elif ext_switch == 4:
            from agent_config import REDQ_NGU_agent_config
            agent_config = REDQ_NGU_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for REDQ")
    else:
        raise ValueError('Please try to correct agent_switch')
    
    return env_config, agent_config