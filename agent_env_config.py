from typing import Dict
from agent_config import *
from extractor_config import *


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
            agent_config = DQN_Vanilla_agent_config

        elif ext_switch == 2:
            agent_config = DQN_ICM_agent_config

        elif ext_switch == 3:
            agent_config = DQN_RND_agent_config

        elif ext_switch == 4:
            agent_config = DQN_NGU_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for DQN")

    ## PPO
    elif agent_switch == 2:
        if isinstance(ext_switch, list) == True:
            if (1 in ext_switch) and (5 in ext_switch):
                agent_config = PPO_Vanilla_ageng_config

            elif (2 in ext_switch) and (5 in ext_switch):
                agent_config = PPO_ICM_ageng_config

            elif (3 in ext_switch) and (5 in ext_switch):
                agent_config = PPO_RND_ageng_config

            elif (4 in ext_switch) and (5 in ext_switch):
                agent_config = PPO_NGU_ageng_config
            else:
                raise ValueError("Please correct aux switch in [(1,5), (2,5), (3,5), (4,5)] for SAC")

            agent_config['extension']['name'] += '_ModelEnsemble'
            agent_config['extension']['use_ModelEnsemble'] = True

        else:
            if ext_switch == 1:
                agent_config = PPO_Vanilla_ageng_config

            elif ext_switch == 2:
                agent_config = PPO_ICM_ageng_config

            elif ext_switch == 3:
                agent_config = PPO_RND_ageng_config

            elif ext_switch == 4:
                agent_config = PPO_NGU_ageng_config

            elif ext_switch == 5:
                agent_config = PPO_Vanilla_ageng_config
                agent_config['extension']['name'] = 'ModelEnsemble'
                agent_config['extension']['use_ModelEnsemble'] = True
            else:
                raise ValueError("Please correct aux switch in [1, 2, 3, 4, 5] for PPO")

    ## SAC
    elif agent_switch == 3:
        if isinstance(ext_switch, list) == True:
            if (1 in ext_switch) and (6 in ext_switch):
                agent_config = SAC_Vanilla_agent_config

            elif (2 in ext_switch) and (6 in ext_switch):
                agent_config = SAC_ICM_agent_config

            elif (3 in ext_switch) and (6 in ext_switch):
                agent_config = SAC_RND_agent_config

            elif (4 in ext_switch) and (6 in ext_switch):
                agent_config = SAC_NGU_agent_config
            else:
                raise ValueError("Please correct aux switch in [(1,6), (2,6), (3,6), (4,6)] for SAC")

            agent_config['extension']['name'] += '_TQC'
            agent_config['extension']['use_TQC'] = True

        else:
            if ext_switch == 1:
                agent_config = SAC_Vanilla_agent_config

            elif ext_switch == 2:
                agent_config = SAC_ICM_agent_config

            elif ext_switch == 3:
                agent_config = SAC_RND_agent_config

            elif ext_switch == 4:
                agent_config = SAC_NGU_agent_config

            elif ext_switch == 6:
                agent_config = SAC_Vanilla_agent_config
                agent_config['extension']['name'] = 'TQC'
                agent_config['extension']['use_TQC'] = True
            else:
                raise ValueError("Please correct aux switch in [1, 2, 3, 4, 6] for SAC")

    ## C51
    elif agent_switch == 4:
        if ext_switch == 1:
            agent_config = C51_Vanilla_agent_config

        elif ext_switch == 2:
            agent_config = C51_ICM_agent_config

        elif ext_switch == 3:
            agent_config = C51_RND_agent_config

        elif ext_switch == 4:
            agent_config = C51_NGU_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for C51")

    ## QR-DQN
    elif agent_switch == 5:
        if ext_switch == 1:
            agent_config = QR_DQN_Vanilla_agent_config
            
        elif ext_switch == 2:
            agent_config = QR_DQN_ICM_agent_config

        elif ext_switch == 3:
            agent_config = QR_DQN_RND_agent_config

        elif ext_switch == 4:
            agent_config = QR_DQN_NGU_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for QR-DQN")

    ## QUOTA
    elif agent_switch == 6:
        if ext_switch == 1:
            agent_config = QUOTA_Vanilla_agent_config

        elif ext_switch == 2:
            agent_config = QUOTA_ICM_agent_config

        elif ext_switch == 3:
            agent_config = QUOTA_RND_agent_config

        elif ext_switch == 4:
            agent_config = QUOTA_NGU_agent_config

        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for QUOTA")

    ## IQN
    elif agent_switch == 7:
        if ext_switch == 1:
            agent_config = IQN_Vanilla_agent_config

        elif ext_switch == 2:
            agent_config = IQN_ICM_agent_config

        elif ext_switch == 3:
            agent_config = IQN_RND_agent_config

        elif ext_switch == 4:
            agent_config = IQN_NGU_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for IQN")

    ## FQF
    elif agent_switch == 8:
        if ext_switch == 1:
            agent_config = FQF_Vanilla_agent_config

        elif ext_switch == 2:
            agent_config = FQF_Vanilla_agent_config
            
        elif ext_switch == 3:
            agent_config = FQF_Vanilla_agent_config

        elif ext_switch == 4:
            agent_config = FQF_Vanilla_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for FQF")

    # MMDQN
    elif agent_switch == 9:
        if ext_switch == 1:
            agent_config = MMDQN_Vanilla_agent_config

        elif ext_switch == 2:
            agent_config = MMDQN_ICM_agent_config

        elif ext_switch == 3:
            agent_config = MMDQN_RND_agent_config

        elif ext_switch == 4:
            agent_config = MMDQN_NGU_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for MMDQN")

    ## C2D
    elif agent_switch == 10:
        if ext_switch == 1:
            agent_config = C2D_Vanilla_agent_config

        elif ext_switch == 2:
            agent_config = C2D_ICM_agent_config

        elif ext_switch == 3:
            agent_config = C2D_RND_agent_config

        elif ext_switch == 4:
            agent_config = C2D_NGU_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for C2D")

    ## Agent57
    elif agent_switch == 11:
        if ext_switch == 1:
            agent_config = Agent57_Vanilla_agent_config
        else:
            raise ValueError("Agent57 should use switch 1")

    ## REDQ
    elif agent_switch == 12:
        if ext_switch == 1:
            agent_config = REDQ_Vanilla_agent_config

        elif ext_switch == 2:
            agent_config = REDQ_ICM_agent_config

        elif ext_switch == 3:
            agent_config = REDQ_RND_agent_config

        elif ext_switch == 4:
            agent_config = REDQ_NGU_agent_config
        else:
            raise ValueError("Please correct aux switch in [1, 2, 3, 4] for REDQ")
    else:
        raise ValueError('Please try to correct agent_switch')
    
    return env_config, agent_config


def agent_network_config(agent_config:Dict, extractor_switch: int, fcn_config:Dict):
    if extractor_switch == 1:
        pass
    else:
        if extractor_switch == 2: # MLP
            extractor_config = MLP_flatten_extractor_config
            feature_dim = MLP_flatten_feature_dim

        elif extractor_switch == 3: # Convolutional
            extractor_config = Convolutional_CNN1d_extractor_config
            feature_dim = Convolutional_CNN1d_feature_dim

        elif extractor_switch == 4: # Recurrent
            extractor_config = REC_lstm_extractor_config
            feature_dim = REC_lstm_feature_dim

        elif extractor_switch == 5: # Attention
            extractor_config = Attention_attention_extractor_config
            feature_dim = Attention_attention_feature_dim

        elif extractor_switch == 6: # Graph
            extractor_config = Graph_GCN_extractor_config
            feature_dim = Graph_GCN_feature_dim

        elif extractor_switch == 7: # Custom
            extractor_config = Custom_simple_gru_extractor_config
            feature_dim = Custom_simple_gru_feature_dim
        else:
            raise ValueError("Please correct the extractor switch in [1, 2, 3, 4, 5, 6, 7]")

        agent_config['is_configurable_critic'] = True

        agent_config['critic_config'] = {}
        agent_config['critic_config']['name'] = extractor_config['name'] + '_critic'

        agent_config['critic_config']['network_config'] = {}
        agent_config['critic_config']['network_config']['feature_extractor_config'] = extractor_config
        agent_config['critic_config']['network_config']['feature_extractor_config']['feature_dim'] = feature_dim
        agent_config['critic_config']['network_config']['fcn_config'] = fcn_config

    return agent_config