# Blank DQN
BLANK_DQN_Vanilla_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'tau': 0.005, 'epsilon': 0.99, 'epsilon_decaying_rate': 0.999995, 'min_epsilon': 0.01, 'update_freq': 4, 'target_update_freq': 20,
                        'batch_size': 128, 'warm_up': 512, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False, 'use_Huber': False, 'example': False}
BLANK_DQN_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'use_DDQN':False, 'use_Dueling': False}

# DQN
DQN_Vanilla_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'tau': 0.005, 'epsilon': 0.99, 'epsilon_decaying_rate': 0.999995, 'min_epsilon': 0.01, 'update_freq': 4, 'target_update_freq': 20,
                        'batch_size': 128, 'warm_up': 512, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False, 'use_Huber': True}
DQN_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'use_DDQN':True}

DQN_ICM_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'tau': 0.005, 'epsilon': 0.99, 'epsilon_decaying_rate': 0.999995, 'min_epsilon': 0.01, 'update_freq': 4, 'target_update_freq': 20,
                        'batch_size': 128, 'warm_up': 512, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False, 'use_Huber': True}
DQN_ICM_agent_config['extension'] = {'name': 'ICM', 'use_DDQN':True, 'icm_update_freq': 5, 'icm_lr': 0.0005, 'icm_feature_dim': 64}

DQN_RND_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'tau': 0.005, 'epsilon': 0.99, 'epsilon_decaying_rate': 0.999995, 'min_epsilon': 0.01, 'update_freq': 4, 'target_update_freq': 20,
                        'batch_size': 128, 'warm_up': 512, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False, 'use_Huber': True}
DQN_RND_agent_config['extension'] = {'name': 'RND', 'use_DDQN':True, 'rnd_update_freq': 5, 'rnd_lr': 0.0005}

DQN_NGU_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'tau': None, 'epsilon': 0.99, 'epsilon_decaying_rate': 0.9999, 'min_epsilon': 0.1, 'update_freq': 2, 'target_update_freq': 3,
                        'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False, 'use_Huber': True}
DQN_NGU_agent_config['extension'] = {'name': 'NGU', 'use_DDQN':True, 'ngu_lr': 0.0005}


# PPO
PPO_Vanilla_agent_config = {'agent_name': 'PPO', 'gamma' : 0.99, 'batch_size': 128, 'epoch_num': 4, 'entropy_coeff': 0.005, 'entropy_reduction_rate': 0.999999, 
		                    'epsilon': 0.2, 'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 512, 'use_GAE': True, 'lambda': 0.95, 'reward_normalize' : False}
PPO_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'std_bound': [0.02, 0.3]}

ME_PPO_agent_config = {'agent_name': 'PPO', 'gamma' : 0.99, 'update_freq': 2, 'batch_size': 128, 'epoch_num': 20, 'eps_clip': 0.2, 'eps_reduction_rate': 0.999999, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_GAE': True, 'lambda': 0.995, 'reward_normalize' : False}
ME_PPO_agent_config['extension'] = {'name': 'TQC', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}


# SAC
SAC_Vanilla_agent_config = {'agent_name': 'SAC', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'actor_update_freq': 2, 'batch_size': 128, 'warm_up': 1024, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
SAC_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}

SAC_TQC_agent_config = {'agent_name': 'SAC', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'actor_update_freq': 2, 'batch_size': 128, 'warm_up': 1024, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
SAC_TQC_agent_config['extension'] = {'name': 'TQC', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}


# QR_DQN
QR_DQN_Vanilla_agent_config = {'agent_name': 'QR_DQN', 'gamma' : 0.99, 'tau': 0.005, 'quantile_num': 51, 'epsilon': 0.9, 'epsilon_decaying_rate': 0.999995, 'min_epsilon': 0.01, 'update_freq': 4, 'target_update_freq': 20,
                        'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False}
QR_DQN_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'use_DDQN':False}

QR_DQN_ICM_agent_config = {'agent_name': 'QR_DQN', 'gamma' : 0.99, 'tau': None, 'quantile_num': 32, 'epsilon': 0.99, 'epsilon_decaying_rate': 0.9999, 'min_epsilon': 0.1, 'update_freq': 3, 'target_update_freq': 3,
                        'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
QR_DQN_ICM_agent_config['extension'] = {'name': 'ICM', 'use_DDQN':True, 'icm_update_freq': 2, 'icm_lr': 0.001, 'icm_feature_dim': 128}

QR_DQN_RND_agent_config = {'agent_name': 'QR_DQN', 'gamma' : 0.99, 'tau': 0.005, 'quantile_num': 51, 'epsilon': 0.9, 'epsilon_decaying_rate': 0.99999, 'min_epsilon': 0.01, 'update_freq': 4, 'target_update_freq': 20,
                        'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False}
QR_DQN_RND_agent_config['extension'] = {'name': 'RND', 'use_DDQN':False, 'rnd_update_freq': 5, 'rnd_lr': 0.0005}

QR_DQN_NGU_agent_config = {'agent_name': 'QR_DQN', 'gamma' : 0.99, 'tau': None, 'quantile_num': 32, 'epsilon': 0.99, 'epsilon_decaying_rate': 0.9999, 'min_epsilon': 0.1, 'update_freq': 3, 'target_update_freq': 3,
                        'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
QR_DQN_NGU_agent_config['extension'] = {'name': 'NGU', 'use_DDQN':True, 'ngu_lr': 0.0005}


# QUOTA
QUOTA_Vanilla_agent_config = {'agent_name': 'QUOTA', 'gamma' : 0.99, 'update_freq': 3, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
QUOTA_Vanilla_agent_config['extension'] = {'name': 'gSDE', 'latent_space': 64, 'n_step_reset': 16}


# IQN
IQN_Vanilla_agent_config = {'agent_name': 'IQN', 'gamma' : 0.99, 'tau': 0.005, 'quantile_dim': 128, 'quantile_num': 51, 'epsilon': 0.9, 'epsilon_decaying_rate': 0.999995, 'min_epsilon': 0.01, 'update_freq': 4, 'target_update_freq': 20,
                        'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False}
IQN_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'use_DDQN':False}

IQN_ICM_agent_config = {'agent_name': 'IQN', 'gamma' : 0.99, 'tau': None, 'quantile_dim': 128, 'quantile_num': 51, 'epsilon': 0.99, 'epsilon_decaying_rate': 0.9999, 'min_epsilon': 0.1, 'update_freq': 3, 'target_update_freq': 3,
                        'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
IQN_ICM_agent_config['extension'] = {'name': 'ICM', 'use_DDQN':True, 'icm_update_freq': 2, 'icm_lr': 0.001, 'icm_feature_dim': 128}

IQN_RND_agent_config = {'agent_name': 'IQN', 'gamma' : 0.99, 'tau': 0.005, 'quantile_dim': 128, 'quantile_num': 51, 'epsilon': 0.9, 'epsilon_decaying_rate': 0.99999, 'min_epsilon': 0.01, 'update_freq': 4, 'target_update_freq': 20,
                        'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False}
IQN_RND_agent_config['extension'] = {'name': 'RND', 'use_DDQN':False, 'rnd_update_freq': 5, 'rnd_lr': 0.0005}

IQN_NGU_agent_config = {'agent_name': 'IQN', 'gamma' : 0.99, 'tau': None, 'quantile_dim': 128, 'quantile_num': 51, 'epsilon': 0.99, 'epsilon_decaying_rate': 0.9999, 'min_epsilon': 0.1, 'update_freq': 3, 'target_update_freq': 3,
                        'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
IQN_NGU_agent_config['extension'] = {'name': 'NGU', 'use_DDQN':True, 'ngu_lr': 0.0005}


# FQF
FQF_Vanilla_agent_config = {'agent_name': 'FQF', 'gamma' : 0.99, 'update_freq': 3, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
FQF_Vanilla_agent_config['extension'] = {'name': 'gSDE', 'latent_space': 64, 'n_step_reset': 16}


# MMDQN
MMDQN_Vanilla_agent_config = {'agent_name': 'MMDQN', 'gamma' : 0.99, 'tau': 0.005, 'particle_num': 200, 'kernel_option': 0, 'kernel_parameter': 1, 'epsilon': 0.99, 'epsilon_decaying_rate': 0.999995, 'min_epsilon': 0.01,
                              'update_freq': 4, 'target_update_freq': 20, 'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False}
MMDQN_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'use_DDQN':False}

MMDQN_ICM_agent_config = {'agent_name': 'MMDQN', 'gamma' : 0.99, 'tau': 0.005, 'particle_num': 200, 'kernel_option': 0, 'kernel_parameter': 0, 'epsilon': 0.9, 'epsilon_decaying_rate': 0.999995, 'min_epsilon': 0.01,
                              'update_freq': 4, 'target_update_freq': 20, 'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False}
MMDQN_ICM_agent_config['extension'] = {'name': 'ICM', 'use_DDQN':True, 'icm_update_freq': 2, 'icm_lr': 0.001, 'icm_feature_dim': 128}

MMDQN_RND_agent_config = {'agent_name': 'MMDQN', 'gamma' : 0.99, 'tau': 0.005, 'particle_num': 200, 'kernel_option': 0, 'kernel_parameter': 0, 'epsilon': 0.9, 'epsilon_decaying_rate': 0.999995, 'min_epsilon': 0.01,
                              'update_freq': 4, 'target_update_freq': 20, 'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False}
MMDQN_RND_agent_config['extension'] = {'name': 'RND', 'use_DDQN':False, 'rnd_update_freq': 5, 'rnd_lr': 0.0005}

MMDQN_NGU_agent_config = {'agent_name': 'MMDQN', 'gamma' : 0.99, 'tau': 0.005, 'particle_num': 200, 'kernel_option': 0, 'kernel_parameter': 0, 'epsilon': 0.9, 'epsilon_decaying_rate': 0.999995, 'min_epsilon': 0.01,
                              'update_freq': 4, 'target_update_freq': 20, 'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False}
MMDQN_NGU_agent_config['extension'] = {'name': 'NGU', 'use_DDQN':True, 'ngu_lr': 0.0005}


# C2D
C2D_Vanilla_agent_config = {'agent_name': 'C2D', 'gamma' : 0.99, 'update_freq': 3, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
C2D_Vanilla_agent_config['extension'] = {'name': 'gSDE', 'latent_space': 64, 'n_step_reset': 16}


# Agent57
Agent57_Vanilla_agent_config = {'agent_name': 'Agent57', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'actor_update_freq': 2, 'batch_size': 128, 'warm_up': 1024, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
Agent57_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}


# REDQ
REDQ_Vanilla_agent_config = {'agent_name': 'REDQ', 'gamma' : 0.99, 'update_freq': 3, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
REDQ_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}

REDQ_ICM_agent_config = {'agent_name': 'REDQ', 'gamma' : 0.99, 'update_freq': 3, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
REDQ_ICM_agent_config['extension'] = {'name': 'TQC', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}

REDQ_RND_agent_config = {'agent_name': 'REDQ', 'gamma' : 0.99, 'update_freq': 3, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
REDQ_RND_agent_config['extension'] = {'name': 'gSDE', 'latent_space': 64, 'n_step_reset': 16}

REDQ_NGU_agent_config = {'agent_name': 'REDQ', 'gamma' : 0.99, 'update_freq': 3, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
REDQ_NGU_agent_config['extension'] = {'name': 'gSDE', 'latent_space': 64, 'n_step_reset': 16}


######### Safe RL #########
# Safe_QR_DQN
Safe_QR_DQN_Vanilla_agent_config = {'agent_name': 'Safe_QR_DQN', 'gamma' : 0.99, 'tau': 0.005, 'quantile_num': 51, 'epsilon': 0.9, 'epsilon_decaying_rate': 0.99999, 'min_epsilon': 0.01, 'update_freq': 4, 'target_update_freq': 10,
                        'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False}
Safe_QR_DQN_Vanilla_agent_config['extension'] = {'name': 'Safe_Vanilla', 'use_DDQN':True, 'safe_option': 8}

Safe_QR_DQN_RND_agent_config = {'agent_name': 'Safe_QR_DQN', 'gamma' : 0.99, 'tau': 0.005, 'quantile_num': 51, 'epsilon': 0.9, 'epsilon_decaying_rate': 0.99999, 'min_epsilon': 0.01, 'update_freq': 4, 'target_update_freq': 10,
                        'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False}
Safe_QR_DQN_RND_agent_config['extension'] = {'name': 'RND', 'use_DDQN':True, 'rnd_update_freq': 5, 'rnd_lr': 0.0005, 'safe_option': 8}


# Safe_IQN
Safe_IQN_Vanilla_agent_config = {'agent_name': 'Safe_IQN', 'gamma' : 0.99, 'tau': 0.005, 'quantile_dim': 128, 'quantile_num': 51, 'epsilon': 0.9, 'epsilon_decaying_rate': 0.999995, 'min_epsilon': 0.01, 'update_freq': 4, 'target_update_freq': 20,
                        'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False}
Safe_IQN_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'use_DDQN':False, 'safe_option': 8}

Safe_IQN_RND_agent_config = {'agent_name': 'Safe_IQN', 'gamma' : 0.99, 'tau': 0.005, 'quantile_dim': 128, 'quantile_num': 51, 'epsilon': 0.9, 'epsilon_decaying_rate': 0.99999, 'min_epsilon': 0.01, 'update_freq': 4, 'target_update_freq': 20,
                        'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False}
Safe_IQN_RND_agent_config['extension'] = {'name': 'RND', 'use_DDQN':False, 'rnd_update_freq': 5, 'rnd_lr': 0.0005, 'safe_option': 8}


# Safe_MMDQN
Safe_MMDQN_Vanilla_agent_config = {'agent_name': 'Safe_MMDQN', 'gamma' : 0.99, 'tau': 0.005, 'particle_num': 200, 'kernel_option': 0, 'kernel_parameter': 0, 'epsilon': 0.9, 'epsilon_decaying_rate': 0.999995, 'min_epsilon': 0.01,
                              'update_freq': 4, 'target_update_freq': 20, 'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False}
Safe_MMDQN_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'use_DDQN':False, 'safe_option': 8}

Safe_MMDQN_RND_agent_config = {'agent_name': 'Safe_MMDQN', 'gamma' : 0.99, 'tau': 0.005, 'particle_num': 200, 'kernel_option': 0, 'kernel_parameter': 0, 'epsilon': 0.9, 'epsilon_decaying_rate': 0.999995, 'min_epsilon': 0.01,
                              'update_freq': 4, 'target_update_freq': 20, 'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.0005, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False}
Safe_MMDQN_RND_agent_config['extension'] = {'name': 'RND', 'use_DDQN':False, 'rnd_update_freq': 5, 'rnd_lr': 0.0005, 'safe_option': 8}

