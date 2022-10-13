
# DQN
DQN_Vanilla_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'epsilon': 0.5, 'epsilon_decaying_rate': 0.9999, 'min_epsilon': 0.1, 'update_freq': 2, 'target_update_freq': 250,
                        'batch_size': 128, 'warm_up': 512, 'lr_critic': 0.0005, 'buffer_size': 100000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False, 'use_Huber': True}
DQN_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'use_DDQN':True}
DQN_Vanilla_agent_config['feature_extractor'] = {'use_GNN': False, 'use_GNN': False, 'use_MPGNN': False, 'use_GCN': False}

DQN_ICM_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'epsilon': 0.99, 'epsilon_decaying_rate': 0.9999, 'min_epsilon': 0.1, 'update_freq': 2, 'target_update_freq': 3,
                        'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False, 'use_Huber': True}
DQN_ICM_agent_config['extension'] = {'name': 'ICM', 'use_DDQN':True, 'icm_update_freq': 2, 'icm_lr': 0.001, 'icm_feature_dim': 128}
DQN_ICM_agent_config['feature_extractor'] = {'use_GNN': False, 'use_GNN': False, 'use_MPGNN': False, 'use_GCN': False}

DQN_RND_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'epsilon': 0.9, 'epsilon_decaying_rate': 0.99, 'min_epsilon': 0.05, 'update_freq': 4, 'target_update_freq': 250,
                        'batch_size': 128, 'warm_up': 512, 'lr_critic': 0.0005, 'buffer_size': 100000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False, 'use_Huber': True}
DQN_RND_agent_config['extension'] = {'name': 'RND', 'use_DDQN':True, 'rnd_update_freq': 5, 'rnd_lr': 0.0005}
DQN_RND_agent_config['feature_extractor'] = {'use_GNN': False, 'use_GNN': False, 'use_MPGNN': False, 'use_GCN': False}

DQN_NGU_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'epsilon': 0.99, 'epsilon_decaying_rate': 0.9999, 'min_epsilon': 0.1, 'update_freq': 2, 'target_update_freq': 3,
                        'batch_size': 128, 'warm_up': 1024, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False, 'use_Huber': True}
DQN_NGU_agent_config['extension'] = {'name': 'NGU', 'use_DDQN':True}
DQN_NGU_agent_config['feature_extractor'] = {'use_GNN': False, 'use_GNN': False, 'use_MPGNN': False, 'use_GCN': False}


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


# Distributional RL
QR_DQN_Vanilla_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'update_freq': 3, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
QR_DQN_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}

IQN_Vanilla_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'update_freq': 3, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
IQN_Vanilla_agent_config['extension'] = {'name': 'TQC', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}

QUOTA_Vanilla_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'update_freq': 3, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
QUOTA_Vanilla_agent_config['extension'] = {'name': 'gSDE', 'latent_space': 64, 'n_step_reset': 16}

IDAC_Vanilla_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'update_freq': 3, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
IDAC_Vanilla_agent_config['extension'] = {'name': 'gSDE', 'latent_space': 64, 'n_step_reset': 16}

# RAINBOW_DQN
RAINBOW_DQN_Vanilla_agent_config = {'agent_name': 'RAINBOW', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
RAINBOW_DQN_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}

RAINBOW_DQN_ICM_agent_config = {'agent_name': 'RAINBOW', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'batch_size': 128, 'warm_up': 1024, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
RAINBOW_DQN_ICM_agent_config['extension'] = {'name': 'TQC', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}

RAINBOW_DQN_RND_agent_config = {'agent_name': 'RAINBOW', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'batch_size': 128, 'warm_up': 1024, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
RAINBOW_DQN_RND_agent_config['extension'] = {'name': 'gSDE', 'latent_space': 64, 'n_step_reset': 16}

RAINBOW_DQN_NGU_agent_config = {'agent_name': 'RAINBOW', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'batch_size': 128, 'warm_up': 1024, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
RAINBOW_DQN_NGU_agent_config['extension'] = {'name': 'gSDE', 'latent_space': 64, 'n_step_reset': 16}

# Agent57
Agent57_Vanilla_agent_config = {'agent_name': 'Agent57', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'actor_update_freq': 2, 'batch_size': 128, 'warm_up': 1024, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
Agent57_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}


# REDQ
REDQ_Vanilla_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'update_freq': 3, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
REDQ_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}

REDQ_ICM_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'update_freq': 3, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
REDQ_ICM_agent_config['extension'] = {'name': 'TQC', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}

REDQ_RND_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'update_freq': 3, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
REDQ_RND_agent_config['extension'] = {'name': 'gSDE', 'latent_space': 64, 'n_step_reset': 16}

REDQ_NGU_agent_config = {'agent_name': 'DQN', 'gamma' : 0.99, 'update_freq': 3, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
REDQ_NGU_agent_config['extension'] = {'name': 'gSDE', 'latent_space': 64, 'n_step_reset': 16}