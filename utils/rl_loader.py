import gym


class RLLoader():
    def __init__(self, env_config, agent_config):
        self.env_config = env_config
        self.agent_config = agent_config

    def env_loader(self):
        if self.env_config['env_name'] == 'LunarLander-v2':
            env = gym.make(self.env_config['env_name'])
            obs_space = env.observation_space.shape
            act_space = env.action_space.n
        elif self.env_config['env_name'] == 'coin-run':
            from procgen import ProcgenEnv
            env = ProcgenEnv(num_envs=1, env_name="coinrun")
            obs_space = env.observation_space['rgb'].shape
            act_space = env.action_space.n
        elif self.env_config['env_name'] == 'highway-env':
            env = gym.make('highway-env-v0')
            obs_space = env.observation_space.shape
            act_space = env.action_space.n
        elif self.env_config['env_name'] == 'domestic':
            env = gym.make('nota-its-v0')
            obs_space = env.observation_space.shape
            act_space = env.action_space.n

        return env, obs_space, act_space


    def agent_loader(self):
        if self.agent_config['agent_name'] == 'DQN':
            if self.agent_config['extension']['name'] == 'ICM':
                from agents.ICM_DQN import Agent
            elif self.agent_config['extension']['name'] == 'RND':
                from agents.RND_DQN import Agent
            elif self.agent_config['extension']['name'] == 'NGU':
                from agents.NGU_DQN import Agent
            else:
                from agents.DQN import Agent

        if self.agent_config['agent_name'] == 'DDQN':
            if self.agent_config['extension']['name'] == 'ICM':
                from agents.ICM_DDQN import Agent
            elif self.agent_config['extension']['name'] == 'RND':
                from agents.RND_DDQN import Agent
            elif self.agent_config['extension']['name'] == 'NGU':
                from agents.NGU_DDQN import Agent
            else:
                from agents.DDQN import Agent

        elif self.agent_config['agent_name'] == 'PPO':
            if self.agent_config['extension']['name'] == 'Model_Ensemble':
                from agents.MEPPO import Agent
            else:
                from agents.PPO import Agent

        elif self.agent_config['agent_name'] == 'SAC':
            if self.agent_config['extension']['name'] == 'TQC':
                from agents.TQC_SAC import Agent
            else:
                from agents.SAC import Agent

        elif self.agent_config['agent_name'] == 'QR_DQN':
            from agents.QR_DQN import Agent
        elif self.agent_config['agent_name'] == 'IQN':
            from agents.IQN import Agent
        elif self.agent_config['agent_name'] == 'QUOTA':
            from agents.QUOTA import Agent
        elif self.agent_config['agent_name'] == 'IDAC':
            from agents.IDAC import Agent

        if self.agent_config['agent_name'] == 'RAINBOW_DQN':
            if self.agent_config['extension']['name'] == 'ICM':
                from agents.ICM_RAINBOW_DQN import Agent
            elif self.agent_config['extension']['name'] == 'RND':
                from agents.RND_RAINBOW_DQN import Agent
            elif self.agent_config['extension']['name'] == 'NGU':
                from agents.NGU_RAINBOW_DQN import Agent
            else:
                from agents.RAINBOW_DQN import Agent

        elif self.agent_config['agent_name'] == 'Agent57':
            from agents.Agent_57 import Agent

        if self.agent_config['agent_name'] == 'REDQ':
            if self.agent_config['extension']['name'] == 'ICM':
                from agents.ICM_REDQ import Agent
            elif self.agent_config['extension']['name'] == 'RND':
                from agents.RND_REDQ import Agent
            elif self.agent_config['extension']['name'] == 'NGU':
                from agents.NGU_REDQ import Agent
            else:
                from agents.REDQ import Agent

        else:
            raise ValueError('Please try to set the correct Agent')

        return Agent