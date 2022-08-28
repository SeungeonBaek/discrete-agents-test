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
        elif self.env_config['env_name'] == 'highway-fast-v0':
            import highway_env

            env = gym.make(self.env_config['env_name'])
            obs_space = env.observation_space.shape
            act_space = env.action_space.n
        elif self.env_config['env_name'] == 'custom_highway-fast-v0':
            import envs.highway_env

            env = gym.make(self.env_config['env_name'])
            obs_space = env.observation_space.shape
            act_space = env.action_space.n

        return env, obs_space, act_space


    def agent_loader(self):
        if self.agent_config['agent_name'] == 'DQN':
            from agents.DQN import Agent

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

        elif self.agent_config['agent_name'] == 'RAINBOW_DQN':
            from agents.RAINBOW_DQN import Agent

        elif self.agent_config['agent_name'] == 'Agent57':
            from agents.Agent_57 import Agent

        elif self.agent_config['agent_name'] == 'REDQ':
            from agents.REDQ import Agent

        else:
            raise ValueError('Please try to set the correct Agent')

        return Agent