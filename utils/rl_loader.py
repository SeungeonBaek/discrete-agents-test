import gym


class RLLoader():
    def __init__(self, env_config, agent_config):
        self.env_config = env_config
        self.agent_config = agent_config

    def env_loader(self):
        if self.env_config['env_name'] == 'LunarLander-v2':
            if self.env_config['render'] == True:
                env = gym.make(self.env_config['env_name'], render_mode='human')
            else:
                try:
                    env = gym.make(self.env_config['env_name'], render_mode=None)
                except:
                    env = gym.make(self.env_config['env_name'])
            obs_space = env.observation_space.shape
            act_space = env.action_space.n
        elif self.env_config['env_name'] == 'coin-run':
            from procgen import ProcgenEnv
            env = ProcgenEnv(num_envs=1, env_name="coinrun")
            obs_space = env.observation_space['rgb'].shape
            act_space = env.action_space.n
        elif self.env_config['env_name'] == 'highway-v0':
            import highway_env

            env = gym.make(self.env_config['env_name'])
            obs_space = env.observation_space.shape
            act_space = env.action_space.n
        elif self.env_config['env_name'] == 'custom_highway-v0':
            import envs.custom_highway_env

            # env = gym.make(self.env_config['env_name'].split('_')[1])
            env = gym.make(self.env_config['env_name'].replace("_", "-"))
            env.config['observation']['type'] = "Kinematics"
            env.config['observation']['vehicles_count'] =5
            env.config['observation']['features'] = ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
            env.config['observation']['features_range'] = {
                                                            "x": [-100, 100],
                                                            "y": [-100, 100],
                                                            "vx": [-20, 20],
                                                            "vy": [-20, 20]
                                                        }
            env.config['observation']['absolute'] = True
            env.config['observation']['normalize'] = True
            env.config['observation']['order'] = "sorted"
            env.config['action'] = { "type": "DiscreteMetaAction",}
            env.config['lanes_count'] = 3
            env.config['duration'] = 80
            env.config['initial_spacing'] = 2
            env.config['simulation_frequency'] = 15
            env.config['policy_frequency'] = 1
            env.config['other_vehicles_type'] = "highway_env.vehicle.behavior.IDMVehicle"
            env.config['ego_vehicle_spd'] = 1 # default
            env.config['other_vehicle_spd'] = 1 # default
            env.config['vehicles_count'] = 10
            env.config['vehicles_density']= 1
            env.config['collision_reward'] = -1
            env.config['reward_speed_range'] = [0,25]
            env.config['scaling'] = 5.5
            env.config['offscreen_rendering']= False
            env.config['ego_spacing']= 2
            env.config['right_lane_reward'] = 0

            _ = env.reset()
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

        elif self.agent_config['agent_name'] == 'RAINBOW_DQN':
            from agents.RAINBOW_DQN import Agent

        elif self.agent_config['agent_name'] == 'Agent57':
            from agents.Agent_57 import Agent

        elif self.agent_config['agent_name'] == 'REDQ':
            from agents.REDQ import Agent

        else:
            raise ValueError('Please try to set the correct Agent')

        return Agent