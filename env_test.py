import gym
import numpy as np

def env_space_check():
    env = gym.make('LunarLander-v2')

    print(f'action_space: {env.action_space}')
    print(f'obs_space: {env.observation_space}')

    obs = env.reset()
    obs = np.array(obs)

    flatten_obs = obs.reshape(-1)

    print('origin_obs: {obs}')
    print('flatten_obs: {flatten_obs}')

if __name__ == '__main__':
    env_space_check()