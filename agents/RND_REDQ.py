import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense


class Critic(Model):
    def __init__(self, obs_size):
        super(Critic,self).__init__()
        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.005)
        
        self.l1 = Dense(64, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2 = Dense(128, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l3 = Dense(64, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l4 = Dense(32, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.value = Dense(obs_size, activation = 'softmax')

    def call(self, state_action):
        l1 = self.l1(state_action) # 확인
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        l4 = self.l4(l3)
        value = self.value(l4)

        return value


class Agent:
    """
    input argument: hyper_param, obs_shape_n, act_shape_n

    hyper_param: lr_critic_main, gamma, tau, update_freq, batch_size, warm_up, gaussian_std, noise_reduce_rate
    """
    def __init__(self, name, hyper_param, obs_shape_n, act_shape_n):
        self.name = name
        
        self.critic_lr_main = hyper_param['lr_critic_main']

        self.gamma = hyper_param['gamma']
        self.tau = hyper_param['tau']
        self.update_freq = hyper_param['update_freq']

        self.replay_buffer = Memory.ExperienceMemory(2000000)
        self.batch_size = hyper_param['batch_size']
        self.warm_up = hyper_param['warm_up']
        self.update_step = 0
        self.std = hyper_param['gaussian_std']
        self.reduce_rate = hyper_param['noise_reduce_rate']

        self.obs_size = obs_shape_n
        self.act_size = act_shape_n
        print('obs_size: {}, act_size: {}'.format(self.obs_size, self.act_size))

        self.critic_main = Critic(self.obs_size)
        self.critic_target = Critic(self.obs_size)
        self.critic_target.set_weights(self.critic_main.get_weights())
        self.critic_opt_main = Adam(self.critic_lr_main)
        self.critic_main.compile(optimizer=self.critic_opt_main)

    def action(self, obs):
        obs = tf.convert_to_tensor([obs], dtype=tf.float32)
        # print('in action, obs: ', np.shape(np.array(obs)))
        mu = self.actor_main(obs)
        # print('in action, mu: ', np.shape(np.array(mu)))

        if self.update_step > self.warm_up:
            std = tf.convert_to_tensor([self.std]*4, dtype=tf.float32)
            dist = tfp.distributions.Normal(loc=mu, scale=std)
            action = tf.squeeze(dist.sample())
            self.std = self.std * self.reduce_rate

        action = mu.numpy()
        # print('in action, action: ', np.shape(np.array(action)))
        action = np.clip(action, -1, 1)
        # print('in action, clipped_action: ', np.shape(np.array(action)))
        
        return action

    def target_action(self, obs):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        # print('in trgt action, obs: ', np.shape(np.array(obs)))
        mu = self.actor_target(obs)
        # print('in trgt action, mu: ', np.shape(np.array(mu)))

        if self.update_step > self.warm_up:
            std = tf.convert_to_tensor([self.std]*4, dtype=tf.float32)
            dist = tfp.distributions.Normal(loc=mu, scale=std)
            action = tf.squeeze(dist.sample())

        action = mu.numpy()
        # print('in trgt action, action: ', np.shape(np.array(action)))
        action = np.clip(action, -1, 1)
        # print('in trgt action, clipped_action: ', np.shape(np.array(action)))

        return action

    def update_target(self):
        critic_weithgs = []
        critic_targets = self.critic_target.get_weights()
        
        for idx, weight in enumerate(self.critic_main.get_weights()):
            critic_weithgs.append(weight * self.tau + critic_targets[idx] * (1 - self.tau))
        self.critic_target.set_weights(critic_weithgs)

    def update(self, steps):
        if self.replay_buffer._len() < self.batch_size:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if not steps % self.update_freq == 0:  # only update every update_freq
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        updated = True
        self.update_step += 1
        
        states, next_states, rewards, actions, dones = self.replay_buffer.sample(self.batch_size)
        
        states = tf.convert_to_tensor(states, dtype = tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype = tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)
        actions = tf.convert_to_tensor(actions, dtype = tf.float32)
        dones = tf.convert_to_tensor(dones, dtype = tf.bool)
        
        critic_variable = self.critic_main.trainable_variables
        with tf.GradientTape() as tape_critic:
            tape_critic.watch(critic_variable)
            target_action = self.target_action(next_states)
            
            target_q_next = tf.squeeze(self.critic_target(tf.concat([next_states,target_action], 1)), 1)

            target_q = rewards + self.gamma * target_q_next * (1.0 - tf.cast(dones, dtype=tf.float32))

            current_q = tf.squeeze(self.critic_main(tf.concat([states,actions], 1)), 1)
        
            critic_loss = tf.keras.losses.MSE(target_q, current_q)

        grads_critic, _ = tf.clip_by_global_norm(tape_critic.gradient(critic_loss, critic_variable), 0.5)

        self.critic_opt_main.apply_gradients(zip(grads_critic, critic_variable))

        target_q_val = target_q.numpy()
        current_q_val = current_q.numpy()
        critic_loss_val = critic_loss.numpy()

        self.update_target()

        return updated, np.mean(critic_loss_val), np.mean(target_q_val), np.mean(current_q_val)

    def save_xp(self, obs, new_obs, reward, action, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add((obs, new_obs, reward, action, done))

    def load_models(self, path):
        print('Load Model Path : ', path)
        self.critic_main.load_weights(path, "_critic_main")
        self.critic_target.load_weights(path, "_critic_target")

    def save_models(self, path, score):
        save_path = str(path) + "score_" + str(score) + "_model"
        print('Save Model Path : ', save_path)
        self.critic_main.save_weights(save_path, "_critic_main")
        self.critic_target.save_weights(save_path, "_critic_target")