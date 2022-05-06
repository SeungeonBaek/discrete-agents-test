import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense

from utils.replay_buffer import ExperienceMemory


class Critic(Model):
    def __init__(self, act_size):
        super(Critic,self).__init__()
        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.005)
        
        self.l1 = Dense(64, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2 = Dense(128, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l3 = Dense(64, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l4 = Dense(32, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.value = Dense(act_size, activation = 'softmax')

    def call(self, state_action):
        l1 = self.l1(state_action) # 확인
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        l4 = self.l4(l3)
        value = self.value(l4)

        return value


class Agent:
    """
    input argument: agent_config, obs_shape_n, act_shape_n

    agent_config: lr_critic_main, gamma, update_freq, batch_size, warm_up
    """
    def __init__(self, agent_config, obs_shape_n, act_shape_n):
        self.name = agent_config['agent_name']
        self.critic_lr_main = agent_config['lr_critic']

        self.gamma = agent_config['gamma']
        self.epsilon = agent_config['epsilon']
        self.epsilon_decaying_rate = agent_config['epsilon_decaying_rate']

        self.update_freq = agent_config['update_freq']
        self.target_update_freq = agent_config['target_update_freq']

        self.replay_buffer = ExperienceMemory(agent_config['buffer_size'])
        self.batch_size = agent_config['batch_size']
        self.warm_up = agent_config['warm_up']
        self.update_step = 0

        self.obs_size = obs_shape_n
        self.act_size = act_shape_n
        print('obs_size: {}, act_size: {}'.format(self.obs_size, self.act_size))

        self.critic_main = Critic(self.act_size)
        self.critic_target = Critic(self.act_size)
        self.critic_target.set_weights(self.critic_main.get_weights())
        self.critic_opt_main = Adam(self.critic_lr_main)
        self.critic_main.compile(optimizer=self.critic_opt_main)

    def action(self, obs):
        obs = tf.convert_to_tensor([obs], dtype=tf.float32)
        # print(f'in action, obs: {np.shape(np.array(obs))}')
        values = self.critic_main(obs)
        # print(f'in action, values: {np.shape(np.array(values))}')

        if self.update_step > self.warm_up:
            if random_val:=np.random.rand() > self.epsilon:
                action = np.argmax(values.numpy())
            else:
                action = np.random.randint(self.act_size)
        else:
            action = np.random.randint(self.act_size)
        # print(f'in action, action: {np.shape(np.array(action))}')

        self.epsilon *= self.epsilon_decaying_rate

        return action

    def update_target(self):
        critic_main_weight = self.critic_main.get_weights()
        self.critic_target.set_weights(critic_main_weight)

    def update(self, steps):
        if self.replay_buffer._len() < self.batch_size:
            return False, 0.0, 0.0, 0.0
        if not steps % self.update_freq == 0:  # only update every update_freq
            return False, 0.0, 0.0, 0.0

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
            # print(f'in update, states: {states.shape}')
            # print(f'in update, next_states: {next_states.shape}')
            # print(f'in update, actions: {actions.shape}')
            # print(f'in update, rewards: {rewards.shape}')
            
            target_q_next = tf.reduce_max(self.critic_target(next_states), axis=1)
            # print(f'in update, target_q_next: {target_q_next.shape}')

            target_q = rewards + self.gamma * target_q_next * (1.0 - tf.cast(dones, dtype=tf.float32))
            # print(f'in update, target_q_next: {target_q_next.shape}')

            current_q = self.critic_main(states)
            # print(f'in update, current_q: {current_q.shape}')
            action_one_hot = tf.one_hot(tf.cast(actions, tf.int32), self.act_size)
            # print(f'in update, action_one_hot: {action_one_hot.shape}')
            current_q = tf.reduce_sum(tf.multiply(current_q, action_one_hot), axis=1)
            # print(f'in update, current_q: {current_q.shape}')
        
            critic_loss = tf.keras.losses.MSE(target_q, current_q)
            # print(f'in update, critic_loss: {critic_loss.shape}')

        grads_critic, _ = tf.clip_by_global_norm(tape_critic.gradient(critic_loss, critic_variable), 0.5)

        self.critic_opt_main.apply_gradients(zip(grads_critic, critic_variable))

        target_q_val = target_q.numpy()
        current_q_val = current_q.numpy()
        critic_loss_val = critic_loss.numpy()

        if self.update_step % self.target_update_freq == 0:
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