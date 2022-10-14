import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense

from utils.replay_buffer import ExperienceMemory


class Critic(Model): # Q network
    def __init__(self, act_size):
        super(Critic,self).__init__()
        # network 형상 정의
        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.005)
        
        self.l1 = Dense(64, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2 = Dense(128, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l3 = Dense(64, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l4 = Dense(32, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.value = Dense(act_size, activation = 'softmax')

    def call(self, state):
        l1 = self.l1(state) # 확인
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        l4 = self.l4(l3)
        value = self.value(l4)

        return value


class DistCritic(Model): # Distributional Q net
    def __init__(self,
                 quantile_num,
                 obs_space,
                 action_space):
        super(DistCritic,self).__init__()
        self.quantile_num = quantile_num

        self.obs_space = obs_space
        self.action_space = action_space

        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.001)
        
        self.l1 = Dense(256, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2 = Dense(256, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l3 = Dense(128, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l4 = Dense(128, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.value = Dense(self.action_space * self.quantile_num, activation = None)

    def call(self, state):
        l1 = self.l1(state) # 확인
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        l4 = self.l4(l3)
        value = self.value(l4)
        value = tf.reshape(value, shape=(state.shape[0], self.action_space, self.quantile_num))

        return value


class Agent: # => Q network를 가지고 있으며, 환경과 상호작용 하는 녀석이다!
    """
    input argument: agent_config, obs_shape_n, act_shape_n

    agent_config: lr_critic_main, gamma, update_freq, batch_size, warm_up
    """
    def __init__(self, agent_config, obs_space, act_space):
        self.agent_config = agent_config
        self.name = self.agent_config['agent_name']

        self.obs_space = obs_space
        self.act_space = act_space
        print(f'obs_space: {self.obs_space}, act_space: {self.act_space}')

        self.critic_lr_main = self.agent_config['lr_critic']

        self.gamma = self.agent_config['gamma']
        self.tau = self.agent_config['tau']
        self.quantile_num = self.agent_config['quantile_num']

        self.update_step = 0
        self.update_freq = self.agent_config['update_freq']
        self.target_update_freq = agent_config['target_update_freq']

        if self.agent_config['use_PER']:
            self.replay_buffer = PrioritizedMemory(self.agent_config['buffer_size'])
        else:
            self.replay_buffer = ExperienceMemory(self.agent_config['buffer_size'])
        self.batch_size = self.agent_config['batch_size']
        self.warm_up = self.agent_config['warm_up']

        self.epsilon = self.agent_config['epsilon']
        self.epsilon_decaying_rate = self.agent_config['epsilon_decaying_rate']

        # network config
        self.critic_lr_main = self.agent_config['lr_critic']

        self.critic_main = DistCritic(self.quantile_num, self.obs_space, self.act_size)
        self.critic_target = DistCritic(self.quantile_num, self.obs_space, self.act_size)
        self.critic_target.set_weights(self.critic_main.get_weights())
        self.critic_opt_main = Adam(self.critic_lr_main)
        self.critic_main.compile(optimizer=self.critic_opt_main)

        # extension config
        self.extension_config = self.agent_config['extension']
        self.extension_name = self.extension_config['name']

        if self.extension_name == 'ICM':
            self.icm_update_freq = self.extension_config['icm_update_freq']

            self.icm_lr = self.extension_config['icm_lr']
            self.icm_feature_dim = self.extension_config['icm_feature_dim']
            self.icm = ICM_model(self.obs_space, self.act_space, self.icm_feature_dim)
            self.icm_opt = Adam(self.icm_lr)

        elif self.extension_name == 'RND':
            self.rnd_update_freq = self.extension_config['rnd_update_freq']

            self.rnd_lr = self.extension_config['rnd_lr']
            self.rnd_target = RND_target(self.obs_space, self.act_space)
            self.rnd_predict = RND_predict(self.obs_space, self.act_space)
            self.rnd_opt = Adam(self.rnd_lr)

        elif self.extension_name == 'NGU':
            self.icm_lr = self.extension_config['icm_lr']

    def action(self, obs): # Todo
        obs = tf.convert_to_tensor([obs], dtype=tf.float32)
        # print(f'in action, obs: {np.shape(np.array(obs))}')
        values = self.critic_main(obs)
        # print(f'in action, values: {np.shape(np.array(values))}')

        random_val = np.random.rand()
        if self.update_step > self.warm_up:
            if random_val > self.epsilon:
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

    def update(self, steps): # Todo
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