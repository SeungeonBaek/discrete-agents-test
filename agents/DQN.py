import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense

from utils.replay_buffer import ExperienceMemory
from utils.prioritized_memory_numpy import PrioritizedMemory

from agents.ICM_model import ICM_model
from agents.RND_model import RND_target, RND_predict

class Critic(Model): # Q network
    def __init__(self, act_space):
        super(Critic,self).__init__()
        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.005)
        
        self.l1 = Dense(64, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2 = Dense(128, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l3 = Dense(64, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l4 = Dense(32, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.value = Dense(act_space, activation = 'softmax')

    def call(self, state_action):
        l1 = self.l1(state_action)
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        l4 = self.l4(l3)
        value = self.value(l4)

        return value


class Agent: # => Q network를 가지고 있으며, 환경과 상호작용 하는 녀석이다!
    """
    Argument:
        agent_config: agent configuration which is realted with RL algorithm => DQN
            agent_config:
                {
                    name, gamma, tau, update_freq, batch_size, warm_up, lr_actor, lr_critic,
                    buffer_size, use_PER, use_ERE, reward_normalize
                    extension = {
                        'gaussian_std, 'noise_clip', 'noise_reduce_rate'
                    }
                }
        obs_shape_n: shpae of observation
        act_shape_n: shape of action

    Methods:
        action: return the action which is mapped with obs in policy
        update_target: update target critic network at user-specified frequency
        update: update main critic network
        save_xp: save transition(s, a, r, s', d) in experience memory
        load_models: load weights
        save_models: save weights
    
    """
    def __init__(self, agent_config, obs_shape_n, act_shape_n):
        self.agent_config = agent_config
        self.name = self.agent_config['agent_name']

        self.obs_space = obs_shape_n
        self.act_space = act_shape_n
        print(f'obs_space: {self.obs_space}, act_space: {self.act_space}')

        self.gamma = self.agent_config['gamma']
        self.epsilon = self.agent_config['epsilon']
        self.epsilon_decaying_rate = self.agent_config['epsilon_decaying_rate']

        self.update_step = 0
        self.update_freq = self.agent_config['update_freq']
        self.target_update_freq = self.agent_config['target_update_freq']

        if self.agent_config['use_PER']:
            self.replay_buffer = PrioritizedMemory(self.agent_config['buffer_size'])
        else:
            self.replay_buffer = ExperienceMemory(self.agent_config['buffer_size'])
        self.batch_size = self.agent_config['batch_size']
        self.warm_up = self.agent_config['warm_up']

        # network config
        self.critic_lr_main = self.agent_config['lr_critic']

        self.critic_main = Critic(self.act_space)
        self.critic_target = Critic(self.act_space)
        self.critic_target.set_weights(self.critic_main.get_weights())
        self.critic_opt_main = Adam(self.critic_lr_main)
        self.critic_main.compile(optimizer=self.critic_opt_main)

        # extension config
        self.extension_config = self.agent_config['extension']
        self.extension_name = self.extension_config['name']

    def action(self, obs):
        obs = tf.convert_to_tensor([obs], dtype=tf.float32)
        # print(f'in action, obs: {np.shape(np.array(obs))}')
        values = self.critic_main(obs)
        # print(f'in action, values: {np.shape(np.array(values))}')

        random_val = np.random.rand()
        if self.update_step > self.warm_up:
            if random_val > self.epsilon:
                action = np.argmax(values.numpy())
            else:
                action = np.random.randint(self.act_space)
        else:
            action = np.random.randint(self.act_space)
        # print(f'in action, action: {np.shape(np.array(action))}')

        self.epsilon *= self.epsilon_decaying_rate

        return action

    def update_target(self):
        critic_main_weight = self.critic_main.get_weights()
        self.critic_target.set_weights(critic_main_weight)

    def update(self):
        if self.replay_buffer._len() < self.batch_size:
            return False, 0.0, 0.0, 0.0
        if not self.update_step % self.update_freq == 0:  # only update every update_freq
            self.update_step += 1
            return False, 0.0, 0.0, 0.0

        updated = True
        self.update_step += 1
        
        if self.agent_config['use_PER']:
            states, next_states, rewards, actions, dones, idxs, is_weight = self.replay_buffer.sample(self.batch_size)

            if self.agent_config['reward_normalize']:
                rewards = np.asarray(rewards)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            states = tf.convert_to_tensor(states, dtype = tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype = tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)
            actions = tf.squeeze(tf.convert_to_tensor(actions, dtype = tf.float32))
            dones = tf.convert_to_tensor(dones, dtype = tf.bool)
            is_weight = tf.convert_to_tensor(is_weight, dtype=tf.float32)
            # print(f'states : {states.shape}')
            # print(f'next_states : {next_states.shape}')
            # print(f'rewards : {rewards.shape}')
            # print(f'actions : {actions.shape}')
            # print(f'dones : {dones.shape}')
            # print(f'is_weight : {is_weight.shape}')
        
        else:
            states, next_states, rewards, actions, dones = self.replay_buffer.sample(self.batch_size)
            
            states = tf.convert_to_tensor(states, dtype = tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype = tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)
            actions = tf.squeeze(tf.convert_to_tensor(actions, dtype = tf.float32))
            dones = tf.convert_to_tensor(dones, dtype = tf.bool)
            # print(f'states : {states.shape}')
            # print(f'next_states : {next_states.shape}')
            # print(f'rewards : {rewards.shape}')
            # print(f'actions : {actions.shape}')

        critic_variable = self.critic_main.trainable_variables
        with tf.GradientTape() as tape_critic:
            tape_critic.watch(critic_variable)

            current_q_next = self.critic_main(next_states)
            # print(f'in update, current_q_next: {current_q_next.shape}')
            next_action = tf.argmax(current_q_next, axis=1)
            # print(f'in update, next_action: {next_action.shape}')
            indices = tf.stack([range(self.batch_size), next_action], axis=1)
            # print(f'in update, indices: {indices.shape}')

            target_q_next = tf.cond(tf.convert_to_tensor(self.extension_config['use_DDQN'], dtype=tf.bool),\
                    lambda: tf.gather_nd(params=self.critic_target(next_states), indices=indices), \
                    lambda: tf.reduce_max(self.critic_target(next_states), axis=1))
            # print(f'in update, target_q_next: {target_q_next.shape}')

            target_q = rewards + self.gamma * target_q_next * (1.0 - tf.cast(dones, dtype=tf.float32))
            target_q = tf.stop_gradient(target_q)
            # print(f'in update, target_q_next: {target_q.shape}')

            current_q = self.critic_main(states)
            # print(f'in update, current_q: {current_q.shape}')
            action_one_hot = tf.one_hot(tf.cast(actions, tf.int32), self.act_space)
            # print(f'in update, action_one_hot: {action_one_hot.shape}')
            current_q = tf.reduce_sum(tf.multiply(current_q, action_one_hot), axis=1)
            # print(f'in update, current_q: {current_q.shape}')

            td_error = tf.subtract(current_q, target_q)
            # print(f'in update, td_error: {td_error.shape}')

            critic_losses = tf.cond(tf.convert_to_tensor(self.agent_config['use_PER'], dtype=tf.bool), \
                    lambda: tf.multiply(is_weight, tf.math.square(td_error)), \
                    lambda: tf.math.square(td_error))
            # print(f'in update, critic_losses : {critic_losses.shape}')
            
            critic_loss = tf.math.reduce_mean(critic_losses)
            # print(f'in update, critic_loss: {critic_loss.shape}')

        grads_critic, _ = tf.clip_by_global_norm(tape_critic.gradient(critic_loss, critic_variable), 0.5)

        self.critic_opt_main.apply_gradients(zip(grads_critic, critic_variable))

        target_q_val = target_q.numpy()
        current_q_val = current_q.numpy()
        critic_loss_val = critic_loss.numpy()

        if self.update_step % self.target_update_freq == 0:
            self.update_target()

        td_error_numpy = np.abs(td_error.numpy())
        if self.agent_config['use_PER']:
            for i in range(self.batch_size):
                self.replay_buffer.update(idxs[i], td_error_numpy[i])

        return updated, np.mean(critic_loss_val), np.mean(target_q_val), np.mean(current_q_val)

    def save_xp(self, state, next_state, reward, action, done):
        # Store transition in the replay buffer.
        if self.agent_config['use_PER']:
            state_tf = tf.convert_to_tensor([state], dtype = tf.float32)
            action_tf = tf.convert_to_tensor([action], dtype = tf.float32)
            next_state_tf = tf.convert_to_tensor([next_state], dtype = tf.float32)
            target_action_tf = self.critic_target(next_state_tf)
            # print(f'state_tf: {state_tf.shape}, {state_tf}')
            # print(f'action_tf: {action_tf.shape}, {action_tf}')
            # print(f'next_state_tf: {next_state_tf.shape}, {next_state_tf}')
            # print(f'target_action_tf: {target_action_tf.shape}, {target_action_tf}')

            current_q_next = self.critic_main(next_state_tf)
            # print(f'current_q_next: {current_q_next.shape}, {current_q_next}')
            next_action = tf.argmax(current_q_next, axis=1)
            # print(f'next_action: {next_action.shape}, {next_action}')
            indices = tf.stack([[0], next_action], axis=1)
            # print(f'indices: {indices.shape}, {indices}')

            target_q_next = self.critic_target(next_state_tf)
            # print(f'target_q_next: {target_q_next.shape}, {target_q_next}')

            target_q_next = tf.cond(tf.convert_to_tensor(self.extension_config['use_DDQN'], dtype=tf.bool),\
                    lambda: tf.gather_nd(params=self.critic_target(next_state_tf), indices=indices), \
                    lambda: tf.reduce_max(self.critic_target(next_state_tf), axis=1))
            # print(f'target_q_next: {target_q_next.shape}, {target_q_next}')

            target_q = reward + self.gamma * target_q_next * (1.0 - tf.cast(done, dtype=tf.float32))
            # print(f'target_q: {target_q.shape}, {target_q}')
            
            current_q = self.critic_main(state_tf)
            # print(f'current_q: {current_q.shape}, {current_q}')
            action_one_hot = tf.one_hot(tf.cast(action_tf, tf.int32), self.act_space)
            # print(f'action_one_hot: {action_one_hot.shape}, {action_one_hot}')
            current_q = tf.reduce_sum(tf.multiply(current_q, action_one_hot), axis=1)
            # print(f'current_q: {current_q.shape}, {current_q}')
            
            td_error = tf.subtract(target_q ,current_q)
            # print(f'td_error: {td_error.shape}, {td_error}')

            td_error_numpy = np.abs(td_error)
            # print(f'td_error_numpy: {td_error_numpy.shape}, {td_error_numpy}')

            self.replay_buffer.add(td_error_numpy[0], (state, next_state, reward, action, done))
        else:
            self.replay_buffer.add((state, next_state, reward, action, done))

    def load_models(self, path):
        print('Load Model Path : ', path)
        self.critic_main.load_weights(path, "_critic_main")
        self.critic_target.load_weights(path, "_critic_target")

    def save_models(self, path, score):
        save_path = str(path) + "score_" + str(score) + "_model"
        print('Save Model Path : ', save_path)
        self.critic_main.save_weights(save_path, "_critic_main")
        self.critic_target.save_weights(save_path, "_critic_target")