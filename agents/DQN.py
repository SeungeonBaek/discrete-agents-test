from typing import Dict, Union, Any
from numpy.typing import NDArray

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
    def __init__(self,
                 act_space: int)-> None:
        super(Critic,self).__init__()
        self.initializer = initializers.orthogonal()
        self.regularizer = regularizers.l2(l=0.001)
        
        self.l1 = Dense(256, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2 = Dense(256, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.value = Dense(act_space, activation = 'None')

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        l1 = self.l1(state)
        l2 = self.l2(l1)
        value = self.value(l2)

        return value


class Agent:
    """
    Argument:
        agent_config: agent configuration which is realted with RL algorithm => DQN
            agent_config:
                {
                    name, gamma, tau, update_freq, batch_size, warm_up, lr_actor, lr_critic,
                    buffer_size, use_PER, use_ERE, reward_normalize
                    extension = {
                        'name', 'use_DDQN'
                    }
                }
        obs_space: shpae of observation
        act_space: shape of action

    Properties:
        agent_config: agent configuration
        name: agent name
        obs_space: shape of observation
        act_space: shape of action
        gamma: discount rate

        epsilon: exploration related hyperparam
        epsilon_decaying_rate: decaying rate of epsilon
        min_epsilon: minimum epsilon(lower bound)

        update_step: current update step for logging
        update_freq: critic updqte freqeuency per env step
        target_update_freq: target network update freqency per update freqency

        replay_buffer: replay buffer which store transition sample (s, a, r, s', d)
        batch_size: mini-batch size
        warm_up: number of warm_up step that uses pure action

        critic_lr_main: learning rate of main critic network
        critic_main: main critic network
        critic_target: target critic network
        critic_opt_main: optimizer of main critic network

        # extension properties
        extension_config: configuration of extention algorithm
        extension_name: name of extention algorithm
            
            # icm - intrinsic curiosity model
            icm_update_freq: update freqency per step for icm model update
            icm_lr: learning rate of icm
            icm_feqture_dim: feature dimension of icm

            icm: icm network
            icm_opt: optimizer of icm network

            # rnd: random network distillation
            rnd_update_freq: update freqency per step for rnd model update
            rnd_lr: learning rate of rnd
            
            rnd_target: target network of rnd
            rnd_predict: predict network of rnd
            rnd_opt: optimizer for predict network of rnd

            # ngu: never give up!
            Todo

    Methods:
        action: return the action which is mapped with obs in policy
        get_intrinsic_reward: return the intrinsic reward
        update_target: update target critic network at user-specified frequency
        update: update main critic network
        save_xp: save transition(s, a, r, s', d) in experience memory
        load_models: load weights
        save_models: save weights
    
    """
    def __init__(self,
                 agent_config: Dict,
                 obs_space: int,
                 act_space: int)-> None:
        self.agent_config = agent_config
        self.name = self.agent_config['agent_name']

        self.obs_space = obs_space
        self.act_space = act_space
        print(f'obs_space: {self.obs_space}, act_space: {self.act_space}')

        self.gamma = self.agent_config['gamma']
        
        self.epsilon = self.agent_config['epsilon']
        self.epsilon_decaying_rate = self.agent_config['epsilon_decaying_rate']
        self.min_epsilon = self.agent_config['min_epsilon']

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
            self.icm_lr = self.extension_config['ngu_lr']

    def action(self, obs: NDArray)-> NDArray:
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
        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon

        return action

    def get_intrinsic_reward(self, state: NDArray, next_state: NDArray, action: NDArray)-> float:
        reward_int = 0
        if self.extension_name == 'ICM':
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
            action = tf.convert_to_tensor([action], dtype=tf.float32)
            
            feature_next_s, pred_feature_next_s, _ = self.icm((state, next_state, action))

            reward_int = tf.clip_by_value(tf.reduce_mean(tf.math.square(tf.subtract(feature_next_s, pred_feature_next_s))), 0, 5)
            reward_int = reward_int.numpy()
        
        elif self.extension_name == 'RND':
            next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
            
            target_value = self.rnd_target(next_state)
            predict_value = self.rnd_predict(next_state)

            reward_int = tf.clip_by_value(tf.reduce_mean(tf.math.square(tf.subtract(predict_value, target_value))), 0, 5)
            reward_int = reward_int.numpy()

        elif self.extension_name == 'NGU':
            pass

        return reward_int

    def update_target(self)-> None:
        critic_main_weight = self.critic_main.get_weights()
        self.critic_target.set_weights(critic_main_weight)

    def update(self)-> None:
        if self.replay_buffer._len() < self.batch_size:
            if self.extension_name == 'ICM':
                return False, 0.0, 0.0, 0.0, 0.0, 0.0
            elif self.extension_name == 'RND':
                return False, 0.0, 0.0, 0.0, 0.0
            elif self.extension_name == 'NGU':
                return False, 0.0, 0.0, 0.0
            else:
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
        
        else:
            states, next_states, rewards, actions, dones = self.replay_buffer.sample(self.batch_size)

            if self.agent_config['reward_normalize']:
                rewards = np.asarray(rewards)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            states = tf.convert_to_tensor(states, dtype = tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype = tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)
            actions = tf.squeeze(tf.convert_to_tensor(actions, dtype = tf.float32))
            dones = tf.convert_to_tensor(dones, dtype = tf.bool)

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
                                tf.cond(tf.convert_to_tensor(self.agent_config['use_Huber'], dtype=tf.bool), \
                                    lambda: tf.multiply(is_weight, tf.where(tf.less(td_error, 1.0), 1/2 * tf.math.square(td_error), 1.0 * tf.abs(td_error - 1.0 * 1/2))), \
                                    lambda: tf.multiply(is_weight, tf.math.square(td_error))), \
                                tf.cond(tf.convert_to_tensor(self.agent_config['use_Huber'], dtype=tf.bool), \
                                    lambda: tf.where(tf.less(td_error, 1.0), 1/2 * tf.math.square(td_error), 1.0 * tf.abs(td_error - 1.0 * 1/2)), \
                                    lambda: tf.math.square(td_error)))
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

        icm_pred_next_s_loss_val, icm_pred_a_loss = 0, 0
        rnd_pred_loss_val = 0
        
        # extensions
        if self.extension_name == 'ICM':
            if self.update_step % self.icm_update_freq == 0:
                icm_variable = self.icm.trainable_variables
                with tf.GradientTape() as tape_icm:
                    tape_icm.watch(icm_variable)

                    feature_next_s, pred_feature_next_s, pred_a = self.icm((states, next_states, actions))

                    icm_pred_next_s_loss = tf.reduce_mean(tf.math.square(tf.subtract(feature_next_s, pred_feature_next_s)))
                    icm_pred_a_loss = tf.reduce_mean(tf.math.square(tf.subtract(actions, pred_a)))

                    icm_pred_loss = tf.add(icm_pred_next_s_loss, icm_pred_a_loss)

                grads_icm, _ = tf.clip_by_global_norm(tape_icm.gradient(icm_pred_loss, icm_variable), 0.5)
                self.icm_opt.apply_gradients(zip(grads_icm, icm_variable))            

                icm_pred_next_s_loss_val = icm_pred_next_s_loss.numpy()
                icm_pred_a_loss_val = icm_pred_a_loss.numpy()

        elif self.extension_name == 'RND':
            if self.update_step % self.rnd_update_freq == 0:
                rnd_variable = self.rnd_predict.trainable_variables
                with tf.GradientTape() as tape_rnd:
                    tape_rnd.watch(rnd_variable)
            
                    predictions = self.rnd_predict(next_states)
                    targets = self.rnd_target(next_states)

                    rnd_pred_loss = tf.reduce_mean(tf.math.square(tf.subtract(predictions, targets)))

                grads_rnd, _ = tf.clip_by_global_norm(tape_rnd.gradient(rnd_pred_loss, rnd_variable), 0.5)
                self.rnd_opt.apply_gradients(zip(grads_rnd, rnd_variable))

                rnd_pred_loss_val = rnd_pred_loss.numpy()


        elif self.extension_name == 'NGU':
            pass

        # PER update
        td_error_numpy = np.abs(td_error.numpy())
        if self.agent_config['use_PER']:
            for i in range(self.batch_size):
                self.replay_buffer.update(idxs[i], td_error_numpy[i])

        if self.extension_name == 'ICM':
            return updated, np.mean(critic_loss_val), np.mean(target_q_val), np.mean(current_q_val), icm_pred_next_s_loss_val, icm_pred_a_loss_val
        elif self.extension_name == 'RND':
            return updated, np.mean(critic_loss_val), np.mean(target_q_val), np.mean(current_q_val), rnd_pred_loss_val
        elif self.extension_name == 'NGU':
            pass
        else:
            return updated, np.mean(critic_loss_val), np.mean(target_q_val), np.mean(current_q_val)

    def save_xp(self, state: NDArray, next_state: NDArray, reward: float, action: int, done: bool)-> None:
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

    def load_models(self, path: str)-> None:
        print('Load Model Path : ', path)
        self.critic_main.load_weights(path, "_critic_main")
        self.critic_target.load_weights(path, "_critic_target")

    def save_models(self, path: str, score: float)-> None:
        save_path = path + "score_" + str(score) + "_model"
        print('Save Model Path : ', save_path)
        self.critic_main.save_weights(save_path, "_critic_main")
        self.critic_target.save_weights(save_path, "_critic_target")