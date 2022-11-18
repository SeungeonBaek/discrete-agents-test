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
from tensorflow.keras.layers import LayerNormalization

from utils.replay_buffer import ExperienceMemory
from utils.prioritized_memory_numpy import PrioritizedMemory

from agents.ICM_model import ICM_model
from agents.RND_model import RND_target, RND_predict


class DistCritic(Model): # Distributional Q network
    def __init__(self,
                 quantile_num: int,
                 obs_space: int,
                 action_space: int)-> None:
        super(DistCritic,self).__init__()
        self.quantile_num = quantile_num

        self.obs_space = obs_space
        self.action_space = action_space

        self.initializer = initializers.glorot_normal()
        self.regularizer = regularizers.l2(l=0.0005)
        
        self.l1 = Dense(256, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l1_ln = LayerNormalization(axis=-1)
        self.l2 = Dense(256, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2_ln = LayerNormalization(axis=-1)
        self.value_dist = Dense(self.action_space * self.quantile_num, activation = None)

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        l1 = self.l1(state) # Todo: check!
        l1_ln = self.l1_ln(l1)
        l2 = self.l2(l1_ln)
        l2_ln = self.l2_ln(l2)
        value_dist = self.value_dist(l2_ln)
        value_dist = tf.reshape(value_dist, shape=(state.shape[0], self.action_space, self.quantile_num)) # check 필요

        return value_dist


class Agent:
    """
    Argument:
        agent_config: agent configuration which is realted with RL algorithm => QR-DQN
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
        tau: polyak update parameter

        quantile_num: number of quantiles for distributional critic
        tau_hat: quantile values

        epsilon: exploration related hyperparam
        epsilon_decaying_rate: decaying rate of epsilon
        min_epsilon: minimum epsilon(lower bound)

        update_step: current update step for logging
        update_freq: critic updqte freqeuency per env step
        target_update_freq: target network update freqency per update freqency

        replay_buffer: replay buffer which store transition sample (s, a, r, s', d)
        batch_size: mini-batch size
        warm_up: number of warm_up step that uses pure action

        critic_lr_main: learning rate of main distributional critic network
        critic_main: main distributional critic network
        critic_target: target distributional critic network
        critic_opt_main: optimizer of main distributional critic network

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
            Todo: Implementation

    Methods:
        action: return the action which is mapped with obs in policy
        get_intrinsic_reward: return the intrinsic reward
        update_target: update target critic network at user-specified frequency
        update: update main distributional critic network
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
        self.tau = self.agent_config['tau']

        self.quantile_num = self.agent_config['quantile_num']
        self.tau_hat = np.array([(2*(i-1) + 1) / (2 * self.quantile_num) for i in range(1, self.quantile_num + 1)], dtype=np.float32)

        self.epsilon = self.agent_config['epsilon']
        self.epsilon_decaying_rate = self.agent_config['epsilon_decaying_rate']
        self.min_epsilon = self.agent_config['min_epsilon']

        self.update_call_step = 0
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

        self.critic_main = DistCritic(self.quantile_num, self.obs_space, self.act_space)
        self.critic_target = DistCritic(self.quantile_num, self.obs_space, self.act_space)
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

    def action(self, obs: NDArray)-> NDArray: # Todo: check!
        obs = tf.convert_to_tensor([obs], dtype=tf.float32)
        # print(f'in action, obs: {np.shape(np.array(obs))}')
        value_dist = self.critic_main(obs)
        # print(f'in action, value_dist: {np.shape(np.array(value_dist))}')

        random_val = np.random.rand()
        if self.update_step > self.warm_up:
            if random_val > self.epsilon:
                mean_value = np.mean(value_dist.numpy(), axis=2) # Todo: CVaR Implementation
                action = np.argmax(mean_value)
            else:
                action = np.random.randint(self.act_space)

            self.epsilon *= self.epsilon_decaying_rate
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon
        else:
            action = np.random.randint(self.act_space)

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
        if self.tau == None:
            critic_main_weight = self.critic_main.get_weights()
            self.critic_target.set_weights(critic_main_weight)
        else:
            critic_weithgs = []
            critic_targets = self.critic_target.get_weights()
            
            for idx, weight in enumerate(self.critic_main.get_weights()):
                critic_weithgs.append(weight * self.tau + critic_targets[idx] * (1 - self.tau))
            self.critic_target.set_weights(critic_weithgs)

    def update(self, inference_mode: bool=False)-> None:
        self.update_call_step += 1

        if inference_mode == True or (self.replay_buffer._len() < self.batch_size) or (self.update_call_step % self.update_freq != 0):
            if self.extension_name == 'ICM':
                return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            elif self.extension_name == 'RND':
                return False, 0.0, 0.0, 0.0, 0.0, 0.0
            elif self.extension_name == 'NGU':
                return False, 0.0, 0.0, 0.0, 0.0
            else:
                return False, 0.0, 0.0, 0.0, 0.0

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

        # print(f'in update, states: {states.shape}')
        # print(f'in update, next_states: {next_states.shape}')
        # print(f'in update, actions: {actions.shape}')
        # print(f'in update, rewards: {rewards.shape}')

        critic_variable = self.critic_main.trainable_variables
        with tf.GradientTape() as tape_critic:  # Todo backpropagation related code
            tape_critic.watch(critic_variable)
            
            # target
            target_q_dists = self.critic_target(next_states)
            target_indices = tf.stack([range(self.batch_size), tf.argmax(tf.reduce_mean(target_q_dists, axis=2), axis=1)], axis=1)
            target_q_dist_next = tf.gather_nd(params=target_q_dists, indices=target_indices)
            # print(f'in update, target_q_dists: {target_q_dists.shape}')
            # print(f'in update, target_indices: {target_indices.shape}')
            # print(f'in update, target_q_dist_next: {target_q_dist_next.shape}')

            target_q_dist = tf.expand_dims(rewards, axis=1) + self.gamma * target_q_dist_next * tf.expand_dims((1.0 - tf.cast(dones, dtype=tf.float32)), axis=1)
            target_q_dist = tf.stop_gradient(target_q_dist)
            target_q_dist_tile = tf.tile(tf.expand_dims(target_q_dist, axis=1), [1, self.quantile_num, 1])
            # print(f'in update, target_q_dist: {target_q_dist.shape}')
            # print(f'in update, target_q_dist_tile: {target_q_dist_tile.shape}')

            # current
            current_q_dists = self.critic_main(states)
            current_indices = tf.stop_gradient(tf.stack([range(self.batch_size), tf.cast(actions, tf.int32)], axis=1))
            current_q_dist = tf.gather_nd(params=current_q_dists, indices=current_indices)
            # print(f'in update, current_q_dists: {current_q_dists.shape}')
            # print(f'in update, current_indices: {current_indices.shape}')
            # print(f'in update, current_q_dist: {current_q_dist.shape}')

            current_q_dist_tile = tf.tile(tf.expand_dims(current_q_dist, axis=2), [1, 1, self.quantile_num])
            # print(f'in update, current_q_dist_tile: {current_q_dist_tile.shape}')

            # loss
            td_error = tf.subtract(target_q_dist_tile, current_q_dist_tile)
            huber_loss = tf.where(tf.less(tf.math.abs(td_error), 1.0), 1/2 * tf.math.square(td_error), 1.0 * tf.abs(td_error) - 1.0 * 1/2)
            # print(f'in update, td_error: {td_error.shape}')
            # print(f'in update, huber_loss: {huber_loss.shape}')
            
            tau = tf.reshape(np.array(self.tau_hat), [1, self.quantile_num])
            inv_tau = 1.0 - tau
            # print(f'in update, tau: {tau.shape}')
            # print(f'in update, inv_tau: {inv_tau.shape}')

            tau_tile = tf.tile(tf.expand_dims(tau, axis=1), [1, self.quantile_num, 1])
            inv_tau_tile = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, self.quantile_num, 1])
            # print(f'in update, tau_tile: {tau_tile.shape}')
            # print(f'in update, inv_tau_tile: {inv_tau_tile.shape}')

            critic_losses = tf.where(tf.less(td_error, 0.0), tf.multiply(inv_tau_tile, huber_loss), tf.multiply(tau_tile, huber_loss))
            critic_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(critic_losses, axis=2), axis=1))
            # print(f'in update, critic_losses: {critic_losses.shape}')
            # print(f'in update, critic_loss: {critic_loss.shape}')

        # value check
        # print(f'in update, target_q_dists: {target_q_dists}')
        # ... omitted

        grads_critic, _ = tf.clip_by_global_norm(tape_critic.gradient(critic_loss, critic_variable), 0.5)

        self.critic_opt_main.apply_gradients(zip(grads_critic, critic_variable))

        target_q_val = tf.reduce_mean(tf.reduce_mean(target_q_dist, axis=1), axis=0).numpy()
        current_q_val = tf.reduce_mean(tf.reduce_mean(current_q_dist, axis=1), axis=0).numpy()
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
        td_error_numpy = tf.reduce_sum(tf.reduce_mean(critic_losses, axis=2), axis=1)
        if self.agent_config['use_PER']:
            for i in range(self.batch_size):
                self.replay_buffer.update(idxs[i], td_error_numpy[i])

        # return for logging
        if self.extension_name == 'ICM':
            return updated, critic_loss_val, target_q_val, current_q_val, self.epsilon, icm_pred_next_s_loss_val, icm_pred_a_loss_val
        elif self.extension_name == 'RND':
            return updated, critic_loss_val, target_q_val, current_q_val, self.epsilon, rnd_pred_loss_val
        elif self.extension_name == 'NGU':
            pass
        else:
            return updated, critic_loss_val, target_q_val, current_q_val, self.epsilon

    def save_xp(self, state: NDArray, next_state: NDArray, reward: float, action: int, done: bool)-> None:
        # Store transition in the replay buffer.
        if self.agent_config['use_PER']:
            state_tf = tf.convert_to_tensor([state], dtype = tf.float32)
            next_state_tf = tf.convert_to_tensor([next_state], dtype = tf.float32)
            reward_tf = tf.convert_to_tensor([reward], dtype=tf.float32)
            action_tf = tf.convert_to_tensor([action], dtype=tf.float32)
            done_tf = tf.convert_to_tensor([done], dtype=tf.bool)
            
            # target
            target_q_dists = self.critic_target(next_state_tf)
            target_indices = tf.stack([[0], tf.argmax(tf.reduce_mean(target_q_dists, axis=2), axis=1)], axis=1)
            target_q_dist_next = tf.gather_nd(params=target_q_dists, indices=target_indices)

            target_q_dist = tf.expand_dims(reward_tf, axis=1) + self.gamma * target_q_dist_next * tf.expand_dims((1.0 - tf.cast(done_tf, dtype=tf.float32)), axis=1)
            target_q_dist_tile = tf.tile(tf.expand_dims(target_q_dist, axis=1), [1, self.quantile_num, 1])

            # current
            current_q_dists = self.critic_main(state_tf)
            current_indices = tf.stop_gradient(tf.stack([[0], tf.cast(action_tf, tf.int32)], axis=1))
            current_q_dist = tf.gather_nd(params=current_q_dists, indices=current_indices)

            current_q_dist_tile = tf.tile(tf.expand_dims(current_q_dist, axis=2), [1, 1, self.quantile_num])

            # loss
            td_error = tf.subtract(target_q_dist_tile, current_q_dist_tile)
            huber_loss = tf.where(tf.less(tf.math.abs(td_error), 1.0), 1/2 * tf.math.square(td_error), 1.0 * tf.abs(td_error) - 1.0 * 1/2)
            
            tau = tf.reshape(np.array(self.tau_hat), [1, self.quantile_num])
            inv_tau = 1.0 - tau

            tau_tile = tf.tile(tf.expand_dims(tau, axis=1), [1, self.quantile_num, 1])
            inv_tau_tile = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, self.quantile_num, 1])

            critic_losses = tf.where(tf.less(td_error, 0.0), tf.multiply(inv_tau_tile, huber_loss), tf.multiply(tau_tile, huber_loss))
            critic_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(critic_losses, axis=2), axis=1))

            self.replay_buffer.add(critic_loss.numpy(), (state, next_state, reward, action, done))
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