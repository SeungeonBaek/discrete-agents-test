

class RLLogger():
    def __init__(self, agent_config, rl_config, summary_writer = None, wandb_session = None):
        self.agent_config = agent_config
        self.rl_config = rl_config
        self.summary_writer = summary_writer
        self.wandb_session = wandb_session
        if self.wandb_session is not None:
            import wandb

    def step_logging(self, Agent: object, reward_int: float = None, inference_mode: bool = False):
        if self.rl_config['tensorboard'] == True:
            self.step_logging_tensorboard(Agent, reward_int, inference_mode)
        if self.rl_config['wandb'] == True:
            self.step_logging_wandb(Agent, reward_int, inference_mode)

    def episode_logging(self, *args, inference_mode:bool = False):
        if self.rl_config['tensorboard'] == True:
            self.episode_logging_tensorboard(*args, inference_mode)
        if self.rl_config['wandb'] == True:
            self.episode_logging_wandb(*args, inference_mode)

    def eval_logging(self, *args):
        if self.rl_config['tensorboard'] == True:
            self.eval_logging_tensorboard(*args)
        if self.rl_config['wandb'] == True:
            self.eval_logging_wandb(*args)

    def step_logging_tensorboard(self, Agent, reward_int = None, inference_mode: bool = False):
        # Update
        if self.agent_config['agent_name'] == 'DQN':
            if self.agent_config['is_configurable_critic']:
                if self.agent_config['critic_config']['network_config']['feature_extractor_config']['name'].lower() in ('autoencoder1d', 'ae1d', 'autoencoder2d', 'ae2d', 'autoencoder', 'ae'):
                    if Agent.extension_name == 'ICM':
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon, icm_state_loss, icm_action_loss, recon_loss = Agent.update(inference_mode)
                    elif Agent.extension_name == 'RND':
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon, rnd_pred_loss, recon_loss = Agent.update(inference_mode)
                    elif Agent.extension_name == 'NGU':
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon, recon_loss = Agent.update(inference_mode)
                    else:
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon, recon_loss = Agent.update(inference_mode)
                else:
                    if Agent.extension_name == 'ICM':
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon, icm_state_loss, icm_action_loss = Agent.update(inference_mode)
                    elif Agent.extension_name == 'RND':
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon, rnd_pred_loss = Agent.update(inference_mode)
                    elif Agent.extension_name == 'NGU':
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon = Agent.update(inference_mode)
                    else:
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon = Agent.update(inference_mode)
            else:
                if Agent.extension_name == 'ICM':
                    updated, critic_loss, trgt_q_mean, critic_value, epsilon, icm_state_loss, icm_action_loss = Agent.update(inference_mode)
                elif Agent.extension_name == 'RND':
                    updated, critic_loss, trgt_q_mean, critic_value, epsilon, rnd_pred_loss = Agent.update(inference_mode)
                elif Agent.extension_name == 'NGU':
                    updated, critic_loss, trgt_q_mean, critic_value, epsilon = Agent.update(inference_mode)
                else:
                    updated, critic_loss, trgt_q_mean, critic_value, epsilon = Agent.update(inference_mode)

        elif self.agent_config['agent_name'] == 'PPO':
            if self.agent_config['extension']['name'] == 'Model_Ensemble':
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value, critic_q_value = Agent.update(inference_mode)
            else:
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value, critic_q_value = Agent.update(inference_mode)

        elif self.agent_config['agent_name'] == 'SAC':
            if self.agent_config['extension']['name'] == 'TQC':
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value, critic_q_value = Agent.update(inference_mode)
            else:
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value, critic_q_value = Agent.update(inference_mode)

        elif self.agent_config['agent_name'] in ('QR_DQN', 'Safe_QR_DQN', 'IQN', 'Safe_IQN', 'MMDQN', 'Safe_MMDQN'):
            if Agent.extension_name == 'ICM':
                updated, critic_loss, trgt_q_mean, critic_value, epsilon, icm_state_loss, icm_action_loss = Agent.update(inference_mode)
            elif Agent.extension_name == 'RND':
                updated, critic_loss, trgt_q_mean, critic_value, epsilon, rnd_pred_loss = Agent.update(inference_mode)
            elif Agent.extension_name == 'NGU':
                updated, critic_loss, trgt_q_mean, critic_value, epsilon = Agent.update(inference_mode)
            else:
                updated, critic_loss, trgt_q_mean, critic_value, epsilon = Agent.update(inference_mode)

        elif self.agent_config['extension']['name'] == 'QUOTA':
            updated, critic_loss, trgt_q_mean, critic_value= Agent.update(inference_mode)

        elif self.agent_config['agent_name'] == 'Agent57':
            updated, critic_loss, trgt_q_mean, critic_value= Agent.update(inference_mode)

        elif self.agent_config['agent_name'] == 'REDQ':
            if self.agent_config['extension']['name'] == 'ICM':
                updated, critic_loss, trgt_q_mean, critic_value= Agent.update(inference_mode)
            elif self.agent_config['extension']['name'] == 'RND':
                updated, critic_loss, trgt_q_mean, critic_value= Agent.update(inference_mode)
            elif self.agent_config['extension']['name'] == 'NGU':
                updated, critic_loss, trgt_q_mean, critic_value= Agent.update(inference_mode)
            else:
                updated, critic_loss, trgt_q_mean, critic_value= Agent.update(inference_mode)

        # Logging
        if self.agent_config['agent_name'] == 'DQN':
            if updated:
                self.summary_writer.add_scalar('01_Step/Epsilon', epsilon, Agent.update_step)
                self.summary_writer.add_scalar('02_Loss/Critic_loss', critic_loss, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Critic_value', critic_value, Agent.update_step)
                if Agent.extension_name == 'ICM':
                    self.summary_writer.add_scalar('04_ICM/intrinsic_reward', reward_int, Agent.update_step)
                    self.summary_writer.add_scalar('04_ICM/ICM_state_loss', icm_state_loss, Agent.update_step)
                    self.summary_writer.add_scalar('04_ICM/ICM_action_loss', icm_action_loss, Agent.update_step)
                elif Agent.extension_name == 'RND':
                    self.summary_writer.add_scalar('04_RND/intrinsic_reward', reward_int, Agent.update_step)
                    self.summary_writer.add_scalar('04_RND/RND_pred_loss', rnd_pred_loss, Agent.update_step)
                elif Agent.extension_name == 'NGU':
                    pass
                if self.agent_config['is_configurable_critic']:
                    if self.agent_config['critic_config']['network_config']['feature_extractor_config']['name'] in ('AutoEncoder1D', 'autoencoder1D', 'AE1D', 'AE1d', 'ae1D', 'ae1d', 'AutoEncoder2D', 'autoencoder2D', 'AE2D', 'AE2d', 'ae2D', 'ae2d'):
                        self.summary_writer.add_scalar('02_Loss/Recon_loss', recon_loss, Agent.update_step)

        elif self.agent_config['agent_name'] == 'PPO':
            if updated:
                self.summary_writer.add_scalar('02_Loss/Critic_1_loss', critic_loss, Agent.update_step)
                self.summary_writer.add_scalar('02_Loss/Actor_loss', actor_loss, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Critic_value', critic_value, Agent.update_step)

        elif self.agent_config['agent_name'] == 'SAC':
            if updated:
                self.summary_writer.add_scalar('02_Loss/Critic_1_loss', critic_loss, Agent.update_step)
                self.summary_writer.add_scalar('02_Loss/Actor_loss', actor_loss, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Critic_value', critic_value, Agent.update_step)

        elif self.agent_config['agent_name'] in ('QR_DQN', 'Safe_QR_DQN', 'IQN', 'Safe_IQN', 'MMDQN', 'Safe_MMDQN'):
            if updated:
                self.summary_writer.add_scalar('01_Step/Epsilon', epsilon, Agent.update_step)
                self.summary_writer.add_scalar('02_Loss/Critic_loss', critic_loss, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Critic_value', critic_value, Agent.update_step)
                if Agent.extension_name == 'ICM':
                    self.summary_writer.add_scalar('04_ICM/intrinsic_reward', reward_int, Agent.update_step)
                    self.summary_writer.add_scalar('04_ICM/ICM_state_loss', icm_state_loss, Agent.update_step)
                    self.summary_writer.add_scalar('04_ICM/ICM_action_loss', icm_action_loss, Agent.update_step)
                elif Agent.extension_name == 'RND':
                    self.summary_writer.add_scalar('04_RND/intrinsic_reward', reward_int, Agent.update_step)
                    self.summary_writer.add_scalar('04_RND/RND_pred_loss', rnd_pred_loss, Agent.update_step)
                elif Agent.extension_name == 'NGU':
                    pass

        elif self.agent_config['agent_name'] == 'QUOTA':
            if updated:
                self.summary_writer.add_scalar('02_Loss/Critic_1_loss', critic_loss, Agent.update_step)
                self.summary_writer.add_scalar('02_Loss/Actor_loss', actor_loss, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Critic_value', critic_value, Agent.update_step)

        elif self.agent_config['agent_name'] == 'Agent57':
            if updated:
                self.summary_writer.add_scalar('02_Loss/Critic_1_loss', critic_loss, Agent.update_step)
                self.summary_writer.add_scalar('02_Loss/Actor_loss', actor_loss, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Critic_value', critic_value, Agent.update_step)

        elif self.agent_config['agent_name'] == 'REDQ':
            if updated:
                self.summary_writer.add_scalar('02_Loss/Critic_1_loss', critic_loss, Agent.update_step)
                self.summary_writer.add_scalar('02_Loss/Actor_loss', actor_loss, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Critic_value', critic_value, Agent.update_step)


    def step_logging_wandb(self, Agent, reward_int, inference_mode: bool = False):
        # Update
        if self.agent_config['agent_name'] == 'DQN':
            if self.agent_config['is_configurable_critic']:
                if self.agent_config['critic_config']['network_config']['feature_extractor_config']['name'].lower() in ('autoencoder1d', 'ae1d', 'autoencoder2d', 'ae2d', 'autoencoder', 'ae'):
                    if Agent.extension_name == 'ICM':
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon, icm_state_loss, icm_action_loss, recon_loss = Agent.update(inference_mode)
                    elif Agent.extension_name == 'RND':
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon, rnd_pred_loss, recon_loss = Agent.update(inference_mode)
                    elif Agent.extension_name == 'NGU':
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon, recon_loss = Agent.update(inference_mode)
                    else:
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon, recon_loss = Agent.update(inference_mode)
                else:
                    if Agent.extension_name == 'ICM':
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon, icm_state_loss, icm_action_loss = Agent.update(inference_mode)
                    elif Agent.extension_name == 'RND':
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon, rnd_pred_loss = Agent.update(inference_mode)
                    elif Agent.extension_name == 'NGU':
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon = Agent.update(inference_mode)
                    else:
                        updated, critic_loss, trgt_q_mean, critic_value, epsilon = Agent.update(inference_mode)
            else:
                if Agent.extension_name == 'ICM':
                    updated, critic_loss, trgt_q_mean, critic_value, epsilon, icm_state_loss, icm_action_loss = Agent.update(inference_mode)
                elif Agent.extension_name == 'RND':
                    updated, critic_loss, trgt_q_mean, critic_value, epsilon, rnd_pred_loss = Agent.update(inference_mode)
                elif Agent.extension_name == 'NGU':
                    updated, critic_loss, trgt_q_mean, critic_value, epsilon = Agent.update(inference_mode)
                else:
                    updated, critic_loss, trgt_q_mean, critic_value, epsilon = Agent.update(inference_mode)

        elif self.agent_config['agent_name'] == 'Blank_DQN':
            if Agent.extension_name == 'ICM':
                updated, critic_loss, trgt_q_mean, critic_value, epsilon, icm_state_loss, icm_action_loss = Agent.update(inference_mode)
            elif Agent.extension_name == 'RND':
                updated, critic_loss, trgt_q_mean, critic_value, epsilon, rnd_pred_loss = Agent.update(inference_mode)
            elif Agent.extension_name == 'NGU':
                updated, critic_loss, trgt_q_mean, critic_value, epsilon = Agent.update(inference_mode)
            else:
                updated, critic_loss, trgt_q_mean, critic_value, epsilon = Agent.update(inference_mode)

        elif self.agent_config['agent_name'] == 'PPO':
            if self.agent_config['extension']['name'] == 'Model_Ensemble':
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value, critic_q_value = Agent.update(inference_mode)
            else:
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value, critic_q_value = Agent.update(inference_mode)

        elif self.agent_config['agent_name'] == 'SAC':
            if self.agent_config['extension']['name'] == 'TQC':
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value, critic_q_value = Agent.update(inference_mode)
            else:
                updated, actor_loss, critic_loss, trgt_q_mean, critic_value, critic_q_value = Agent.update(inference_mode)

        elif self.agent_config['agent_name'] in ('QR_DQN', 'Safe_QR_DQN', 'IQN', 'Safe_IQN', 'MMDQN', 'Safe_MMDQN'):
            if Agent.extension_name == 'ICM':
                updated, critic_loss, trgt_q_mean, critic_value, epsilon, icm_state_loss, icm_action_loss = Agent.update(inference_mode)
            elif Agent.extension_name == 'RND':
                updated, critic_loss, trgt_q_mean, critic_value, epsilon, rnd_pred_loss = Agent.update(inference_mode)
            elif Agent.extension_name == 'NGU':
                updated, critic_loss, trgt_q_mean, critic_value, epsilon = Agent.update(inference_mode)
            else:
                updated, critic_loss, trgt_q_mean, critic_value, epsilon = Agent.update(inference_mode)

        elif self.agent_config['extension']['name'] == 'QUOTA':
            updated, critic_loss, trgt_q_mean, critic_value= Agent.update(inference_mode)

        elif self.agent_config['agent_name'] == 'Agent57':
            updated, critic_loss, trgt_q_mean, critic_value= Agent.update(inference_mode)

        elif self.agent_config['agent_name'] == 'REDQ':
            if self.agent_config['extension']['name'] == 'ICM':
                updated, critic_loss, trgt_q_mean, critic_value= Agent.update(inference_mode)
            elif self.agent_config['extension']['name'] == 'RND':
                updated, critic_loss, trgt_q_mean, critic_value= Agent.update(inference_mode)
            elif self.agent_config['extension']['name'] == 'NGU':
                updated, critic_loss, trgt_q_mean, critic_value= Agent.update(inference_mode)
            else:
                updated, critic_loss, trgt_q_mean, critic_value= Agent.update(inference_mode)

        # Logging
        if self.agent_config['agent_name'] == 'DQN':
            if updated:
                self.wandb_session.log({
                    "01_Step/Epsilon": epsilon,
                    "02_Loss/Critic_loss": critic_loss,
                    '03_Critic/Target_Q_mean': trgt_q_mean, 
                    '03_Critic/Critic_value': critic_value
                }, step=Agent.update_step)
                if Agent.extension_name == 'ICM':
                    self.wandb_session.log({
                        "04_ICM/Intrinsic_reward": reward_int,
                        '04_ICM/ICM_state_loss': icm_state_loss, 
                        '04_ICM/ICM_action_loss': icm_action_loss
                    }, step=Agent.update_step)
                elif Agent.extension_name == 'RND':
                    self.wandb_session.log({
                        "04_RND/Intrinsic_reward": reward_int,
                        '04_RND/RND_pred_loss': rnd_pred_loss, 
                    }, step=Agent.update_step)
                elif Agent.extension_name == 'NGU':
                    pass

        elif self.agent_config['agent_name'] == 'PPO':
            if updated:
                self.wandb_session.log({
                    "02_Loss/Critic_1_loss": critic_loss,
                    '03_Critic/Target_Q_mean': trgt_q_mean, 
                    '03_Critic/Critic_value': critic_value
                }, step=Agent.update_step)
                
        elif self.agent_config['agent_name'] == 'SAC':
            if updated:
                self.wandb_session.log({
                    "02_Loss/Critic_1_loss": critic_loss,
                    '03_Critic/Target_Q_mean': trgt_q_mean, 
                    '03_Critic/Critic_value': critic_value
                }, step=Agent.update_step)

        elif self.agent_config['agent_name'] in ('QR_DQN', 'Safe_QR_DQN', 'IQN', 'Safe_IQN', 'MMDQN', 'Safe_MMDQN'):
            if updated:
                self.wandb_session.log({
                    "01_Step/Epsilon": epsilon,
                    "02_Loss/Critic_loss": critic_loss,
                    '03_Critic/Target_Q_mean': trgt_q_mean, 
                    '03_Critic/Critic_value': critic_value
                }, step=Agent.update_step)
                if Agent.extension_name == 'ICM':
                    self.wandb_session.log({
                        "04_ICM/Intrinsic_reward": reward_int,
                        '04_ICM/ICM_state_loss': icm_state_loss, 
                        '04_ICM/ICM_action_loss': icm_action_loss
                    }, step=Agent.update_step)
                elif Agent.extension_name == 'RND':
                    self.wandb_session.log({
                        "04_RND/Intrinsic_reward": reward_int,
                        '04_RND/RND_pred_loss': rnd_pred_loss, 
                    }, step=Agent.update_step)
                elif Agent.extension_name == 'NGU':
                    pass

        elif self.agent_config['agent_name'] == 'QUOTA':
            if updated:
                self.wandb_session.log({
                    "02_Loss/Critic_1_loss": critic_loss,
                    '03_Critic/Target_Q_mean': trgt_q_mean, 
                    '03_Critic/Critic_value': critic_value
                }, step=Agent.update_step)

        elif self.agent_config['agent_name'] == 'Agent57':
            if updated:
                self.wandb_session.log({
                    "02_Loss/Critic_1_loss": critic_loss,
                    '03_Critic/Target_Q_mean': trgt_q_mean, 
                    '03_Critic/Critic_value': critic_value
                }, step=Agent.update_step)

        elif self.agent_config['agent_name'] == 'REDQ':
            if updated:
                self.wandb_session.log({
                    "02_Loss/Critic_1_loss": critic_loss,
                    '03_Critic/Target_Q_mean': trgt_q_mean, 
                    '03_Critic/Critic_value': critic_value
                }, step=Agent.update_step)

    def episode_logging_tensorboard(self, Agent, episode_score, episode_step, episode_num, episode_rewards, inference_mode: bool = False):
        if self.agent_config['agent_name'] == 'PPO':
            if self.agent_config['extension']['name'] == 'MEPPO':
                updated, entropy, ratio, actor_loss, advantage, target_val, critic_value, critic_loss = Agent.update(inference_mode)
            elif self.agent_config['extension']['name'] == 'gSDE':
                updated, entropy, ratio, actor_loss, advantage, target_val, critic_value, critic_loss = Agent.update(inference_mode)
            else:
                updated, entropy, ratio, actor_loss, advantage, target_val, critic_value, critic_loss = Agent.update(inference_mode)

            if updated:
                self.summary_writer.add_scalar('02_Loss/Critic_loss', critic_loss, Agent.update_step)
                self.summary_writer.add_scalar('02_Loss/Actor_loss', actor_loss, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Advantage', advantage, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Target_value', target_val, Agent.update_step)
                self.summary_writer.add_scalar('03_Critic/Critic_value', critic_value, Agent.update_step)
                self.summary_writer.add_scalar('04_Actor/Entropy', entropy, Agent.update_step)
                self.summary_writer.add_scalar('04_Actor/Ratio', ratio, Agent.update_step)

        self.summary_writer.add_scalar('01_Episode/Score', episode_score, episode_num)
        self.summary_writer.add_scalar('01_Episode/Average_reward', episode_score/episode_step, episode_num)
        self.summary_writer.add_scalar('01_Episode/Steps', episode_step, episode_num)

        self.summary_writer.add_histogram('Reward_histogram', episode_rewards, episode_num)

    def episode_logging_wandb(self, Agent, episode_score, episode_step, episode_num, episode_rewards, inference_mode: bool = False):
        if self.agent_config['agent_name'] == 'PPO':
            if self.agent_config['extension']['name'] == 'MEPPO':
                updated, entropy, ratio, actor_loss, advantage, target_val, critic_value, critic_loss = Agent.update(inference_mode)
            elif self.agent_config['extension']['name'] == 'gSDE':
                updated, entropy, ratio, actor_loss, advantage, target_val, critic_value, critic_loss = Agent.update(inference_mode)
            else:
                updated, entropy, ratio, actor_loss, advantage, target_val, critic_value, critic_loss = Agent.update(inference_mode)

            if updated:
                self.wandb_session.log({
                    '02_Loss/Critic_loss': critic_loss,
                    '02_Loss/Actor_loss': actor_loss, 
                    '03_Critic/Advantage': advantage,
                    '03_Critic/Target_value': target_val,
                    '03_Critic/Critic_value': critic_value,
                    '04_Actor/Entropy': entropy,
                    '04_Actor/Ratio': ratio
                }, step=Agent.update_step)

        self.wandb_session.log({
            '01_Episode/Average_reward': episode_score/episode_step,
            "01_Episode/Score": episode_score,
            '01_Episode/Steps': episode_step,
            "episode_num": episode_num
        })

        histogram = wandb.Histogram(episode_rewards)
        self.wandb_session.log({"Reward_histogram": histogram})

    def eval_logging_tensorboard(self, Agent, episode_score, episode_step, episode_num):
        if self.agent_config['agent_name'] == 'PPO':
            if self.agent_config['extension']['name'] == 'MEPPO':
                pass # Todo: Consider the PPO which update in every episode
            elif self.agent_config['extension']['name'] == 'gSDE':
                pass # Todo: Consider the PPO which update in every episode
            else:
                pass # Todo: Consider the PPO which update in every episode

        self.summary_writer.add_scalar('00_Eval/Score', episode_score, episode_num)
        self.summary_writer.add_scalar('00_Eval/Average_reward', episode_score/episode_step, episode_num)
        self.summary_writer.add_scalar('00_Eval/Steps', episode_step, episode_num)

    def eval_logging_wandb(self, Agent, episode_score, episode_step, episode_num):
        if self.agent_config['agent_name'] == 'PPO':
            if self.agent_config['extension']['name'] == 'MEPPO':
                pass # Todo: Consider the PPO which update in every episode
            elif self.agent_config['extension']['name'] == 'gSDE':
                pass # Todo: Consider the PPO which update in every episode
            else:
                pass # Todo: Consider the PPO which update in every episode

        self.wandb_session.log({
            '00_Eval/Average_reward': episode_score/episode_step,
            "00_Eval/Score": episode_score,
            '00_Eval/Steps': episode_step,
            "episode_num": episode_num
        })
