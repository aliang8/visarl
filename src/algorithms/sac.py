import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import util.general_utils as utils
import src.algorithms.modules as m
import wandb
from src.modules.mmae import initialize_mmae
import os
import util.constants as constants


class SAC(nn.Module):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__()
        self.args = args
        self.discount = args.discount
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.encoder_tau
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.encoder = self.build_encoders()

        if self.encoder is not None:
            input_dim = self.encoder.out_dim
        else:
            if args.pretrained_encoder:
                input_dim = constants.encoder_to_dim_mapping[args.encoder_type]
            else:
                input_dim = obs_shape[0]

        self.actor = m.Actor(
            input_dim,
            action_shape,
            args.hidden_dim,
            args.actor_log_std_min,
            args.actor_log_std_max,
        ).cuda()
        self.critic = m.Critic(input_dim, action_shape, args.hidden_dim).cuda()
        self.critic_target = deepcopy(self.critic)

        self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=args.encoder_lr
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=args.critic_lr,
            betas=(args.critic_beta, 0.999),
            weight_decay=args.critic_weight_decay,
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def build_encoders(self):
        if "image" in self.args.observation_type:
            if self.args.encoder_type == "mmae":
                multimae = initialize_mmae(
                    load_from_ckpt=self.args.load_encoder_from_ckpt,
                    ckpt_path=self.args.encoder_ckpt_path,
                )
                encoder = m.MMAEFeatureExtractor(self.args, multimae)
                encoder = encoder.cuda()
            elif self.args.encoder_type == "cnn":
                shared_cnn = m.SharedCNN(
                    self.obs_shape, self.args.num_shared_layers, self.args.num_filters
                ).cuda()
                head_cnn = m.HeadCNN(
                    shared_cnn.out_shape,
                    self.args.num_head_layers,
                    self.args.num_filters,
                ).cuda()
                encoder = m.Encoder(
                    shared_cnn,
                    head_cnn,
                    m.RLProjection(head_cnn.out_shape, self.args.projection_dim),
                )
                encoder = encoder.cuda()
            else:
                encoder = None
        else:
            encoder = None
        return encoder

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def eval(self):
        self.train(False)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _obs_to_input(self, obs):
        if isinstance(obs, utils.LazyFrames):
            _obs = np.array(obs)
        else:
            _obs = obs

        if isinstance(_obs, torch.Tensor):
            _obs = _obs.float().cuda()
        else:
            _obs = torch.FloatTensor(_obs).cuda()
        if _obs.shape[0] != 1:
            _obs = _obs.unsqueeze(0)
        return _obs

    def select_action(self, obs):
        _obs = self._obs_to_input(obs)

        with torch.no_grad():
            if (
                "image" in self.args.observation_type
                and self.encoder is not None
                and not self.args.pretrained_encoder
            ):
                _obs = self.encoder(_obs)

            mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        _obs = self._obs_to_input(obs)

        with torch.no_grad():
            if (
                "image" in self.args.observation_type
                and self.encoder is not None
                and not self.args.pretrained_encoder
            ):
                _obs = self.encoder(_obs)

            mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        if L is not None:
            L.log("train_critic/loss", critic_loss, step)

            if self.args.use_wandb:
                wandb.log(
                    {
                        "train/critic_loss": critic_loss,
                        "train/max_Q1": current_Q1.max(),
                        "train/max_Q2": current_Q2.max(),
                        "train/min_Q1": current_Q1.min(),
                        "train/min_Q2": current_Q2.min(),
                    },
                    step=step,
                )

        # update critic and encoder
        if not self.args.pretrained_encoder:
            self.encoder_optimizer.zero_grad(set_to_none=True)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if not self.args.pretrained_encoder:
            self.encoder_optimizer.step()

    def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
        _, pi, log_pi, log_std = self.actor(obs)
        actor_Q1, actor_Q2 = self.critic(obs, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            L.log("train_actor/loss", actor_loss, step)
            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
                dim=-1
            )
        else:
            entropy = 0.0

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log("train_alpha/loss", alpha_loss, step)
                L.log("train_alpha/value", self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0)
            self.alpha = torch.tensor(0.0)

        if self.args.use_wandb:
            wandb.log(
                {
                    "train/actor_loss": actor_loss,
                    "train/alpha_loss": alpha_loss,
                    "train/alpha": self.alpha,
                    "train/entropy": entropy,
                },
                step=step,
            )

        return actor_loss, alpha_loss, self.alpha

    def soft_update_critic_target(self):
        utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
        utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)

    def update(self, replay_buffer, L, step, oracle_agent=None, latents=False):
        batch = next(replay_buffer)
        obs, action, reward, done, next_obs = utils.to_torch(batch, "cuda")
        not_done = 1.0 - done

        if self.args.observation_type == "image" and not self.args.pretrained_encoder:
            obs = self.encoder(obs)

            with torch.no_grad():
                next_obs = self.encoder(next_obs)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            if self.args.observation_type == "image":
                obs = obs.detach()

            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

    def load(self, model_dir, step=None):
        if step is None:
            ckpt_path = sorted(glob.glob(f"{model_dir}/model/*.pt"))[-1]

        self.load_state_dict(torch.load(ckpt_path), strict=True)

        print(f"loading from : {ckpt_path}")
