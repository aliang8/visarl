import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
import numpy as np

from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

import sys

from torch.nn.modules.batchnorm import BatchNorm1d

sys.path.append(constants.MMAE)
from run_pretraining_multimae import DOMAIN_CONF
from torchvision import transforms

import os
import util.constants as constants
from src.modules.picanet import Unet
from train_encoder_picanet import picanet_cfg


def _get_out_shape_cuda(in_shape, layers):
    x = torch.randn(*in_shape).cuda().unsqueeze(0)
    return layers(x).squeeze(0).shape


def _get_out_shape(in_shape, layers):
    x = torch.randn(*in_shape).unsqueeze(0)
    return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability"""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Truncated normal distribution, see https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf"""

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class CenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        # assert size in {84, 100}, f"unexpected size: {size}"
        self.size = size

    def forward(self, x):
        # import ipdb

        # ipdb.set_trace()
        assert x.ndim == 4, "input must be a 4D tensor"
        if x.size(2) == self.size and x.size(3) == self.size:
            return x
        assert x.size(3) == 100, f"unexpected size: {x.size(3)}"
        if self.size == 84:
            p = 8
        return x[:, :, p:-p, p:-p]


class NormalizeImg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / 255.0


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class RLProjection(nn.Module):
    def __init__(self, in_shape, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_shape[0], out_dim), nn.LayerNorm(out_dim), nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, x):
        y = self.projection(x)
        return y


class SharedCNN(nn.Module):
    def __init__(self, obs_shape, num_layers=11, num_filters=32):
        super().__init__()
        assert len(obs_shape) == 3
        self.num_layers = num_layers
        self.num_filters = num_filters

        self.layers = [
            CenterCrop(size=64),
            NormalizeImg(),
            nn.Conv2d(obs_shape[0], num_filters, 3, stride=2),
        ]
        for _ in range(1, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(obs_shape, self.layers)
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x)


class HeadCNN(nn.Module):
    def __init__(self, in_shape, num_layers=0, num_filters=32):
        super().__init__()
        self.layers = []
        for _ in range(0, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers.append(Flatten())
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(in_shape, self.layers)
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x)


class MMAEFeatureExtractor(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
        self.out_dim = 256

        self.tasks_loss_fn = {
            domain: DOMAIN_CONF[domain]["loss"](
                patch_size=self.args.patch_size,
                stride=DOMAIN_CONF[domain]["stride_level"],
            )
            for domain in self.args.out_domains
        }

        if self.args.extra_norm_pix_loss:
            self.tasks_loss_fn["norm_rgb"] = DOMAIN_CONF["rgb"]["loss"](
                patch_size=args.patch_size,
                stride=DOMAIN_CONF["rgb"]["stride_level"],
                norm_pix=True,
            )

        self.transforms = transforms.Compose(
            [
                lambda x: x / 255.0,
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # load saliency predictor
        if args.use_saliency_input:
            cfg = picanet_cfg[args.picanet_cfg]
            self.saliency_predictor = Unet(cfg).eval().cuda()
            ckpt_file = os.path.join(
                constants.ROOT,
                f"pretrained_models/{self.args.task_name}/fixation_model_picanet_cfg_{args.picanet_cfg}.pt",
            )
            print(f"loading saliency predictor from: {ckpt_file}")
            self.saliency_predictor.load_state_dict(torch.load(ckpt_file))

            self.upsample_m = torch.nn.modules.upsampling.Upsample(
                scale_factor=(224 / 64, 224 / 64), mode="nearest"
            ).to("cuda")
            self.downsample_m = torch.nn.modules.upsampling.Upsample(
                scale_factor=(64 / 224, 64 / 224), mode="nearest"
            ).to("cuda")

    def compute_loss(self, observations, saliency=None):
        x = {"rgb": observations}

        if saliency is not None:
            x["saliency"] = saliency

        tasks_dict = {
            task: tensor.to("cuda", non_blocking=True) for task, tensor in x.items()
        }

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in self.args.in_domains
        }

        # our observation size is 64 x 64, patch size of 8
        # each modality has 64 tokens * 2 = 128, if we mask 0.75 then we encode 32 tokens
        num_encoded_tokens = 32
        alphas = 1.0

        loss_on_unmasked = False

        preds, masks = self.model(
            input_dict,
            mask_inputs=True,
            num_encoded_tokens=num_encoded_tokens,
            alphas=alphas,
            output_latent=False,
        )

        if self.args.extra_norm_pix_loss:
            tasks_dict["norm_rgb"] = tasks_dict["rgb"]
            masks["norm_rgb"] = masks.get("rgb", None)

        task_losses = {}
        for task in preds:
            if task == "saliency" and saliency is None:
                continue

            target = tasks_dict[task]

            if loss_on_unmasked:
                task_losses[task] = self.tasks_loss_fn[task](
                    preds[task].float(), target
                )
            else:
                task_losses[task] = self.tasks_loss_fn[task](
                    preds[task].float(), target, mask=masks.get(task, None)
                )

        loss = sum(task_losses.values())
        return loss, task_losses

    def forward(self, observation, saliency=None):
        if saliency is None and self.args.use_saliency_input:
            # compute saliency using saliency predictor and use
            # as input into model
            with torch.no_grad():
                inp_ = observation / 255.0
                inp_ = self.upsample_m(inp_)
                pred, _ = self.saliency_predictor(inp_)
                # B 1 H W
                saliency = self.downsample_m(pred[-1])

        # apply preprocessing to the input frames 0-255
        observation = self.transforms(observation)

        input_dict = {}
        input_dict["rgb"] = observation

        if saliency is not None:
            input_dict["saliency"] = saliency

        # there are 64 tokens total, do not mask anything for this
        num_encoded_tokens = 64  # the number of visible tokens
        if saliency is not None:
            num_encoded_tokens = 128

        alphas = 1.0  # Dirichlet concentration parameter

        latents, _ = self.model.forward(
            input_dict,
            mask_inputs=False,  # True if forward pass should sample random masks
            num_encoded_tokens=num_encoded_tokens,
            alphas=alphas,
            output_latent=True,
        )
        # take average
        output_embedding = latents.mean(dim=1)

        return output_embedding


class Encoder(nn.Module):
    def __init__(self, shared_cnn, head_cnn, projection):
        super().__init__()
        self.shared_cnn = shared_cnn
        self.head_cnn = head_cnn
        self.projection = projection
        self.out_dim = projection.out_dim

    def forward(self, x, detach=False):
        x = self.shared_cnn(x)
        x = self.head_cnn(x)
        if detach:
            x = x.detach()
        return self.projection(x)


class Actor(nn.Module):
    def __init__(
        self,
        input_dim,
        action_shape,
        hidden_dim,
        log_std_min,
        log_std_max,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0]),
        )
        self.mlp.apply(weight_init)
        # self.norm_layer = partial(nn.LayerNorm, eps=1e-6)(1024)

    def forward(
        self,
        x,
        compute_pi=True,
        compute_log_pi=True,
        detach=False,
        compute_attrib=False,
    ):
        mu, log_std = self.mlp(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weight_init)

    def forward(self, obs, action):
        if action is not None:

            assert obs.size(0) == action.size(0)
            return self.trunk(torch.cat([obs, action], dim=1))
        else:
            return self.trunk(obs)


def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, detach=False):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        q1, q2 = forward(self, observations, actions, detach)
        if multiple_actions:
            q1 = q1.reshape(batch_size, -1)
            q2 = q2.reshape(batch_size, -1)

        return q1, q2

    return wrapped


class Critic(nn.Module):
    def __init__(self, input_dim, action_shape, hidden_dim):
        super().__init__()
        self.Q1 = QFunction(input_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(input_dim, action_shape[0], hidden_dim)

    @multiple_action_q_function
    def forward(self, x, action, detach=False):
        return self.Q1(x, action), self.Q2(x, action)


class ValueFunction(nn.Module):
    def __init__(self, input_dim, action_shape, hidden_dim):
        super().__init__()
        self.Q1 = QFunction(input_dim, 0, hidden_dim)

    def forward(self, x, detach=False):
        return self.Q1(x, None)


def extend_and_repeat(tensor, dim, repeat):
    # Extend and repeast the tensor along dim axie and repeat it
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)


def soft_target_update(network, target_network, soft_target_update_rate):
    target_network_params = {k: v for k, v in target_network.named_parameters()}
    for k, v in network.named_parameters():
        target_network_params[k].data = (
            1 - soft_target_update_rate
        ) * target_network_params[k].data + soft_target_update_rate * v.data


def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values

    return wrapped


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, arch="256-256", orthogonal_init=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init

        d = input_dim
        modules = []
        hidden_sizes = [int(h) for h in arch.split("-")]

        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            if orthogonal_init:
                nn.init.orthogonal_(fc.weight, gain=np.sqrt(2))
                nn.init.constant_(fc.bias, 0.0)
            modules.append(fc)
            modules.append(nn.ReLU())
            d = hidden_size

        last_fc = nn.Linear(d, output_dim)
        if orthogonal_init:
            nn.init.orthogonal_(last_fc.weight, gain=1e-2)
        else:
            nn.init.xavier_uniform_(last_fc.weight, gain=1e-2)

        nn.init.constant_(last_fc.bias, 0.0)
        modules.append(last_fc)

        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor):
        return self.network(input_tensor)


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(self, log_std_min=-20.0, log_std_max=2.0, no_tanh=False):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(self, mean, log_std, sample):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(self, mean, log_std, deterministic=False):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        arch="256-256",
        log_std_multiplier=1.0,
        log_std_offset=-1.0,
        orthogonal_init=False,
        no_tanh=False,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = FullyConnectedNetwork(
            observation_dim, 2 * action_dim, arch, orthogonal_init
        )
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations, actions):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, observations, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian(mean, log_std, deterministic)
