import torchvision.transforms.functional as TF
from functools import partial
import sys
import torch
import os
from einops import rearrange
import wandb

import util.constants as constants
import util.general_utils as utils

import sys

sys.path.append(constants.ROOT)
sys.path.append(constants.SRC)
sys.path.append(constants.MMAE)
from multimae.input_adapters import PatchedInputAdapter
from multimae.output_adapters import SpatialOutputAdapter
from multimae.multimae import pretrain_multimae_base, pretrain_multimae_large
from multimae.multimae import MultiMAE
from utils.data_constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.algorithms.sac import SAC
from torchvision import transforms
import numpy as np
import src.algorithms.modules as m
import torchshow as ts
import matplotlib.pyplot as plt


class SAC_MAE(SAC):
    """
    Instead of learning the encoder, use fixed visual representation
    from a pretrained MultiMAE model.
    """

    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

        tb_dir = os.path.join(
            args.log_dir,
            args.domain_name + "_" + args.task_name,
            args.algorithm,
            str(args.seed),
            "tensorboard",
        )

        # Load pretrained m MultiMAE model
        if self.args.multimae:
            print("loading mmae model")
            DOMAIN_CONF = {
                "rgb": {
                    "channels": 3,
                    "stride_level": 1,
                    "input_adapter": partial(
                        PatchedInputAdapter, num_channels=3, stride_level=1
                    ),
                    "output_adapter": partial(
                        SpatialOutputAdapter, num_channels=3, stride_level=1
                    ),
                },
                "saliency": {
                    "channels": 1,
                    "stride_level": 1,
                    "input_adapter": partial(
                        PatchedInputAdapter, num_channels=1, stride_level=1
                    ),
                    "output_adapter": partial(
                        SpatialOutputAdapter, num_channels=1, stride_level=1
                    ),
                },
            }

            DOMAINS = ["rgb", "saliency"]

            input_adapters = {
                domain: dinfo["input_adapter"](
                    patch_size_full=16,
                )
                for domain, dinfo in DOMAIN_CONF.items()
            }
            output_adapters = {
                domain: dinfo["output_adapter"](
                    patch_size_full=16,
                    dim_tokens=256,
                    use_task_queries=True,
                    depth=2,
                    context_tasks=DOMAINS,
                    task=domain,
                )
                for domain, dinfo in DOMAIN_CONF.items()
            }

            # don't need output adapters if we are just using it for encoding
            pretrained_mdl_cls = (
                pretrain_multimae_base
                if args.pretrained_model == "base"
                else pretrain_multimae_large(input_adapters, output_adapters)
            )
            multimae = pretrained_mdl_cls(
                input_adapters=input_adapters,
                output_adapters=None,
            )

            input_adapters = {
                domain: dinfo["input_adapter"](
                    patch_size_full=16,
                )
                for domain, dinfo in DOMAIN_CONF.items()
            }
            output_adapters = {
                domain: dinfo["output_adapter"](
                    patch_size_full=16,
                    dim_tokens=256,
                    use_task_queries=True,
                    depth=2,
                    context_tasks=DOMAINS,
                    task=domain,
                )
                for domain, dinfo in DOMAIN_CONF.items()
            }

            if args.pretrained_model == "base":
                multimae.out_dim = 768
            else:
                multimae.out_dim = 1024

            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt_file = (
                f"{constants.MMAE}/output_dir/{args.task_name}/checkpoint-233.pth"
            )
            ckpt = torch.load(ckpt_file)

            multimae.load_state_dict(ckpt["model"], strict=False)
            encoder = multimae.to(device).eval().cuda()

        encoder.transform = transforms.Compose(
            [
                transforms.Resize(224),
                lambda x: x / 255.0,
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.encoder = encoder

    def log_tensorboard(self, obs, action, step, prefix="original"):
        selected_obs = obs[0:1]

        if self.args.multimae:

            with torch.no_grad():
                input_dict = {}
                input_dict["rgb"] = self.multimae_viz.transform(selected_obs).cuda()
                saliency = torch.zeros_like(input_dict["rgb"])[:, 0:1]
                input_dict["saliency"] = saliency

                num_encoded_tokens = 392  # the number of visible tokens
                alphas = 1.0  # Dirichlet concentration parameter

                preds, masks = self.multimae_viz.forward(
                    input_dict,
                    mask_inputs=True,  # True if forward pass should sample random masks
                    num_encoded_tokens=num_encoded_tokens,
                    alphas=alphas,
                )

                masks_ = masks
                masks_["saliency"].fill_(1)

                preds, masks = self.multimae_viz.forward(
                    input_dict,
                    mask_inputs=True,  # True if forward pass should sample random masks
                    task_masks={k: v.to("cuda") for k, v in masks_.items()},
                )

                mask_rgb = utils.get_masked_image(
                    denormalize(input_dict["rgb"]), masks_["rgb"]
                )
                mask_saliency = utils.get_masked_image(
                    input_dict["saliency"], masks_["saliency"]
                )

                pred_img = utils.get_pred_with_input(
                    denormalize(input_dict["rgb"]), preds["rgb"], masks_["rgb"]
                )
                pred_saliency = utils.get_pred_with_input(
                    input_dict["saliency"], preds["saliency"], masks_["saliency"]
                ).cuda()

                if self.args.use_wandb:
                    to_show = list(
                        zip(
                            selected_obs,
                            mask_rgb,
                            pred_img,
                            saliency,
                            mask_saliency,
                        )
                    )
                    to_show = [list(elem) for elem in to_show]
                    ts.show(to_show, figsize=(24, 24))
                    wandb.log({"train/reconstructions": plt})

                    to_show = list(
                        zip(
                            saliency.repeat(1, 3, 1, 1) * input_dict["rgb"],
                            pred_saliency.repeat(1, 3, 1, 1) * input_dict["rgb"],
                            pred_saliency.repeat(1, 3, 1, 1) * input_dict["rgb"],
                            pred_saliency * pred_img.cuda(),
                        )
                    )

                    to_show = [list(elem) for elem in to_show]
                    ts.show(to_show, figsize=(24, 24))
                    wandb.log({"train/reconstructions_pred": plt})

                else:
                    self.writer.add_image(
                        "train/orig", make_obs_grid(selected_obs), global_step=step
                    )
                    self.writer.add_image(
                        "train/mask", make_obs_grid(mask_rgb * 255.0), global_step=step
                    )
                    self.writer.add_image(
                        "train/pred_img",
                        make_obs_grid(denormalize(pred_img) * 255.0),
                        global_step=step,
                    )
                    self.writer.add_image(
                        "train/pred_saliency",
                        make_obs_grid(pred_saliency * 255.0),
                        global_step=step,
                    )
                    # self.writer.add_image(
                    #     "train/saliency_x_img",
                    #     make_obs_grid(saliency_x_img * 255.0),
                    #     global_step=step,
                    # )
        else:
            model = self.encoder
            selected_obs = model.transform(selected_obs)

            with torch.no_grad():
                loss, y, mask = model(selected_obs.float(), mask_ratio=0)

                y = model.unpatchify(y)

                mask = mask.unsqueeze(-1).repeat(
                    1, 1, model.patch_embed.patch_size[0] ** 2 * 3
                )
                mask = model.unpatchify(mask)

                # masked image
                im_masked = denormalize(selected_obs) * (1 - mask)

                # MAE reconstruction pasted with visible patches
                im_paste = (
                    denormalize(selected_obs) * (1 - mask) + denormalize(y) * mask
                )
                self.writer.add_image(
                    "train/orig",
                    make_obs_grid(denormalize(selected_obs) * 255.0),
                    global_step=step,
                )
                self.writer.add_image(
                    "train/mask", make_obs_grid(mask * 255.0), global_step=step
                )
                self.writer.add_image(
                    "train/im_masked",
                    make_obs_grid(im_masked * 255.0),
                    global_step=step,
                )
                self.writer.add_image(
                    "train/im_paste", make_obs_grid(im_paste * 255.0), global_step=step
                )

    def build_encoders(self):
        actor_encoder = m.Encoder(
            torch.nn.Identity(),
            torch.nn.Identity(),
            m.RLProjection((768,), self.args.projection_dim),
        )
        critic_encoder = m.Encoder(
            torch.nn.Identity(),
            torch.nn.Identity(),
            m.RLProjection((768,), self.args.projection_dim),
        )
        return actor_encoder, critic_encoder

    def update(self, replay_buffer, L, step, oracle_agent=None):
        (
            obs,
            action,
            reward,
            next_obs,
            not_done,
            latents,
            next_latents,
        ) = replay_buffer.sample_sacai()

        self.update_critic(latents, action, reward, next_latents, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(latents, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
