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
from src.modules.picanet import Unet
from train_encoder_picanet import picanet_cfg
from src.algorithms.sac import SAC


class SACMaskedObs(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

        # use saliency to mask the observation
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

    def apply_saliency(self, obs):
        # apply saliency predictor
        with torch.no_grad():
            inp_ = obs / 255.0
            inp_ = self.upsample_m(inp_)
            pred, _ = self.saliency_predictor(inp_)
            # B 1 H W
            saliency = self.downsample_m(pred[-1])

        # multiply saliency by RGB
        if self.args.rgb_x_saliency:
            out = obs * saliency.repeat(1, 3, 1, 1)
        else:
            out = saliency.repeat(1, 3, 1, 1)

        if self.args.saliency_as_depth_channel:
            out = torch.cat([obs, saliency], dim=1)
        return out
