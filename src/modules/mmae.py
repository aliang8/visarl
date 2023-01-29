import sys
import util.constants as constants

sys.path.append(constants.ROOT)
sys.path.append(constants.SRC)
sys.path.append(constants.MMAE)

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from functools import partial
import sys
import torch
from einops import rearrange
from multimae.input_adapters import (
    PatchedInputAdapter,
    SemSegInputAdapter,
    PatchedInputAdapterConvStem,
)
from multimae.output_adapters import SpatialOutputAdapter
from multimae.multimae import (
    pretrain_multimae_base,
    pretrain_multimae_large,
    pretrain_multimae_small,
)
from multimae.criterion import MaskedCrossEntropyLoss, MaskedL1Loss, MaskedMSELoss

import utils
from utils.data_constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from run_pretraining_multimae import get_model


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def initialize_mmae(load_from_ckpt=True, ckpt_path=""):
    if load_from_ckpt:
        print(f"load encoder from {ckpt_path} ...")
        state_dict = torch.load(ckpt_path)
        model = get_model(state_dict["args"])
        utils.checkpoint.load_state_dict(model, state_dict["model"])
    else:
        args = AttrDict(
            patch_size=8,
            decoder_dim=256,
            decoder_depth=3,
            decoder_num_heads=4,
            decoder_use_task_queries=True,
            in_domains=["rgb"],
            out_domains=["rgb"],
            decoder_use_xattn=True,
            model="pretrain_multimae_small",
            extra_norm_pix_loss=False,
            num_global_tokens=1,
            drop_path=0.0,
        )
        model = get_model(args)

    model = model.cuda()

    return model