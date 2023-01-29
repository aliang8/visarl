import torch
import numpy as np
import os
import glob
import json
import random
import subprocess
import pickle
from datetime import datetime

import time
import sys
import util.constants as constants

sys.path.append(constants.ROOT)
sys.path.append(constants.SRC)
sys.path.append(constants.MMAE)

from utils.data_constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def cat(x, y, axis=0):
    return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_info(args, fp):
    data = {
        "timestamp": str(datetime.now()),
        "git": subprocess.check_output(["git", "describe", "--always"])
        .strip()
        .decode(),
        "args": vars(args),
    }
    with open(fp, "w") as f:
        json.dump(data, f, indent=4, separators=(",", ": "))


def load_config(key=None):
    path = os.path.join("setup", "config.cfg")
    with open(path) as f:
        data = json.load(f)
    if key is not None:
        return data[key]
    return data


def make_dir(dir_path, exist_ok=False):
    try:
        os.makedirs(dir_path, exist_ok=exist_ok)
    except OSError:
        pass
    return dir_path


def listdir(dir_path, filetype="jpg", sort=True):
    fpath = os.path.join(dir_path, f"*.{filetype}")
    fpaths = glob.glob(fpath, recursive=True)
    if sort:
        return sorted(fpaths)
    return fpaths


def prefill_memory(obses, capacity, obs_shape, dtype=np.uint8):
    """Reserves memory for replay buffer"""
    for _ in range(capacity):
        frame = np.ones(obs_shape, dtype=dtype)
        obses.append(frame)
    return obses


def prefill_memory_latent(obses, capacity, obs_shape, dtype=np.uint8):
    """Reserves memory for replay buffer"""
    for _ in range(capacity):
        frame = np.ones(obs_shape, dtype=dtype)
        obses.append(frame)
    return obses


class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0] // 3

    def frame(self, i):
        return self._force()[i * 3 : (i + 1) * 3]


def count_parameters(net, as_int=False):
    """Returns total number of params in a network"""
    count = sum(p.numel() for p in net.parameters())
    if as_int:
        return count
    return f"{count:,}"


def denormalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    return TF.normalize(
        img.clone(), mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )


def get_masked_image(img, mask, image_size=224, patch_size=16, mask_value=0.0):
    img_token = rearrange(
        img.detach().cpu(),
        "b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
        nh=image_size // patch_size,
        nw=image_size // patch_size,
    )
    img_token[mask.detach().cpu() != 0] = mask_value
    img = rearrange(
        img_token,
        "b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)",
        ph=patch_size,
        pw=patch_size,
        nh=image_size // patch_size,
        nw=image_size // patch_size,
    )
    return img


def get_pred_with_input(gt, pred, mask, image_size=224, patch_size=16):
    gt_token = rearrange(
        gt.detach().cpu(),
        "b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
        nh=image_size // patch_size,
        nw=image_size // patch_size,
    )
    pred_token = rearrange(
        pred.detach().cpu(),
        "b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
        nh=image_size // patch_size,
        nw=image_size // patch_size,
    )
    pred_token[mask.detach().cpu() == 0] = gt_token[mask.detach().cpu() == 0]
    img = rearrange(
        pred_token,
        "b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)",
        ph=patch_size,
        pw=patch_size,
        nh=image_size // patch_size,
        nw=image_size // patch_size,
    )
    return img


import h5py


def load_h5_dataset(dataset_file, task_name):
    dataset = {}
    with h5py.File(dataset_file, "r") as f:
        demos = f[task_name]
        for i in range(len(demos)):
            for i in range(len(demos)):
                dataset["observations"] = np.array(demos[f"demo_{i}/frames"])
                dataset["states"] = np.array(demos[f"demo_{i}/states"])
                dataset["actions"] = np.array(demos[f"demo_{i}/actions"])
                dataset["rewards"] = np.array(demos[f"demo_{i}/rewards"])
                dataset["terminals"] = np.array(demos[f"demo_{i}/dones"])
                dataset["next_observations"] = np.array(demos[f"demo_{i}/frames"])[1:]
    return dataset


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def parition_batch_train_test(batch, train_ratio):
    train_indices = (
        np.random.rand(np.array(batch["observations"]).shape[0]) < train_ratio
    )
    train_batch = index_batch(batch, train_indices)
    test_batch = index_batch(batch, ~train_indices)
    return train_batch, test_batch


def subsample_batch(batch, size):
    indices = np.random.randint(batch["observations"].shape[0], size=size)
    return index_batch(batch, indices)


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate(
            [batch[key] for batch in batches], axis=0
        ).astype(np.float32)
    return concatenated


def prefix_metrics(metrics, prefix):
    return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}


class Timer(object):
    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


def batch_to_torch(batch, device):
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True)
        for k, v in batch.items()
    }


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def create_video_grid(videos, height=64, width=64, max_columns=5):
    # The input expects list of videos in BHWC
    # wandb needs videos to be in BxCxHxW

    # assert len(videos) % max_columns == 0
    if len(videos) % max_columns != 0:
        # need to pad with some black videos
        extra_videos_needed = ((len(videos) // max_columns) + 1) * max_columns - len(
            videos
        )
        for i in range(extra_videos_needed):
            videos.append(np.zeros_like(videos[0]))

    assert len(videos) % max_columns == 0

    max_seq_length = max([video.shape[0] for video in videos])

    # first resize videos and pad them to max length
    for i, video in enumerate(videos):
        all_frames = []
        for frame in video:
            if frame.shape[0] == 1:
                frame = (
                    frame.reshape((frame.shape[1], frame.shape[2], 1)).repeat(
                        3, axis=-1
                    )
                    * 256
                ).astype(np.uint8)
            frame = Image.fromarray(frame)
            frame = np.array(frame)
            # frame = np.array(frame.resize((height, width)))
            all_frames.append(frame)
        all_frames = np.array(all_frames).transpose(0, 3, 1, 2)
        video = all_frames

        if video.shape[0] < max_seq_length:
            padded_video = np.zeros((max_seq_length, *all_frames.shape[1:]))
            padded_video[: video.shape[0]] = video
            videos[i] = padded_video
        else:
            videos[i] = video

    max_columns = 5
    num_rows = int(len(videos) / max_columns)
    chunks = list(split(videos, num_rows))

    rows = []
    for chunk in chunks:
        # stick the videos into a grid and concatenate the videos on width
        row = np.concatenate(chunk, axis=-1)
        rows.append(row)

    videos = np.concatenate(rows, axis=-2)  # concat over height
    return videos


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
