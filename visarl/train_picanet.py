"""
Script to train a saliency predictor model on a dataset of annotated frames.    
"""


import os
import cv2
import tqdm
import pickle
import sys
import torch
import json
import glob
import numpy as np
import argparse
import subprocess
import random
import torchvision
import einops
from pathlib import Path

from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics import Dice

from functools import partial
import torchshow as ts
import torchvision.transforms as T

import visarl.util.constants as constants
import sys
from pprint import pprint
from visarl.src.modules.picanet import Unet
from visarl.util.saliency_util import get_heatmap_from_ann_file
from visarl.util.general_utils import chunks, count_trainable_parameters, MetricMeter

picanet_cfg = {
    0: {
        "PicaNet": "GGLLL",
        "Size": [28, 28, 28, 56, 112, 224],
        "Channel": [1024, 512, 512, 256, 128, 64],
        "loss_ratio": [0.5, 0.5, 0.5, 0.8, 0.8, 1],
        "en_indices": [5, 4, 3, 2, 1, 0],
    },
    1: {
        "PicaNet": "GLLL",
        "Size": [28, 28, 56, 112, 224],
        "Channel": [512, 512, 256, 128, 64],
        "loss_ratio": [0.5, 0.5, 0.8, 0.8, 1],
        "en_indices": [5, 4, 3, 2, 1, 0],
    },
    2: {
        "PicaNet": "LLL",
        "Size": [28, 56, 112, 224],
        "Channel": [512, 256, 128, 64],
        "loss_ratio": [0.5, 0.8, 0.8, 1],
        "en_indices": [3, 2, 1, 0],
    },
}


class Trainer:
    def __init__(self, args):
        self.args = args
        root_dir = os.environ["ROOT_DIR"]

        # make model files
        self.picanet_logs = Path(root_dir) / "eval_results" / self.args.task
        self.picanet_logs.mkdir(parents=True, exist_ok=True)
        self.saliency_mdl_ckpt = Path(root_dir, "pretrained_models", self.args.task)
        self.saliency_mdl_ckpt.mkdir(parents=True, exist_ok=True)

        # Set up models and training
        # remove a global attention layer
        self.transforms = T.Compose(
            [
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            ]
        )

        self.train_dataset, self.train_dataloader = self.setup_dataloader("train")
        self.test_dataset, self.test_dataloader = self.setup_dataloader("test")

        self.model, self.optimizer, self.scheduler = self.setup_model_and_optimizer()

        # Setup metric trackers
        self.metrics = MetricMeter()

        self.loss_fn = lambda input, target: (
            (1 / (1.05 - target)) * (input - target) ** 2
        ).mean()

    def setup_dataloader(self, split="train"):
        images, saliency_maps = self.get_dataset(split=split)
        dataset = CustomTensorDataset(
            [images, saliency_maps], transform=self.transforms
        )
        dataloader = DataLoader(
            dataset, batch_size=self.args.batch_size, pin_memory=True
        )
        return dataset, dataloader

    def setup_model_and_optimizer(self):
        cfg = picanet_cfg[self.args.picanet_cfg]
        pprint(cfg)

        model = Unet(cfg)
        vgg = torchvision.models.vgg16(pretrained=True)
        model.encoder.seq.load_state_dict(vgg.features.state_dict())
        model = model.to("cuda")
        print(model)
        print(
            f"==> Number of trainable parameters: {count_trainable_parameters(model)}"
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=0.005
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=100, factor=0.1, threshold=0.001, min_lr=1e-6
        )
        return model, optimizer, scheduler

    def train(self):
        pbar = tqdm.tqdm(range(self.args.num_epochs), position=0, leave=True)
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")

            self.run_single_epoch()
            if (epoch + 1) % self.args.log_every == 0:
                print(f"EPOCH {epoch + 1}")
                print(self.metrics)

            if (epoch + 1) % self.args.eval_every == 0:
                self.run_single_epoch(training=False)

        # save model
        print("==> Saving model...")
        torch.save(
            self.model.state_dict(),
            self.saliency_mdl_ckpt
            / f"{self.args.pred_type}_model{self.args.model_suffix}_cfg_{self.args.picanet_cfg}.pt",
        )

    def run_single_epoch(self, training=True):
        if training:
            self.model.train()
        else:
            self.model.eval()

        epoch_loss = 0.0
        self.metrics.reset()

        dataloader = self.train_dataloader if training else self.test_dataloader
        for batch in dataloader:
            obss, seg_masks = batch

            obss = obss.to("cuda")
            seg_masks = seg_masks.to("cuda")

            if training:
                # compute loss
                self.optimizer.zero_grad()
                pred, loss = self.model(obss, seg_masks)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss.item())
            else:
                with torch.no_grad():
                    pred, loss = self.model(obss, seg_masks)

            if self.args.pred_type == "segmentation" or args.binary_fixation:
                dice_fn = Dice().cuda()
                pred_seg_mask = (pred > 0.5).int()
                targets = seg_masks.detach().flatten().int()
                dice_score = dice_fn(pred_seg_mask.flatten(), targets)

                self.metrics["dice"] += dice_score.item()

            self.metrics.update({"train/loss": loss.item()})

    def evaluate(self):
        print("Running eval...")
        self.model.load_state_dict(
            torch.load(
                self.saliency_mdl_ckpt
                / f"{self.args.pred_type}_model{self.args.model_suffix}_cfg_{self.args.picanet_cfg}.pt",
            )
        )

        self.model.eval()

        chunk_size = 10
        to_show = []

        for obs, gt_mask in self.test_dataset:
            with torch.no_grad():
                pred, loss = self.model(
                    obs.unsqueeze(0).cuda(), gt_mask.unsqueeze(0).cuda()
                )

                pred = pred[-1].squeeze()

                masked_obs = pred.cpu() * obs
                to_show.append([pred, obs, gt_mask, masked_obs])

        to_show_chunks = list(chunks(to_show, chunk_size))
        for idx, chunk in enumerate(to_show_chunks):
            ts.show(chunk, figsize=(12, 3 * len(chunk)))
            plt.savefig(
                self.picanet_logs
                / f"mask_visualizations_{self.args.pred_type}_{idx}{self.args.model_suffix}.png"
            )

    def get_dataset(self, split="train"):
        fixation_maps = []
        frames = []

        annotation_dir = Path(self.args.annotation_dir) / self.args.task / split

        # Load frames
        for frame_f in sorted(list(annotation_dir.glob("*.jpg"))):
            frame = plt.imread(frame_f)[:, :, :3]
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            frames.append(frame)

        obs = torch.from_numpy(np.stack(frames)).float()
        # reshape
        obs = einops.rearrange(obs, "b h w c -> b c h w")

        # normalize
        if obs.max() > 1.0:
            obs /= 255.0

        assert (
            obs.max() <= 1.0 and obs.min() >= 0
        ), f"obs needs to be between 0 and 1, max: {obs.max()}"

        assert obs.shape[1:] == (3, 224, 224), "obs wrong shape"

        # Load annotations as fixtation maps
        if split == "train":
            for annotation_f in sorted(list(annotation_dir.glob("*.txt"))):
                fixation_map = get_heatmap_from_ann_file(annotation_f)[
                    None
                ]  # 1 x H x W
                fixation_maps.append(fixation_map)

            fixation_maps = torch.from_numpy(np.stack(fixation_maps)) / 255.0
            assert (
                fixation_maps.max() <= 1.0 and fixation_maps.min() >= 0
            ), f"fixation_maps needs to be between 0 and 1, max: {fixation_maps.max()}"
            assert fixation_maps.shape[1:] == (1, 224, 224), "fixation_maps wrong shape"
        else:
            fixation_maps = torch.zeros((obs.shape[0], 1, 224, 224))

        if split == "train":
            masked_obs = fixation_maps * obs

            ts.save(
                [
                    [obs_, fixation_map_, masked_obs_]
                    for obs_, fixation_map_, masked_obs_ in zip(
                        obs, fixation_maps, masked_obs
                    )
                ],
                path=self.picanet_logs / f"eval_results/{self.args.task}/viz_mask.png",
                figsize=(15, 3 * obs.shape[0]),
            )

        return obs, fixation_maps


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        obs = self.tensors[0][index]
        masks = self.tensors[1][index]

        if self.transform and isinstance(self.transform, list):
            obs = self.transform(obs)
            masks = self.transform(masks)

        hflip = random.random() < 0.5
        if hflip:
            obs = T.RandomHorizontalFlip(p=1.0)(obs)
            masks = T.RandomHorizontalFlip(p=1.0)(masks)

        vflip = random.random() < 0.5
        if vflip:
            obs = T.RandomVerticalFlip(p=1.0)(obs)
            masks = T.RandomVerticalFlip(p=1.0)(masks)

        return obs, masks

    def __len__(self):
        return self.tensors[0].size(0)


def main(args):
    trainer = Trainer(args)
    trainer.train()
    trainer.evaluate()


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--dataset_size", default=10000)
    parser.add_argument("--train_frac", default=0.02)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--eval_every", default=20, type=int)
    parser.add_argument("--log_every", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--pred_type", default="segmentation", type=str)
    parser.add_argument("--annotation_dir", default="annotations", type=str)
    parser.add_argument("--task", default="drawer-open-v2", type=str)
    parser.add_argument("--sample", default=False, type=bool)
    parser.add_argument("--binary_fixation", default=False, type=bool)
    parser.add_argument("--model_suffix", default="", type=str)
    parser.add_argument("--picanet_cfg", default=0, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
