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

from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics import Dice

from functools import partial
import torchshow as ts
import torchvision.transforms as T

import util.constants as constants
import sys
from pprint import pprint

sys.path.append(constants.ROOT)
sys.path.append(constants.SRC)
from src.modules.picanet import Unet
from util.saliency_util import get_heatmap_from_ann_file


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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


def train_epoch(
    model, epoch, dataloader, optimizer, scheduler, loss_fn, pred_type, training=True
):
    if training:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    metrics = {"dice": 0}

    for batch in dataloader:
        obss, seg_masks = batch

        obss = obss.to("cuda")
        seg_masks = seg_masks.to("cuda")

        if training:
            # compute loss
            optimizer.zero_grad()
            pred, loss = model(obss, seg_masks)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
        else:
            with torch.no_grad():
                pred, loss = model(obss, seg_masks)

        if pred_type == "segmentation" or args.binary_fixation:
            dice_fn = Dice().cuda()
            pred_seg_mask = (pred > 0.5).int()
            targets = seg_masks.detach().flatten().int()
            dice_score = dice_fn(pred_seg_mask.flatten(), targets)

            metrics["dice"] += dice_score.item()

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    metrics["dice"] /= len(dataloader)
    return epoch_loss, metrics


def evaluate(args, model, dataset):
    print("running eval...")
    model.eval()

    chunk_size = 10
    to_show = []

    for idx in range(min(30, len(dataset))):
        obs, gt_mask = dataset[idx]

        with torch.no_grad():
            if args.sample:
                mean, var = model(obs.unsqueeze(0).cuda(), None)
                mean = mean.squeeze()
                var = var.squeeze()

                if args.pred_type == "segmentation" or args.binary_fixation:
                    pred = (torch.sigmoid(mean) > 0.5).int()
                else:
                    pred = mean
                print(f"average variance: {var.mean()}")
            else:
                pred, loss = model(obs.unsqueeze(0).cuda(), gt_mask.unsqueeze(0).cuda())

                pred = pred[-1].squeeze()

            masked_obs = pred.cpu() * obs
            to_show.append([pred, obs, gt_mask, masked_obs])

    to_show_chunks = list(chunks(to_show, chunk_size))
    for idx, chunk in enumerate(to_show_chunks):
        ts.show(chunk, figsize=(12, 3 * len(chunk)))
        plt.savefig(
            f"eval_results/{args.task}/mask_visualizations_{args.pred_type}_{idx}{args.model_suffix}.png"
        )


def get_dataset(args):
    dataset_path = os.path.join(
        constants.REPLAY_BUFFER_DIR, f"{args.task}_step_1000000_small.pkl"
    )
    print(f"loading dataset from: {dataset_path}")
    dataset = pickle.load(open(dataset_path, "rb"))

    for k, v in dataset.items():
        dataset[k] = np.array(dataset[k])
    all_obs = dataset["frames"][:, 0]

    # visualize heatmaps
    fixation_maps = []
    frames = []

    annotation_dir = os.path.join(args.annotation_dir, args.task)

    train_indices = list(np.arange(30))
    test_indices = list(np.random.randint(0, len(all_obs), 30))

    num_examples = 0
    for i in train_indices:
        annotation_file = os.path.join(
            annotation_dir, f"{args.pred_type}/annotations_{i:03d}.txt"
        )
        if os.path.exists(annotation_file):
            fixation_map = get_heatmap_from_ann_file(annotation_file)
            fixation_maps.append(fixation_map)

            frame = plt.imread(os.path.join(annotation_dir, f"frame_{i:03d}.png"))[
                :, :, :3
            ]
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            num_examples += 1
        else:
            print(f"skipping index: {i}")

    train_masks = np.stack(fixation_maps)
    train_obs = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    train_masks = torch.from_numpy(train_masks).unsqueeze(1) / 255.0

    test_obs = torch.from_numpy(all_obs[torch.tensor(test_indices)] / 255.0).float()
    upsample_m = torch.nn.modules.Upsample(
        scale_factor=(224 / 64, 224 / 64), mode="nearest"
    ).to("cuda")
    test_obs = upsample_m(test_obs)
    test_masks = torch.zeros((len(test_indices), 1, 224, 224)).float()

    if train_masks.max() > 1:
        train_masks /= 255.0
        train_masks /= 255.0

    if args.binary_fixation:
        train_masks = (train_masks > 0).float()

    if train_obs.max() > 1:
        train_obs /= 255.0
        test_obs /= 255.0

    assert (
        train_obs.max() <= 1.0 and train_obs.min() >= 0
    ), f"obs needs to be between 0 and 1, max: {train_obs.max()}"
    assert (
        train_masks.max() <= 1.0 and train_masks.min() >= 0
    ), f"fixation_maps needs to be between 0 and 1, max: {train_obs.max()}"

    assert train_obs.shape[1:] == (3, 224, 224), "train_obs wrong shape"
    assert test_obs.shape[1:] == (
        3,
        224,
        224,
    ), f"test_obs wrong shape, {test_obs.shape}"
    assert train_masks.shape[1:] == (1, 224, 224), "train_masks wrong shape"
    assert test_masks.shape[1:] == (
        1,
        224,
        224,
    ), f"test_masks wrong shape, {test_masks.shape}"

    masked_train_obs = train_masks * train_obs

    ts.save(
        [
            [train_obs[i], train_masks[i], masked_train_obs[i]]
            for i in range(num_examples)
        ],
        path=f"eval_results/{args.task}/viz_{args.pred_type}_mask.png",
        figsize=(15, 3 * num_examples),
    )

    return train_obs, train_masks, test_obs, test_masks


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
    # make model files
    os.makedirs(f"eval_results/{args.task}", exist_ok=True)
    os.makedirs(f"pretrained_models/{args.task}", exist_ok=True)

    # remove a global attention layer
    cfg = picanet_cfg[args.picanet_cfg]
    pprint(cfg)

    # set up models and training
    model = Unet(cfg)
    vgg = torchvision.models.vgg16(pretrained=True)
    model.encoder.seq.load_state_dict(vgg.features.state_dict())

    model = model.to("cuda")
    print(model)
    print(f"number of trainable parameters: {count_trainable_parameters(model)}")

    train_obs, train_masks, test_obs, test_masks = get_dataset(args)

    transforms = T.Compose(
        [
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        ]
    )
    train_dataset = CustomTensorDataset([train_obs, train_masks], transform=transforms)
    test_dataset = CustomTensorDataset([test_obs, test_masks], transform=transforms)

    print(f"train_dataset: {len(train_dataset)}, test_dataset: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, pin_memory=True
    )

    if args.mode == "train":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.005
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=100, factor=0.1, threshold=0.001, min_lr=1e-6
        )

        loss_fn = lambda input, target: (
            (1 / (1.05 - target)) * (input - target) ** 2
        ).mean()

        for epoch in tqdm.tqdm(range(args.num_epochs)):
            loss, metrics = train_epoch(
                model,
                epoch,
                train_dataloader,
                optimizer,
                scheduler,
                loss_fn,
                args.pred_type,
                training=True,
            )

            print(
                f"epoch: {epoch}, train loss: {round(loss, 5)}, train metrics: {metrics}, curr_lr: {scheduler.optimizer.param_groups[0]['lr']}"
            )

            if epoch % args.eval_every == 0:
                eval_loss, metrics = train_epoch(
                    model,
                    epoch,
                    test_dataloader,
                    optimizer,
                    scheduler,
                    loss_fn,
                    args.pred_type,
                    training=False,
                )
                print(
                    f"epoch: {epoch}, eval loss: {round(eval_loss, 5)}, eval metrics: {metrics}"
                )

        # save model
        print("saving model...")
        torch.save(
            model.state_dict(),
            f"pretrained_models/{args.task}/{args.pred_type}_model{args.model_suffix}_cfg_{args.picanet_cfg}.pt",
        )

    model.load_state_dict(
        torch.load(
            f"pretrained_models/{args.task}/{args.pred_type}_model{args.model_suffix}_cfg_{args.picanet_cfg}.pt"
        )
    )
    evaluate(args, model, test_dataset)


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--dataset_size", default=10000)
    parser.add_argument("--train_frac", default=0.02)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--eval_every", default=20, type=int)
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
