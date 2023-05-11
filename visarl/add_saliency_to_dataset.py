"""
Script to add saliency to a dataset of frames. This is then used for training the MultiMAE visual encoder model.    
"""
import os
import pickle
import numpy as np
import torch
import einops
import tqdm
from pathlib import Path
import torchvision
import visarl.util.constants as constants
from pprint import pprint
from pathlib import Path
from visarl.src.modules.picanet import Unet
from visarl.train_picanet import picanet_cfg

if __name__ == "__main__":
    dataset_name = "pickup_apple_all_images"
    root_dir = Path(os.environ["ROOT_DIR"])
    image_data = np.load(root_dir / f"data/{dataset_name}.npy")

    print(image_data.shape)

    # Load saliency predictor model from checkpoint
    model_dir = root_dir / "pretrained_models/pickup_apple"
    print(f"==> Loading model from {model_dir}")
    ckpt_file = list(model_dir.glob("*.pt"))[0]
    device = "cuda"
    print(f"==> Loading model from {ckpt_file}")

    cfg = picanet_cfg[2]
    model = Unet(cfg)
    vgg = torchvision.models.vgg16(pretrained=True)
    model.encoder.seq.load_state_dict(vgg.features.state_dict())
    model = model.to("cuda")
    model.load_state_dict(torch.load(ckpt_file))
    model.eval()

    chunk_size = 10

    saliency_predictions = []

    for obs in tqdm.tqdm(image_data):
        obs = torch.from_numpy(obs).float().to(device)
        # reshape obs
        obs = einops.rearrange(obs, "h w c -> 1 c h w")
        gt_mask = torch.zeros_like(obs)[:, 0:1].to(device)

        with torch.no_grad():
            pred, loss = model(obs, gt_mask)

            pred = pred[-1].squeeze()
            pred = pred.cpu().numpy()
            saliency_predictions.append(pred)

    # Convert to numpy array
    # N x H x W
    saliency_predictions = np.array(saliency_predictions)

    # Save original frames and saliency annotations into a dataset
    # Image data is N x H x W x C
    pickle.dump(
        {"saliency": saliency_predictions, "frames": image_data},
        open(root_dir / f"data/{dataset_name}_and_saliency.pkl", "wb"),
    )
