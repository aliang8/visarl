# ViSARL: Visual Saliency Reinforcement Learning

Codebase to replicate the results for https://sites.google.com/view/visarl. 

# Instructions

Create a new conda environment: `conda create --prefix ./venv python=3.9`

Install packages: `pip3 install requirements.txt`

Download and install Metaworld: git clone https://github.com/rlworkgroup/metaworld.git

Download and install MultiMAE


# ViSARL Framework 

See `notebooks/visualize_multimae.ipynb` to see visualizations for saliency prediction
and multimae reconstruction.

Download pretrained saliency predictor and multimae checkpoint files for different Metaworld tasks from:
https://drive.google.com/drive/folders/1PxdjgIljydijMPWu6547pN5D7nbPZv3W?usp=sharing

Put the downloaded checkpoint under `pretrained_models`. 

Your file directory should look like: `pretrained_models/{task}/model.ckpt`

You can directly skip to step 5 and run downstream RL training with the pretrained saliency and MultiMAE checkpoints.


Here are the steps to run your own experiments: 

## 1. Collect some RGB images from environment 

``` 
CUDA_VISIBLE_DEVICES=0 python scripts/collect_data.py \
    --algorithm sac \
    --seed 1 \
    --domain_name metaworld \
    --task_name pick-place-v2 \
    --log_dir logs/oracle \
    --eval_mode train \
    --save_video \
    --episode_length 500 \
    --eval_episodes 10
```

## 2. Annotate saliency maps
b. Annotate the frames with the UI (index.html) by clicking on salient regions. 

## 3.Train PiCANet saliency predictor

```
CUDA_VISIBLE_DEVICES=0 python3 train_picanet.py \
    --lr 3e-4 \
    --num_epochs 1000 \
    --mode train \
    --pred_type fixation \
    --annotation_dir annotations \
    --model_suffix _picanet \
    --task drawer-open-v2 \
    --batch_size 8 \
    --picanet_cfg 2
```

## 4.Pretrain MultiMAE 

``` 
CUDA_VISIBLE_DEVICES=0 python3 run_pretraining_multimae.py \
    --output_dir=output_dir/drawer-open-v2_with_saliency \
    --task_name=drawer-open-v2 \
    --in_domains=rgb-saliency \
    --out_domains=rgb-saliency \
    --model=pretrain_multimae_small \
    --num_encoded_tokens=32 \
    --batch_size=512 \
    --patch_size=8 \
    --input_size=64 \
    --save_ckpt_freq 5 \
    --decoder_depth 3 \
    --extra_norm_pix_loss \
    --decoder_num_heads 4 \
    --epochs 200 \
    --data_path /PATH/TO/DATASET/dataset.pkl \
    --log_wandb
```

## 5.Run downstream RL experiments 

Example of running SAC for the Drawer Open task

```
CUDA_VISIBLE_DEVICES=2 python online_rl_train.py \
    --algorithm sac \
    --seed 1 \
    --domain metaworld \
    --task_name drawer-open-v2 \
    --train_steps 1M \
    --log_dir logs/online_rl \
    --save_freq 100k \
    --use_wandb 1 \
    --observation_type image \
    --image_size 64 \
    --save_replay_buffer 1 \
    --exp_name online_rl_image_based \
    --encoder_type mmae \
    --freeze_encoder 1 \
    --load_encoder_from_ckpt 1 \
    --pretrained_encoder 1 \
    --in_domains rgb saliency \
    --use_saliency_input 1 \
    --cameras 0 \
    --picanet_cfg 2 \
    --encoder_ckpt_path /PATH/TO/ENCODER/CKPT.pth \
    --notes "RGB + saliency and use saliency as input during inference time" \
    --skip_first_eval 1
```


# Extra packages and troubleshooting for mujoco

sudo apt-get install libx11-dev libglew-dev libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
