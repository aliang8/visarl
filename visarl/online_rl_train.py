import torch
import os
import numpy as np
import gym

from src.algorithms.factory import make_agent

import time
from src.arguments import parse_args
from src.env.wrappers import make_env, make_env_mw

import sys
import glob
import pickle
from tqdm import tqdm
import warnings
import wandb
import imageio
import shutil

warnings.filterwarnings("ignore")

import util.constants as constants
import util.general_utils as utils
from util.logger import Logger
from util.replay_buffer import ReplayBufferStorage, make_replay_loader

sys.path.append(constants.MMAE)
from multimae.multimae import MultiMAE
from dm_env import specs
from pathlib import Path


def evaluate(
    args, env, agent, num_episodes, step, L=None, save_video=False, log_wandb=True
):
    episode_rewards = []
    successes = []
    episode_steps = []
    videos = []

    if not os.path.exists(os.path.join(args.work_dir, "videos")):
        os.makedirs(os.path.join(args.work_dir, "videos"))

    for i in tqdm(range(num_episodes)):
        frames = []

        obs, _ = env.reset()

        if "image" not in args.observation_type or "r3m" in args.encoder_type:
            frame = env.unwrapped.sim.render(
                camera_name="corner2",
                width=args.image_size,
                height=args.image_size,
            ).transpose(2, 0, 1)
            frames.append(frame)
        else:
            frames.append(np.array(obs))

        if args.pretrained_encoder:
            input_ = agent._obs_to_input(obs)

            if args.algorithm == "sac_masked_obs":
                input_ = agent.apply_saliency(input_)

            if agent.encoder is not None:
                obs = agent.encoder(input_)
            else:
                obs = input_
        else:
            input_ = agent._obs_to_input(obs)
            if args.algorithm == "sac_masked_obs":
                obs = agent.apply_saliency(input_)

        done = False
        episode_reward = 0
        episode_step = 0
        traj_success = False
        while not done:
            with torch.no_grad():
                with utils.eval_mode(agent):
                    action = agent.select_action(obs)

                obs, _, reward, done, info = env.step(action)

                if "image" not in args.observation_type or "r3m" in args.encoder_type:
                    frame = env.unwrapped.sim.render(
                        camera_name="corner2",
                        width=args.image_size,
                        height=args.image_size,
                    ).transpose(2, 0, 1)
                    frames.append(frame)
                else:
                    frames.append(np.array(obs))

                if args.pretrained_encoder:
                    input_ = agent._obs_to_input(obs)

                    if args.algorithm == "sac_masked_obs":
                        input_ = agent.apply_saliency(input_)

                    if agent.encoder is not None:
                        obs = agent.encoder(input_)
                    else:
                        obs = input_
                else:
                    input_ = agent._obs_to_input(obs)
                    if args.algorithm == "sac_masked_obs":
                        obs = agent.apply_saliency(input_)

                if info["success"]:
                    traj_success = True

                if traj_success:
                    break

                episode_reward += reward
                episode_step += 1

        videos.append(np.array(frames).transpose(0, 2, 3, 1))
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        successes.append(traj_success)
        print(i, episode_reward, traj_success)

        if save_video and "image" not in args.observation_type:
            path = os.path.join(args.work_dir, "videos", f"eval_{i}.mp4")
            print("saving video to: ", path)
            imageio.mimsave(path, np.array(frames).transpose(0, 2, 3, 1), fps=30)

    if L:
        L.log("eval/episode_reward", np.mean(episode_rewards), step)
        L.log("eval/num_successes", np.mean(successes), step)
        L.log("eval/episode_steps", np.mean(episode_steps), step)

    if args.use_wandb and log_wandb:
        wandb.log(
            {
                "eval/episode_reward_mean": np.mean(episode_rewards),
                "eval/episode_reward_std": np.std(episode_rewards),
                "eval/num_successes": np.mean(successes),
                "eval/episode_steps_mean": np.mean(episode_steps),
                "eval/episode_steps_std": np.std(episode_steps),
            },
            step=step,
        )

        # log the videos
        video_grid = utils.create_video_grid(videos, height=84, width=84)
        wandb.log(
            {
                "eval/rollouts": wandb.Video(
                    video_grid,
                    caption=f"train_steps_{step}",
                    fps=20,
                    format="gif",
                )
            },
            step=step,
        )
    print(successes)
    return np.mean(episode_rewards), np.mean(successes)


def main(args):
    home = os.environ["HOME"]
    os.environ["MJKEY_PATH"] = f"{home}/.mujoco/mujoco210_linux/bin/mjkey.txt"
    os.environ["MUJOCO_GL"] = "egl"

    # Set seed
    utils.set_seed_everywhere(args.seed)

    if args.cameras == 0:
        cameras = ["corner"]
    elif args.cameras == 1:
        cameras = ["corner2"]
    elif args.cameras == 2:
        cameras = ["corner", "corner2", "topview"]
    else:
        raise Exception("Current Camera Pose Not Supported.")

    # Initialize environments
    gym.logger.set_level(40)
    env = make_env_mw(
        domain_name="metaworld",
        task_name=args.task_name,
        seed=args.seed,
        episode_length=500,
        n_substeps=args.n_substeps,
        frame_stack=args.frame_stack,
        image_size=args.image_size,
        mode="train",
        cameras=cameras,  # ['third_person', 'first_person']
        observation_type=args.observation_type,
        action_space=args.action_space,
        pretrained_encoder=args.encoder_type,
    )

    env.unwrapped.model.cam_pos[2][:] = [0.75, 0.075, 0.7]

    run_name = f"{args.exp_name}_{args.domain_name}_{args.task_name}_{args.algorithm}_{args.observation_type}_{args.train_steps_str}"
    if "image" in args.observation_type:
        run_name += f"_{args.encoder_type}"
        run_name += f"_{','.join(args.in_domains)}"
        run_name += f"_freeze-{args.freeze_encoder}"
        run_name += f"_ft-{args.finetune_encoder}"
        run_name += f"_cam-{cameras[0]}"
        run_name += f"_sal_input-{args.use_saliency_input}"
        if args.use_saliency_input:
            run_name += f"_pcfg-{args.picanet_cfg}"
        if args.algorithm == "sac_masked_obs":
            run_name += f"_rxs-{args.rgb_x_saliency}"
            run_name += f"_depth-{args.saliency_as_depth_channel}"

    # Create working directory
    work_dir = os.path.join(
        args.log_dir,
        run_name,
        str(args.seed),
    )
    args.work_dir = work_dir
    print("Working directory:", work_dir)

    if not args.overwrite:
        assert not os.path.exists(
            os.path.join(work_dir, "train.log")
        ), "specified working directory already exists"

    if args.overwrite:
        print("overwriting old dir...")
        shutil.rmtree(work_dir)

    utils.make_dir(work_dir, exist_ok=True)
    model_dir = utils.make_dir(os.path.join(work_dir, "model"))
    utils.write_info(args, os.path.join(work_dir, "info.log"))

    if args.use_wandb:
        wandb.init(
            name=run_name,
            # group=group_name,
            project="visual_saliency",
            config=args,
            entity="glamor",
        )

    # Prepare agent
    assert torch.cuda.is_available(), "must have cuda enabled"

    if args.pretrained_encoder:
        latent_dim = constants.encoder_to_dim_mapping[args.encoder_type]
    else:
        latent_dim = None

    # use this replay buffer to avoid using too much RAM
    data_specs = [
        specs.Array((3, 64, 64), np.uint8, "observation"),
        specs.Array((4,), np.float32, "action"),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "done"),
        specs.Array((1,), np.float32, "done_bool"),
    ]

    if args.algorithm == "sac_masked_obs":
        # we are storing rgb x saliency which is floating point
        if args.saliency_as_depth_channel:
            num_channels = 4
        else:
            num_channels = 3
        data_specs[0] = specs.Array((num_channels, 64, 64), np.float32, "observation")

    if args.pretrained_encoder:
        data_specs.append(specs.Array((latent_dim,), np.float32, "latent"))

    replay_storage = ReplayBufferStorage(
        data_specs, Path(os.path.join(work_dir, "buffer"))
    )

    replay_loader = make_replay_loader(
        Path(os.path.join(work_dir, "buffer")),
        args.train_steps,
        args.batch_size,
        args.num_workers,
        True,  # save_snapshot
        1,
        args.discount,
    )
    replay_iter = None

    if args.observation_type == "state":
        obs_shape = (39,)
    elif "image" in args.observation_type:
        if args.saliency_as_depth_channel:
            obs_shape = (
                4 * args.frame_stack,
                args.image_crop_size,
                args.image_crop_size,
            )
        else:
            obs_shape = (
                3 * args.frame_stack,
                args.image_crop_size,
                args.image_crop_size,
            )
    print("Observations:", env.observation_space.shape)
    print("Cropped observations:", obs_shape)
    agent = make_agent(
        obs_shape=obs_shape, action_shape=env.action_space.shape, args=args
    )

    start_step, episode, episode_reward, episode_step, done = 0, 0, 0, 0, True
    L = Logger(work_dir)
    start_time = time.time()

    print(agent.actor)
    print(agent.critic)

    print(args)
    print(f"training for {args.train_steps} steps")

    print(
        sum(p.numel() for p in agent.critic.parameters() if p.requires_grad),
        " critic parameters",
    )

    for step in range(start_step, args.train_steps + 1):
        if done:
            if step > start_step:
                L.log("train/duration", time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # Evaluate agent periodically
            if step % args.eval_freq == 0:
                print("Evaluating:", work_dir)
                L.log("eval/episode", episode, step)
                if step == 0 and args.skip_first_eval:
                    pass
                else:
                    evaluate(args, env, agent, args.eval_episodes, step, L)
                L.dump(step)

            # Save agent periodically
            if step > start_step and step % args.save_freq == 0:
                print("saving @!!!!!!!")
                torch.save(
                    agent.state_dict(),
                    os.path.join(model_dir, f"ckpt_{step}.pt"),
                )
                if "sgsac" in args.algorithm:
                    torch.save(
                        agent.attribution_predictor.state_dict(),
                        os.path.join(model_dir, f"attrib_predictor_{step}.pt"),
                    )

            L.log("train/episode_reward", episode_reward, step)

            if args.use_wandb:
                wandb.log(
                    {
                        "train/episode": episode,
                        "train/step": step,
                        "train/episode_step": episode_step,
                        "train/episode_reward": episode_reward,
                    },
                    step=step,
                )

            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log("train/episode", episode, step)

            # encode obs
            if args.pretrained_encoder:
                with torch.no_grad():  # don't update the encoder
                    input_ = agent._obs_to_input(obs)
                    if agent.encoder is not None:
                        latent = agent.encoder(input_)
                    else:
                        latent = input_

            if args.algorithm == "sac_masked_obs":
                input_ = agent._obs_to_input(obs)
                with torch.no_grad():
                    obs = agent.apply_saliency(input_)

            if args.save_replay_buffer and args.observation_type == "state":
                frame = env.unwrapped.sim.render(
                    camera_name=cameras[0],
                    width=args.image_size,
                    height=args.image_size,
                ).transpose(2, 0, 1)
            else:
                frame = None

        # Sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                if args.pretrained_encoder:
                    action = agent.sample_action(latent)
                else:
                    action = agent.sample_action(obs)

        # Run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1

            if step == args.init_steps:
                print("warmup updates...")

                if replay_iter is None:
                    replay_iter = iter(replay_loader)

                for _ in tqdm(range(num_updates)):
                    agent.update(
                        replay_iter,
                        L,
                        step,
                        None,
                        latents=args.pretrained_encoder,
                    )
            else:
                for _ in range(num_updates):
                    agent.update(
                        replay_iter,
                        L,
                        step,
                        None,
                        latents=args.pretrained_encoder,
                    )

        # Take step
        next_obs, _, reward, done, _ = env.step(action)
        if args.pretrained_encoder:
            with torch.no_grad():
                input_ = agent._obs_to_input(next_obs)
                if agent.encoder is not None:
                    next_latent = agent.encoder(input_)
                else:
                    next_latent = input_

        # compute saliency map and cache it
        if args.algorithm == "sac_masked_obs":
            input_ = agent._obs_to_input(next_obs)
            with torch.no_grad():
                next_obs = agent.apply_saliency(input_)

        if args.save_replay_buffer and args.observation_type == "state":
            next_frame = env.unwrapped.sim.render(
                camera_name=cameras[0],
                width=args.image_size,
                height=args.image_size,
            ).transpose(2, 0, 1)
        else:
            next_frame = None

        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)

        if isinstance(obs, torch.Tensor):
            obs_save = obs.squeeze(0).cpu().numpy()
        else:
            obs_save = np.array(obs)

        to_add = {
            "observation": obs_save,
            "action": action,
            "reward": reward,
            "done": done,
            "done_bool": done_bool,
            "discount": args.discount,
        }

        if args.pretrained_encoder:
            to_add["latent"] = latent.squeeze().detach().cpu().numpy()
        else:
            if frame is not None:
                to_add["frame"] = frame

        replay_storage.add(to_add)

        episode_reward += reward
        obs = next_obs
        frame = next_frame

        if args.pretrained_encoder:
            latent = next_latent

        episode_step += 1

    # save the replay buffer at the end
    # if args.save_replay_buffer:
    #     save_path = os.path.join(
    #         constants.REPLAY_BUFFER_DIR, f"{args.task_name}_step_{args.train_steps}.pkl"
    #     )
    #     replay_buffer.save(save_path)

    print("Completed training for", work_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
