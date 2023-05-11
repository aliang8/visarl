import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--domain_name", default="robot")
    parser.add_argument("--task_name", default="reach")
    parser.add_argument("--frame_stack", default=1, type=int)
    parser.add_argument("--episode_length", default=50, type=int)
    parser.add_argument("--n_substeps", default=20, type=int)
    parser.add_argument("--eval_mode", default="none", type=str)
    parser.add_argument("--action_space", default="xy", type=str)
    parser.add_argument(
        "--cameras", default="0", type=int
    )  # 0: 3rd person, 1: 1st person, 2: both
    parser.add_argument("--observation_type", default="image", type=str)
    # agent
    parser.add_argument("--algorithm", default="sgsac", type=str)
    parser.add_argument("--train_steps", default="100k", type=str)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--init_steps", default=1000, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--hidden_dim", default=1024, type=int)

    # actor
    parser.add_argument("--actor_lr", default=1e-4, type=float)
    parser.add_argument("--actor_beta", default=0.9, type=float)
    parser.add_argument("--actor_log_std_min", default=-10, type=float)
    parser.add_argument("--actor_log_std_max", default=2, type=float)
    parser.add_argument("--actor_update_freq", default=2, type=int)

    # critic
    parser.add_argument("--critic_lr", default=1e-4, type=float)
    parser.add_argument("--critic_beta", default=0.9, type=float)
    parser.add_argument("--critic_tau", default=0.005, type=float)
    parser.add_argument("--critic_target_update_freq", default=2, type=int)
    parser.add_argument("--critic_weight_decay", default=0, type=float)

    # architecture
    parser.add_argument("--num_shared_layers", default=11, type=int)
    parser.add_argument("--num_head_layers", default=0, type=int)
    parser.add_argument("--num_filters", default=32, type=int)
    parser.add_argument("--projection_dim", default=512, type=int)
    parser.add_argument("--encoder_tau", default=0.05, type=float)

    # entropy maximization
    parser.add_argument("--init_temperature", default=0.1, type=float)
    parser.add_argument("--alpha_lr", default=1e-4, type=float)
    parser.add_argument("--alpha_beta", default=0.9, type=float)

    # auxiliary tasks
    parser.add_argument("--aux_lr", default=3e-4, type=float)
    parser.add_argument("--aux_beta", default=0.9, type=float)
    parser.add_argument("--aux_update_freq", default=2, type=int)

    # soda
    parser.add_argument("--soda_batch_size", default=256, type=int)
    parser.add_argument("--soda_tau", default=0.005, type=float)

    # svea
    parser.add_argument("--svea_alpha", default=0.5, type=float)
    parser.add_argument("--svea_beta", default=0.5, type=float)
    parser.add_argument("--svea_norm_coeff", default=0.1, type=float)
    parser.add_argument("--attrib_coeff", default=0.25, type=float)
    parser.add_argument("--consistency", default=1, type=int)

    # sgsac
    parser.add_argument("--sgsac_quantile", default=0.95, type=float)

    # eval
    parser.add_argument("--save_freq", default="100k", type=str)
    parser.add_argument("--eval_freq", default="10k", type=str)
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--distracting_cs_intensity", default=0.0, type=float)

    # misc
    parser.add_argument("--seed", default=10081, type=int)
    parser.add_argument("--log_dir", default="logs", type=str)
    parser.add_argument("--save_video", default=False, action="store_true")

    parser.add_argument("--save_replay_buffer", default=0, type=int)

    # data collection
    parser.add_argument("--random_policy_data", default=1, type=int)
    parser.add_argument("--only_save_success", default=0, type=int)
    parser.add_argument("--act_noise_pct", default=0.0, type=float)

    # offline rl training
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--n_epochs", default=2000, type=int)
    parser.add_argument("--bc_epochs", default=0, type=int)
    parser.add_argument("--n_train_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--save_every_epoch", default=10, type=int)
    parser.add_argument("--eval_every_epoch", default=10, type=int)
    parser.add_argument("--encoder_lr", default=1e-4, type=float)
    parser.add_argument("--load_encoder_from_ckpt", default=0, type=int)

    parser.add_argument("--encoder_update_freq", default=10, type=int)
    parser.add_argument("--skip_first_eval", default=0, type=int)
    parser.add_argument("--patch_size", default=8, type=int)
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--image_crop_size", default=64, type=int)
    parser.add_argument("--freeze_encoder", default=0, type=int)
    parser.add_argument("--dataset_path", default="", type=str)
    parser.add_argument("--extra_norm_pix_loss", default=1, type=int)
    parser.add_argument("--finetune_encoder", default=0, type=int)
    parser.add_argument("--exp_name", default="test", type=str)
    parser.add_argument("--notes", default="", type=str)
    parser.add_argument("--encoder_ckpt_path", default="", type=str)
    parser.add_argument("--use_saliency_input", default=0, type=int)
    parser.add_argument("--rgb_x_saliency", default=0, type=int)
    parser.add_argument("--saliency_as_depth_channel", default=0, type=int)

    parser.add_argument("--picanet_cfg", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument("--in_domains", default=["rgb"], type=str, nargs="+")
    parser.add_argument("--out_domains", default=["rgb"], type=str, nargs="+")

    # for evaluation use
    parser.add_argument("--run_dir", default="", type=str)

    args = parser.parse_args()

    assert args.seed is not None, "must provide seed for experiment"
    assert args.log_dir is not None, "must provide a log directory for experiment"

    intensities = {0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
    assert (
        args.distracting_cs_intensity in intensities
    ), f"distracting_cs has only been implemented for intensities: {intensities}"

    args.train_steps_str = args.train_steps
    args.train_steps = args.train_steps.replace("k", "000")
    args.train_steps = int(args.train_steps.replace("M", "000000"))
    args.save_freq = int(args.save_freq.replace("k", "000"))
    args.eval_freq = int(args.eval_freq.replace("k", "000"))

    if args.eval_mode == "none":
        args.eval_mode = None

    return args
