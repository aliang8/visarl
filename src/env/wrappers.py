import numpy as np
from numpy.random import randint
import os
import gym
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from gym.wrappers import TimeLimit
from src.env.robot import registration
import util.general_utils as utils
from collections import deque
from mujoco_py import modder

from metaworld.envs import (
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
    ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
)

from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from PIL import Image
import torchvision.models as models
from torch.nn.modules.linear import Identity
import torchvision.transforms as T
from gym.spaces.box import Box


def make_env_mw(
    domain_name,
    task_name,
    seed=0,
    episode_length=50,
    n_substeps=20,
    frame_stack=3,
    image_size=84,
    cameras=["third_person", "first_person"],
    mode="train",
    observation_type="image",
    action_space="xyzw",
    test=None,
    pretrained_encoder=None,
):
    env_cls = ALL_V2_ENVIRONMENTS[task_name]
    env = env_cls()
    env._freeze_rand_vec = False  # randomize objects
    env._set_task_called = True
    env._partially_observable = False  # doesn't matter

    # env.seed(seed)
    env = TimeLimit(env, max_episode_steps=episode_length)
    env = SuccessWrapper(env, any_success=True)

    env = ObservationSpaceWrapperMW(
        env, observation_type=observation_type, image_size=image_size, cameras=cameras
    )
    # env = ActionSpaceWrapper(env, action_space=action_space)

    if pretrained_encoder and pretrained_encoder not in ["mmae", "cnn"]:
        env = StateEmbedding(
            env,
            observation_type,
            image_size,
            cameras,
            proprio=0,
            load_path=pretrained_encoder,
        )

    if "image" in observation_type:
        env = FrameStack(env, frame_stack)
    return env


class ObservationSpaceWrapperMW(gym.Wrapper):
    def __init__(self, env, observation_type, image_size, cameras):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.observation_type = observation_type
        self.image_size = image_size
        self.cameras = cameras
        self.num_cams = len(self.cameras)

        if self.observation_type in {"image", "state+image"}:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(3 * self.num_cams, image_size, image_size),
                dtype=np.uint8,
            )

        elif self.observation_type == "state":
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(39,),
                dtype=np.float32,
            )

    def reset(self):
        obs = self.env.reset()
        return self._get_obs(obs), obs if "state" in self.observation_type else None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return (
            self._get_obs(obs),
            obs if "state" in self.observation_type else None,
            reward,
            done,
            info,
        )

    def _get_obs(self, obs_dict):
        if self.observation_type in {"image", "state+image"}:
            obs = np.empty(
                (3 * self.num_cams, self.image_size, self.image_size),
                dtype=np.uint8,
            )
            for ob in range(len(self.cameras)):
                obs[3 * ob : 3 * (ob + 1)] = self.env.unwrapped.sim.render(
                    camera_name=self.cameras[ob],
                    width=self.image_size,
                    height=self.image_size,
                ).transpose(2, 0, 1)

        elif self.observation_type == "state":
            obs = obs_dict

        return obs


def _get_embedding(embedding_name="resnet34", load_path="", *args, **kwargs):
    if load_path == "random":
        prt = False
    else:
        prt = True
    if embedding_name == "resnet34":
        model = models.resnet34(pretrained=prt, progress=False)
        embedding_dim = 512
    elif embedding_name == "resnet18":
        model = models.resnet18(pretrained=prt, progress=False)
        embedding_dim = 512
    elif embedding_name == "resnet50":
        model = models.resnet50(pretrained=prt, progress=False)
        embedding_dim = 2048
    else:
        print("Requested model not available currently")
        raise NotImplementedError
    # make FC layers to be identity
    # NOTE: This works for ResNet backbones but should check if same
    # template applies to other backbone architectures
    model.fc = Identity()
    model = model.eval()
    return model, embedding_dim


class ClipEnc(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, im):
        e = self.m.encode_image(im)
        return e


class StateEmbedding(gym.Wrapper):
    """
    This wrapper places a convolution model over the observation.
    From https://pytorch.org/vision/stable/models.html
    All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    where H and W are expected to be at least 224.
    Args:
        env (Gym environment): the original environment,
        embedding_name (str, 'baseline'): the name of the convolution model,
        device (str, 'cuda'): where to allocate the model.
    """

    def __init__(
        self,
        env,
        observation_type,
        image_size,
        cameras,
        proprio=0,
        load_path="r3m",
        device="cuda",
    ):
        gym.Wrapper.__init__(self, env)

        self.proprio = proprio
        self.load_path = load_path
        self.start_finetune = False
        self._max_episode_steps = env._max_episode_steps
        if load_path == "clip":
            import clip

            model, cliptransforms = clip.load("RN50", device="cuda")
            embedding = ClipEnc(model)
            embedding.eval()
            embedding_dim = 1024
            self.transforms = cliptransforms
        elif (load_path == "random") or (load_path == ""):
            embedding, embedding_dim = _get_embedding(
                embedding_name=embedding_name, load_path=load_path
            )
            self.transforms = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),  # ToTensor() divides by 255
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif "r3m" in load_path:
            from r3m import load_r3m

            rep = load_r3m(load_path.replace("r3m_", ""))
            rep.eval()
            embedding_dim = rep.module.outdim
            embedding = rep
            self.transforms = T.Compose(
                [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
            )  # ToTensor() divides by 255
        else:
            raise NameError("Invalid Model")
        embedding.eval()

        if device == "cuda" and torch.cuda.is_available():
            print("Using CUDA.")
            device = torch.device("cuda")
        else:
            print("Not using CUDA.")
            device = torch.device("cpu")
        self.device = device
        embedding.to(device=device)

        self.embedding, self.embedding_dim = embedding, embedding_dim
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.embedding_dim + self.proprio,)
        )

    def observation(self, observation):
        ### INPUT SHOULD BE [0,255]
        if self.embedding is not None:
            inp = self.transforms(
                Image.fromarray(observation.transpose(1, 2, 0).astype(np.uint8))
            ).reshape(-1, 3, 224, 224)
            if "r3m" in self.load_path:
                ## R3M Expects input to be 0-255, preprocess makes 0-1
                inp *= 255.0
            inp = inp.to(self.device)
            with torch.no_grad():
                emb = (
                    self.embedding(inp)
                    .view(-1, self.embedding_dim)
                    .to("cpu")
                    .numpy()
                    .squeeze()
                )

            ## IF proprioception add it to end of embedding
            if self.proprio:
                try:
                    proprio = self.env.unwrapped.get_obs()[: self.proprio]
                except:
                    proprio = self.env.unwrapped._get_obs()[: self.proprio]
                emb = np.concatenate([emb, proprio])

            return emb
        else:
            return observation

    def encode_batch(self, obs, finetune=False):
        ### INPUT SHOULD BE [0,255]
        inp = []
        for o in obs:
            i = self.transforms(Image.fromarray(o.astype(np.uint8))).reshape(
                -1, 3, 224, 224
            )
            if "r3m" in self.load_path:
                ## R3M Expects input to be 0-255, preprocess makes 0-1
                i *= 255.0
            inp.append(i)
        inp = torch.cat(inp)
        inp = inp.to(self.device)
        if finetune and self.start_finetune:
            emb = self.embedding(inp).view(-1, self.embedding_dim)
        else:
            with torch.no_grad():
                emb = (
                    self.embedding(inp)
                    .view(-1, self.embedding_dim)
                    .to("cpu")
                    .numpy()
                    .squeeze()
                )
        return emb

    def reset(self):
        obs = self.env.reset()
        return self._get_obs(obs), obs if "state" in self.observation_type else None

    def step(self, action):
        obs, state, reward, done, info = self.env.step(action)
        return (
            self._get_obs(obs),
            obs if "state" in self.observation_type else None,
            reward,
            done,
            info,
        )

    def _get_obs(self, obs_dict):
        observation = self.env._get_obs(obs_dict)

        if self.embedding is not None:
            observation = self.observation(observation)

        return observation


def make_env(
    domain_name,
    task_name,
    seed=0,
    episode_length=50,
    n_substeps=20,
    frame_stack=3,
    image_size=84,
    cameras=["third_person", "first_person"],
    mode="train",
    observation_type="image",
    action_space="xyzw",
    test=None,
):
    """Make environment for experiments"""
    assert (
        domain_name == "robot"
    ), f'expected domain_name "robot", received "{domain_name}"'
    assert action_space in {
        "xy",
        "xyz",
        "xyzw",
    }, f'unexpected action space "{action_space}"'

    registration.register_robot_envs(
        n_substeps=n_substeps,
        observation_type=observation_type,
        image_size=image_size,
        use_xyz=action_space.replace("w", "") == "xyz",
    )
    randomizations = {}
    if test == None:
        env_id = "Robot" + task_name.capitalize() + "-v0"
    else:
        env_id = "Robot" + task_name.capitalize() + f"_test_{test}" + "-v0"

    env = gym.make(
        env_id, cameras=cameras, render=False, observation_type=observation_type
    )
    env.seed(seed)
    env = TimeLimit(env, max_episode_steps=episode_length)
    env = SuccessWrapper(env, any_success=True)

    env = ObservationSpaceWrapper(
        env, observation_type=observation_type, image_size=image_size, cameras=cameras
    )
    env = ActionSpaceWrapper(env, action_space=action_space)
    env = FrameStack(env, frame_stack)

    return env


class FrameStack(gym.Wrapper):
    """Stack frames as observation"""

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        if len(shp) == 3:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=((shp[0] * k,) + shp[1:]),
                dtype=env.observation_space.dtype,
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(shp[0] * k,),
                dtype=env.observation_space.dtype,
            )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs, state_obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), state_obs

    def step(self, action):
        obs, state_obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), state_obs, reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return utils.LazyFrames(list(self._frames))


class SuccessWrapper(gym.Wrapper):
    def __init__(self, env, any_success=True):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.any_success = any_success
        self.success = False

    def reset(self):
        self.success = False
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        any_success = info["is_success"] if "is_success" in info else info["success"]
        if self.any_success:
            self.success = self.success or bool(any_success)
        else:
            self.success = bool(any_success)
        info["is_success"] = self.success
        return obs, reward, done, info


class ObservationSpaceWrapper(gym.Wrapper):
    def __init__(self, env, observation_type, image_size, cameras):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.observation_type = observation_type
        self.image_size = image_size
        self.cameras = cameras
        self.num_cams = len(self.cameras)

        if self.observation_type in {"image", "state+image"}:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(3 * self.num_cams, image_size, image_size),
                dtype=np.uint8,
            )

        elif self.observation_type == "state":
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=env.unwrapped.state_dim,
                dtype=np.float32,
            )

    def reset(self):
        obs = self.env.reset()
        return self._get_obs(obs), obs["state"] if "state" in obs else None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return (
            self._get_obs(obs),
            obs["state"] if "state" in obs else None,
            reward,
            done,
            info,
        )

    def _get_obs(self, obs_dict):
        if self.observation_type in {"image", "state+image"}:
            if self.num_cams == 1:
                return obs_dict["observation"][0].transpose(2, 0, 1)
            obs = np.empty(
                (3 * self.num_cams, self.image_size, self.image_size),
                dtype=obs_dict["observation"][0].dtype,
            )
            for ob in range(obs_dict["observation"].shape[0]):
                obs[3 * ob : 3 * (ob + 1)] = obs_dict["observation"][ob].transpose(
                    2, 0, 1
                )

        elif self.observation_type == "state":
            obs = obs_dict["observation"]

        return obs


class ActionSpaceWrapper(gym.Wrapper):
    def __init__(self, env, action_space):
        assert action_space in {
            "xy",
            "xyz",
            "xyzw",
        }, "task must be one of {xy, xyz, xyzw}"
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.action_space_dims = action_space
        self.use_xyz = "xyz" in action_space
        self.use_gripper = "w" in action_space
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 + self.use_xyz + self.use_gripper,),
            dtype=np.float32,
        )

    def step(self, action):
        assert (
            action.shape == self.action_space.shape
        ), "action shape must match action space"
        action = np.array(
            [
                action[0],
                action[1],
                action[2] if self.use_xyz else 0,
                action[3] if self.use_gripper else 1,
            ],
            dtype=np.float32,
        )
        return self.env.step(action)
