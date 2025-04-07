import collections
import copy
import time
from typing import Dict, List, Tuple, Union

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb
from matplotlib import cm, font_manager
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

from robot_learning.data.dataclasses import Transition
from robot_learning.envs.utils import make_envs
from robot_learning.models.image_embedder import ImageEmbedder
from robot_learning.models.policy.action_chunking_transformer_decoder import (
    ACTTemporalEnsembler,
)
from robot_learning.utils.general_utils import to_numpy
from robot_learning.utils.logger import log


def annotate_video(
    video: np.ndarray,
    annotations: Dict,
    img_size: Tuple[int, int] = (256, 256),
):
    # load a nice big readable font
    font = font_manager.FontProperties(family="sans-serif", weight="bold")
    file = font_manager.findfont(font)
    font = ImageFont.truetype(file, 12)

    annotated_imgs = []
    for step, frame in enumerate(video):
        frame = Image.fromarray(frame)
        frame = frame.resize(img_size)

        # add border on top of image
        extra_border_height = 100
        annotated_img = Image.new(
            "RGB",
            (frame.width, frame.height + extra_border_height),
            color=(255, 255, 255),
        )
        annotated_img.paste(frame, (0, extra_border_height))
        draw = ImageDraw.Draw(annotated_img)

        count = 0
        lines = []
        to_display = ""
        num_keys_per_line = 2

        for key, values in annotations.items():
            if isinstance(values[step], np.ndarray):
                values = np.round(values[step], 3)
                to_add = f"{key}: {values}  "
            elif isinstance(values[step], float) or isinstance(
                values[step], np.float32
            ):
                to_add = f"{key}: {values[step]:.3f}  "
            else:
                to_add = f"{key}: {values[step]}  "

            if count < num_keys_per_line:
                to_display += to_add
                count += 1
            else:
                lines.append(to_display)
                count = 1
                to_display = to_add

        # add the last line
        if lines:
            lines.append(to_display)

        for i, line in enumerate(lines):
            # make font size bigger
            draw.text((10, 10 + i * 20), line, fill="black", font=font)

        # convert to numpy array
        annotated_imgs.append(np.array(annotated_img))

    return np.array(annotated_imgs)


def run_eval_rollouts(cfg: DictConfig, model: nn.Module, wandb_run=None):
    """
    Run evaluation rollouts
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollout_time = time.time()
    cfg.env.num_envs = cfg.num_eval_rollouts
    envs = make_envs(training=False, **cfg.env)

    use_pretrained_img_embed = False
    for modality in cfg.model.input_modalities:
        if "embedding" in modality:
            use_pretrained_img_embed = True
            break

    if use_pretrained_img_embed:
        img_embedder = ImageEmbedder(
            model_name=cfg.model.embedding_model,
            device=device,
            # feature_map_layer=cfg.model.resnet_feature_map_layer,
        )
    else:
        img_embedder = None

    eval_metrics, rollouts = rollout_helper(
        cfg,
        envs,
        model=model,
        img_embedder=img_embedder,
        log_videos=False,
        device=device,
    )

    envs.viewer = None
    envs.close()
    del envs

    rollout_time = time.time() - rollout_time

    eval_metrics["time"] = rollout_time

    # compute normalized returns if d4rl mujoco env
    if cfg.env.env_name == "mujoco":
        # create a dummy env
        env_cfg = copy.deepcopy(cfg.env)
        env_cfg.num_envs = 1
        dummy_env = make_envs(training=False, **env_cfg)
        normalized_return = dummy_env.envs[0].get_normalized_score(
            eval_metrics["ep_ret"]
        )
        normalized_return_std = dummy_env.envs[0].get_normalized_score(
            eval_metrics["ep_ret_std"]
        )
        eval_metrics["normalized_ret"] = normalized_return
        eval_metrics["normalized_ret_stderr"] = normalized_return_std / np.sqrt(
            cfg.num_eval_rollouts
        )

    # for some environments, we need to run it sequentially to generate the videos
    # this is just for the rendering rollout videos
    if cfg.log_rollout_videos:
        cfg = copy.deepcopy(cfg)
        cfg.env.num_envs = 1
        video_env = make_envs(training=False, **cfg.env)

        rollout_videos = []
        for _ in tqdm.tqdm(
            range(cfg.num_eval_rollouts_render), desc="generating videos"
        ):
            _, rollouts = rollout_helper(
                cfg,
                video_env,
                model,
                img_embedder=img_embedder,
                log_videos=cfg.log_rollout_videos,
                device=device,
            )

            # squeeze the N envs dimension
            info = rollouts["info"]
            rollouts = {k: v.squeeze() for k, v in rollouts.items() if k != "info"}
            rollouts["info"] = info
            num_steps = rollouts["observation"].shape[0]
            ret = rollouts["reward"].cumsum(axis=0)

            annotations = {
                "S": np.arange(num_steps),
                "Ret": ret,
                "R": rollouts["reward"],
                "D": rollouts["done"],
                "A": rollouts["action"],
            }

            # select keys from the environment infos to log in our rollout video
            if cfg.env.env_name == "metaworld":
                env_ann = {
                    "Grasp": rollouts["info"]["grasp_success"].squeeze().astype(int),
                    "Dist": rollouts["info"]["obj_to_target"].squeeze(),
                    "SUC": rollouts["info"]["success"].squeeze().astype(int),
                }
            elif cfg.env.env_name == "mujoco":
                env_ann = {
                    "Rew_ctrl": rollouts["info"]["reward_ctrl"].squeeze(),
                    "Rew_run": rollouts["info"]["reward_run"].squeeze(),
                }
            else:
                env_ann = {}

            annotations.update(env_ann)
            annotated_video = annotate_video(rollouts["image"], annotations)
            rollout_videos.append(annotated_video)

        rollout_videos = np.array(rollout_videos)

        if wandb_run is not None:
            rollout_videos = einops.rearrange(rollout_videos, "n t h w c -> n t c h w")
            wandb_run.log(
                {"rollout_videos/": wandb.Video(rollout_videos, fps=cfg.video_fps)}
            )

        video_env.viewer = None
        video_env.close()
        del video_env

    return eval_metrics, rollouts


def rollout_helper(
    cfg: DictConfig,
    env,
    model: nn.Module,
    img_embedder: ImageEmbedder,
    log_videos: bool = False,
    device: torch.device = torch.device("cuda"),
) -> Union[Dict[str, np.ndarray], List[Transition]]:
    """
    Return:
        eval_metrics: Dict of evaluation metrics
        transitions: List of transitions
    """
    # gym version needs to be gym==0.23.1 for this to work
    log(f"running rollouts, log videos: {log_videos}", color="green")

    obs = env.reset()
    if cfg.env.env_name == "metaworld":
        obs, info = obs

    n_envs = obs.shape[0]

    curr_timestep = 0
    dones = np.zeros((cfg.num_eval_rollouts,))
    ep_returns = np.zeros((cfg.num_eval_rollouts,))
    ep_lengths = np.zeros((cfg.num_eval_rollouts,))
    ep_success = np.zeros((cfg.num_eval_rollouts,))

    rollouts = collections.defaultdict(list)

    use_pretrained_img_embed = False
    for modality in cfg.model.input_modalities:
        if "embedding" in modality:
            use_pretrained_img_embed = True
            break

    if cfg.model.name == "act_transformer":
        model.reset(n_envs)
    elif cfg.model.name == "ac_decoder":
        temporal_ensembler = ACTTemporalEnsembler(
            temporal_ensemble_coeff=cfg.model.temporal_ensemble_coeff,
            chunk_size=cfg.data.seq_len,
        )
        temporal_ensembler.reset()

    while not np.all(dones):
        # break after max timesteps
        if curr_timestep >= cfg.env.max_episode_steps:
            break

        obs = torch.from_numpy(obs).to(device).float()

        timesteps = (
            torch.arange(cfg.model.seq_len).unsqueeze(0).to(device).repeat(n_envs, 1)
        )
        # TODO: fix
        model_inputs = {
            "states": obs,
            "images": obs,
            "image_embeddings": None,
            "timesteps": timesteps,
        }

        # TODO: want to make this a wrapper
        if cfg.env.env_name == "calvin":
            model_inputs["states"] = model_inputs["states"][:, :15]  # robot state

            if n_envs == 1:
                image_dict = env.call("get_camera_obs")[0][0]
                images = image_dict["rgb_static"]
                images = images[None, :]
            else:
                image_dicts = env.env_method("get_camera_obs")
                images = [image_dict[0]["rgb_static"] for image_dict in image_dicts]

            # to tensor
            images = np.array(images)
            images = torch.from_numpy(images).to(device).float()

            # normalize and reshape so channel comes first
            images = images / 255.0
            images = images.permute(0, 3, 1, 2)
            model_inputs["images"] = images

            # also add embeddings
            image_embeddings = img_embedder(images)
            model_inputs["image_embeddings"] = image_embeddings

        if cfg.model.name == "act_transformer":
            if cfg.env.image_obs:
                # here we assume the obs from the environment are images
                states = info["state"][:, :4]  # only hand eef and gripper
                states = torch.from_numpy(states).to(device).float()

                if cfg.env.image_obs and use_pretrained_img_embed:
                    image_embeddings = img_embedder(obs).unsqueeze(1)
                else:
                    image_embeddings = None

                # the unsqueeze is to match training time shape of chunk size
                actions = model.select_action(
                    states=states.unsqueeze(1),
                    images=obs.unsqueeze(1),
                    image_embeddings=image_embeddings,
                )
            else:
                actions = model.select_action(states=obs.unsqueeze(1))

            # predicts a chunk, let's take the first one for now
            # TODO: implement temporal ensembling
            actions = actions[:, 0]
        elif cfg.model.name == "ac_decoder" or cfg.model.name == "mlp":
            # the unsqueeze is to match training time shape of chunk size
            for k, v in model_inputs.items():
                if v is not None:
                    model_inputs[k] = v.unsqueeze(1)

            actions = model.select_action(model_inputs, sample=False)
        else:
            actions = model(obs, decode_latent_action=True)

        if cfg.model.use_temporal_ensembling:
            action = temporal_ensembler.update(actions)
            action = to_numpy(action)

        actions = to_numpy(actions)

        # step in the environment
        if cfg.env.env_name == "metaworld":
            obs, reward, done, _, info = env.step(action)
        else:
            if cfg.data.seq_len > 1:
                if cfg.model.use_temporal_ensembling:
                    obs, reward, done, info = env.step(action)
                else:
                    # open loop
                    for i in range(cfg.data.seq_len // 2):
                        action = actions[:, i]
                        obs, reward, done, info = env.step(action)
            else:
                obs, reward, done, info = env.step(actions)

        # update episode returns and lengths
        dones = np.logical_or(dones, done)
        ep_rew = reward * (1 - dones)
        step = np.ones_like(dones) * (1 - dones)
        ep_returns += ep_rew
        ep_lengths += step

        if cfg.env.env_name == "metaworld":
            if "final_info" in info:
                info = rollouts["info"][-1]  # use the last info
            else:
                ep_success = np.logical_or(ep_success, info["success"])
        else:
            # this is for CALVIN
            success = [info[i]["success"] for i in range(n_envs)]
            ep_success = np.logical_or(ep_success, success)

        # generate image frames for the video
        if log_videos:
            if cfg.env.env_name == "procgen":
                image = obs
            else:
                if cfg.env.env_name == "metaworld":
                    image = env.call("render")
                else:
                    # need to render each env separately
                    image = env.call("render", mode="rgb_array")
                # tuple to array
                image = np.array(image)

                # if cfg.env.env_name == "metaworld":
                #     # flip the image vertically
                #     # test
                #     image[0] = np.flipud(image[0])
        else:
            image = None

        # fix the last info timestep for mujoco hc because DMC env returns
        # extra keys in the info dict
        if cfg.env.env_name == "mujoco" and "terminal_observation" in info[0]:
            for i in range(n_envs):
                info[i].pop("terminal_observation")
                info[i].pop("TimeLimit.truncated")

        rollouts["observation"].append(obs)
        rollouts["action"].append(action)
        rollouts["reward"].append(reward)
        rollouts["done"].append(done)
        rollouts["image"].append(image)
        rollouts["info"].append(info)
        curr_timestep += 1

    # convert to numpy arrays
    for k, v in rollouts.items():
        if not isinstance(v[0], dict) and v[0] is not None:
            rollouts[k] = np.array(v)
            rollouts[k] = np.swapaxes(rollouts[k], 0, 1)

    # infos is a list of dicts
    infos = rollouts["info"]

    if cfg.env.env_name == "metaworld":
        # flatten the timesteps
        infos = {k: np.array([info[k] for info in infos]) for k in infos[0].keys()}

    else:
        # first flatten the envs
        infos_env = []
        for step in infos:
            _infos_env = {
                k: np.array([info[k] for info in step]) for k in step[0].keys()
            }
            infos_env.append(_infos_env)

        # then flatten to timestep
        infos = {
            k: np.array([info[k] for info in infos_env]) for k in infos_env[0].keys()
        }
    rollouts["info"] = infos

    eval_metrics = {
        "ep_ret": np.mean(ep_returns),
        "ep_ret_std": np.std(ep_returns),
        "avg_len": np.mean(ep_lengths),
        "std_len": np.std(ep_lengths),
        "max_len": np.max(ep_lengths),
        "min_len": np.min(ep_lengths),
        "success_rate": np.mean(ep_success),
    }

    return eval_metrics, rollouts
