"""
Script to convert replay buffer data to TensorFlow Datasets (TFDS) format.

Example usage:
python3 scripts/convert_buffer_to_tfds.py \
    task_name=mw-assembly \
    ckpt_num=100000 \
    precompute_embeddings=True \
    compute_2d_flow=True \
    embedding_model=resnet50 \
    resnet_feature_map_layer=avgpool \
    data_dir=/scr/shared/clam/datasets/metaworld/tdmpc2_buffer_imgs \
    tfds_data_dir=/scr/shared/clam/tensorflow_datasets \
    dataset_name=tdmpc2-buffer \
    save_imgs=True \
    debug=True

python3 scripts/convert_buffer_to_tfds.py \
    task_name=mw-door-open \
    ckpt_num=100000 \
    precompute_embeddings=True \
    compute_2d_flow=True \
    embedding_model=r3m \
    data_dir=/scr/shared/clam/datasets/metaworld/tdmpc2_buffer_imgs \
    tfds_data_dir=/scr/shared/clam/tensorflow_datasets \
    dataset_name=tdmpc2-buffer \
    save_imgs=True \
    debug=True
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import hydra
import numpy as np
import torch
import tqdm
from cotracker.utils.visualizer import Visualizer
from omegaconf import DictConfig

from robot_learning.data.optical_flow.compute_2d_flow import (
    generate_point_tracks,
    get_seg_mask,
    load_cotracker,
    load_sam_model,
)
from robot_learning.data.utils import save_dataset
from robot_learning.models.image_embedder import ImageEmbedder
from robot_learning.utils.general_utils import to_numpy
from robot_learning.utils.logger import log


@dataclass
class EpisodeData:
    """Container for processed episode data from replay buffer."""

    observations: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[np.ndarray]
    episodes: List[np.ndarray]
    images: Optional[List[np.ndarray]] = None


def load_replay_buffer(cfg) -> Dict[str, torch.Tensor]:
    """Load replay buffer data from disk."""
    replay_buffer = Path(cfg.data_dir) / f"buffer_{cfg.task_name}_{cfg.ckpt_num}.pt"
    return torch.load(str(replay_buffer))


def process_episodes(
    td: Dict[str, torch.Tensor],
) -> EpisodeData:
    """Split replay buffer data into episodes and filter invalid ones.

    Args:
        td: Dictionary containing replay buffer tensors

    Returns:
        EpisodeData containing processed and filtered episode data
    """
    # Convert tensors to numpy and get episode splits
    observations = td["obs"].cpu().numpy()
    actions = td["action"].cpu().numpy()
    rewards = td["reward"].cpu().numpy()
    episodes = td["episode"].cpu().numpy()

    # Split data into episodes
    episode_split = np.where(episodes[1:] != episodes[:-1])[0] + 1
    observations = np.split(observations, episode_split)
    actions = np.split(actions, episode_split)
    rewards = np.split(rewards, episode_split)
    episodes = np.split(episodes, episode_split)

    # Filter episodes to ensure consistent lengths
    episode_lens = [len(ep) for ep in observations]
    valid_episodes_id = np.where(np.array(episode_lens) == episode_lens[100])[0]

    observations = [observations[i] for i in valid_episodes_id]
    actions = [actions[i] for i in valid_episodes_id]
    rewards = [rewards[i] for i in valid_episodes_id]
    episodes = [episodes[i] for i in valid_episodes_id]

    # Handle images if present
    images = None
    if "image" in td:
        images = np.split(td["image"].cpu().numpy(), episode_split)
        images = [images[i] for i in valid_episodes_id]

    return EpisodeData(
        observations=observations,
        actions=actions,
        rewards=rewards,
        episodes=episodes,
        images=images,
    )


def compute_image_embeddings(
    images: List[np.ndarray], embedder: ImageEmbedder
) -> List[np.ndarray]:
    """Compute embeddings for a sequence of images using the specified embedder."""
    log(f"Computing image embeddings using {embedder.model_name}")
    embeddings = []

    for video in tqdm.tqdm(images, desc="computing embeddings"):
        with torch.no_grad():
            emb = embedder(np.array(video))
        embeddings.append(emb.cpu().numpy())

    return embeddings


def compute_flow_features(
    cfg, images: List[np.ndarray], save_file: Path
) -> List[Dict[str, np.ndarray]]:
    """Compute optical flow features for a sequence of images."""
    point_tracking_results = []

    # Initialize models
    sam, image_predictor = load_sam_model(cfg.flow)
    cotracker = load_cotracker(cfg.flow)

    # Initialize visualizer
    vis = Visualizer(
        save_dir=save_file.parent / (save_file.name + "_point_tracks"),
        pad_value=10,
        linewidth=2,
    )

    for indx, video in enumerate(tqdm.tqdm(images, desc="computing 2d flow")):
        # Segment out the table
        table_mask = get_seg_mask(
            cfg.flow, sam, image_predictor, video=video, text="table."
        )

        # Get object segmentation mask
        segm_mask = get_seg_mask(cfg.flow, sam, image_predictor, video=video)

        # Make all positive values 1
        segm_mask = (segm_mask > 0).astype(np.uint8)
        table_mask = (table_mask > 0).astype(np.uint8)

        # Zero out table in segm_mask
        segm_mask = segm_mask * (1 - table_mask)

        if cfg.debug:
            # Save binary segmentation mask somewhere
            import matplotlib.pyplot as plt

            ax = plt.subplot(1, 4, 1)
            ax.imshow(video[0])

            ax = plt.subplot(1, 4, 2)
            ax.imshow(table_mask)

            ax = plt.subplot(1, 4, 3)
            ax.imshow(segm_mask)

            ax = plt.subplot(1, 4, 4)
            ax.imshow((segm_mask[..., None] * video[0] / 255.0))
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_file / f"seg_mask_vis_{indx:04d}.png")

        # Track points across video
        points, visibility = generate_point_tracks(
            cfg.flow, cotracker, video, segm_mask
        )

        log(f"Points shape: {points.shape}, visibility shape: {visibility.shape}")

        # Process and store results
        tracked_points = {
            "points": to_numpy(points[0]),
            "visibility": to_numpy(visibility[0]),
        }
        point_tracking_results.append(tracked_points)

        # Visualize tracks
        vis.visualize(
            video=torch.from_numpy(video).permute(0, 3, 1, 2)[None],
            tracks=points,
            visibility=visibility,
            filename=f"point_track_{indx:04d}",
        )

    return point_tracking_results


def process_framestack(cfg, images: List[np.ndarray]) -> List[np.ndarray]:
    """Apply frame stacking to a sequence of images."""
    log(f"Framestacking images with framestack {cfg.framestack}")
    processed_images = []

    for ep_imgs in images:
        # Pad with first frame
        first_frame = np.repeat(ep_imgs[0][None], cfg.framestack - 1, axis=0)
        ep_imgs = np.concatenate([first_frame, ep_imgs], axis=0)

        # Create framestack
        new_imgs = []
        for i in range(len(ep_imgs) - cfg.framestack + 1):
            new_imgs.append(ep_imgs[i : i + cfg.framestack])

        processed_images.append(np.array(new_imgs))

    return processed_images


def create_trajectory(
    cfg: DictConfig,
    obs: np.ndarray,
    act: np.ndarray,
    rew: np.ndarray,
    camera_imgs: Dict[str, np.ndarray],
    camera_embeddings: Dict[str, np.ndarray],
    flow_data: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    """Create a single trajectory dictionary from episode data."""
    trajectory = {
        "states": obs,
        "actions": act,
        "rewards": rew,
        "discount": np.ones_like(rew),
        "is_last": np.zeros_like(rew),
        "is_first": np.zeros_like(rew),
        "is_terminal": np.zeros_like(rew),
    }

    # Set episode boundary flags
    trajectory["is_last"][-1] = 1
    trajectory["is_terminal"][-1] = 1
    trajectory["is_first"][0] = 1

    # Add optional data
    for camera_name, camera_imgs in camera_imgs.items():
        trajectory[f"{camera_name}_imgs"] = camera_imgs

    for camera_name, camera_embeddings in camera_embeddings.items():
        trajectory[f"{camera_name}_img_embeds"] = camera_embeddings

    if flow_data is not None:
        points = flow_data["points"][:-1]
        viz = flow_data["visibility"][:-1]
        mask = np.ones_like(viz)

        # Pad or truncate to max_query_points
        num_points = points.shape[1]
        if num_points < cfg.flow.max_query_points:
            pad = cfg.flow.max_query_points - num_points
            points = np.pad(points, ((0, 0), (0, pad), (0, 0)))
            viz = np.pad(viz, ((0, 0), (0, pad)))
            mask = np.pad(mask, ((0, 0), (0, pad)))
        elif num_points > cfg.flow.max_query_points:  # crop to max_query_points
            points = points[:, : cfg.flow.max_query_points]
            viz = viz[:, : cfg.flow.max_query_points]
            mask = mask[:, : cfg.flow.max_query_points]

        trajectory.update(
            {
                "points": points,
                "points_viz": viz,
                "points_mask": mask,
            }
        )

    return trajectory


def create_dataset_name(cfg) -> str:
    """Create descriptive dataset name based on important configuration parameters.

    Args:
        cfg: Configuration object containing dataset parameters

    Returns:
        String containing formatted dataset name with relevant parameters
    """
    # Start with base name
    name_parts = [cfg.dataset_name]

    # Add data feature indicators
    if cfg.save_imgs:
        name_parts.append("imgs")
        if cfg.black_white:
            name_parts.append("bw")
        if cfg.framestack > 1:
            name_parts.append(f"fs{cfg.framestack}")

    if cfg.precompute_embeddings:
        name_parts.append(f"emb-{cfg.embedding_model}")

        if "resnet" in cfg.embedding_model:
            name_parts.append(f"l-{cfg.resnet_feature_map_layer}")

    if cfg.compute_2d_flow:
        name_parts.append("flow")

    # Add debug indicator if in debug mode
    if cfg.debug:
        name_parts.append("debug")

    # Join all parts with hyphens
    return "_".join(name_parts)


@hydra.main(version_base=None, config_name="convert_to_tfds", config_path="../cfg")
def main(cfg):
    """Main function to convert replay buffer to TFDS format."""
    # Generate dataset name
    dataset_name = create_dataset_name(cfg)

    # Initialize embedder if requested
    embedder = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}", color="blue")
    if cfg.precompute_embeddings:
        embedder = ImageEmbedder(
            model_name=cfg.embedding_model,
            device=device,
            feature_map_layer=cfg.resnet_feature_map_layer,
        )

    # Load and process data
    td = load_replay_buffer(cfg)
    episode_data = process_episodes(td)

    # Use episode_data attributes instead of unpacking tuple
    observations = episode_data.observations
    actions = episode_data.actions
    rewards = episode_data.rewards
    images = episode_data.images

    # Create save directory
    save_dir = Path(cfg.tfds_data_dir) / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log(f"Logging to {save_dir}")

    # Set up save path
    save_file = save_dir / cfg.task_name
    save_file.mkdir(parents=True, exist_ok=True)
    if cfg.debug:
        images = images[:3] if images is not None else None

    # Process images if available
    save_imgs = cfg.save_imgs and images is not None
    image_embeddings = None
    flow_features = None

    if save_imgs:
        if cfg.precompute_embeddings:
            image_embeddings = compute_image_embeddings(images, embedder)
        if cfg.compute_2d_flow:
            flow_features = compute_flow_features(cfg, images, save_file)
        if cfg.framestack > 1:
            images = process_framestack(cfg, images)

    # Create trajectories
    trajectories = []
    returns = []
    num_trajs = 3 if cfg.debug else len(observations)

    for idx in tqdm.tqdm(range(num_trajs), desc="processing trajectories"):
        # Skip first timestep
        obs = observations[idx][:-1]
        act = actions[idx][1:]
        rew = rewards[idx][1:]

        # Get optional data for this trajectory
        traj_imgs = images[idx][:-1] if save_imgs else None
        traj_emb = image_embeddings[idx][:-1] if image_embeddings else None
        traj_flow = flow_features[idx] if flow_features else None

        # Create and store trajectory
        trajectory = create_trajectory(
            cfg, obs, act, rew, traj_imgs, traj_emb, traj_flow
        )
        trajectories.append(trajectory)
        returns.append(np.sum(rew))

    dataset_kwargs = {
        "env_name": cfg.env_name,
        "save_imgs": save_imgs,
        "framestack": cfg.framestack,
        "precompute_embeddings": cfg.precompute_embeddings,
        "has_flow": cfg.compute_2d_flow,
        "embedding_dim": embedder.output_dim if embedder is not None else 0,
    }

    # Save full dataset
    save_dataset(trajectories, save_file, **dataset_kwargs)

    # Save filtered datasets based on returns
    sorted_returns = np.argsort(returns)

    # Save medium/random trajectories
    medium_trajs = [
        traj for traj, ret in zip(trajectories, returns) if ret < cfg.max_return
    ]
    save_file = Path(cfg.tfds_data_dir) / dataset_name / f"{cfg.task_name}-al"
    save_dataset(medium_trajs, save_file, **dataset_kwargs)

    # Save expert trajectories
    expert_trajs = [trajectories[i] for i in sorted_returns[-cfg.num_samples :]]
    save_file = Path(cfg.tfds_data_dir) / dataset_name / f"{cfg.task_name}-expert"
    save_dataset(expert_trajs, save_file, **dataset_kwargs)

    # Save random/medium trajectories of different sizes
    for num_samples in [cfg.num_samples, cfg.num_samples * 2]:
        random_med = [trajectories[i] for i in sorted_returns[:num_samples]]
        save_file = (
            Path(cfg.tfds_data_dir) / dataset_name / f"{cfg.task_name}-rm-{num_samples}"
        )
        save_dataset(random_med, save_file, **dataset_kwargs)


if __name__ == "__main__":
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    main()
