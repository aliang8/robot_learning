"""
Script to convert robot demonstration data into tfds format.

Usage:
    python3 scripts/convert_robot_to_tfds.py \
        dataset_name=robot_play \
        task_name=pick_up_green_object_expert \
        precompute_embeddings=True \
        save_imgs=True \
        embedding_model=r3m
"""

import os
import re
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import tensorflow as tf
import torch
import tqdm
from omegaconf import DictConfig
from PIL import Image

from robot_learning.data.utils import (
    create_dataset_name,
    get_base_trajectory,
    save_dataset,
)
from robot_learning.models.image_embedder import ImageEmbedder
from robot_learning.utils.general_utils import to_numpy
from robot_learning.utils.logger import log


def load_images(
    cfg, image_dir: str, is_depth: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load, process images and compute embeddings if needed.

    Args:
        image_dir: Directory containing images
        cfg: Configuration object
        embedder: Optional image embedder model
        is_depth: Whether the images are depth images
    """
    # Get sorted image paths with appropriate extension
    ext = ".png" if is_depth else ".jpg"
    image_paths = sorted(
        [f for f in os.listdir(image_dir) if f.endswith(ext)],
        key=lambda x: int(re.search(r"\d+", x).group()),
    )

    # Load images
    images = [Image.open(Path(image_dir) / img_path) for img_path in image_paths]
    images = np.array([np.array(img) for img in images])[:-1]

    # Process images differently based on type
    if is_depth:
        # For depth images, use numpy-based resizing to preserve depth values
        processed_images = []
        for img in images:
            # Center crop [480, 640] -> [480, 480]
            h, w = img.shape
            start_w = (w - h) // 2
            cropped = img[:, start_w : start_w + h]

            # Resize to target size using nearest neighbor to preserve depth values
            target_size = cfg.image_size
            from scipy.ndimage import zoom

            scale = (
                target_size[0] / cropped.shape[0],
                target_size[1] / cropped.shape[1],
            )
            resized = zoom(cropped, scale, order=0)  # order=0 for nearest neighbor

            processed_images.append(resized)
        processed_images = np.array(processed_images)
    else:
        # For RGB images, use tensorflow resizing
        processed_images = [
            tf.image.resize(
                tf.image.crop_to_bounding_box(img, 0, 80, 480, 480), cfg.image_size
            )
            for img in images
        ]
        processed_images = np.array(processed_images)

    return processed_images


def load_metadata(data_file: str) -> Dict:
    """Load and process metadata from file."""
    data = np.load(data_file, allow_pickle=True)
    if isinstance(data, list):  # For policy output
        return {k: np.array([p[k] for p in data]) for k in data[0].keys()}
    return {k: data[k][:-1] for k in data.keys()}  # For observation dict


def get_available_cameras(data_files: List[str]) -> Dict[str, str]:
    """
    Get mapping of available camera types to their directories.

    Args:
        data_files: List of paths in the trajectory directory
    Returns:
        Dictionary mapping camera names to their directory paths
    """
    camera_mapping = {}
    for file_path in data_files:
        if "depth_images" in file_path:
            camera_mapping["depth"] = file_path
        elif "external" in file_path:
            camera_mapping["external"] = file_path
        elif "over_shoulder" in file_path:
            camera_mapping["over_shoulder"] = file_path
        elif "wrist" in file_path:
            camera_mapping["wrist"] = file_path
    return camera_mapping


def load_robot_data(cfg: DictConfig, data_dir: str):
    """
    Assumes robot data is stored in the following format:
    data_dir/
        traj_0/
            obs_dict.pkl
            policy_out.pkl
            depth_images/
            external_images/
    """
    traj_dirs = sorted(glob(str(data_dir) + "/traj*"))
    # filter only folders
    traj_dirs = [d for d in traj_dirs if os.path.isdir(d)]
    log(f"Processing {len(traj_dirs)} trajectories", "yellow")

    if cfg.debug:
        traj_dirs = traj_dirs[:2]

    trajectories = []
    for traj_dir in tqdm.tqdm(traj_dirs, desc="Processing traj groups"):
        data_files = sorted(glob(traj_dir + "/*"))

        # Check for required files
        obs_dict_file = Path(traj_dir) / "obs_dict.pkl"
        policy_out_file = Path(traj_dir) / "policy_out.pkl"

        if not (obs_dict_file and policy_out_file):
            log(f"Skipping {traj_dir} - missing required files", "red")
            continue

        # Get available cameras
        camera_mapping = get_available_cameras(data_files)
        if not camera_mapping:
            log(f"Skipping {traj_dir} - no camera data found", "red")
            continue

        # Initialize storage for images and embeddings
        camera_imgs = {}

        # Process each available camera
        for camera_type, camera_dir in camera_mapping.items():
            is_depth = camera_type == "depth"
            imgs = load_images(cfg, camera_dir, is_depth=is_depth)

            if imgs is not None:
                camera_imgs[f"{camera_type}_images"] = imgs

        # Load metadata
        obs_dict = load_metadata(obs_dict_file)
        policy_out = load_metadata(policy_out_file)
        trajectories.append([obs_dict, policy_out, camera_imgs])

    return trajectories


@hydra.main(
    version_base=None, config_name="convert_robot_to_tfds", config_path="../cfg"
)
def main(cfg):
    """Main function to convert replay buffer to TFDS format."""
    # Generate dataset name
    dataset_name = create_dataset_name(cfg)

    # Initialize embedder if requested
    embedder = None
    if cfg.precompute_embeddings:
        embedder = ImageEmbedder(
            model_name=cfg.embedding_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    # Create save directory
    save_dir = Path(cfg.tfds_data_dir) / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_dir / cfg.task_name

    log(
        f"------------------- Saving dataset to {save_file} -------------------", "blue"
    )

    data_dir = Path(cfg.data_dir) / cfg.dataset_name / cfg.task_name

    num_transitions = 0
    processed_trajs = []
    trajectories = load_robot_data(cfg, data_dir)
    for traj in trajectories:
        obs_dict, policy_out, camera_imgs = traj

        # Compute embeddings if requested (only for RGB images)
        camera_embeddings = {}
        if cfg.precompute_embeddings and embedder is not None:
            for camera_type, images in camera_imgs.items():
                embeddings = None
                if "depth" not in camera_type:
                    camera_type = camera_type.replace("_images", "")
                    # Split images to avoid memory issues
                    imgs_split = np.array_split(images, 2)
                    embeddings = []
                    for img_batch in imgs_split:
                        embeddings.append(to_numpy(embedder(img_batch)))
                        # clean up gpu memory
                        torch.cuda.empty_cache()

                    embeddings = np.concatenate(embeddings)
                    camera_embeddings[f"{camera_type}_img_embeds"] = embeddings

        # Create trajectory data dictionary
        traj_data = {
            "states": obs_dict["state"],
            "actions": policy_out["actions"],
            "rewards": np.zeros(len(policy_out["actions"])),
            **camera_imgs,
            **camera_embeddings,
        }
        base_trajectory = get_base_trajectory(traj_data["rewards"])
        traj_data = {**base_trajectory, **traj_data}

        num_transitions += len(traj_data["rewards"])
        processed_trajs.append(traj_data)

        if cfg.debug:
            break

    if processed_trajs:
        traj = processed_trajs[0]
        for k, v in traj.items():
            log(f"{k}: {v.shape}", "yellow")

    log(f"Total transitions: {num_transitions} collected", "green")
    save_dataset(processed_trajs, save_file)


if __name__ == "__main__":
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    main()
