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
from PIL import Image

from robot_learning.data.utils import (
    create_dataset_name,
    get_base_trajectory,
    save_dataset,
)
from robot_learning.models.image_embedder import ImageEmbedder
from robot_learning.utils.logger import log


def load_and_process_images(
    image_dir: str, cfg, embedder=None, is_depth: bool = False
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

    if not cfg.save_imgs:
        return None, None

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

    # Compute embeddings if requested (only for RGB images)
    embeddings = None
    if cfg.precompute_embeddings and embedder is not None and not is_depth:
        # Split images to avoid memory issues
        imgs_split = np.array_split(images, 2)
        embeddings = []
        for img_batch in imgs_split:
            embeddings.append(embedder(img_batch).detach().cpu().numpy())
            # clean up gpu memory
            torch.cuda.empty_cache()

        embeddings = np.concatenate(embeddings)

    return processed_images, embeddings


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
    traj_dirs = sorted(glob(str(data_dir) + "/*"))
    trajectories = []

    log(f"Processing {len(traj_dirs)} trajectory groups", "yellow")

    num_transitions = 0

    for traj_dir in tqdm.tqdm(traj_dirs, desc="Processing traj groups"):
        data_files = sorted(glob(traj_dir + "/*"))

        # Check for required files
        obs_dict_file = [f for f in data_files if "obs_dict.pkl" in f]
        policy_out_file = [f for f in data_files if "policy_out.pkl" in f]

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
        camera_embeddings = {}

        # Process each available camera
        for camera_type, camera_dir in camera_mapping.items():
            is_depth = camera_type == "depth"
            imgs, embeddings = load_and_process_images(
                camera_dir, cfg, embedder if not is_depth else None, is_depth=is_depth
            )

            if imgs is not None:
                camera_imgs[f"{camera_type}_images"] = imgs
            if embeddings is not None:
                camera_embeddings[f"{camera_type}_img_embeds"] = embeddings

        # Load metadata
        obs_dict = load_metadata(obs_dict_file[0])
        policy_out = load_metadata(policy_out_file[0])

        # Create trajectory data dictionary
        traj_data = {
            "states": obs_dict["state"],
            "actions": policy_out["actions"],
            "rewards": np.zeros(len(policy_out["actions"])),
            **camera_imgs,
            **camera_embeddings,
        }
        base_trajectory = get_base_trajectory(traj_data["rewards"])
        traj = {**base_trajectory, **traj_data}

        num_transitions += len(traj_data["rewards"])
        trajectories.append(traj)

        if cfg.debug:
            break

    if trajectories:
        traj = trajectories[0]
        for k, v in traj.items():
            log(f"{k}: {v.shape}", "yellow")

    log(f"Total transitions: {num_transitions} collected", "green")

    # Save dataset
    save_dataset(trajectories, save_file)


if __name__ == "__main__":
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    main()
