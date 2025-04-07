from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import tensorflow as tf

from robot_learning.utils.logger import log


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


def get_base_trajectory(rew: np.ndarray):
    trajectory = {
        "discount": np.ones_like(rew),
        "is_last": np.zeros_like(rew),
        "is_first": np.zeros_like(rew),
        "is_terminal": np.zeros_like(rew),
    }

    # Set episode boundary flags
    trajectory["is_last"][-1] = 1
    trajectory["is_terminal"][-1] = 1
    trajectory["is_first"][0] = 1
    return trajectory


def save_dataset(trajectories, save_file: Path, save_imgs: bool = False):
    """Save trajectory data as TFDS."""
    log(f"Saving dataset to: {save_file}", "green")
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # Create tfds from generator
    def generator():
        for trajectory in trajectories:
            yield trajectory

    # Create features dict
    feature_keys = trajectories[0].keys()
    # remove batch dimension
    feature_shapes = {k: trajectories[0][k].shape[1:] for k in feature_keys}
    features_dict = {
        k: tf.TensorSpec(
            shape=(None, *feature_shapes[k]), dtype=trajectories[0][k].dtype
        )
        for k in feature_keys
    }
    log(f"Features: {features_dict}", "yellow")
    trajectory_tfds = tf.data.Dataset.from_generator(
        generator, output_signature=features_dict
    )
    tf.data.experimental.save(trajectory_tfds, str(save_file))
