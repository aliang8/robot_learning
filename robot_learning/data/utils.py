from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import tensorflow as tf

from robot_learning.utils.logger import log


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
    tf.data.Dataset.save(trajectory_tfds, str(save_file))
