from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import tensorflow as tf

from robot_learning.utils.logger import log

# Base image sizes
HALFCHEETAH_IMAGE_SIZE = (64, 64, 3)
METAWORLD_IMAGE_SIZE = (84, 84, 3)
CALVIN_IMAGE_SIZE = (200, 200, 3)
MAX_QUERY_POINTS = 40

# Base feature specifications for different environments
BASE_FEATURES = {
    "metaworld": {
        "observations": tf.TensorSpec(shape=(None, 39), dtype=np.float32),
        "actions": tf.TensorSpec(shape=(None, 4), dtype=np.float32),
    },
    "halfcheetah": {
        "observations": tf.TensorSpec(shape=(None, 17), dtype=np.float32),
        "actions": tf.TensorSpec(shape=(None, 6), dtype=np.float32),
    },
    "calvin": {
        "observations": tf.TensorSpec(shape=(None, 15), dtype=np.float32),
        "actions": tf.TensorSpec(shape=(None, 7), dtype=np.float32),
        "scene_obs": tf.TensorSpec(shape=(None, 24), dtype=np.float32),
    },
}

# Common features across all environments
COMMON_FEATURES = {
    "discount": tf.TensorSpec(shape=(None,), dtype=np.float32),
    "rewards": tf.TensorSpec(shape=(None,), dtype=np.float32),
    "is_first": tf.TensorSpec(shape=(None,), dtype=np.bool_),
    "is_last": tf.TensorSpec(shape=(None,), dtype=np.bool_),
    "is_terminal": tf.TensorSpec(shape=(None,), dtype=np.bool_),
}


def get_features_dict(
    env_name: str,
    save_imgs: bool = False,
    framestack: int = 0,
    black_white: bool = False,
    has_flow: bool = False,
    save_costs: bool = False,
    embedding_dim: Union[int, List[int]] = 2048,
) -> Dict:
    """Dynamically construct features dictionary based on available data.

    Args:
        env_name: Name of the environment ('metaworld' or 'halfcheetah')
        save_imgs: Whether image data is included
        framestack: Number of frames to stack (if > 1)
        black_white: Whether images are grayscale
        has_flow: Whether optical flow features are included
        emb_output_dim: Dimensionality of image embeddings

    Returns:
        Dictionary of TensorSpec features
    """
    # Start with base and common features
    features = {**BASE_FEATURES[env_name], **COMMON_FEATURES}

    # Add image features if needed
    if save_imgs:
        if env_name == "calvin":
            img_size = CALVIN_IMAGE_SIZE
        else:
            raise ValueError(f"Unknown environment: {env_name}")

        features["images"] = tf.TensorSpec(
            shape=(None, *img_size),
            dtype=np.uint8,
        )

        features["wrist_images"] = tf.TensorSpec(
            shape=(None, *img_size),
            dtype=np.uint8,
        )

    # Add embedding features if needed
    if save_imgs:
        if isinstance(embedding_dim, int):
            emb_shape = (
                (None, framestack, embedding_dim)
                if framestack > 1
                else (None, embedding_dim)
            )
        else:
            emb_shape = (
                (None, framestack, *embedding_dim)
                if framestack > 1
                else (None, *embedding_dim)
            )
        features["image_embeddings"] = tf.TensorSpec(shape=emb_shape, dtype=np.float32)
        features["wrist_image_embeddings"] = tf.TensorSpec(
            shape=emb_shape, dtype=np.float32
        )

    if save_costs:
        features["costs"] = tf.TensorSpec(shape=(None,), dtype=np.float32)

    if has_flow:
        features["flow"] = tf.TensorSpec(shape=(None, 2), dtype=np.float32)

        features["gmflow"] = tf.TensorSpec(
            shape=(None, *CALVIN_IMAGE_SIZE[:-1], 2), dtype=np.float32
        )

    return features


def save_dataset(
    trajectories,
    save_file: Path,
    env_name: str,
    save_imgs: bool = False,
    framestack: int = 0,
    black_white: bool = False,
    embedding_dim: int = None,
    save_costs: bool = False,
    save_flow: bool = False,
):
    """Save trajectory data as a TensorFlow dataset.

    Features included in the dataset are determined dynamically based on the
    available data and specified parameters.
    """
    log(f"Saving dataset to: {save_file}", "green")
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # Create tfds from generator
    def generator():
        for trajectory in trajectories:
            yield trajectory

    # Get appropriate features dictionary
    features_dict = get_features_dict(
        env_name=env_name,
        save_imgs=save_imgs,
        framestack=framestack,
        black_white=black_white,
        embedding_dim=embedding_dim,
        save_costs=save_costs,
        has_flow=save_flow,
    )

    trajectory_tfds = tf.data.Dataset.from_generator(
        generator, output_signature=features_dict
    )
    tf.data.experimental.save(trajectory_tfds, str(save_file))