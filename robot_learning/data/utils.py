from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import tensorflow as tf
from clam.utils.logger import log

# Base image sizes
HALFCHEETAH_IMAGE_SIZE = (64, 64, 3)
METAWORLD_IMAGE_SIZE = (84, 84, 3)
MAX_QUERY_POINTS = 100

# ROBOT_IMAGE_SIZE = (480, 640, 3) # OG image size
ROBOT_IMAGE_SIZE = (128, 128, 3)
ROBOT_DEPTH_IMAGE_SIZE = (128, 128)


# Base feature specifications for different environments
BASE_FEATURES = {
    "metaworld": {
        "states": tf.TensorSpec(shape=(None, 39), dtype=np.float32),
        "actions": tf.TensorSpec(shape=(None, 4), dtype=np.float32),
    },
    "halfcheetah": {
        "states": tf.TensorSpec(shape=(None, 17), dtype=np.float32),
        "actions": tf.TensorSpec(shape=(None, 6), dtype=np.float32),
    },
    "robot": {
        "states": tf.TensorSpec(shape=(None, 7), dtype=np.float32),
        "actions": tf.TensorSpec(shape=(None, 7), dtype=np.float32),
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
    precompute_embeddings: bool = False,
    camera_names: List[str] = [],
    camera_embed_dims: List[int] = [],
) -> Dict:
    """Dynamically construct features dictionary based on available data.

    Args:
        env_name: Name of the environment ('metaworld' or 'halfcheetah')
        save_imgs: Whether image data is included
        framestack: Number of frames to stack (if > 1)
        black_white: Whether images are grayscale
        precompute_embeddings: Whether image embeddings are included
        has_flow: Whether optical flow features are included

    Returns:
        Dictionary of TensorSpec features
    """
    # Start with base and common features
    features = {**BASE_FEATURES[env_name], **COMMON_FEATURES}

    # Add image features if needed
    if save_imgs:
        if env_name == "robot":
            for camera_name in camera_names:
                if "depth" in camera_name:
                    img_size = ROBOT_DEPTH_IMAGE_SIZE
                else:
                    img_size = ROBOT_IMAGE_SIZE

                features[f"{camera_name}_imgs"] = tf.TensorSpec(
                    shape=(None, *img_size), dtype=np.uint8
                )

            if precompute_embeddings:
                # remove depth from camera_names
                camera_names = [name for name in camera_names if "depth" not in name]
                for camera_name, embed_dim in zip(camera_names, camera_embed_dims):
                    if "depth" in camera_name:
                        continue
                    features[f"{camera_name}_img_embeds"] = tf.TensorSpec(
                        shape=(None, embed_dim), dtype=np.float32
                    )
        else:
            if env_name == "halfcheetah":
                img_size = HALFCHEETAH_IMAGE_SIZE
            else:
                img_size = METAWORLD_IMAGE_SIZE

            if framestack > 1:
                if black_white:
                    features["images"] = tf.TensorSpec(
                        shape=(None, framestack, *img_size[:-1], 1),
                        dtype=np.uint8,
                    )
                else:
                    features["images"] = tf.TensorSpec(
                        shape=(None, framestack, *img_size),
                        dtype=np.uint8,
                    )
            else:
                features["images"] = tf.TensorSpec(
                    shape=(None, *img_size), dtype=np.uint8
                )
    # Add flow features if needed
    if has_flow:
        features.update(
            {
                "points": tf.TensorSpec(
                    shape=(None, MAX_QUERY_POINTS, 2), dtype=np.float32
                ),
                "points_viz": tf.TensorSpec(
                    shape=(None, MAX_QUERY_POINTS), dtype=np.uint8
                ),
                "points_mask": tf.TensorSpec(
                    shape=(None, MAX_QUERY_POINTS), dtype=np.uint8
                ),
            }
        )

    log(f"Features: {features}", "yellow")
    return features


def save_dataset(
    trajectories,
    save_file: Path,
    env_name: str,
    save_imgs: bool = False,
    framestack: int = 0,
    black_white: bool = False,
    has_flow: bool = False,
    precompute_embeddings: bool = False,
    camera_names: List[str] = [],
    camera_embed_dims: List[int] = [],
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
        has_flow=has_flow,
        camera_names=camera_names,
        camera_embed_dims=camera_embed_dims,
        precompute_embeddings=precompute_embeddings,
    )

    trajectory_tfds = tf.data.Dataset.from_generator(
        generator, output_signature=features_dict
    )
    tf.data.Dataset.save(trajectory_tfds, str(save_file))
