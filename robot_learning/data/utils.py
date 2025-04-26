import pickle as pkl
from pathlib import Path
from typing import Dict, List, Union

import blosc
import numpy as np
import tensorflow as tf
import tqdm

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
    name_parts.append("imgs")
    if cfg.black_white:
        name_parts.append("bw")
    if cfg.framestack > 1:
        name_parts.append(f"fs{cfg.framestack}")

    name_parts.append(f"emb-{cfg.embedding_model}")

    if "resnet" in cfg.embedding_model:
        name_parts.append(f"l-{cfg.resnet_feature_map_layer}")

    name_parts.append(f"flow-{cfg.flow_suffix}")

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


def raw_data_to_tfds(
    traj_dirs: List[str],
    save_file: str,
    embedding_model: str,
    resnet_feature_map_layer: str = "avgpool",
    flow_suffix: str = "all",
):
    num_transitions = 0

    # Load trajectories
    available_cameras = []
    traj_dir = traj_dirs[0]
    for dat_file in Path(traj_dir).glob("*.dat"):
        if "images" in dat_file.name and "processed" not in dat_file.name:
            available_cameras.append(
                dat_file.name.split("_images")[0].replace(".dat", "")
            )

    log(f"Available cameras: {available_cameras}", "yellow")

    processed_trajs = []
    for traj_dir in tqdm.tqdm(traj_dirs, desc="Loading trajectories"):
        traj_dir = Path(traj_dir)
        traj_data = load_data_compressed(traj_dir / "traj_data.dat")
        num_transitions += len(traj_data["actions"])

        for camera_type in available_cameras:
            if camera_type == "depth":
                continue

            if camera_type == "images":
                images_file = traj_dir / "images.dat"
                key = "images"
            else:
                images_file = traj_dir / f"{camera_type}_processed_images.dat"
                key = f"{camera_type}_images"

            if images_file.exists():
                images = load_data_compressed(images_file)
                traj_data[key] = images

            if "resnet" in embedding_model:
                img_embeds_file = (
                    traj_dir
                    / f"{camera_type}_img_embeds_{embedding_model}_{resnet_feature_map_layer}.dat"
                )
            else:
                img_embeds_file = (
                    traj_dir / f"{camera_type}_img_embeds_{embedding_model}.dat"
                )

            if camera_type == "images":
                img_embeds_file = Path(
                    str(img_embeds_file).replace(f"{camera_type}_", "")
                )
                key = "images_embeds"
            else:
                key = f"{camera_type}_images_embeds"

            if img_embeds_file.exists():
                img_embeds = load_data_compressed(img_embeds_file)
                traj_data[key] = img_embeds

        flow_file = traj_dir / f"2d_flow_{flow_suffix}.dat"
        if flow_file.exists():
            flow_data = load_data_compressed(flow_file)
            traj_data.update(flow_data)

        log("=" * 100)
        for k, v in traj_data.items():
            if isinstance(v, np.ndarray):
                log(f"{k}: {v.shape}")

        processed_trajs.append(traj_data)

    for idx, traj in enumerate(processed_trajs):
        base_trajectory = get_base_trajectory(traj["rewards"])
        traj = {**base_trajectory, **traj}
        processed_trajs[idx] = traj

    traj = processed_trajs[0]
    for k, v in traj.items():
        if isinstance(v, np.ndarray):
            log(f"{k}: {v.shape}")

    log(f"Total number of transitions: {num_transitions} collected", "green")
    save_dataset(processed_trajs, save_file)


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
    for k, v in features_dict.items():
        log(f"{k}: {v}", "yellow")

    trajectory_tfds = tf.data.Dataset.from_generator(
        generator, output_signature=features_dict
    )
    tf.data.Dataset.save(trajectory_tfds, str(save_file))


def save_data_compressed(path, data):
    log(f"Saving to {path}", "yellow")
    with open(path, "wb") as f:
        compressed_data = blosc.compress(pkl.dumps(data))
        f.write(compressed_data)


def load_data_compressed(path):
    log(f"Loading from {path}", "yellow")
    with open(path, "rb") as f:
        compressed_data = f.read()
        data = pkl.loads(blosc.decompress(compressed_data))
        return data
