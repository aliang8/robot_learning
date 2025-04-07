import json
import os
from functools import partial
from pathlib import Path
from typing import List

import einops
import rlds
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from robot_learning.utils.logger import log

os.environ["TFDS_DATA_DIR"] = "/scr/shared/prompt_dtla/tensorflow_datasets"


def episode_to_step_custom(episode, size, shift):
    episode = tf.data.Dataset.from_tensor_slices(episode)
    return rlds.transformations.batch(
        episode, size=size, shift=shift, drop_remainder=True
    )


# add additional fields to the dataset
def add_new_fields(x, cfg):
    x["mask"] = tf.ones_like(x["actions"])
    x["timestep"] = tf.range(tf.shape(x["actions"])[0])

    if "points" in x:
        log("normalizing points", "yellow")
        # normalize by dividing by image size
        # TODO: do i need to account for the padding here?
        x["points"] = x["points"] / 84

    x = del_keys(x)

    return x


def del_keys(x):
    # TODO: for now just get rid of unused keys, taking a lot of time when moving device
    del x["is_terminal"]
    del x["is_last"]
    del x["is_first"]

    del x["mask"]
    del x["scene_obs"]
    del x["timestep"]

    if "rewards" in x:
        del x["rewards"]
    if "images" in x:
        del x["images"]
    if "discount" in x:
        del x["discount"]
    if "flow" in x:
        del x["flow"]
    if "wrist_images" in x:
        del x["wrist_images"]

    return x


def process_image(
    x,
    channel_first: bool = False,
    image_shape: List[int] = [84, 84],
    image_key: str = "images",
):
    if image_key not in x:
        return x

    images = x[image_key]
    if channel_first:
        # has framestack
        has_framestack = len(images.shape) == 5

        if has_framestack:
            # take the last frame first
            images = einops.rearrange(images, "B F H W C -> B (F C) H W")
        else:
            # reshape images here
            # TODO: remove this later
            images = tf.image.resize(images, OmegaConf.to_container(image_shape))
            images = tf.transpose(images, perm=[0, 3, 1, 2])

    if tf.reduce_max(images) > 1:
        images = tf.cast(images, tf.float32) / 255.0
    else:
        images = tf.cast(images, tf.float32)

    x[image_key] = images
    return x


def process_state(x, cfg, env_name):
    x["states"] = x["observations"]

    states = x["states"]
    has_framestack = len(states.shape) == 3

    if has_framestack:
        # take the last state!
        states = states[:, -1]

    return x


def pad_dataset(x, pad):
    # create a dataset from tensor with padded shapes
    for key in x:
        x[key] = tf.concat([x[key], pad[key]], axis=0)
    return x


# remove trajectories where the number of steps is less than 2
def filter_fn(traj):
    return tf.math.greater(tf.shape(traj["states"])[0], 2)


def process_dataset(
    cfg: DictConfig,
    ds: tf.data.Dataset,
    shuffle: bool = True,
    env_name: str = None,
    drop_remainder: bool = False,
):
    """
    Applies transformations to base tfds such as batching, shuffling, etc.
    """
    # ds = ds.filter(filter_fn)

    # caching the dataset makes it faster in the next iteration
    # ds = ds.cache()

    # the buffer size is important for memory usage and affects speed
    # shuffle here is for trajectories
    if shuffle:
        ds = ds.shuffle(100, reshuffle_each_iteration=False)

    # limit the number of trajectories that we use
    ds = ds.take(cfg.num_trajs)
    log(f"\ttaking {cfg.num_trajs} trajectories")

    ds = ds.map(partial(add_new_fields, cfg=cfg), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        partial(process_state, cfg=cfg, env_name=env_name),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # process image in case we need to use it
    for key in cfg.input_modalities:
        if ("img" in key or "image" in key) and "embed" not in key:
            ds = ds.map(
                partial(
                    process_image,
                    channel_first=True,
                    image_shape=cfg.image_shape,
                    image_key=key,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

    # maybe pad the dataset here
    if cfg.pad_dataset:
        # TODO: what happens if we use transitions here
        # add extra timesteps to each trajectory
        log("padding dataset", "yellow")
        pad = rlds.transformations.zeros_from_spec(ds.element_spec)

        def repeat_padding(pad, batch_size):
            return tf.nest.map_structure(
                lambda x: tf.repeat(x, repeats=batch_size, axis=0), pad
            )

        # padding needed when we use shift
        pad = repeat_padding(pad, cfg.shift)
        ds = ds.map(partial(pad_dataset, pad=pad), num_parallel_calls=tf.data.AUTOTUNE)

    if cfg.data_type == "n_step":
        ds = ds.flat_map(
            partial(episode_to_step_custom, size=cfg.seq_len, shift=cfg.shift)
        )
    elif cfg.data_type == "transitions":
        ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
    else:
        raise ValueError(f"unknown data type: {cfg.data_type}")

    # shuffle the full dataset one more time
    if shuffle:  # shuffle here is for transitions
        log("\tshuffling dataset")
        ds = ds.shuffle(1000, reshuffle_each_iteration=False)

    if cfg.num_examples != -1:
        log(f"\ttaking {cfg.num_examples} examples")
        ds = ds.take(cfg.num_examples)

    ds = ds.batch(cfg.batch_size, drop_remainder=drop_remainder)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_dataloader(
    cfg: DictConfig,
    dataset_names: List[str],
    dataset_split: List[int],
    shuffle: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    local_rank: int = 0,
):
    """
    Returns a dictionary containing the training and validation datasets.
    Validation dataset is a dictionary of {env_id: dataset}
    """
    data_cfg = cfg.data
    data_dir = Path(data_cfg.data_dir) / "tensorflow_datasets"
    env_id = cfg.env.env_id

    if cfg.retrieval:
        data_dir = data_dir / "retrieved_dataset"

    log(f"Loading tfds dataset from: {data_dir}, env id: {env_id}")
    log(f"Dataset names: {dataset_names}")
    log(f"Dataset split: {dataset_split}")

    datasets = {}
    dataset_split = dataset_split[: len(dataset_names)]
    # convert this into a ratio
    dataset_ratio = [x / sum(dataset_split) for x in dataset_split]
    log(f"Dataset ratio: {dataset_ratio}")

    total_trajs = 0
    ds_to_len = {}
    for i, ds_name in enumerate(dataset_names):
        if cfg.retrieval:
            ds_name = f"{cfg.method}/{ds_name}_N-{cfg.N}_K-{cfg.K}"
            dataset_names[i] = ds_name

        save_file = data_dir / cfg.data.dataset_name / ds_name
        ds = tf.data.Dataset.load(str(save_file))
        if data_cfg.load_latent_actions:
            mapping_file = save_file / "la_map.json"

            if mapping_file.exists():
                log(f"Loading latent actions mapping from {mapping_file}", "yellow")
                with open(mapping_file, "r") as f:
                    la_map = json.load(f)
            else:
                raise ValueError(
                    f"Latent actions mapping file not found: {mapping_file}"
                )

            if not hasattr(cfg, "lam_ckpt"):
                raise ValueError("lam_ckpt not found in config")

            id_, save_dir = la_map[cfg.lam_ckpt][str(cfg.lam_ckpt_step)]
            la_file = Path(save_dir) / f"latent_actions_{id_}"

            log(f"Loading latent actions relabelled from {la_file}", "yellow")

            if la_file.exists():
                latent_actions_ds = tf.data.Dataset.load(str(la_file))
            else:
                raise ValueError(f"Latent actions file not found: {la_file}")

            combined_ds = tf.data.Dataset.zip((ds, latent_actions_ds))
            combined_ds = combined_ds.map(
                lambda x, y: {
                    **x,
                    "latent_actions": y["latent_actions"],
                    # "prequantized_las": (
                    #     y["prequantized_las"] if "prequantized_las" in y else None
                    # ),
                }
            )
            ds = combined_ds

        datasets[ds_name] = ds
        total_trajs += len(ds)
        ds_to_len[ds_name] = len(ds)

    log(f"Total trajectories: {total_trajs}")

    for ds_name in dataset_names:
        log(f"\t{ds_name}: {ds_to_len[ds_name]} trajs")

    # split dataset into train and eval
    train_ds = {}
    eval_ds = {}

    log("=" * 100)
    log("Split dataset into train and eval: ")
    for i, ds_name in enumerate(dataset_names):
        num_take = int(ds_to_len[ds_name] * cfg.data.train_frac)
        num_eval = ds_to_len[ds_name] - num_take
        log(f"\t{ds_name}: num train trajs: {num_take}, num eval trajs: {num_eval}")
        train_ds[ds_name] = datasets[ds_name].take(num_take)
        eval_ds[ds_name] = datasets[ds_name].skip(num_take)

    log("=" * 100)
    log("Creating train datasets")
    for i, ds_name in enumerate(dataset_names):
        cfg_train = cfg.data.copy()
        if cfg.data.num_trajs != -1:
            cfg_train.num_trajs = int(cfg.data.num_trajs * dataset_ratio[i])
        if cfg.data.num_examples != -1:
            cfg_train.num_examples = int(cfg.data.num_examples * dataset_ratio[i])

        log(
            f"\t{ds_name}: num_trajs: {cfg_train.num_trajs}, num_examples: {cfg_train.num_examples}"
        )

        ds = train_ds[ds_name]

        # Apply sharding before processing if distributed
        if distributed:
            ds = ds.shard(num_shards=world_size, index=local_rank)
            log(
                f"Rank {local_rank}: Using 1/{world_size} of the data for {ds_name}",
                "blue",
            )

        train_ds[ds_name] = process_dataset(
            cfg_train, ds, env_name=cfg.env.env_name, shuffle=shuffle
        )

    log("Creating eval datasets")
    # use all the trajectories in the eval dataset
    cfg_eval = cfg.data.copy()
    cfg_eval.num_trajs = -1
    cfg_eval.num_examples = -1
    for i, ds_name in enumerate(dataset_names):
        eval_ds[ds_name] = process_dataset(
            cfg_eval, eval_ds[ds_name], env_name=cfg.env.env_name, shuffle=False
        )

    return train_ds, eval_ds


if __name__ == "__main__":
    pass
