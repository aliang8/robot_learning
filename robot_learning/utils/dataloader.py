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
    return x


def _apply_image_augmentation(
    x: tf.Tensor,
    image_key: str = "images",
    random_flip: bool = True,
    random_crop: bool = True,
    color_jitter: bool = True,
    min_scale: float = 0.5,
    max_scale: float = 1.0,
    orig_size: List[int] = [84, 84],
    brightness_factor: float = 0.3,
    contrast_factor: float = 0.3,
    saturation_factor: float = 0.3,
    hue_factor: float = 0.1,
) -> tf.Tensor:
    """Apply image augmentations including random horizontal flip, random resize crop, and color jitter.

    Args:
        x: Input dictionary containing image tensor
        image_key: Key for accessing images in the dictionary
        random_flip: Whether to apply random horizontal flipping
        random_crop: Whether to apply random resize cropping
        color_jitter: Whether to apply color jittering
        min_scale: Minimum scale for random resize crop (0.5 means crop to 50% of original size)
        max_scale: Maximum scale for random resize crop (1.0 means use original size)
        orig_size: Original image size to crop back to
        brightness_factor: Maximum brightness adjustment factor
        contrast_factor: Maximum contrast adjustment factor
        saturation_factor: Maximum saturation adjustment factor
        hue_factor: Maximum hue adjustment factor
    """
    if image_key not in x:
        return x

    images = x[image_key]
    channel_first = len(images.shape) == 4 and images.shape[1] == 3

    # Convert orig_size to both float and int tensors to avoid casting issues
    orig_size_float = tf.convert_to_tensor(orig_size, dtype=tf.float32)
    orig_size_int = tf.convert_to_tensor(orig_size, dtype=tf.int32)

    if channel_first:
        images = tf.transpose(images, perm=[0, 2, 3, 1])

    def _random_crop(img):
        # Random crop using tf's built-in function
        scale = tf.random.uniform([], min_scale, max_scale)
        scaled_size = tf.cast(orig_size_float * scale, tf.int32)
        img = tf.image.resize_with_crop_or_pad(img, scaled_size[0], scaled_size[1])
        img = tf.image.random_crop(img, [scaled_size[0], scaled_size[1], 3])
        img = tf.image.resize(img, orig_size_int)
        return img

    def _random_flip(img):
        return tf.image.random_flip_left_right(img)

    def _color_jitter(img):
        # Apply color jittering in random order
        jitter_order = tf.random.shuffle(tf.range(4))

        for idx in jitter_order:
            if idx == 0:  # brightness
                img = tf.image.random_brightness(img, brightness_factor)
            elif idx == 1:  # contrast
                img = tf.image.random_contrast(
                    img, 1 - contrast_factor, 1 + contrast_factor
                )
            elif idx == 2:  # saturation
                img = tf.image.random_saturation(
                    img, 1 - saturation_factor, 1 + saturation_factor
                )
            else:  # hue
                img = tf.image.random_hue(img, hue_factor)

        # Ensure values are in valid range
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img

    # Apply augmentations
    if random_crop:
        images = tf.map_fn(_random_crop, images, fn_output_signature=tf.float32)

    if random_flip:
        images = tf.map_fn(_random_flip, images, fn_output_signature=tf.float32)

    if color_jitter:
        images = tf.map_fn(_color_jitter, images, fn_output_signature=tf.float32)

    if channel_first:
        images = tf.transpose(images, perm=[0, 3, 1, 2])

    x[image_key] = images
    return x


def process_image(
    x,
    channel_first: bool = False,
    image_shape: List[int] = [84, 84],
    image_key: str = "images",
):
    """Process images with optional augmentation."""
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
            images = tf.image.resize(images, image_shape)
            images = tf.transpose(images, perm=[0, 3, 1, 2])

    if tf.reduce_max(images) > 1:
        images = tf.cast(images, tf.float32) / 255.0
    else:
        images = tf.cast(images, tf.float32)

    x[image_key] = images
    return x


def process_state(x, cfg, env_name):
    states = x["states"]
    has_framestack = len(states.shape) == 3

    if has_framestack:
        # take the last state!
        states = states[:, -1]

    # for calvin, add scene_obs
    if env_name == "calvin":
        x["states"] = tf.concat([x["states"], x["scene_obs"]], axis=-1)
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
    apply_image_augmentation: bool = False,
):
    """
    Applies transformations to base tfds such as batching, shuffling, etc.
    """
    ds = ds.filter(filter_fn)

    # caching the dataset makes it faster in the next iteration
    # ds = ds.cache()

    # the buffer size is important for memory usage and affects speed
    # shuffle here is for trajectories
    if shuffle:
        ds = ds.shuffle(100, reshuffle_each_iteration=False)

    # limit the number of trajectories that we use
    ds = ds.take(cfg.num_trajs)
    log(f"\ttaking {cfg.num_trajs} trajectories")

    # compute return of the trajectories
    # TODO: commented this out because it was taking too long to run
    # if "rewards" in ds.element_spec:
    #     returns = list(
    #         ds.map(
    #             lambda episode: tf.reduce_sum(episode["rewards"])
    #         ).as_numpy_iterator()
    #     )
    #     traj_lens = list(
    #         ds.map(lambda episode: tf.shape(episode["rewards"])[0]).as_numpy_iterator()
    #     )
    #     if len(returns) > 0:
    #         log(
    #             f"\tN: {len(returns)} | Average return: {sum(returns) / len(returns)} | Max return: {max(returns)} | Min return: {min(returns)} | Average traj len: {sum(traj_lens) / len(traj_lens)}",
    #             "yellow",
    #         )
    #     log("done with rewards", "yellow")

    ds = ds.map(partial(add_new_fields, cfg=cfg), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        partial(process_state, cfg=cfg, env_name=env_name),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Process images WITHOUT augmentation first
    img_shape = OmegaConf.to_container(cfg.image_shape)
    for key in cfg.input_modalities:
        if ("img" in key or "image" in key) and "embed" not in key:
            ds = ds.map(
                partial(
                    process_image,
                    channel_first=True,
                    image_shape=img_shape,
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

    # Now apply augmentation AFTER n-step processing
    if apply_image_augmentation:
        for key in cfg.input_modalities:
            if ("img" in key or "image" in key) and "embed" not in key:
                ds = ds.map(
                    partial(
                        _apply_image_augmentation,
                        image_key=key,
                        orig_size=img_shape,
                        **cfg.augmentation_kwargs,
                    ),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )

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

    log(f"Loading tfds dataset from: {data_dir}, env id: {cfg.env.env_id}")
    log(f"Dataset names: {dataset_names}")
    log(f"Dataset split: {dataset_split}")

    datasets = {}
    dataset_split = dataset_split[: len(dataset_names)]
    # convert this into a ratio
    dataset_ratio = [x / sum(dataset_split) for x in dataset_split]
    log(f"Dataset ratio: {dataset_ratio}")

    total_trajs = 0
    ds_to_len = {}
    for ds_name in dataset_names:
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
            cfg_train,
            ds,
            env_name=cfg.env.env_name,
            shuffle=shuffle,
            apply_image_augmentation=cfg.data.apply_image_augmentation,
        )

    log("Creating eval datasets")
    # use all the trajectories in the eval dataset
    cfg_eval = cfg.data.copy()
    cfg_eval.num_trajs = -1
    cfg_eval.num_examples = -1
    for i, ds_name in enumerate(dataset_names):
        eval_ds[ds_name] = process_dataset(
            cfg_eval,
            eval_ds[ds_name],
            env_name=cfg.env.env_name,
            shuffle=False,
            apply_image_augmentation=False,
        )

    return train_ds, eval_ds


if __name__ == "__main__":
    pass
