"""
Script to convert robot demonstration data into tfds format.

Usage:
    python3 -m robot_learning.real_robot.scripts.convert_robot_to_tfds \
        env_name=robot \
        dataset_name=playdata0 \
        compute_2d_flow=True \
        flow.text_prompt="robot. objects." \
        debug=True
"""

import os
import re
from email.mime import image
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import clip
import hydra
import numpy as np
import tensorflow as tf
import torch
import tqdm
from omegaconf import DictConfig
from PIL import Image, ImageDraw

from robot_learning.data.molmo_utils import (
    get_center_of_hand,
    load_molmo_model,
)
from robot_learning.data.optical_flow.compute_flow_cotracker_util import (
    load_cotracker,
    load_sam_model,
)
from robot_learning.data.preprocess import (
    compute_flow_features,
    compute_image_embeddings,
)
from robot_learning.data.utils import (
    load_data_compressed,
    save_data_compressed,
)
from robot_learning.models.image_embedder import ImageEmbedder
from robot_learning.utils.logger import log

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def load_images(
    image_dir: str, is_depth: bool = False
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
    return images


def center_crop_depth_image(depth_image: np.ndarray) -> np.ndarray:
    shortest_edge = min(depth_image.shape[0], depth_image.shape[1])
    pad_h = (depth_image.shape[0] - shortest_edge) // 2
    pad_w = (depth_image.shape[1] - shortest_edge) // 2
    # center crop to shortest edge
    if shortest_edge == depth_image.shape[0]:
        depth_image = depth_image[pad_h : pad_h + shortest_edge, :]
    else:
        depth_image = depth_image[:, pad_w : pad_w + shortest_edge]
    return depth_image


def center_crop_rgb_image(
    image: np.ndarray,
    y_offset: int = 120,
) -> np.ndarray:
    shortest_edge = min(image.shape[0], image.shape[1])
    image = tf.image.crop_to_bounding_box(
        image, 0, y_offset, shortest_edge, shortest_edge
    )
    return image


def center_crop_and_resize_depth_images(
    images: np.ndarray,
    image_size: List[int] = [480, 480],
) -> np.ndarray:
    processed_images = []
    for img in images:
        cropped = center_crop_depth_image(img)

        target_size = image_size
        from scipy.ndimage import zoom

        scale = (
            target_size[0] / cropped.shape[0],
            target_size[1] / cropped.shape[1],
        )

        resized = zoom(cropped, scale, order=0)  # order=0 for nearest neighbor
        processed_images.append(resized)
    processed_images = np.array(processed_images)
    return processed_images


def center_crop_rgb_images(images: np.ndarray, y_offset: int = 120) -> np.ndarray:
    processed_images = []
    for img in images:
        cropped = center_crop_rgb_image(img, y_offset)
        processed_images.append(cropped)
    processed_images = np.array(processed_images)
    return processed_images


def load_metadata(data_file: str) -> Dict:
    """Load and process metadata from file."""
    data = np.load(data_file, allow_pickle=True)
    if isinstance(data, list):  # For policy output
        return {k: np.array([p[k] for p in data]) for k in data[0].keys()}
    return {k: data[k][:-1] for k in data.keys()}  # For observation dict


def get_available_cameras(data_files: List[Path]) -> Dict[str, str]:
    """
    Get mapping of available camera types to their directories.

    Args:
        data_files: List of paths in the trajectory directory
    Returns:
        Dictionary mapping camera names to their directory paths
    """
    camera_mapping = {}
    for file_path in data_files:
        if "depth_images" in file_path.name:
            camera_mapping["depth"] = file_path
        elif "external" in file_path.name:
            camera_mapping["external"] = file_path
        elif "over_shoulder" in file_path.name:
            camera_mapping["over_shoulder"] = file_path
        elif "wrist" in file_path.name:
            camera_mapping["wrist"] = file_path
    return camera_mapping


def preprocess_robot_data(cfg: DictConfig, data_dir: Path):
    """
    Assumes robot data is stored in the following format:

        data_dir/
            traj0/
                obs_dict.pkl
                policy_out.pkl
                depth_images/
                external_images/

    Creates a new directory in data_dir/processed_trajs/ with the following format
    with .dat files for each type of information:

        data_dir/
            processed_trajs/
                traj_000000/
                    depth_images.dat
                    external_images.dat
    """
    traj_dirs = sorted(data_dir.glob("traj*"))
    # filter only folders
    traj_dirs = [d for d in traj_dirs if os.path.isdir(d)]
    log(f"Processing {len(traj_dirs)} trajectories", "yellow")

    if cfg.debug:
        traj_dirs = traj_dirs[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize necessary models!
    # Load SAM 2 and Grounding DINO if we are computing 2d flow
    sam, image_predictor = load_sam_model(
        cfg.flow.sam2_checkpoint_file, cfg.flow.model_cfg_file
    )
    cotracker = load_cotracker(cfg.flow.cotracker_ckpt_file)
    cotracker = cotracker.to(device)

    # If we are doing hand tracking, then we load molmo model
    # to get the center of the hand
    if "hand" in data_dir.name:
        log("Loading molmo model for hand tracking", "yellow")
        processor, molmo = load_molmo_model()

    # Make different image embedders
    image_embedders = {}
    image_embedders["dinov2_vitb14"] = ImageEmbedder(
        model_name="dinov2_vitb14", device=device
    )

    for embed_type in ["resnet18", "resnet50"]:
        for feature_map_layer in ["layer4", "avgpool"]:
            image_embedders[f"{embed_type}_{feature_map_layer}"] = ImageEmbedder(
                model_name=embed_type,
                device=device,
                feature_map_layer=feature_map_layer,
            )

    for embed_type in image_embedders:
        image_embedders[embed_type] = image_embedders[embed_type].to(device)

    for traj_idx, traj_dir in enumerate(
        tqdm.tqdm(traj_dirs, desc="Processing trajectories")
    ):
        data_files = sorted(traj_dir.glob("*"))

        # Check for required files
        obs_dict_file = Path(traj_dir) / "obs_dict.pkl"
        policy_out_file = Path(traj_dir) / "policy_out.pkl"

        if not (obs_dict_file and policy_out_file):
            log(f"Skipping {traj_dir} - missing required files", "red")
            continue

        # Save to .dat format
        new_traj_dir = data_dir / "processed_trajs" / f"traj_{traj_idx:06d}"
        new_traj_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        save_file = new_traj_dir / "traj_data.dat"
        if not save_file.exists():
            if obs_dict_file.exists():
                obs_dict = load_metadata(obs_dict_file)
                policy_out = load_metadata(policy_out_file)

                # Save metadata for each trajectory
                traj_data = {
                    "states": obs_dict["state"],
                    "actions": policy_out["actions"],
                    "rewards": np.zeros(len(policy_out["actions"])),
                    "qvel": obs_dict["qvel"],  # need this for retrieval
                }
                save_data_compressed(save_file, traj_data)

        # Get available cameras files
        camera_mapping = get_available_cameras(data_files)
        if not camera_mapping:
            log(f"Skipping {traj_dir} - no camera data found", "red")
            continue

        # Initialize storage for images and embeddings
        camera_imgs = {}
        processed_camera_imgs = {}

        # Process each available camera
        for camera_type, camera_dir in camera_mapping.items():
            # skipping depth camera for now
            if camera_type == "depth":
                continue

            is_depth = camera_type == "depth"
            y_offset = 120 if camera_type == "external" else 80

            # Load raw images
            img_file = new_traj_dir / f"{camera_type}_images.dat"

            if not img_file.exists():
                imgs = load_images(camera_dir, is_depth=is_depth)
                camera_imgs[f"{camera_type}"] = imgs
            else:
                camera_imgs[f"{camera_type}"] = load_data_compressed(img_file)

            # Center crop and resize
            processed_img_file = new_traj_dir / f"{camera_type}_processed_images.dat"
            if not processed_img_file.exists():
                if is_depth:
                    processed_imgs = center_crop_and_resize_depth_images(
                        camera_imgs[f"{camera_type}"], cfg.image_size
                    )
                else:
                    # first center crop to shortest edge
                    processed_imgs = center_crop_rgb_images(
                        camera_imgs[f"{camera_type}"], y_offset
                    )
                    # then resize to target size
                    processed_imgs = tf.image.resize(processed_imgs, cfg.image_size)
                processed_camera_imgs[f"{camera_type}"] = processed_imgs

        # Save processed images
        for camera_type, images in camera_imgs.items():
            img_file = new_traj_dir / f"{camera_type}_processed_images.dat"

            if not img_file.exists():
                save_data_compressed(img_file, processed_camera_imgs[camera_type])

            img_file = new_traj_dir / f"{camera_type}_images.dat"
            if not img_file.exists():
                save_data_compressed(img_file, camera_imgs[camera_type])

            # save the raw images to a folder as jpgs TODO: this is for labeling the language
            raw_images_dir = new_traj_dir / f"{camera_type}_raw_images"
            raw_images_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(camera_imgs[camera_type]):
                img = Image.fromarray(img)
                img.save(raw_images_dir / f"{i:06d}.jpg")

            # Process all embedding types that we want to save
            embedding_model = "dinov2_vitb14"

            img_embed_file = (
                new_traj_dir / f"{camera_type}_img_embeds_{embedding_model}.dat"
            )
            if not img_embed_file.exists() and camera_type != "depth":
                img_embeds = compute_image_embeddings(
                    embedder=image_embedders["dinov2_vitb14"], images=[images]
                )[0]
                save_data_compressed(img_embed_file, img_embeds)

            # Process all embedding types that we want to save
            embedding_model = "clip_vitb32"

            img_embed_file = (
                new_traj_dir / f"{camera_type}_img_embeds_{embedding_model}.dat"
            )
            if not img_embed_file.exists() and camera_type != "depth":
                image =  torch.stack([
                    preprocess(Image.fromarray(image)).to(device)
                    for image in images
                ])

                with torch.no_grad():
                    img_embeds = clip_model.encode_image(image)
                
                save_data_compressed(img_embed_file, img_embeds)

            resnet_embedding_models = ["resnet18", "resnet50"]
            resnet_feature_map_layers = ["layer4", "avgpool"]

            for resnet_embedding_model in resnet_embedding_models:
                for resnet_feature_map_layer in resnet_feature_map_layers:
                    img_embed_file = (
                        new_traj_dir
                        / f"{camera_type}_img_embeds_{resnet_embedding_model}_{resnet_feature_map_layer}.dat"
                    )
                    if not img_embed_file.exists() and camera_type != "depth":
                        img_embeds = compute_image_embeddings(
                            embedder=image_embedders[
                                f"{resnet_embedding_model}_{resnet_feature_map_layer}"
                            ],
                            images=[images],
                        )[0]
                        save_data_compressed(img_embed_file, img_embeds)

        # Compute flow information and perform SAM 2 point tracking
        object_flow_file = new_traj_dir / "2d_flow_all.dat"
        point_tracking_file = new_traj_dir / "2d_flow_query.dat"

        # if object_flow_file.exists() and point_tracking_file.exists():
        #     continue

        video = camera_imgs["external"]
        # TODO: this is hard-coded for external camera
        # y_offset = 120
        # # run flow tracking on the CROPPED video
        # video = center_crop_rgb_images(video, y_offset)
        # import ipdb; ipdb.set_trace()

        # if not object_flow_file.exists():
        flow_traj_data, renders = compute_flow_features(
            image_predictor=image_predictor,
            cotracker=cotracker,
            text=cfg.flow.text_prompt,
            queries=None,
            grounding_model_id=cfg.flow.grounding_model_id,
            videos=[video],
            device=device,
        )
        save_data_compressed(object_flow_file, flow_traj_data[0])

        # save renders as png files
        for indx, render in enumerate(renders):
            render.savefig(new_traj_dir / "flow_visualization_all.png")

        # if not point_tracking_file.exists():
            # if cfg.flow.queries:
            #     queries = np.array(cfg.flow.queries)
            # else:
            #     queries = None

        h, w = video.shape[1], video.shape[2]  # 1080, 1920

        if "hand" not in data_dir.name:
            # queries = np.array([[0, 561, 282]])
            queries = np.array([[0, 448, 272]])  # post cropping
        else:
            # Calculate target height for 1920 width to match 480:640 aspect ratio
            # 640/480 = 1920/target_h
            # We need this for processing videos recorded on the iphone
            # target_h = int(1920 * (480 / 640))  # = 1440
            # # Calculate padding needed
            # pad_h = target_h - h  # 1440 - 1080 = 360
            # pad_top = pad_h // 2  # 180
            # pad_bottom = pad_h - pad_top  # 180

            # # Add padding to top and bottom (black padding)
            # video = np.pad(
            #     video,
            #     (
            #         (0, 0),  # time dimension
            #         (pad_top, pad_bottom),  # height dimension
            #         (0, 0),  # width dimension
            #         (0, 0),
            #     ),  # channels
            #     mode="constant",
            #     constant_values=0,
            # )

            # take a middle frame
            # hopefully the hand is visible from this frame
            # and get the center of the hand using molmo
            query_frame = video[len(video) // 2]
            query_frame = Image.fromarray(query_frame)
            center_of_hand = get_center_of_hand(processor, molmo, query_frame)

            query_frame_file = new_traj_dir / "query_frame.png"
            # plot the center of the hand
            draw = ImageDraw.Draw(query_frame)
            draw.circle((center_of_hand[0], center_of_hand[1]), 20, fill="red")
            query_frame.save(query_frame_file)

            queries = np.array(
                [[len(video) // 2, center_of_hand[0], center_of_hand[1]]]
            )

        flow_traj_data, renders = compute_flow_features(
            image_predictor=image_predictor,
            cotracker=cotracker,
            text=cfg.flow.text_prompt,
            queries=queries,
            grounding_model_id=cfg.flow.grounding_model_id,
            videos=[video],
            device=device,
        )
        save_data_compressed(point_tracking_file, flow_traj_data[0])

        # save renders as png files
        for indx, render in enumerate(renders):
            render.savefig(new_traj_dir / "flow_visualization_query.png")


@hydra.main(version_base=None, config_name="convert_to_tfds", config_path="../../cfg")
def main(cfg):
    """Main function to convert replay buffer to TFDS format."""
    data_dir = Path(cfg.data_dir)
    log(f"Processing data from {data_dir}", "yellow")
    preprocess_robot_data(cfg, data_dir)


if __name__ == "__main__":
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    main()
