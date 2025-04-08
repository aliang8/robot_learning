from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from robot_learning.data.optical_flow.compute_flow_cotracker_util import (
    generate_point_tracks,
    get_seg_mask,
)
from robot_learning.models.image_embedder import ImageEmbedder
from robot_learning.utils.general_utils import to_numpy
from robot_learning.utils.logger import log


def compute_image_embeddings(
    embedding_model: str,
    images: List[np.ndarray],
    device: str,
    resnet_feature_map_layer: str = None,
) -> List[np.ndarray]:
    """Compute embeddings for a sequence of images using the specified embedder.

    Args:
        images: List of images, [T, H, W, C]
        device: Device to run the embeddings on
    """
    embedder = ImageEmbedder(
        model_name=embedding_model,
        device=device,
        feature_map_layer=resnet_feature_map_layer,
    )

    log(f"Computing image embeddings using {embedder.model_name}")
    embeddings = []

    for video in tqdm.tqdm(images, desc="computing embeddings"):
        with torch.no_grad():
            emb = embedder(np.array(video))
        embeddings.append(emb.cpu().numpy())

    return embeddings


def compute_flow_features(
    image_predictor,
    cotracker,
    text: str,
    grounding_model_id: str,
    grid_size: int = 25,
    max_query_points: int = 100,
    images: List[np.ndarray] = None,
    device: str = "cuda",
    visualize_segmentation: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """Compute optical flow features for a sequence of images."""
    point_tracking_results = []
    seg_masks = []

    for indx, video in enumerate(tqdm.tqdm(images, desc="computing 2d flow")):
        # Segment out the table
        table_mask = get_seg_mask(
            image_predictor,
            grounding_model_id,
            text="table.",
            video=video,
            device=device,
        )

        # Get object segmentation mask
        segm_mask = get_seg_mask(
            image_predictor, grounding_model_id, text=text, video=video, device=device
        )

        # Make all positive values 1
        segm_mask = (segm_mask > 0).astype(np.uint8)
        table_mask = (table_mask > 0).astype(np.uint8)

        # Zero out table in segm_mask
        segm_mask = segm_mask * (1 - table_mask)

        # Track points across video
        points, visibility = generate_point_tracks(
            cotracker, video, segm_mask=segm_mask, grid_size=grid_size, device=device
        )

        log(f"Points shape: {points.shape}, visibility shape: {visibility.shape}")

        # Process and store results
        tracked_points = {
            "points": to_numpy(points[0]),
            "visibility": to_numpy(visibility[0]),
        }

        points = tracked_points["points"]
        viz = tracked_points["visibility"]
        mask = np.ones_like(viz)

        if visualize_segmentation:
            image = video[0] / 255.0
            # Create figure with 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle("Segmentation Visualization")

            # Original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title("Original Image")

            # Table segmentation mask
            axes[0, 1].imshow(table_mask, cmap="gray")
            axes[0, 1].set_title("Table Segmentation")

            # Object segmentation mask
            axes[1, 0].imshow(segm_mask, cmap="viridis")
            axes[1, 0].set_title("Object Segmentation")

            # Overlay segmentation on original image
            overlay = image.copy()
            # Add colored overlay for table (red) and object (blue)
            overlay[table_mask > 0.5] = [1.0, 0.0, 0.0]  # Red for table
            overlay[segm_mask > 0.5] = [0.0, 0.0, 1.0]  # Blue for object
            # Blend with original image
            overlay = 0.7 * image + 0.3 * overlay

            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title("Overlay")

            for ax in axes.flatten():
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(f"debug_flow_{indx}.png")
            plt.close()

        # Pad or truncate to max_query_points
        num_points = points.shape[1]
        if num_points < max_query_points:
            pad = max_query_points - num_points
            points = np.pad(points, ((0, 0), (0, pad), (0, 0)))
            viz = np.pad(viz, ((0, 0), (0, pad)))
            mask = np.pad(mask, ((0, 0), (0, pad)))
        elif num_points > max_query_points:  # crop to max_query_points
            points = points[:, :max_query_points]
            viz = viz[:, :max_query_points]
            mask = mask[:, :max_query_points]

        point_tracking_results.append(
            {
                "points": points,
                "points_visibility": viz,
                "points_mask": mask,
            }
        )

        seg_masks.append(segm_mask)

    return point_tracking_results, seg_masks


def process_framestack(cfg, images: List[np.ndarray]) -> List[np.ndarray]:
    """Apply frame stacking to a sequence of images."""
    log(f"Framestacking images with framestack {cfg.framestack}")
    processed_images = []

    for ep_imgs in images:
        # Pad with first frame
        first_frame = np.repeat(ep_imgs[0][None], cfg.framestack - 1, axis=0)
        ep_imgs = np.concatenate([first_frame, ep_imgs], axis=0)

        # Create framestack
        new_imgs = []
        for i in range(len(ep_imgs) - cfg.framestack + 1):
            new_imgs.append(ep_imgs[i : i + cfg.framestack])

        processed_images.append(np.array(new_imgs))

    return processed_images
