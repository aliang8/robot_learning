"""
Given a video, we use GroundedSAM2 to generate
object masks for the first frame of the video.

The object masks are used to filter query points for CoTracker.
We apply CoTracker to track the query points across the video
to generate the 2D "object/scene" flow.

The output is a set of points (N, 2) for each video where N is the desired number
of points to track. We also get the visibility of each point across the video,
note not all the points are visible at each frame because of occlusions.
"""

import os
import sys
import time
from base64 import b64encode
from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import torch
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from robot_learning.utils.logger import log

# ===============================================
# HELPER FUNCTIONS
# ===============================================


def load_sam_model(sam_ckpt_path: str, model_cfg_file: str):
    log("Building SAM2 model")
    start = time.time()
    sam2_image_model = build_sam2(model_cfg_file, sam_ckpt_path)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    log(f"Built SAM2 model in {time.time() - start:.2f}s")

    return sam2_image_model, image_predictor


def load_cotracker(cotracker_ckpt_path: str):
    log("Initializing CoTracker model")
    model = CoTrackerPredictor(checkpoint=cotracker_ckpt_path)
    return model


def get_seg_mask(
    image_predictor: SAM2ImagePredictor,
    grounding_model_id: str,
    text: str,
    video: np.ndarray = None,
    image: np.ndarray = None,
    device: str = "cuda",
):
    """
    Args:
        image: (H, W, 3) numpy array
        video: (T, H, W, 3) numpy array
    """

    """
    Step 1: Environment settings and model initialization
    """
    # init grounding dino model from huggingface
    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        grounding_model_id
    ).to(device)

    if video is not None:
        # image is first frame of the video
        image = Image.fromarray(video[0].astype(np.uint8))
    else:
        image = Image.fromarray(image.astype(np.uint8))

    """
    Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
    """

    # run Grounding DINO on the image
    log(f"Running Grounding DINO on the image, with text: {text}")
    start = time.time()
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    log(f"Finished running Grounding DINO, took {time.time() - start:.2f}s")

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process the detection results
    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]
    log(f"Inferred boxes: {input_boxes}")
    log(f"Detected objects: {OBJECTS}")

    # prompt SAM 2 image predictor to get the mask for the object
    log("Running SAM2 image predictor to get the mask for the object")
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # convert the mask shape to (n, H, W)
    if masks.ndim == 2:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)
    log(f"Got masks of shape: {masks.shape}")
    log(f"Object scores: {scores}")

    # Combine all the masks to generate one mask of relevant objects
    combined_mask = np.zeros_like(masks[0])
    for mask in masks:
        combined_mask += mask

    log(f"Mask min: {combined_mask.min()}, max: {combined_mask.max()}")
    return combined_mask


def generate_point_tracks(
    cotracker: CoTrackerPredictor,
    video: np.ndarray,
    segm_mask: np.ndarray = None,
    queries: np.ndarray = None,
    grid_size: int = 25,
    device: str = "cuda",
):
    """
    Args:
        video: (T, H, W, 3) numpy array
    """
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    video = video.to(device)

    log("Running CoTracker on the video")
    start = time.time()

    if queries is not None:
        log(f"Using provided queries: {queries}")
        queries = torch.from_numpy(queries).float().to(device)

    if segm_mask is None:
        pred_tracks, pred_visibility = cotracker(
            video,
            segm_mask=torch.from_numpy(segm_mask)[None, None],
            queries=queries[None],
        )
    else:
        pred_tracks, pred_visibility = cotracker(
            video,
            grid_size=grid_size,
            segm_mask=torch.from_numpy(segm_mask)[None, None],
        )
    log(
        f"Predicted tracks shape: {pred_tracks.shape}, visibility shape: {pred_visibility.shape}"
    )
    log(f"Finished running CoTracker, took {time.time() - start:.2f}s")
    return pred_tracks, pred_visibility
