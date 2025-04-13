"""
Usage:
    python -m robot_learning.data.optical_flow.visualize_flow \
        --data_dir=
"""

import os
from pathlib import Path

import cv2
import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tensorflow as tf
from omegaconf import DictConfig
from tqdm import tqdm

from robot_learning.data.utils import load_data_compressed
from robot_learning.utils.logger import log


def add_flow_to_image(image, flow):
    """
    Image: [H, W, 3]
    Flow: [T, 2]
    """
    T = flow.shape[0]
    for t in range(T):
        flow_vec = flow[t]
        x, y = flow_vec[0], flow_vec[1]
        alpha = (t / T) if T > 0 else 1.0

        blue = 0
        green = int(128 * alpha)
        red = int(255 * alpha)

        cv2.circle(
            image,
            center=(int(x), int(y)),
            radius=5,
            color=(blue, green, red),
            thickness=-1,
        )
    return image


def add_flow_to_video(video, flow):
    """
    Video: [T, H, W, 3]
    Flow: [T, 2]

    Add flow as points on the video with temporal color gradient trail in orange.
    """
    # make circle size adaptive to video size
    T, H, W = video.shape[0], video.shape[1], video.shape[2]
    circle_size = max(H, W) // 100

    # Store all previous points for each frame
    for i in range(T):
        frame = video[i]

        # Draw all points up to current timestep
        for t in range(i + 1):
            flow_vec = flow[t]
            x, y = flow_vec[0], flow_vec[1]

            # Calculate color based on temporal distance from current frame
            # Newer points are bright orange, older points are more faded
            alpha = (t / i) if i > 0 else 1.0  # avoid division by zero

            # True orange in BGR: (0, 128, 255)
            blue = 0
            green = int(128 * alpha)
            red = int(255 * alpha)

            cv2.circle(
                frame,
                center=(int(x), int(y)),
                radius=circle_size,
                color=(blue, green, red),  # fading orange
                thickness=-1,  # filled circle
            )

        video[i] = frame
    return video


def visualize_flow(data_dir: str):
    traj_dirs = sorted(list(Path(data_dir).glob("*traj_*")))
    num_trajs = len(traj_dirs)

    log(f"Found {num_trajs} trajectories", "green")
    for i, traj in tqdm(
        enumerate(traj_dirs[:8]), desc="Processing trajectories", total=num_trajs
    ):
        flow_file = traj / "2d_flow_query.dat"
        if not flow_file.exists():
            log(f"Skipping {traj} because it does not have a flow file", "red")
            continue

        # TODO: this changes based on the env
        img_file = traj / "external_images.dat"
        if not img_file.exists():
            log(f"Skipping {traj} because it does not have an image file", "red")
            continue

        flow_data = load_data_compressed(flow_file)
        images = load_data_compressed(img_file)
        points = flow_data["points"]
        normalized_points = flow_data["points_normalized"]
        log(f"Images shape: {images.shape}")
        log(f"Points shape: {points.shape}")

        # make video of frames and points and flow at each step
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plt.tight_layout()
        ax.axis("off")

        def animate(i):
            ax.clear()
            ax.axis("off")
            ax.imshow(images[i])
            # Plot points at current timestep
            ax.scatter(points[i, :, 0], points[i, :, 1], color="orange", s=30)
            # Plot flow vectors
            if i < len(images) - 1:
                flow = points[i + 1] - points[i]
                ax.quiver(
                    points[i, :, 0],
                    points[i, :, 1],
                    flow[:, 0],
                    flow[:, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=0.15,
                    color="green",
                )
            ax.set_title(f"Frame {i}")

        anim = animation.FuncAnimation(
            fig, animate, frames=len(images), interval=100, blit=False
        )

        # Save animation
        writer = animation.FFMpegWriter(fps=10)
        anim.save(traj / f"traj_flow_{i}.mp4", writer=writer)
        plt.close()

        if i > 10:
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    visualize_flow(args.data_dir)
