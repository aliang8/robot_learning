"""
Usage:
    python -m robot_learning.data.optical_flow.visualize_flow \
        --data_dir=
"""

import os
from pathlib import Path

import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tensorflow as tf
from omegaconf import DictConfig
from tqdm import tqdm

from robot_learning.data.utils import load_data_compressed
from robot_learning.utils.logger import log


def visualize_flow(data_dir: str):
    traj_dirs = sorted(list(Path(data_dir).glob("traj_*")))
    num_trajs = len(traj_dirs)

    log(f"Found {num_trajs} trajectories", "green")
    for i, traj in tqdm(
        enumerate(traj_dirs), desc="Processing trajectories", total=num_trajs
    ):
        flow_file = traj / "2d_flow.dat"
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
