"""
Script to render a trajectory from dataset collected
from the real world WidowX robot.

Expected file structure:

data_dir/
    dataset_name/
        task_name/
            traj_0/
                images0/
                images1/
                ...
            traj_1/
                ...

python3 -m robot_learning.real_robot.scripts.visualize_trajectory \
    --data_dir=
"""

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from robot_learning.utils.logger import log


def load_trajectory_data(traj_dir: Path) -> Tuple[List[Path], List[Path], List[dict]]:
    """Load and sort image paths and metadata from trajectory directory."""
    external_imgs = sorted(
        list(traj_dir.glob("external_imgs/*")),
        key=lambda x: int(x.stem.split("_")[-1]),
    )
    over_shoulder_imgs = (
        sorted(
            list(traj_dir.glob("over_shoulder_imgs/*")),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        if (traj_dir / "over_shoulder_imgs").exists()
        else None
    )

    wrist_imgs = (
        sorted(
            list(traj_dir.glob("wrist_imgs/*")),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        if (traj_dir / "wrist_imgs").exists()
        else None
    )

    depth_imgs = (
        sorted(
            list(traj_dir.glob("depth_images0/*")),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        if (traj_dir / "depth_images0").exists()
        else None
    )

    # Load metadata
    obs_dict_path = traj_dir / "obs_dict.pkl"
    obs_dict = np.load(obs_dict_path, allow_pickle=True)

    # Load policy output
    policy_out_path = traj_dir / "policy_out.pkl"
    policy_out = np.load(policy_out_path, allow_pickle=True)
    policy_out = {k: np.array([p[k] for p in policy_out]) for k in policy_out[0].keys()}

    # Load agent data
    # agent_data_path = traj_dir / "agent_data.pkl"
    # agent_data = pickle.load(open(agent_data_path, "rb"))

    # Combine metadata
    metadata = {**obs_dict, **policy_out}

    return external_imgs, over_shoulder_imgs, wrist_imgs, depth_imgs, metadata


def visualize_trajectory(data_dir: Path, output_path: Path = None, fps: int = 30):
    """
    Create a video visualization of trajectory images using matplotlib animation.

    Args:
        data_dir: Directory containing trajectory folders
        output_path: Path to save the video (default: same as data_dir)
        fps: Frames per second for the output video
    """
    if output_path is None:
        output_path = data_dir / "trajectory_visualization.mp4"

    # Get sorted trajectory directories
    traj_dirs = sorted(list(data_dir.glob("traj*")))
    traj_dirs = [
        traj_dir for traj_dir in traj_dirs if not traj_dir.name.startswith("traj_")
    ]
    if not traj_dirs:
        raise ValueError(f"No trajectory directories found in {data_dir}")

    # Load all image paths and metadata
    all_images = []
    all_metadata = []
    for traj_idx, traj_dir in enumerate(tqdm.tqdm(traj_dirs[:1])):
        external_imgs, over_shoulder_imgs, wrist_imgs, depth_imgs, metadata = (
            load_trajectory_data(traj_dir)
        )
        for i, img0_path in enumerate(external_imgs):
            img1_path = (
                over_shoulder_imgs[i]
                if over_shoulder_imgs and i < len(over_shoulder_imgs)
                else None
            )
            img2_path = wrist_imgs[i] if wrist_imgs and i < len(wrist_imgs) else None
            depth_img_path = (
                depth_imgs[i] if depth_imgs and i < len(depth_imgs) else None
            )
            all_images.append((img0_path, img1_path, img2_path, depth_img_path))
            # Create per-timestep metadata
            timestep_metadata = {
                "traj_idx": traj_idx,
                "timestep": i,
                "state": metadata["state"][i],
                "action": metadata["actions"][i]
                if i < len(metadata["actions"])
                else None,
            }
            all_metadata.append(timestep_metadata)

    log(f"Loaded {len(all_images)} images", "yellow")

    # Create figure and axes with minimal spacing in a 2x2 grid
    fig = plt.figure(figsize=(16, 16), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,  # Changed to 2x2 grid
        width_ratios=[1, 1],
        height_ratios=[1, 1],
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.99,
        wspace=0.02,  # Minimal space between images
        hspace=0.02,  # Minimal space between rows
    )

    ax1 = fig.add_subplot(gs[0, 0])  # Top left
    ax2 = fig.add_subplot(gs[0, 1])  # Top right
    ax3 = fig.add_subplot(gs[1, 0])  # Bottom left
    ax4 = fig.add_subplot(gs[1, 1])  # Bottom right

    # Remove axes and frames
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    # Initialize with first frame
    external_img_path, over_shoulder_img_path, wrist_img_path, depth_img_path = (
        all_images[0]
    )
    metadata = all_metadata[0]
    external_img = plt.imread(external_img_path)
    im1 = ax1.imshow(external_img)

    # Initialize other views (assuming they exist in the metadata)
    if over_shoulder_img_path:
        over_shoulder_img = plt.imread(over_shoulder_img_path)
        im2 = ax2.imshow(over_shoulder_img)
    else:
        im2 = None
        ax2.remove()

    if wrist_img_path:
        wrist_img = plt.imread(wrist_img_path)
        im3 = ax3.imshow(wrist_img)
    else:
        im3 = None
        ax3.remove()

    if depth_img_path:
        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
        depth_img = cv2.convertScaleAbs(depth_img, alpha=255.0 / depth_img.max())
        im4 = ax4.imshow(depth_img)
    else:
        im4 = None
        ax4.remove()

    # Initialize text overlay (now in top-left corner)
    text_left = ax1.text(
        0.02,
        0.98,
        "",
        transform=ax1.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontfamily="monospace",
        fontsize=18,
        fontweight="bold",
        bbox=dict(
            facecolor="black",
            alpha=0.7,
            edgecolor="none",
            pad=3,
        ),
        color="white",
    )

    def format_array(arr, label):
        """Format numpy array with label and rounded values."""
        if arr is None:
            return f"{label}: None"
        arr = np.round(arr, 3)
        return f"{label}: {arr}"

    def init():
        """Initialize animation."""
        outputs = [im1, text_left]
        if im2 is not None:
            outputs.append(im2)
        if im3 is not None:
            outputs.append(im3)
        if im4 is not None:
            outputs.append(im4)
        return outputs

    def update(frame):
        """Update function for animation."""
        external_img_path, over_shoulder_img_path, wrist_img_path, depth_img_path = (
            all_images[frame]
        )
        metadata = all_metadata[frame]

        # Update first image
        external_img = plt.imread(external_img_path)
        im1.set_array(external_img)

        outputs = [im1]

        # Update second image if it exists
        if img1_path and im2 is not None:
            over_shoulder_img = plt.imread(over_shoulder_img_path)
            im2.set_array(over_shoulder_img)
            outputs.append(im2)

        # Update third image if it exists
        if img2_path and im3 is not None:
            wrist_img = plt.imread(wrist_img_path)
            im3.set_array(wrist_img)
            outputs.append(im3)

        # Update fourth image if it exists
        if depth_img_path and im4 is not None:
            depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
            depth_img = cv2.convertScaleAbs(depth_img, alpha=255.0 / depth_img.max())
            im4.set_array(depth_img)
            outputs.append(im4)

        # Update text with trajectory, state and action information
        state = metadata["state"][:-1]  # Exclude last element
        action = metadata["action"]

        text_content = (
            f"Traj: {metadata['traj_idx']}\n"
            f"T: {metadata['timestep']}\n"
            f"{format_array(state, 'State')}\n"
            f"{format_array(action, 'Action')}"
        )

        text_left.set_text(text_content)
        outputs.append(text_left)

        return outputs

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(all_images),
        interval=1000 / fps,
        blit=True,
    )

    # Save animation
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist="Me"), bitrate=3000)
    anim.save(str(output_path), writer=writer)
    plt.close()

    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    visualize_trajectory(args.data_dir, args.output_path, args.fps)
