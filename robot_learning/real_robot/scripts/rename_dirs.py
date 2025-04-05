"""
Script to rename image directories in trajectory data.

Usage:
    python robot_learning.real_robot.scripts.rename_image_dirs <base_directory>
"""

import os
import sys
from glob import glob
from pathlib import Path

from clam.utils.logger import log


def rename_image_directories(base_dir: str):
    """
    Rename image directories from imagesX format to specific camera names.

    Mapping:
        images0 -> external_imgs
        images1 -> wrist_imgs
    """
    traj_dirs = sorted(glob(str(Path(base_dir) / "*")))

    for traj_dir in traj_dirs:
        # Skip if not a directory
        if not os.path.isdir(traj_dir):
            continue

        log(f"Processing: {traj_dir}", "yellow")

        # Check for image directories
        for old_name in ["images0", "images1"]:
            old_path = Path(traj_dir) / old_name
            if not old_path.exists():
                continue

            # Determine new name
            new_name = (
                "external_imgs" if old_name == "images0" else "over_shoulder_imgs"
            )
            new_path = Path(traj_dir) / new_name

            # Rename directory
            try:
                old_path.rename(new_path)
                log(f"  Renamed {old_name} -> {new_name}", "green")
            except Exception as e:
                log(f"  Error renaming {old_name}: {str(e)}", "red")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_image_dirs.py <base_directory>")
        sys.exit(1)

    base_dir = sys.argv[1]
    if not os.path.exists(base_dir):
        print(f"Directory does not exist: {base_dir}")
        sys.exit(1)

    log(f"Starting directory renaming in: {base_dir}", "blue")
    rename_image_directories(base_dir)
    log("Finished renaming directories", "blue")
