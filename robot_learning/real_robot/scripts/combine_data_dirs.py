"""
python3 -m robot_learning.real_robot.scripts.combine_data_dirs \
    /scr/shared/clam/datasets/robot/play4 \
    /scr/shared/clam/datasets/robot/play5
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List

from robot_learning.utils.logger import log


def combine_data_dirs(data_dirs: List[str]):
    counter = 0

    # take the first data dir as the base
    new_data_dir = Path(data_dirs[0]).parent / "reach_green_block"
    new_data_dir.mkdir(parents=True, exist_ok=True)

    for data_dir in data_dirs:
        log(f"Combining data from {data_dir}", "yellow")
        # Load data from each directory
        for traj_dir in sorted(Path(data_dir).glob("traj*")):
            log(f"Processing {traj_dir}", "yellow")
            counter += 1

            # rename traj dir
            new_traj_dir = Path(new_data_dir) / f"traj{counter}"
            try:
                shutil.move(str(traj_dir), str(new_traj_dir))
            except Exception as e:
                log(f"Error moving {traj_dir} to {new_traj_dir}: {e}", "red")
                # Optionally handle the error or continue

    log(f"Combined {counter} trajectories", "green")


if __name__ == "__main__":
    data_dirs = sys.argv[1:]
    if not data_dirs:
        print("Usage: python combine_data_dirs.py <data_dir1> <data_dir2> ...")
        sys.exit(1)

    combine_data_dirs(data_dirs)
    log("Finished combining data directories", "blue")
