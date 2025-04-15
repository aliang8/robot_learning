"""
The first dir is the base dir.

python3 -m robot_learning.real_robot.scripts.combine_data_dirs \
    /project2/biyik_1165/aliang80/datasets/robot/play1 \
    /project2/biyik_1165/aliang80/datasets/robot/play2 \
    /project2/biyik_1165/aliang80/datasets/robot/play3 \
    /project2/biyik_1165/aliang80/datasets/robot/play4 \
    /project2/biyik_1165/aliang80/datasets/robot/play5

python3 -m robot_learning.real_robot.scripts.combine_data_dirs \
    /scr/shared/clam/datasets/robot/play \
    /scr/shared/clam/datasets/robot/play2
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List

from robot_learning.utils.logger import log


def combine_data_dirs(data_dirs: List[str]):
    # take the first data dir as the base
    base_data_dir = data_dirs[0]

    # count number of trajs in base data dir
    traj_dirs = list(Path(base_data_dir).glob("traj*"))
    traj_ct = [int(path.name.split("traj")[1]) for path in traj_dirs]
    max_traj_ct = max(traj_ct)
    log(f"Using {base_data_dir} as base data dir", "yellow")
    counter = max_traj_ct + 1

    for data_dir in data_dirs[1:]:
        log(f"Combining data from {data_dir}", "yellow")

        # Load data from each directory
        for traj_dir in sorted(Path(data_dir).glob("traj*")):
            counter += 1
            log(f"\tProcessing {traj_dir}", "yellow")

            # rename traj dir
            new_traj_dir = Path(base_data_dir) / f"traj{counter}"
            log(f"\tNew dir: {new_traj_dir}")
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

    output_dir = sys.argv[-1]
    combine_data_dirs(data_dirs)
    log("Finished combining data directories", "blue")
