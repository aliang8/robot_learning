"""
python3 -m robot_learning.real_robot.scripts.download_model \
    --server aliang80@snoopy1.usc.edu \
    --source_root=/scr/aliang80/p-llm-hf/hand_demos/results \
    --target_root=/home/liralab-widowx/p-llm-hf/hand_demos/results \
    --run_ids=reach_block_hpt_no_state

Ckpt structure:
    root/
        run_id/
            name/
                hp_name/
                    model_ckpts/
"""

#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

from robot_learning.utils.logger import log


def get_paths_to_download(server_name, source_root, run_id, ckpt_steps=None):
    """Get paths to model checkpoints for specified run_id.

    Args:
        server_name: Server to check
        source_root: Root directory containing runs
        run_id: Specific run ID to download from
        ckpt_steps: List of specific checkpoint steps to download, or None for all
    """
    # List contents of run directory to get model paths
    exp_path = Path(source_root) / run_id

    # Get all the subdirectories for the experiment
    cmd = ["ssh", "-i", "~/.ssh/id_rsa", server_name, f"ls {exp_path}/*"]

    log(f"Getting model paths for run_id: {run_id}")
    log(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        exp_dirs = result.stdout.strip().split("\n")

        # filter log.txt and multirun.yaml
        exp_dirs = [
            d for d in exp_dirs if "log.txt" not in d and "multirun.yaml" not in d
        ]

        if ckpt_steps is None:
            ckpt_paths = [exp_path / d / "model_ckpts" / "latest.pkl" for d in exp_dirs]
        else:
            ckpt_paths = [
                exp_path / d / "model_ckpts" / f"ckpt_{step:06d}.pkl"
                for d in exp_dirs
                for step in ckpt_steps
            ]
        # also need the config paths
        config_paths = [exp_path / d / "config.yaml" for d in exp_dirs]
        paths = ckpt_paths + config_paths
        return paths

    except subprocess.CalledProcessError as e:
        log(f"Error getting model paths: {e}")
        return []


def setup_rsync(source_root, target_root, source_path, server_name):
    """
    Sets up rsync from server to local, creating necessary directories.

    Args:
        source_root: Root directory on server (e.g., '/scr/my_project/')
        target_root: Root directory on local machine (e.g., '/Users/me/projects/')
        source_path: Path to sync (e.g., '/scr/my_project/data/images/')
        server_name: Server to sync from (e.g., 'username@server.com')
    """
    # Convert paths to Path objects
    source_root = Path(source_root)
    target_root = Path(target_root)
    source_path = Path(source_path)

    # Get relative path from source_root to source_path
    rel_path = source_path.relative_to(source_root)

    # Create target directory structure
    target_path = target_root / rel_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Construct rsync command
    source = f"{server_name}:{source_path}"
    target = str(target_path)

    # Run rsync with progress flag and archive mode
    cmd = [
        "rsync",
        "-avP",  # archive mode, verbose, show progress
        "--relative",  # use relative paths
        "-e",
        "ssh -i ~/.ssh/id_rsa",  # specify ssh key
        source,
        target,
    ]

    log(f"Syncing from: {source}")
    log(f"Syncing to: {target}")
    log(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        log("Sync completed successfully!")
    except subprocess.CalledProcessError as e:
        log(f"Error during sync: {e}")


def download_models(server_name, source_root, target_root, run_ids, ckpt_steps=None):
    """Download specific model checkpoints for a run.

    Args:
        server_name: Server to download from
        source_root: Root directory on server
        target_root: Root directory on local machine
        run_id: Specific run ID to download from
        ckpt_steps: List of specific checkpoint steps to download, or None for all
    """
    # Get paths to all relevant checkpoints
    paths_to_download = []
    for run_id in run_ids:
        paths = get_paths_to_download(server_name, source_root, run_id, ckpt_steps)
        paths_to_download.extend(paths)

    # Download each checkpoint
    for path in paths_to_download:
        log(f"Downloading checkpoint: {path}")
        setup_rsync(
            source_root=source_root,
            target_root=target_root,
            source_path=path,
            server_name=server_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download specific model checkpoints from server"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="aliang80@snoopy.usc.edu",
        help="Server name (e.g., username@server.com)",
    )
    parser.add_argument(
        "--source_root", type=str, required=True, help="Root directory on server"
    )
    parser.add_argument(
        "--target_root", type=str, required=True, help="Root directory on local machine"
    )
    parser.add_argument(
        "--run_ids",
        type=str,
        required=True,
        nargs="+",
        help="Specific run ID to download from",
    )
    parser.add_argument(
        "--ckpt_steps",
        type=int,
        nargs="+",
        help="Specific checkpoint steps to download (e.g., 1000 2000 3000)",
    )

    args = parser.parse_args()

    download_models(
        server_name=args.server,
        source_root=args.source_root,
        target_root=args.target_root,
        run_ids=args.run_ids,
        ckpt_steps=args.ckpt_steps,
    )
