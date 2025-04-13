"""
python3 -m robot_learning.real_robot.scripts.download_model \
    --server aliang80@snoopy.usc.edu \
    --source-root /scr/aliang80/p-llm-hf/hand_demos/results \
    --target-root /home/liralab-widowx/p-llm-hf/hand_demos/results \

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

from robot_learning.utils.logging import log


def get_model_paths(server_name, source_root, run_id, ckpt_steps=None):
    """Get paths to model checkpoints for specified run_id.

    Args:
        server_name: Server to check
        source_root: Root directory containing runs
        run_id: Specific run ID to download from
        ckpt_steps: List of specific checkpoint steps to download, or None for all
    """
    # List contents of run directory to get model paths
    run_path = Path(source_root) / run_id
    cmd = ["ssh", server_name, f"ls {run_path}/*/*/model_ckpts/"]

    log(f"Getting model paths for run_id: {run_id}")
    log(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        ckpt_paths = result.stdout.strip().split("\n")

        # Filter checkpoints if specific steps requested
        if ckpt_steps:
            filtered_paths = []
            for path in ckpt_paths:
                for step in ckpt_steps:
                    if f"step_{step}" in path:
                        filtered_paths.append(path)
            return filtered_paths
        return ckpt_paths

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


def download_models(server_name, source_root, target_root, run_id, ckpt_steps=None):
    """Download specific model checkpoints for a run.

    Args:
        server_name: Server to download from
        source_root: Root directory on server
        target_root: Root directory on local machine
        run_id: Specific run ID to download from
        ckpt_steps: List of specific checkpoint steps to download, or None for all
    """
    # Get paths to all relevant checkpoints
    ckpt_paths = get_model_paths(server_name, source_root, run_id, ckpt_steps)

    if not ckpt_paths:
        log(f"No checkpoints found for run_id: {run_id}")
        return

    # Download each checkpoint
    for path in ckpt_paths:
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
        "--source-root", type=str, required=True, help="Root directory on server"
    )
    parser.add_argument(
        "--target-root", type=str, required=True, help="Root directory on local machine"
    )
    parser.add_argument(
        "--run-id", type=str, required=True, help="Specific run ID to download from"
    )
    parser.add_argument(
        "--ckpt-steps",
        type=int,
        nargs="+",
        help="Specific checkpoint steps to download (e.g., 1000 2000 3000)",
    )

    args = parser.parse_args()

    download_models(
        server_name=args.server,
        source_root=args.source_root,
        target_root=args.target_root,
        run_id=args.run_id,
        ckpt_steps=args.ckpt_steps,
    )
