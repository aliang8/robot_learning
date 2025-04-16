"""
python3 -m robot_learning.real_robot.scripts.download_model \
    --server aliang80@hpc-transfer2.usc.edu \
    --source_root=/project2/biyik_1165/aliang80/clam/results \
    --target_root=/home/liralab-widowx/continuous_lam/clam/results \
    --run_ids=carc_0003_act_0000

python3 -m robot_learning.real_robot.scripts.download_model \
    --server aliang80@hpc-transfer2.usc.edu \
    --source_root=/project2/biyik_1165/aliang80/clam/results \
    --target_root=/home/liralab-widowx/continuous_lam/clam/results \
    --run_ids=carc_0003 \
    --multirun


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


def get_paths_to_download(
    server_name, source_root, run_id, ckpt_steps=None, multirun=False
):
    """Get paths to model checkpoints for specified run_id.

    Args:
        server_name: Server to check
        source_root: Root directory containing runs
        run_id: Specific run ID to download from
        ckpt_steps: List of specific checkpoint steps to download, or None for all
        multirun: Whether to look for checkpoints in multirun structure (extra depth)
    """
    # List contents of run directory to get model paths
    exp_path = Path(source_root) / run_id

    # Different commands for different servers
    if multirun:
        cmd = ["ssh", "-i", "~/.ssh/id_rsa", server_name, f"readlink -f {exp_path}/*/*"]
    else:
        cmd = ["ssh", "-i", "~/.ssh/id_rsa", server_name, f"readlink -f {exp_path}/*"]

    log(f"Getting model paths for run_id: {run_id}")
    log(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    exp_dirs = result.stdout.strip().split("\n")

    # filter log.txt and multirun.yaml
    exp_dirs = [d for d in exp_dirs if "log.txt" not in d and "multirun.yaml" not in d]

    if ckpt_steps is None:
        ckpt_paths = [Path(d) / "model_ckpts" / "latest.pkl" for d in exp_dirs]
    else:
        ckpt_paths = [
            Path(d) / "model_ckpts" / f"ckpt_{step:06d}.pkl"
            for d in exp_dirs
            for step in ckpt_steps
        ]
    # also need the config paths
    config_paths = [Path(d) / "config.yaml" for d in exp_dirs]
    paths = ckpt_paths + config_paths
    return paths


def setup_rsync(source_root, target_root, source_path, server_name, no_confirm=False):
    """
    Sets up rsync from server to local, creating necessary directories.

    Args:
        source_root: Root directory on server (e.g., '/scr/my_project/')
        target_root: Root directory on local machine (e.g., '/Users/me/projects/')
        source_path: Path to sync (file or directory)
        server_name: Server to sync from (e.g., 'username@server.com')
        no_confirm: Skip confirmation for overwriting
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

    # Check if target file exists
    if target_path.exists():
        if not no_confirm:
            log(f"\nWarning: Target file already exists: {target_path}")
            response = input("Do you want to overwrite? [y/N] ").lower()
            if response != "y":
                log("Skipping file...\n")
                return

    # Construct rsync command
    source = f"{server_name}:{source_path}"

    # For files, we want to sync to the parent directory with a trailing slash
    # For directories, we want to sync to the directory itself
    if ".pkl" in str(source_path) or ".yaml" in str(source_path):
        target = str(target_path.parent) + "/"
    else:
        target = str(target_path)

    # Different rsync commands for different servers
    if "snoopy" in server_name:
        cmd = [
            "rsync",
            "-avP",  # archive mode, verbose, show progress
            "-e",
            "ssh -i ~/.ssh/id_rsa",  # specify ssh key
            source,
            target,
        ]
    else:  # CARC
        cmd = [
            "rsync",
            "-rltvh",
            "-e",
            f"ssh -i {os.path.expanduser('~/.ssh/id_rsa')}",
            source,
            target,
        ]

    log("\n" + "=" * 100)
    log("SYNCING FILE:")
    log("-" * 50)
    log(f"FROM: {source}")
    log(f"TO:   {target}")
    log("-" * 50)
    log(f"COMMAND: {' '.join(cmd)}")
    log("=" * 100 + "\n")

    try:
        subprocess.run(cmd, check=True)
        log("✓ Sync completed successfully!\n")
    except subprocess.CalledProcessError as e:
        log(f"✗ Error during sync: {e}\n")


def download_models(
    server_name,
    source_root,
    target_root,
    run_ids,
    ckpt_steps=None,
    multirun=False,
    no_confirm=False,
):
    """Download specific model checkpoints for a run."""
    # Get paths to all relevant checkpoints
    paths_to_download = []
    for run_id in run_ids:
        log("\n" + "#" * 100)
        log(f"Processing run ID: {run_id}")
        log("#" * 100 + "\n")

        paths = get_paths_to_download(
            server_name, source_root, run_id, ckpt_steps, multirun
        )
        paths_to_download.extend(paths)

    # Show summary of files to download
    log("\n" + "*" * 100)
    log(f"Found {len(paths_to_download)} files to download:")
    for i, path in enumerate(paths_to_download, 1):
        log(f"{i:3d}. {path}")
    log("*" * 100 + "\n")

    # Download each checkpoint
    for i, path in enumerate(paths_to_download, 1):
        log(f"\nFile {i}/{len(paths_to_download)}:")
        setup_rsync(
            source_root=source_root,
            target_root=target_root,
            source_path=path,
            server_name=server_name,
            no_confirm=no_confirm,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download specific model checkpoints from server"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="aliang80@snoopy1.usc.edu",
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
    parser.add_argument(
        "--multirun",
        action="store_true",
        help="Look for checkpoints in multirun structure (extra depth)",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation for overwriting existing files",
    )

    args = parser.parse_args()

    download_models(
        server_name=args.server,
        source_root=args.source_root,
        target_root=args.target_root,
        run_ids=args.run_ids,
        ckpt_steps=args.ckpt_steps,
        multirun=args.multirun,
        no_confirm=args.no_confirm,
    )
