#! /usr/bin/python3
# This is the highest level interface to interact with the widowx setup.

import argparse
import pickle as pkl
import queue
import sys
import termios
import threading
import time
import tty
from collections import defaultdict, deque, namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T

# install from: https://github.com/youliangtan/edgeml
from edgeml.action import ActionClient, ActionConfig, ActionServer
from edgeml.internal.utils import compute_hash, jpeg_to_mat, mat_to_jpeg
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from robot_learning.models.image_embedder import EMBEDDING_DIMS, ImageEmbedder
from robot_learning.models.policy.action_chunking_transformer_decoder import (
    ActionChunkingTransformerPolicy,
    ACTTemporalEnsembler,
)
from robot_learning.trainers import trainer_to_cls
from robot_learning.utils.general_utils import to_numpy
from robot_learning.utils.logger import log
from widowx_envs.utils.exceptions import Environment_Exception
from widowx_envs.utils.raw_saver import RawSaver
from widowx_envs.widowx_env_service import WidowXClient, WidowXStatus, show_video


class WidowXConfigs:
    DefaultEnvParams = {
        "fix_zangle": 0.1,
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": [
            [0.1, -0.15, -0.1, -1.57, 0],
            [0.45, 0.25, 0.25, 1.57, 0],
        ],
        "catch_environment_except": False,
        "start_state": [
            0.11865137,
            -0.01696823,
            0.24405071,
            -0.03702571,
            -0.11837727,
            0.03907566,
            0.9994886,
        ],
        "skip_move_to_neutral": False,
        "return_full_image": False,
        "camera_topics": [
            {"name": "/D435/color/image_raw"},
            {"name": "/blue/image_raw"},
            # {"name": "/yellow/image_raw"},
        ],
    }

    DefaultActionConfig = ActionConfig(
        port_number=5556,
        action_keys=["init", "move", "gripper", "reset", "step_action", "reboot_motor"],
        observation_keys=[
            "image",
            "state",
            "over_shoulder_img",
            "external_img",
            "wrist_img",
        ],
        broadcast_port=5557,
    )


def wait_for_observation(client: WidowXClient, timeout: int = 60) -> Dict:
    """Wait for and return a valid observation from the robot."""
    start_time = time.time()
    while True:
        obs = client.get_observation()
        if obs is not None:
            log("✓ Received valid observation from robot", "green")
            return obs

        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"No observation received from robot after {timeout}s")

        time.sleep(1)
        log(f"⏳ Waiting for robot observation... (elapsed: {elapsed:.1f}s)", "yellow")


def get_user_feedback() -> Tuple[bool, str]:
    """Get feedback from the user about the trajectory.

    Returns:
        Tuple of (success: bool, notes: str)
    """
    while True:
        response = input("\nWas this trajectory successful? [y/n]: ").lower().strip()
        if response in ["y", "n"]:
            success = response == "y"
            notes = input(
                "Any notes about this trajectory? (press enter to skip): "
            ).strip()
            return success, notes
        print("Please enter 'y' or 'n'")


def convert_listofdicts2dictoflists(list):
    obs_dict = {}
    for key in list[0].keys():
        vecs = []
        for tstep in list:
            vecs.append(tstep[key])
        obs_dict[key] = np.stack(vecs, 0)
    return obs_dict


def init_robot(cfg: DictConfig) -> WidowXClient:
    # Initialize robot client
    log(f"Connecting to robot at {cfg.ip}:{cfg.port}...", "blue")
    widowx_client = WidowXClient(host=cfg.ip, port=cfg.port)
    log("✓ Connected to robot", "green")

    log("Initializing robot with default parameters...", "blue")
    status = widowx_client.init(
        WidowXConfigs.DefaultEnvParams, image_size=cfg.image_size
    )
    log("✓ Robot initialized", "green")

    # Wait for robot to be ready and reset
    log("Waiting for initial observation...", "blue")
    wait_for_observation(widowx_client)
    log("Resetting robot position...", "blue")
    widowx_client.reset()
    wait_for_observation(widowx_client)

    show_video(widowx_client, duration=2.5)
    log("Robot is ready for operation", "green")
    input("Press [Enter] to start the experiment.")
    return status, widowx_client


def collect_trajectory_data(
    obs_list: List[Dict], actions_list: List[np.ndarray], success: bool, notes: str
) -> Dict:
    """Collate trajectory data into a format suitable for saving.

    Args:
        obs_list: List of observation dictionaries
        actions_list: List of actions taken
        success: Whether the trajectory was successful
        notes: User notes about the trajectory
    """
    # Convert list of observations into arrays
    obs_dict = convert_listofdicts2dictoflists(obs_list)

    # Create agent data dictionary
    agent_data = {
        "actions": np.stack(actions_list),
        "success": success,
        "notes": notes,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    }

    return agent_data, obs_dict


def get_key_async(key_queue: queue.Queue, stop_event: threading.Event):
    """Monitors keyboard input in a separate thread."""

    def _get_key():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    while not stop_event.is_set():
        key = _get_key()
        key_queue.put(key)
        if key == "\x03":  # Ctrl+C
            break


def wait_for_key(
    key_queue: queue.Queue, target_key: Optional[str] = None
) -> Optional[str]:
    """Check for a specific key press without blocking.

    Args:
        key_queue: Queue containing keyboard inputs
        target_key: If specified, only return if this key is pressed. If None, return any key.
    """
    try:
        while not key_queue.empty():
            key = key_queue.get_nowait()
            if target_key is None or key.lower() == target_key:
                return key
    except queue.Empty:
        pass
    return None


def run_eval_rollout(
    cfg: DictConfig,
    widowx_client: WidowXClient,
    agent: torch.nn.Module,
    img_embedder: Optional[ImageEmbedder],
    input_modalities: List[str],
    modality_mapping: Dict[str, str],
    model_cfg: DictConfig,
    device: str,
    max_steps: int = 100,
) -> Tuple[List[Dict], List[np.ndarray], bool, str]:
    """Run a single evaluation rollout with the robot."""
    # Reset environment at start of episode
    log("Resetting robot position...", "blue")
    widowx_client.reset()
    obs = wait_for_observation(widowx_client)
    log(f"Observation keys: {obs.keys()}", "blue")
    log("✓ Robot reset complete", "green")

    # Make sure we're in normal terminal mode for input
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    try:
        input("Press [Enter] to start this episode...")
    except Exception as e:
        print(f"Input error: {e}")
        # Ensure terminal is reset even if input fails
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        raise

    # Initialize keyboard monitoring
    key_queue = queue.Queue()
    stop_event = threading.Event()
    keyboard_thread = threading.Thread(
        target=get_key_async, args=(key_queue, stop_event), daemon=True
    )
    keyboard_thread.start()

    if model_cfg.model.name == "ac_decoder":
        temporal_ensembler = ACTTemporalEnsembler(
            temporal_ensemble_coeff=cfg.temporal_ensemble_coeff,
            chunk_size=model_cfg.data.seq_len,
        )
        temporal_ensembler.reset()
    else:
        temporal_ensembler = None

    try:
        # this is for framestacking
        input_mode_queues = {
            modality: deque(maxlen=model_cfg.data.seq_len)
            for modality in input_modalities
        }

        # embed the first state
        for modality in input_modalities:
            input_ = obs[modality_mapping[modality]]

            if "img" in modality and img_embedder is not None:
                input_ = img_embedder(input_)

            for _ in range(model_cfg.data.seq_len):
                input_mode_queues[modality].append(input_)

        # TODO: fix this!
        # Initialize trajectory storage
        obs_list = []
        actions_list = []

        # Run episode
        episode_start = time.time()
        for timestep in range(max_steps):
            # Check for key presses
            key = wait_for_key(key_queue, target_key=None)  # Check for any key
            if key:
                if key.lower() == "r":
                    log("\n⚠️ Reset requested by user", "yellow")
                    widowx_client.reset()
                    wait_for_observation(widowx_client)
                    return obs_list, actions_list, False, "Reset requested by user"
                elif key.lower() == "s":
                    log("\n💾 Save and continue requested by user", "yellow")
                    return obs_list, actions_list, True, "Saved mid-trajectory by user"

            obs = wait_for_observation(widowx_client)
            obs_list.append(obs)  # Store observation

            log(
                f"""
                    Step {timestep + 1}/{max_steps}:
                    • Episode Time: {time.time() - episode_start:.1f}s
                    • Robot State: {np.array2string(obs["state"], precision=3, suppress_small=True)}
                    • Controls: [R]eset trajectory, [S]ave and continue
                """,
                "blue",
            )

            for modality in input_modalities:
                input_ = obs[modality_mapping[modality]]

                if "img" in modality and img_embedder is not None:
                    input_ = img_embedder(input_)

                input_mode_queues[modality].append(input_)

            # stack the frames to create the final input into the model
            inputs = {}

            # for action chunking, each input modality is [B, T, ...]
            for modality in input_modalities:
                if modality == "states":
                    inputs[modality] = (
                        torch.from_numpy(
                            np.stack(list(input_mode_queues[modality]), axis=0)
                        )
                        .float()
                        .to(device)
                        .unsqueeze(0)  # add batch dimension
                    )

                    # take only the first 3 and last dim for gripper state
                    if model_cfg.model.use_only_gripper_state:
                        inputs[modality] = torch.cat(
                            [inputs[modality][:, :, :3], inputs[modality][:, :, -1:]],
                            dim=-1,
                        )
                elif "embed" in modality:
                    inputs[modality] = torch.stack(
                        list(input_mode_queues[modality]), dim=1
                    ).to(device)

            # add timesteps to input
            # inputs["timesteps"] = (
            #     torch.arange(model_cfg.data.seq_len).to(device).unsqueeze(0) + timestep
            # )

            # Get action from model and execute
            with torch.no_grad():
                # For action chunking decoder, we actually want the last thing in the sequence
                # for each modality. TODO: fix this
                if model_cfg.model.name == "ac_decoder":
                    inputs = {k: v[:, -1:] for k, v in inputs.items() if k != "actions"}

                inference_start = time.time()
                actions = agent.select_action(inputs)
                inference_time = time.time() - inference_start

                if cfg.use_temporal_ensembling:
                    action = temporal_ensembler.update(actions)

                actions = to_numpy(actions).squeeze()
                actions_list.append(actions)  # Store action

            if model_cfg.data.seq_len > 1:
                if cfg.use_temporal_ensembling:
                    widowx_client.step_action(action)
                else:
                    # Open loop execution for action chunking
                    for i in range(model_cfg.data.seq_len // 2):
                        action = actions[i]
                        widowx_client.step_action(action)
                        time.sleep(0.1 - inference_time)  # 10Hz control loop
            else:
                widowx_client.step_action(actions)

        # Get feedback after episode
        log("\nEpisode complete! Collecting feedback...", "yellow")

        # Restore normal terminal mode for feedback
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        success, notes = get_user_feedback()

        return obs_list, actions_list, success, notes

    finally:
        # Clean up keyboard thread
        stop_event.set()
        keyboard_thread.join(timeout=0.1)
        # Ensure terminal is reset to normal mode
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


@hydra.main(version_base=None, config_name="robot_inference", config_path="../cfg")
def main(cfg: DictConfig) -> None:
    # Load model and config
    cfg_file = Path(cfg.ckpt_file) / "config.yaml"
    log(f"Loading config from {cfg_file}", "blue")

    model_cfg = OmegaConf.load(cfg_file)
    model_cfg.load_from_ckpt = True
    model_cfg.ckpt_step = cfg.ckpt_step
    model_cfg.ckpt_file = cfg.ckpt_file

    # Load model from checkpoint
    trainer = trainer_to_cls[model_cfg.name](model_cfg)
    trainer.model.eval()

    agent = trainer.model

    # Initialize embedder if needed
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize saver to store the eval trajectories
    save_dir = Path(cfg.save_dir) / model_cfg.run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    saver = RawSaver(str(save_dir))

    # Start up the robot
    status, widowx_client = init_robot(cfg)

    input_modalities = model_cfg.model.input_modalities
    log(f"Input modalities: {input_modalities}", "blue")
    use_pretrained_img_embed = False
    for modality in input_modalities:
        if "embed" in modality:
            use_pretrained_img_embed = True
            break

    img_embedder = None
    if use_pretrained_img_embed:
        log("Initializing image embedder...", "blue")
        img_embedder = ImageEmbedder(
            model_name=model_cfg.model.embedding_model, device=device
        )
        image_embedding_dim = EMBEDDING_DIMS[model_cfg.model.embedding_model]

    # Map modalities to observation keys
    modality_mapping = {
        "external_img_embeds": "external_img",
        "over_shoulder_img_embeds": "over_shoulder_img",
        "wrist_img_embeds": "wrist_img",
        "states": "state",
    }

    try:
        episode = 0
        total_timesteps = 0
        start_time = time.time()
        successes = 0
        resets = 0

        # Store original terminal settings
        fd = sys.stdin.fileno()
        original_settings = termios.tcgetattr(fd)

        while True:  # Run indefinitely until keyboard interrupt
            try:
                # Ensure terminal is in normal mode at the start of each episode
                termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)

                episode += 1
                episode_start = time.time()

                log(f"\n{'=' * 50}\nStarting Episode {episode}\n{'=' * 50}", "blue")

                # Run evaluation rollout
                obs_list, actions_list, success, notes = run_eval_rollout(
                    cfg=cfg,
                    widowx_client=widowx_client,
                    agent=agent,
                    img_embedder=img_embedder,
                    input_modalities=input_modalities,
                    modality_mapping=modality_mapping,
                    model_cfg=model_cfg,
                    device=device,
                    max_steps=50,
                )

                if notes == "Reset requested by user":
                    resets += 1
                    log(f"Manual resets: {resets}", "yellow")
                    # Ensure terminal is reset after a manual reset
                    termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)
                    continue

                total_timesteps += len(obs_list)

                # Save trajectory data
                log("Saving trajectory data...", "blue")
                agent_data, obs_dict = collect_trajectory_data(
                    obs_list, actions_list, success, notes
                )
                saver.save_traj(episode - 1, agent_data=agent_data, obs_dict=obs_dict)
                log("✓ Trajectory saved", "green")

                if success:
                    successes += 1
                    log("✓ Episode marked as successful", "green")
                else:
                    log("✗ Episode marked as unsuccessful", "red")

                if notes:
                    log(f"Notes: {notes}", "blue")

                log(
                    f"""
                        Episode {episode} Summary:
                        • Success: {"Yes" if success else "No"}
                        • Duration: {time.time() - episode_start:.1f}s
                        • Success Rate: {successes}/{episode} ({(successes / episode) * 100:.1f}%)
                    """,
                    "cyan",
                )

            except Exception as e:
                log(f"Episode error: {e}", "red")
                # Ensure terminal is reset even if episode fails
                termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)
                raise

    except KeyboardInterrupt:
        log("\n⚠️ Experiment interrupted by user", "yellow")
    except Exception as e:
        log(f"❌ Error occurred: {str(e)}", "red")
        raise
    finally:
        # Ensure terminal is reset at program exit
        termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)
        # Save experiment metadata
        metadata = {
            "total_episodes": episode,
            "successful_episodes": successes,
            "manual_resets": resets,
            "success_rate": (successes / episode) * 100 if episode > 0 else 0,
            "total_steps": total_timesteps,
            "total_time": time.time() - start_time,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        }

        with open(save_dir / "metadata.pkl", "wb") as f:
            pkl.dump(metadata, f)

        log("✓ Experiment metadata saved", "green")

        log("Stopping robot...", "blue")
        widowx_client.stop()
        log("✓ Robot stopped successfully", "green")

        total_time = time.time() - start_time
        log(
            f"""
                Final Experiment Summary:
                • Total Episodes: {episode}
                • Successful Episodes: {successes}
                • Manual Resets: {resets}
                • Success Rate: {(successes / episode) * 100:.1f}%
                • Total Steps: {total_timesteps}
                • Total Time: {total_time:.1f}s
                • Average FPS: {total_timesteps / total_time:.2f}
            """,
            "green",
        )


if __name__ == "__main__":
    main()
