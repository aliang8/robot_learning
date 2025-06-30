from pathlib import Path

import hydra

from robot_learning.data.utils import create_dataset_name, raw_data_to_tfds
from robot_learning.utils.logger import log


@hydra.main(version_base=None, config_name="convert_to_tfds", config_path="../../cfg")
def main(cfg):
    """Main function to convert replay buffer to TFDS format."""
    # Generate dataset name
    dataset_name = create_dataset_name(cfg)

    # Create save directory
    save_dir = Path(cfg.tfds_data_dir) / cfg.env_name
    save_file = save_dir / dataset_name
    save_file.mkdir(parents=True, exist_ok=True)

    log(
        f"------------------- Saving dataset to {save_file} -------------------", "blue"
    )
    data_dir = Path(cfg.data_dir)
    processed_traj_dirs = list((Path(data_dir) / "processed_trajs").glob("traj_*"))
    # processed_traj_dirs = list((Path(data_dir) / "subtraj_data").glob("traj_*"))
    raw_data_to_tfds(
        processed_traj_dirs,
        save_file=save_file,
        embedding_model=cfg.embedding_model,
        resnet_feature_map_layer=cfg.resnet_feature_map_layer,
        flow_suffix=cfg.flow_suffix,
    )


if __name__ == "__main__":
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    main()
