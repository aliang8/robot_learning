from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from clam.trainers import trainer_to_cls
from clam.utils.logger import log


@hydra.main(version_base=None, config_name="robot_inference", config_path="../cfg")
def main(cfg: DictConfig) -> None:
    # Load model and config
    cfg_file = Path(cfg.ckpt_file) / "config.yaml"
    log(f"Loading config from {cfg_file}", "blue")

    model_cfg = OmegaConf.load(cfg_file)
    model_cfg.load_from_ckpt = True
    model_cfg.ckpt_step = None
    model_cfg.ckpt_file = cfg.ckpt_file

    # Load model from checkpoint
    trainer = trainer_to_cls[model_cfg.name](model_cfg)
    trainer.model.eval()

    agent = trainer.model
    log("Model loaded successfully", "green")


if __name__ == "__main__":
    main()
