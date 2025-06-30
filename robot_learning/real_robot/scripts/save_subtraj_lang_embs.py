import json
from pathlib import Path

import clip
import hydra
import torch

from robot_learning.data.utils import (
    create_dataset_name,
    load_data_compressed,
    raw_data_to_tfds,
    save_data_compressed,
)
from robot_learning.utils.logger import log

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

@hydra.main(version_base=None, config_name="convert_to_tfds", config_path="../../cfg")
def main(cfg):
    """Main function to convert replay buffer to TFDS format."""
    # Generate dataset name
    data_dir = Path(cfg.data_dir)
    processed_traj_dirs = list((Path(data_dir) / "subtraj_data").glob("subtraj_*"))

    lang_file = Path(data_dir) / "lang_ann.json"
    
    # json file that maps subtraj names to language annotations
    with open(lang_file, "r") as f:
        lang_ann = json.load(f)

    for traj_dir in processed_traj_dirs:
        traj_name = traj_dir.name
        traj_len = load_data_compressed(traj_dir / "external_images.dat").shape[0]
        if traj_name in lang_ann:
            lang = lang_ann[traj_name]
        else:
            assert False, f"Language annotation for {traj_name} not found in {lang_file}"

        lang = clip.tokenize([lang]).to(device)
        with torch.no_grad():
            lang_emb = model.encode_text(lang)

        lang_emb = lang_emb.repeat(traj_len, 1)
        lang_emb = lang_emb.cpu().numpy()

        lang_emb_file = traj_dir / "lang_embedding.dat"
        # if not lang_emb_file.exists():
        lang_emb_file.parent.mkdir(parents=True, exist_ok=True)
        log(f"Saving language embedding for {traj_name} with traj_len: {traj_len} to {lang_emb_file}", "green")
        save_data_compressed(lang_emb_file, lang_emb)
        
    # Generate dataset name
    dataset_name = create_dataset_name(cfg)

    # Create save directory
    save_dir = Path(cfg.tfds_data_dir) / cfg.env_name
    save_file = save_dir / dataset_name
    save_file.mkdir(parents=True, exist_ok=True)

    raw_data_to_tfds(
        processed_traj_dirs,
        save_file=save_file,
        embedding_model=cfg.embedding_model,
        resnet_feature_map_layer=cfg.resnet_feature_map_layer,
        flow_suffix=cfg.flow_suffix,
        save_lang_embeds=True,  # Save language embeddings
    )


if __name__ == "__main__":
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    main()
