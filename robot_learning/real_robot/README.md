# Real Robot Experiment w/ WidowX

## Install some Bridge related dependencies 

```
# Install edgeml
git clone https://github.com/youliangtan/edgeml
cd edgeml
pip install -e .
```


## Convert Robot Demonstrations to TFDS

# train BC
```
python3 -m robot_learning.real_robot.scripts.convert_robot_to_tfds \
    dataset_name=robot_play \
    task_name=pick_up_green_object_expert \
    precompute_embeddings=True \
    save_imgs=True \
    embedding_model=dinov2_vitb14
```

## Run policy training

# train action chunking transformer decoder policy
```
python3 -m robot_learning.main --config-name=train_bc_ac \
    env=robot \
    data.dataset_name=robot_play_imgs_emb-dinov2_vitb14 \
    data.datasets=[pick_up_green_block_lighting2] \
    data.num_trajs=-1 \
    run_eval_rollouts=False \
    model.input_modalities=[external_img_embeds,over_shoulder_img_embeds] \
    model.gaussian_policy=True \
    model.embedding_model=dinov2_vitb14 \
    model.gripper_loss_weight=5 \
    data.seq_len=5 \
    wandb.project=clam-robot \
    model.use_only_gripper_state=False
```

## Deploying the robot

```
# test loading the agent
python3 scripts/test_model_loading.py \
    ckpt_file=/path/to/ckpt

python3 scripts/robot_inference.py \
    ckpt_file=/path/to/ckpt
```
