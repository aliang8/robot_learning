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

```
# train CNN policy from scratch
python3 main.py --config-name=train_bc \
    env=robot \
    data.dataset_name=robot_play_imgs_emb-r3m \
    data.datasets=[pick_up_red_object] \
    data.num_trajs=-1 \
    run_eval_rollouts=False \
    num_updates=5000

# space-time attention CLAM
python3 main.py --config-name=train_st_clam \
    env=robot \
    data.dataset_name=robot_play_imgs_emb-r3m \
    data.datasets=[pick_up_green_object_expert] \
    joint_action_decoder_training=True \
    data.action_labeled_ds=[pick_up_green_object_expert] \
    clam_model.input_modalities=[external_imgs,over_shoulder_imgs]
```


# train AC decoder policy
```
python3 main.py --config-name=train_bc_ac \
    env=robot \
    data.dataset_name=robot_play_imgs_emb-dinov2_vitb14 \
    data.datasets=[pick_up_green_block_lighting] \
    data.num_trajs=-1 \
    run_eval_rollouts=False \
    model.input_modalities=[states,external_img_embeds,over_shoulder_img_embeds] \
    model.gaussian_policy=True \
    model.embedding_model=dinov2_vitb14 \
    model.gripper_loss_weight=5 \
    data.seq_len=20 \
    wandb.project=clam-robot \
    model.use_only_gripper_state=True

# mse loss
python3 main.py --config-name=train_bc_ac \
    env=robot \
    data.dataset_name=robot_play_imgs_emb-dinov2_vitb14 \
    data.datasets=[pick_up_green_object_expert] \
    data.num_trajs=-1 \
    run_eval_rollouts=False \
    model.input_modalities=[external_img_embeds,over_shoulder_img_embeds] \
    model.gaussian_policy=False \
    model.embedding_model=dinov2_vitb14 \
    model.gripper_loss_weight=5 \
    data.seq_len=20 \
    wandb.project=clam-robot

# gaussian loss
python3 main.py --config-name=train_bc_ac \
    env=robot \
    data.dataset_name=robot_play_imgs_emb-dinov2_vitb14 \
    data.datasets=[pick_up_green_object_expert] \
    data.num_trajs=-1 \
    run_eval_rollouts=False \
    model.input_modalities=[states,external_img_embeds,over_shoulder_img_embeds] \
    model.gaussian_policy=True \
    model.embedding_model=dinov2_vitb14 \
    model.gripper_loss_weight=5 \
    data.seq_len=20 \
    wandb.project=clam-robot \
    model.use_only_gripper_state=True

# mlp policy
python3 main.py --config-name=train_bc \
    env=robot \
    data.dataset_name=robot_play_imgs_emb-dinov2_vitb14 \
    data.datasets=[pick_up_green_object_expert] \
    data.num_trajs=-1 \
    run_eval_rollouts=False \
    model.input_modalities=[external_img_embeds,over_shoulder_img_embeds] \
    model.gaussian_policy=False \
    model.embedding_model=dinov2_vitb14 \
    wandb.project=clam-robot
```

## Deploying the robot

```
# test loading the agent
python3 scripts/test_model_loading.py \
    ckpt_file=/path/to/ckpt

python3 scripts/robot_inference.py \
    ckpt_file=/path/to/ckpt
```
