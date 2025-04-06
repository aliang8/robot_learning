from pathlib import Path

import einops
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from robot_learning.models.image_embedder import MultiInputEmbedder
from robot_learning.models.lora import apply_lora
from robot_learning.models.policy import POLICY_CLS_MAP
from robot_learning.trainers.offline_trainer import OfflineTrainer
from robot_learning.utils.logger import log


def gaussian_nll_loss(
    actions: torch.Tensor,
    means: torch.Tensor,
    logvars: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the negative log-likelihood loss using PyTorch's Normal distribution.

    Args:
        actions: Target actions tensor of shape (batch_size, action_dim)
        means: Predicted mean tensor of shape (batch_size, action_dim)
        logvars: Predicted log-variance tensor of shape (batch_size, action_dim)

    Returns:
        nll: Negative log-likelihood loss tensor of shape (batch_size, action_dim)
    """
    # Input validation
    if not (actions.shape == means.shape == logvars.shape):
        raise ValueError(
            f"Shape mismatch: actions {actions.shape}, "
            f"means {means.shape}, logvars {logvars.shape}"
        )

    # Clamp logvars for numerical stability
    logvars = torch.clamp(logvars, min=-10, max=2)
    # Convert logvar to std dev
    stds = torch.exp(0.5 * logvars)
    # Clamp stds for numerical stability
    stds = torch.clamp(stds, min=1e-6)

    # Create normal distribution
    dist = torch.distributions.Normal(means, stds)

    # Compute log probability
    log_prob = dist.log_prob(actions)

    # Sum over action dimensions
    nll = -torch.sum(log_prob, dim=-1)
    return nll


class BCTrainer(OfflineTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Get loss configuration
        self.use_separate_gripper = getattr(self.cfg, "use_separate_gripper", False)
        self.gripper_dim = getattr(self.cfg, "gripper_dim", -1)  # default to last dim

        if self.use_separate_gripper:
            self.gripper_pos_weight = getattr(self.cfg, "gripper_pos_weight", 1.0)
            self.gripper_pos_weight = torch.tensor(self.gripper_pos_weight)
            self.gripper_loss_fn = nn.BCEWithLogitsLoss(
                reduction="none", pos_weight=self.gripper_pos_weight
            )

        #  TODO: create loss functions depending on configuration
        # if self.model.is_gaussian:
        #     self.loss_fn = gaussian_nll_loss
        # else:
        #     # self.loss_fn = nn.MSELoss(reduction="none")
        self.loss_fn = nn.L1Loss(reduction="none")

    def setup_model(self, action_dim: int = None, action_activation: str = "Tanh"):
        # Input action dim in case we are doing CLAM policy training

        state_dim = self.cfg.env.state_dim
        if self.cfg.model.use_only_gripper_state:
            state_dim = 4

        embedder = MultiInputEmbedder(
            cfg=self.cfg.model,
            input_modalities=self.cfg.model.input_modalities,
            state_dim=state_dim,
            seq_len=self.cfg.data.seq_len if self.cfg.model.name == "mlp" else 1,
            image_shape=(3, *self.cfg.env.image_shape),
        )

        if self.cfg.model.name not in POLICY_CLS_MAP:
            raise ValueError(f"Unknown model {self.cfg.model.name}")

        policy_cls = POLICY_CLS_MAP[self.cfg.model.name]
        model = policy_cls(
            cfg=self.cfg.model,
            embedder=embedder,
            output_dim=self.cfg.env.action_dim if action_dim is None else action_dim,
        )

        if self.cfg.load_from_ckpt:
            log("Loaded backbone from checkpoint", "green")
            cfg, ckpt = model.load_from_ckpt(
                self.cfg.ckpt_file, ckpt_step=self.cfg.ckpt_step
            )

            # Maybe apply lora here
            # TODO: finish this
            apply_lora(
                self.model.model,  # apply to just the transformer
                lora_r=self.cfg.lora.r,
                lora_alpha=self.cfg.lora.alpha,
            )

        return model

    def compute_loss(self, batch, train: bool = True):
        metrics = {}

        states = batch.states
        if (
            "states" in self.cfg.model.input_modalities
            and self.cfg.model.use_only_gripper_state
        ):
            if self.cfg.env.env_name == "metaworld":
                if self.cfg.data.seq_len == 1:
                    states = states[:, :4]
                else:
                    states = states[:, :, :4]
            elif self.cfg.env.env_name == "robot":
                # combine first three and last dim
                if self.cfg.data.seq_len == 1:
                    states = torch.cat([states[:, :3], states[:, -1:]], dim=-1)
                else:
                    states = torch.cat([states[:, :, :3], states[:, :, -1:]], dim=-1)

        model_inputs = {k: getattr(batch, k) for k in self.cfg.model.input_modalities}
        model_inputs["states"] = states
        model_inputs["timesteps"] = batch.timestep

        # TODO:
        # action_preds = self.model(model_inputs)

        model_inputs["actions"] = batch.actions
        model_inputs["image_embeddings"] = batch.image_embeddings

        metrics = self.model(model_inputs)

        if self.use_separate_gripper:
            # Split predictions and targets into arm and gripper components
            if self.model.is_gaussian:
                means = action_preds.mean
                logvars = action_preds.logvar

                arm_means_pred = means[..., :-1]
                arm_logvars_pred = logvars[..., :-1]
                # this is the ground truth actions
                arm_actions = batch.actions[..., :-1]

                # Compute arm loss using NLL
                # compute sum over timesteps of chunk
                arm_loss = self.loss_fn(arm_actions, arm_means_pred, arm_logvars_pred)
                arm_loss = arm_loss.sum(dim=1)  # sum over T
                arm_loss = arm_loss.mean()  # mean over batch

                # Compute gripper loss using BCE
                gripper_preds = means[..., -1:]
                gripper_targets = batch.actions[..., -1:]

                gripper_loss = self.gripper_loss_fn(gripper_preds, gripper_targets)
                gripper_loss = gripper_loss.sum(dim=1)  # sum over T
                gripper_loss = gripper_loss.mean()  # mean over batch

            else:
                arm_preds = action_preds.actions[..., :-1]
                arm_targets = batch.actions[..., :-1]

                # Compute losses
                arm_loss = (self.loss_fn(arm_preds, arm_targets)).mean()

                gripper_preds = action_preds.actions[..., -1:]
                gripper_targets = batch.actions[..., -1:]

                gripper_loss = self.gripper_loss_fn(
                    gripper_preds, gripper_targets
                ).mean()

            # Combine losses
            loss = (
                self.cfg.model.arm_loss_weight * arm_loss
                + self.cfg.model.gripper_loss_weight * gripper_loss
            )

            # Log separate losses
            metrics["arm_loss"] = arm_loss.item()
            metrics["gripper_loss"] = gripper_loss.item()

            # Add binary accuracy for gripper predictions
            with torch.no_grad():
                if self.model.is_gaussian:
                    gripper_preds = torch.sigmoid(gripper_preds) > 0.5
                else:
                    gripper_preds = torch.sigmoid(gripper_preds) > 0.5
                gripper_acc = (gripper_preds == gripper_targets).float().mean()
                metrics["gripper_accuracy"] = gripper_acc.item()

        else:
            # TODO: i handle the loss calculation in the model, maybe change this to match the other models
            # Use single loss function for all dimensions
            # if self.model.is_gaussian:
            #     loss = self.loss_fn(
            #         batch.actions,
            #         action_preds.mean,
            #         action_preds.logvar,
            #         reduction="mean",
            #     )
            # else:
            #     loss = self.loss_fn(action_preds.actions, batch.actions)
            #     loss = loss.mean()
            # import ipdb; ipdb.set_trace()
            loss = metrics["loss"]

            metrics["loss"] = loss.item()

            pass


        return metrics, loss
