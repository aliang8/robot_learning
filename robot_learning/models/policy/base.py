from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from robot_learning.models.base import BaseModel
from robot_learning.utils.logger import log


@dataclass
class ActionOutput:
    """Container for policy outputs."""

    actions: torch.Tensor
    mean: torch.Tensor | None = None
    logvar: torch.Tensor | None = None
    std: torch.Tensor | None = None

    @property
    def is_gaussian(self) -> bool:
        return self.mean is not None

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ActionHead(nn.Module):
    """Modular action head that handles both deterministic and Gaussian outputs."""

    def __init__(self, cfg, input_dim: int, output_dim: int):
        super().__init__()
        self.cfg = cfg
        self.gaussian_output = cfg.gaussian_policy
        self.use_separate_gripper = getattr(cfg, "use_separate_gripper", False)

        if self.use_separate_gripper:
            # Separate output layers for arm and gripper
            self.arm_dim = output_dim - 1
            self.gripper_dim = 1

            # Arm output layer (Gaussian needs 2x outputs for mean/logvar)
            arm_output_size = self.arm_dim * 2 if self.gaussian_output else self.arm_dim
            self.arm_output = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, arm_output_size),
            )

            # Gripper always uses deterministic output
            self.gripper_output = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, self.gripper_dim),
            )
        else:
            # Original combined output layer
            output_size = output_dim * 2 if self.gaussian_output else output_dim
            self.output_layer = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, output_size),
            )

        if hasattr(cfg, "action_activation") and cfg.action_activation:
            self.action_activation = getattr(nn, cfg.action_activation)()
            log(f"Using action activation: {cfg.action_activation}", "red")
        else:
            self.action_activation = None

    def forward(self, x: torch.Tensor) -> ActionOutput:
        """
        Args:
            x: Input features [B, input_dim]
        Returns:
            ActionOutput containing actions and distribution parameters if gaussian
        """
        if self.use_separate_gripper:
            # Process arm and gripper separately
            if self.gaussian_output:
                arm_output = self.arm_output(x)
                mean, logvar = torch.chunk(arm_output, 2, dim=-1)
                mean = self.action_activation(mean)
                logvar = torch.clamp(logvar, min=-20, max=2)

                # Gripper output (deterministic)
                gripper = self.gripper_output(x)

                # Combine outputs
                full_mean = torch.cat([mean, gripper], dim=-1)
                full_logvar = torch.cat([logvar, torch.zeros_like(gripper)], dim=-1)

                return ActionOutput(
                    actions=full_mean,  # By default, return mean as action
                    mean=full_mean,
                    logvar=full_logvar,
                    std=torch.exp(0.5 * full_logvar),
                )
            else:
                # Deterministic outputs for both
                arm = self.action_activation(self.arm_output(x))
                gripper = self.gripper_output(x)
                output = torch.cat([arm, gripper], dim=-1)
                return ActionOutput(actions=output)
        else:
            # Original behavior remains unchanged
            if self.gaussian_output:
                output = self.output_layer(x)
                mean, logvar = torch.chunk(output, 2, dim=-1)
                mean = self.action_activation(mean)
                logvar = torch.clamp(logvar, min=-20, max=2)

                return ActionOutput(
                    actions=mean,  # By default, return mean as action
                    mean=mean,
                    logvar=logvar,
                    std=torch.exp(0.5 * logvar),
                )

            output = self.output_layer(x)
            output = self._apply_activation(output)
            return ActionOutput(actions=output)


class BasePolicy(BaseModel):
    """Base class for all policies."""

    def __init__(
        self,
        cfg: DictConfig,
        input_dim: int = None,
        output_dim: int = None,
    ):
        super().__init__(cfg, input_dim)
        self.action_head = ActionHead(
            cfg=cfg,
            input_dim=input_dim,
            output_dim=output_dim,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the policy network.
        Args:
            inputs: Dictionary containing input tensors
        Returns:
            features: Extracted features before action head
        """
        raise NotImplementedError

    def select_action(
        self, inputs: Dict[str, torch.Tensor], sample: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Select action based on input.
        Args:
            inputs: Dictionary containing input tensors
            sample: If True and using gaussian policy, sample from distribution
        Returns:
            If sample=False:
                actions tensor
            If sample=True:
                actions tensor, info dictionary
        """
        action_out = self.forward(inputs)

        if action_out.is_gaussian and sample:
            eps = torch.randn_like(action_out.std)
            sampled_actions = action_out.mean + eps * action_out.std
            return sampled_actions, action_out.to_dict()

        actions = action_out.actions
        if self.cfg.use_separate_gripper:
            # batched or no batched
            if actions.ndim == 2:
                arm_actions = actions[:, :-1]
                gripper_actions = actions[:, -1].unsqueeze(-1)
            else:
                arm_actions = actions[:, :, :-1]
                gripper_actions = actions[:, :, -1].unsqueeze(-1)

            # apply sigmoid to gripper actions
            gripper_actions = torch.sigmoid(gripper_actions)
            gripper_actions = torch.where(gripper_actions > 0.5, 1.0, gripper_actions)
            gripper_actions = torch.where(gripper_actions < 0.5, 0.0, gripper_actions)

            actions = torch.cat([arm_actions, gripper_actions], dim=-1)

        return actions

    @property
    def is_gaussian(self) -> bool:
        """Whether the policy outputs a Gaussian distribution."""
        return (
            self.action_head.gaussian_output if self.action_head is not None else False
        )
