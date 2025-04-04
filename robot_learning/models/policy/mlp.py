from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from robot_learning.models.policy.base import BasePolicy
from robot_learning.models.utils.transformer_utils import make_mlp


class MLPPolicy(BasePolicy):
    def __init__(
        self,
        cfg: DictConfig,
        embedder: nn.Module,
        output_dim: int = None,
    ):
        super().__init__(
            cfg,
            input_dim=cfg.net.hidden_dims[-1],
            output_dim=output_dim,
        )
        self.name = "MLPPolicy"
        self.embedder = embedder

        # Create feature extractor MLP
        self.backbone, _ = make_mlp(
            input_dim=self.embedder.output_dim,
            net_kwargs=cfg.net,
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the policy network.

        Args:
            inputs: Dictionary containing input tensors
        Returns:
            If gaussian_output:
                tuple(mean, logvar) each of shape [B, output_dim]
            else:
                action of shape [B, output_dim]
        """
        # Get embeddings for images and states
        embeddings = self.embedder(inputs)

        # Extract features
        features = self.backbone(embeddings)

        # Get action outputs
        return self.action_head(features)
