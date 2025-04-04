"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.

Taken from: https://github.com/KhaledSharif/robot-transformers/blob/main/lerobot/common/policies/act/modeling_act.py
"""

import math
from collections import deque
from itertools import chain
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import DictConfig
from torch import Tensor, nn

from robot_learning.models.image_embedder import EMBEDDING_DIMS
from robot_learning.models.transformer import TransformerDecoder, TransformerEncoder
from robot_learning.models.utils.transformer_utils import (
    SinusoidalPositionEmbedding2d,
    create_sinusoidal_pos_embedding,
)


class ACTPolicy(nn.Module, PyTorchModelHubMixin):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    def __init__(
        self,
        cfg: DictConfig,
        image_embedder,
        state_input_dim: int,
        act_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.act_dim = act_dim
        self.model = ACT(
            cfg=cfg.net,
            image_embedder=image_embedder,
            state_input_dim=state_input_dim,
            act_dim=act_dim,
        )

    def reset(self, num_envs: int):
        """This should be called whenever the environment is reset."""
        # if self.cfg.n_action_steps is not None:
        #     self._action_queue = deque([], maxlen=self.cfg.n_action_steps)
        max_timesteps = 200
        num_queries = self.cfg.n_action_steps
        self.t = 0
        self.all_time_actions = torch.zeros(
            [num_envs, max_timesteps, max_timesteps + num_queries, self.act_dim]
        ).cuda()

    @torch.no_grad
    def select_action(
        self,
        states: Tensor,
        images: Tensor = None,
        image_embeddings: Tensor = None,
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        self.eval()

        # if len(self._action_queue) == 0:
        # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
        # effectively has shape (batch_size, n_action_steps, *), hence the transpose.
        actions = self.model(
            states=states, images=images, image_embeddings=image_embeddings
        )[0][:, : self.cfg.n_action_steps]

        if self.cfg.temporal_agg:
            # apply this over a batch of actions
            self.all_time_actions[
                :, self.t, self.t : self.t + self.cfg.n_action_steps
            ] = actions
            actions_for_curr_step = self.all_time_actions[:, :, self.t]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=2)
            # apply batch select to actions_for_curr_step
            B = actions_for_curr_step.shape[0]
            actions_for_curr_step = actions_for_curr_step[actions_populated].reshape(
                B, -1, self.all_time_actions.shape[-1]
            )

            # TODO: verify this is correct
            if len(actions_for_curr_step.shape) == 2:
                actions_for_curr_step = actions_for_curr_step.unsqueeze(dim=1)

            k = 0.01
            exp_weights = np.exp(-k * np.arange(actions_for_curr_step.shape[1]))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
            raw_action = (actions_for_curr_step * exp_weights).sum(dim=1, keepdim=True)
        else:
            raw_action = actions

        self.t += 1
        # self._action_queue.append(actions)
        # return self._action_queue.popleft()
        return raw_action

    def forward(
        self,
        states: Tensor,
        images: Tensor = None,
        actions: Tensor = None,
        image_embeddings: Tensor = None,
        # action_is_pad: Tensor,
    ) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(
            states=states,
            images=images,
            actions=actions,
            image_embeddings=image_embeddings,
        )

        l1_loss = (
            F.l1_loss(actions, actions_hat, reduction="none")
            # * ~action_is_pad.unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.cfg.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://arxiv.org/abs/1312.6114 for more details).
            mean_kld = (
                (
                    -0.5
                    * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())
                )
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss_dict["loss"] = l1_loss + mean_kld * self.cfg.kl_weight
        else:
            loss_dict["loss"] = l1_loss

        return loss_dict


class ACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └───▲───┘             │
                  │       │     │     │                 │
                inputs    └─────┼─────┘                 │
                                │                       │
                                └───────────────────────┘
    """

    def __init__(
        self,
        cfg: DictConfig,
        image_embedder,
        state_input_dim: int,
        act_dim: int,
    ):
        super().__init__()
        self.cfg = cfg

        # BERT style VAE encoder with input [cls, *joint_space_configuration, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        self.latent_dim = self.cfg.latent_dim
        if self.cfg.use_vae:
            self.vae_encoder = TransformerEncoder(self.cfg.vae_encoder)
            self.vae_encoder_cls_embed = nn.Embedding(1, self.cfg.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            self.vae_encoder_robot_state_input_proj = nn.Linear(
                state_input_dim, self.cfg.dim_model
            )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(act_dim, self.cfg.dim_model)
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(
                self.cfg.dim_model, self.latent_dim * 2
            )
            # Fixed sinusoidal positional embedding the whole input to the VAE encoder. Unsqueeze for batch
            # dimension.
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(
                    1 + 1 + self.cfg.chunk_size, self.cfg.dim_model
                ).unsqueeze(0),
            )

        if self.cfg.use_precomputed_img_embeds:
            self.backbone = image_embedder
        else:
            self.backbone = nn.Identity()

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = TransformerEncoder(self.cfg.encoder)
        self.decoder = TransformerDecoder(self.cfg.decoder)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, robot_state, image_feature_map_pixels].
        self.encoder_robot_state_input_proj = nn.Linear(
            state_input_dim, self.cfg.dim_model
        )

        if self.cfg.use_vae:
            self.encoder_latent_input_proj = nn.Linear(
                self.latent_dim, self.cfg.dim_model
            )

        if self.cfg.image_obs:
            # TODO: this is only for resnet50 at the moment
            if (
                self.cfg.embedding_model == "resnet50"
                and self.cfg.resnet_feature_map_layer == "layer4"
            ):
                self.encoder_img_feat_input_proj = nn.Conv2d(
                    EMBEDDING_DIMS[self.cfg.embedding_model],
                    self.cfg.dim_model,
                    kernel_size=1,
                )
            else:
                # just use a linear projection from the pretrained embedding of the image
                self.encoder_img_feat_input_proj = nn.Linear(
                    EMBEDDING_DIMS[self.cfg.embedding_model], self.cfg.dim_model
                )
        else:
            self.encoder_img_feat_input_proj = None

        # Transformer encoder positional embeddings.
        self.encoder_robot_and_latent_pos_embed = nn.Embedding(2, self.cfg.dim_model)
        if self.cfg.image_obs:
            if (
                self.cfg.embedding_model == "resnet50"
                and self.cfg.resnet_feature_map_layer == "layer4"
            ):
                self.encoder_cam_feat_pos_embed = SinusoidalPositionEmbedding2d(
                    self.cfg.dim_model // 2
                )
            elif self.cfg.embedding_model == "r3m":
                self.encoder_cam_feat_pos_embed = nn.Embedding(1, self.cfg.dim_model)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(self.cfg.chunk_size, self.cfg.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(self.cfg.dim_model, act_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        states: Tensor,
        images: Tensor = None,
        actions: Tensor = None,
        image_embeddings: Tensor = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        states: [B, T, D]
        images: [B, T, C, H, W]
        actions: [B, T, D] not necessary during inference time

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.cfg.use_vae and self.training:
            assert actions is not None, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = states.shape[0]

        states = states[:, 0]

        if self.cfg.image_obs:
            images = images[:, 0]
        else:
            images = None

        # Prepare the latent for input to the transformer encoder.
        if self.cfg.use_vae and actions is not None:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            robot_state_embed = self.vae_encoder_robot_state_input_proj(
                states
            ).unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(actions)  # (B, S, D)
            vae_encoder_input = torch.cat(
                [cls_embed, robot_state_embed, action_embed], axis=1
            )  # (B, S+2, D)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(vae_encoder_input, pos_embed=pos_embed)[
                :, 0
            ]  # select the class token, with shape (B, D)

            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(
                [batch_size, self.latent_dim], dtype=torch.float32
            ).to(states.device)

        # Prepare all other transformer encoder inputs.
        # Camera observation features and positional embeddings.
        # all_cam_features = []
        # all_cam_pos_embeds = []

        if self.cfg.image_obs:
            if not self.cfg.use_precomputed_img_embeds:
                # do not compute gradients for the image backbone
                with torch.no_grad():
                    cam_features = self.backbone(images)
            else:
                cam_features = image_embeddings[:, 0]

            if self.cfg.resnet_feature_map_layer == "avgpool":
                # TODO debug this
                import ipdb

                ipdb.set_trace()
                cam_features = cam_features.unsqueeze(-1).unsqueeze(-1)

            # we only need this pos embed if we are using resnet feature maps
            if (
                self.cfg.embedding_model == "resnet50"
                and self.cfg.resnet_feature_map_layer == "layer4"
            ):
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(
                    dtype=cam_features.dtype
                )

            cam_features = self.encoder_img_feat_input_proj(
                cam_features
            )  # (B, C, h, w)
            # all_cam_features.append(cam_features)
            # all_cam_pos_embeds.append(cam_pos_embed)
            # # Concatenate camera observation feature maps and positional embeddings along the width dimension.
            # encoder_in = torch.cat(all_cam_features, axis=3)
            # cam_pos_embed = torch.cat(all_cam_pos_embeds, axis=3)
            encoder_in = cam_features
        else:
            encoder_in = None

        # Get positional embeddings for robot state and latent.
        robot_state_embed = self.encoder_robot_state_input_proj(states)

        if self.cfg.use_vae:
            latent_embed = self.encoder_latent_input_proj(latent_sample)
        else:
            latent_embed = None

        # Stack encoder input and positional embeddings moving to (B, S, C).
        if self.cfg.image_obs:
            if self.cfg.embedding_model == "r3m":
                encoder_in = torch.cat(
                    [
                        torch.stack([latent_embed, robot_state_embed], axis=1),
                        encoder_in.unsqueeze(1),
                    ],
                    axis=1,
                )
                pos_embed = torch.cat(
                    [
                        self.encoder_robot_and_latent_pos_embed.weight.unsqueeze(0),
                        self.encoder_cam_feat_pos_embed.weight.unsqueeze(0),
                    ],
                    axis=1,
                )
            else:
                encoder_in = torch.cat(
                    [
                        torch.stack([latent_embed, robot_state_embed], axis=1),
                        encoder_in.flatten(2).permute(0, 2, 1),
                    ],
                    axis=1,
                )
                pos_embed = torch.cat(
                    [
                        self.encoder_robot_and_latent_pos_embed.weight.unsqueeze(0),
                        cam_pos_embed.flatten(2).permute(0, 2, 1),
                    ],
                    axis=1,
                )

        else:
            # (B, S+1, D)
            if latent_embed is not None:
                encoder_in = torch.stack([latent_embed, robot_state_embed], axis=1)
                # (1, S+1, D)
                pos_embed = self.encoder_robot_and_latent_pos_embed.weight.unsqueeze(0)

            else:
                # add a time dimension for input to encoder
                encoder_in = robot_state_embed.unsqueeze(1)
                pos_embed = self.encoder_robot_and_latent_pos_embed.weight[
                    0:1
                ].unsqueeze(0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in, pos_embed=pos_embed)
        decoder_in = torch.zeros(
            (batch_size, self.cfg.chunk_size, self.cfg.dim_model),
            dtype=pos_embed.dtype,
            device=pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(0),
            causal=False,  # TODO: is this right
        )

        actions = self.action_head(decoder_out)

        # apply tanh activation
        # TODO: maybe fix this
        actions = torch.tanh(actions)
        return actions, (mu, log_sigma_x2)
