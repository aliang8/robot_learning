from typing import Dict

import einops
import torch
import torch.nn as nn
from omegaconf import DictConfig

from robot_learning.models.policy.base import BasePolicy
from robot_learning.models.utils.transformer_utils import get_pos_encoding


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://arxiv.org/abs/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[:i+1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(
            -temporal_ensemble_coeff * torch.arange(chunk_size)
        )
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(
            device=actions.device
        )
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1),
                dtype=torch.long,
                device=self.ensembled_actions.device,
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[
                self.ensembled_actions_count - 1
            ]
            self.ensembled_actions += (
                actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            )
            self.ensembled_actions /= self.ensemble_weights_cumsum[
                self.ensembled_actions_count
            ]
            self.ensembled_actions_count = torch.clamp(
                self.ensembled_actions_count + 1, max=self.chunk_size
            )
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat(
                [self.ensembled_actions, actions[:, -1:]], dim=1
            )
            self.ensembled_actions_count = torch.cat(
                [
                    self.ensembled_actions_count,
                    torch.ones_like(self.ensembled_actions_count[-1:]),
                ]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ActionChunkingTransformerPolicy(BasePolicy):
    """
    Transformer decoder policy that predicts multiple timesteps of actions at once.
    """

    def __init__(self, cfg: DictConfig, embedder: nn.Module, output_dim: int):
        super().__init__(cfg, input_dim=cfg.d_model, output_dim=output_dim)
        self.name = "ActionChunkingTransformerPolicy"

        self.embedder = embedder

        # Policy is a transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=cfg.d_model, nhead=cfg.nhead, dropout=0.1, batch_first=True
            ),
            num_layers=cfg.num_layers,
            norm=nn.LayerNorm(cfg.d_model),
        )
        self.positional_encoding = get_pos_encoding(
            cfg.pos_enc, embedding_dim=cfg.d_model, max_len=200
        )
        self.decoder_pos_encoding = get_pos_encoding(
            cfg.pos_enc, embedding_dim=cfg.d_model, max_len=200
        )
        self.action_embed = nn.Linear(cfg.action_dim, cfg.d_model)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the policy network.

        Args:
            inputs: Dictionary mapping from input names to tensors.
            inputs should have [B, T, *] for each modality
        """
        # we just need the first timestep for each modality
        embed_inputs = {k: v[:, 0] for k, v in inputs.items()}

        # Get embeddings for each input and then predict a sequence of actions
        embeddings = self.embedder(embed_inputs)
        embeddings = einops.repeat(embeddings, "B E -> B T E", T=self.cfg.seq_len)

        # Pass through the transformer decoder repeated for each timestep
        # pos_encoding = self.positional_encoding(inputs["timesteps"])
        pos_encoding = self.positional_encoding.weight[: self.cfg.seq_len].unsqueeze(0)

        # [B, T, E]
        memory = embeddings + pos_encoding

        # create dummy actions to use as query for the transformer decoder
        B = embeddings.shape[0]

        # [B, T, E]
        action_embeddings = torch.zeros(
            B, self.cfg.seq_len, self.cfg.d_model, device=embeddings.device
        )
        action_pos_encoding = self.decoder_pos_encoding.weight[
            : self.cfg.seq_len
        ].unsqueeze(0)

        action_embeddings = action_embeddings + action_pos_encoding

        # TODO: should we apply causal mask?
        tgt_mask = torch.triu(
            torch.ones(self.cfg.seq_len, self.cfg.seq_len), diagonal=1
        ).to(embeddings.device)

        # [B, T, E] -> [B, T, E]
        output = self.transformer_decoder(
            tgt=action_embeddings,
            memory=memory,
            # tgt_mask=tgt_mask,
            # tgt_is_causal=True,
        )

        # Apply action head
        output = self.action_head(output)
        return output
