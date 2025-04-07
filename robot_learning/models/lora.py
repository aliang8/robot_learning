import importlib
import math
from functools import partial
from typing import Optional

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from robot_learning.utils.logger import log


# taken from https://github.com/sirluk/pytora
class LoraLayer(nn.Module):
    def __init__(self, weight, r, alpha=1, dropout_prob=0, fan_in_fan_out=False):
        super().__init__()

        if fan_in_fan_out:
            self.in_features = weight.shape[0]
            self.out_features = weight.shape[1]
        else:
            self.in_features = weight.shape[1]
            self.out_features = weight.shape[0]
        self.alpha = alpha
        self.fan_in_fan_out = fan_in_fan_out

        if dropout_prob > 0.0:
            self.lora_dropout = nn.Dropout(p=dropout_prob)
        else:
            self.lora_dropout = nn.Identity()

        self._init_lora(r, weight_dtype=weight.dtype)

    def _init_lora(self, r, weight_dtype=None):
        # Actual trainable parameters
        if r > 0:
            if weight_dtype == None:
                weight_dtype = self.lora_A.dtype
            self.register_parameter(
                "lora_A",
                nn.Parameter(torch.empty((self.in_features, r), dtype=weight_dtype)),
            )
            self.register_parameter(
                "lora_B",
                nn.Parameter(torch.zeros((r, self.out_features), dtype=weight_dtype)),
            )
            self.scaling = self.alpha / r
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            try:
                # ensure parameters do not exist if they are zero
                delattr(self, "lora_A")
                delattr(self, "lora_B")
                delattr(self, "scaling")
            except AttributeError:
                pass
        self.r = r

    def change_lora_rank(self, new_rank):
        if new_rank != self.r:
            self._init_lora(new_rank)

    def forward(self, X):
        if self.r == 0:
            return X
        else:
            lora = self.lora_dropout(self.lora_A @ self.lora_B * self.scaling)
            if not self.fan_in_fan_out:
                lora = lora.T
            return X + lora


_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
if _TRANSFORMERS_AVAILABLE:
    from transformers import Conv1D


def module_name_check(
    name: str,
    include_names: Optional[list[str]] = None,
    exclude_names: Optional[list[str]] = None,
):
    if include_names is not None:
        inclusion = [n == name[-len(n) :] for n in include_names]
        return any(inclusion)

    if exclude_names is not None:
        exclusion = [n == name[-len(n) :] for n in exclude_names]
        return not any(exclusion)

    return True


@torch.no_grad()
def apply_lora(
    model: nn.Module,
    lora_r: int = 4,
    lora_alpha: int = 1,
    lora_dropout: float = 0.0,
    include_names: Optional[list[str]] = None,
    exclude_names: Optional[list[str]] = None,
):
    log("Applying LoRA to model and freezing non-LoRA weights", "green")

    check = partial(
        module_name_check, include_names=include_names, exclude_names=exclude_names
    )

    for name, module in model.named_modules():

        if "embedder" in name:
            log(f"Skipping {name} as it is an embedding layer", "green")
        elif check(name):
            if type(module) == torch.nn.Linear:
                l = LoraLayer(
                    weight=module.weight,
                    r=lora_r,
                    alpha=lora_alpha,
                    dropout_prob=lora_dropout,
                ).to(module.weight.device)
                register_parametrization(module, "weight", l)
            elif _TRANSFORMERS_AVAILABLE and type(module) == Conv1D:
                # same as linear layer, was implemented to keep gpt2 style
                l = LoraLayer(
                    weight=module.weight,
                    r=lora_r,
                    alpha=lora_alpha,
                    dropout_prob=lora_dropout,
                    fan_in_fan_out=True,
                ).to(module.weight.device)
                register_parametrization(module, "weight", l)

        # freeze the non-lora weights
        if hasattr(module, "weight"):
            module.weight.requires_grad = False
