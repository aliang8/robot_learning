from dataclasses import dataclass
from itertools import repeat
from typing import Dict

import numpy as np
from jaxtyping import Array, Float, Key, Union

BT = Float[Array, "B T"]
BTD = Float[Array, "B T D"]
BTHWC = Float[Array, "B T H W C"]
BTX = Union[BTD, BTHWC]
PRNGKeyDict = Dict[str, Key]


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    # assert not torch.is_floating_point(obs)
    return obs.astype(np.float64) / 255 - 0.5


def unnormalize_obs(obs: np.ndarray) -> np.ndarray:
    return ((obs + 0.5) * 255).astype(np.uint8)


@dataclass
class Transition:
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    image: np.ndarray = None
    info: Dict[str, np.ndarray] = None


@dataclass
class Batch:
    states: BTX
    actions: BT
    rewards: BT = None
    dones: np.ndarray = None
    timestep: BT = None
    latent_actions: np.ndarray = None
    prequantized_las: np.ndarray = None
    is_first: BT = None
    is_last: BT = None
    is_terminal: BT = None
    discount: BT = None

    # Additional arbitrary arguments
    _extra_fields: dict = None

    def __post_init__(self):
        if self._extra_fields is None:
            self._extra_fields = {}

    def __getattr__(self, name):
        if name in self._extra_fields:
            return self._extra_fields[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    @classmethod
    def create(cls, states, actions, **kwargs):
        """Factory method to create a Batch with arbitrary additional fields."""
        known_fields = cls.__dataclass_fields__.keys()
        standard_kwargs = {k: v for k, v in kwargs.items() if k in known_fields}
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in known_fields}
        return cls(
            states=states,
            actions=actions,
            **standard_kwargs,
            _extra_fields=extra_kwargs,
        )
