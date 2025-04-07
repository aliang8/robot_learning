from functools import partial
from typing import List

import numpy as np
from omegaconf import OmegaConf


def fix_ds_name(ds_name: List[str], dataset_name: str, dataset_split: List[int]) -> str:
    ds_names = []

    mapping = {
        "peg-insert-side": "peg-side",
        "assembly": "assem",
        "hammer": "ham",
        "door-open": "do-op",
        "window-open": "win-op",
        "random": "R",
        "medium": "M",
        "expert": "E",
        "mw-": "",
        "relative": "R",
        "absolute": "A",
        "close_drawer": "close-d",
    }

    # shorten the dataset names
    for name in ds_name:
        shortened_name = name.split("/")[-1]

        # replace the name with the mapping
        for key, val in mapping.items():
            if key in shortened_name:
                shortened_name = shortened_name.replace(key, val)

        ds_names.append(shortened_name)

    ds_name = "-".join(ds_names)
    ds_name = dataset_name + "-" + ds_name
    ds_name = ds_name + "-".join([str(x) for x in dataset_split])

    return ds_name


def fix_env_hp_name(
    hp_name: str = "", image_obs: bool = False, image_extra: str = ""
) -> str:
    if image_obs:
        if hp_name == "":
            hp_name = image_extra
        else:
            hp_name += f"-{image_extra}"

    return hp_name


OmegaConf.register_new_resolver(
    "multiply", lambda a, b: a * b, use_cache=False, replace=True
)
OmegaConf.register_new_resolver(
    "concat", lambda l: ",".join(l[:2]), use_cache=False, replace=True
)
OmegaConf.register_new_resolver(
    "eval", lambda x: eval(x), use_cache=False, replace=True
)
OmegaConf.register_new_resolver(
    "fix_ds_name", fix_ds_name, use_cache=False, replace=True
)
OmegaConf.register_new_resolver(
    "fix_env_hp_name", fix_env_hp_name, use_cache=False, replace=True
)
