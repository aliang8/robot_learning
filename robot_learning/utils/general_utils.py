import collections
import pickle as pkl
import random
from typing import Dict

import blosc
import numpy as np
import torch
from omegaconf import DictConfig
from robot_learning.utils.logger import log

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_dict_keys(dictionary, format_fn):
    """Returns new dict with `format_fn` applied to keys in `dictionary`."""
    return collections.OrderedDict(
        [(format_fn(key), value) for key, value in dictionary.items()]
    )


def prefix_dict_keys(dictionary, prefix):
    """Add `prefix` to keys in `dictionary`."""
    return format_dict_keys(dictionary, lambda key: "%s%s" % (prefix, key))


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


def print_dict(val, nesting: int = -4, start: bool = True):
    """Outputs a nested dictionory."""
    if type(val) == dict:
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        print(val)


def to_device(x, device):
    if isinstance(x, np.ndarray):
        return torch.tensor(x).to(device)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: torch.tensor(v).to(device) for k, v in x.items()}
    else:
        import ipdb

        ipdb.set_trace()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    elif isinstance(x, dict):
        return {k: v.cpu().detach().numpy() for k, v in x.items()}
    else:
        return x
    



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_data(path):
    with open(path, "rb") as f:
        data = pkl.loads(blosc.decompress(f.read()))

    # return np.array(data) # TODO: cant do this right now since some 2D flow is dict
    return data

def save_data(path, data):
    log(f"Saving to {path}", "yellow")
    with open(path, "wb") as f:
        compressed_data = blosc.compress(pkl.dumps(data))
        f.write(compressed_data)
