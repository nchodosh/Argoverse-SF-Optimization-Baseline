import numpy as np
import torch


def rmap_iftype(X, fn, type):
    if isinstance(X, dict):
        for k, v in X.items():
            X[k] = rmap_iftype(v, fn, type)
    elif isinstance(X, list):
        for i, e in enumerate(X):
            X[i] = rmap_iftype(e, fn, type)
    elif isinstance(X, tuple) and hasattr(X, "_fields"):  # collections.namedtuple
        dd = asdict(X)
        dd = rmap_iftype(dd, fn, type)
        return type(X)(**dd)
    elif isinstance(X, tuple):
        X = tuple([rmap_iftype(e, fn, type) for e in X])
    elif isinstance(X, type):
        return fn(X)
    return X


def move_to_device(X, device):
    return rmap_iftype(X, lambda x: x.to(device=device), torch.Tensor)


def torch_to_numpy(X):
    return rmap_iftype(X, lambda x: x.detach().cpu().numpy(), torch.Tensor)


def numpy_to_torch(X):
    return rmap_iftype(X, lambda x: torch.from_numpy(x), np.ndarray)
