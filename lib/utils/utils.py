import operator
import os
import sys
from functools import reduce
from typing import *

import torch
from torch import Tensor


def prod(x: Iterable):
    """return the product of an Iterable"""
    if len(x):
        return reduce(operator.mul, x)
    else:
        return 0


def flatten(x: Tensor):
    """return x.view(x.size(0), -1)"""
    return x.view(x.size(0), -1)


def batch_reduce(x: Tensor):
    """return x.view(x.size(0), -1).sum(1)"""
    return flatten(x).sum(1)


def logging_sep(char="-"):
    """fill a terminal line with a ´char´"""
    return os.get_terminal_size().columns * char


class Header():
    """
    Print a message between two separators
    """

    def __init__(self, message: str = None):
        self.message = message

    def __enter__(self):
        if self.message is not None:
            print(f"{logging_sep('=')}\n{self.message}\n{logging_sep('-')}")
        else:
            print(logging_sep('='))

    def __exit__(self, *args):
        print(f"{logging_sep('=')}")


def available_device() -> torch.device:
    """return torch cuda device if available"""
    return torch.device("cuda:0") if torch.cuda.device_count() else torch.device("cpu")


class ManualSeed():
    """A simple class to execute a statement with a manual random seed without breaking the randomness.
    Another random seed is sampled and set when exiting the `with` statement. Usage:
    ```python
    with ManualSeed(seed=42):
        # code to execute with the random seed 42
        print(torch.rand((1,)))
    #  code to run independently of the seed 42
    print(torch.rand((1,)))
    ```
    """

    def __init__(self, seed: Optional[int] = 1):
        """define the manual seed (setting seed=None allows skipping the setting of the manual seed)"""
        self.seed = seed
        self.new_seed = None

    def __enter__(self):
        """set the random seed `seed`"""
        if self.seed is not None:
            self.new_seed = int(torch.randint(1, sys.maxsize, (1,)).item())
            torch.manual_seed(self.seed)

    def __exit__(self, type, value, traceback):
        """set the random random seed `new_seed`"""
        if self.seed is not None:
            torch.manual_seed(self.new_seed)


def preprocess(batch, device):
    """preprocess a batch of data received from the DataLoader"""
    if isinstance(batch, Tensor):
        x = batch.to(device)
        return x, None
    else:
        x, y = batch  # assume receiving a tuple (x,y)
        x = x.to(device)
        y = y.to(device)
        return x, y


def infer_device(model) -> torch.device:
    return next(iter(model.parameters())).device
