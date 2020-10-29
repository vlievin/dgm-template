from copy import copy
from typing import *

from torch import nn, Tensor

from lib.utils import Diagnostic


class GradientEstimator(nn.Module):
    """
    An abstract class defining a gradient estimator.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        A base gradient estimator class.
        :param kwargs: additional keyword arguments
        """
        super().__init__()
        self.config = {
            **kwargs
        }

    def get_runtime_config(self, **kwargs):
        """update `self.config` with kwargs"""
        if len(kwargs):
            config = copy(self.config)
            config.update(**kwargs)
            return config
        else:
            return copy(self.config)

    def forward(self,
                model: nn.Module,
                x: Tensor,
                return_diagnostic: bool = True,
                **kwargs) -> Tuple[Tensor, Diagnostic, Dict]:
        """
        Compute the loss given the `model` and a batch of data `x`.
        Returns the loss per datapoint, diagnostics and the model's output

        :param model: nn.Module
        :param x: batch of data
        :param return_diagnostic: turn off diagnostics to save computation
        :param kwargs: parameters for the forward pass
        :return: loss, diagnostics, model's output
        """
        # update the `config` object given the `kwargs`
        config = self.get_runtime_config(**kwargs)

        # do some computation
        raise NotImplementedError
