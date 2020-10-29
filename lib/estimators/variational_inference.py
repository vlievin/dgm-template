from typing import *

from torch import nn, Tensor

from lib.utils import Diagnostic
from lib.utils import batch_reduce, prod
from .base import GradientEstimator


class VariationalInference(GradientEstimator):

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
        config = self.get_runtime_config(**kwargs)  # for instance, beta, K, ...

        # retrieve the number of dimensions
        n_pixels = prod(x.shape[1:])

        # forward pass through the model
        output = model(x, **config)
        px, qz, pz, z = [output[k] for k in ['px', 'qz', 'pz', 'z']]

        # evaluate the log probabilities
        log_qz = batch_reduce(qz.log_prob(z))
        log_pz = batch_reduce(pz.log_prob(z))
        log_px = batch_reduce(px.log_prob(x))

        # elbo
        kl = log_qz - log_pz
        elbo = log_px - kl

        # loss
        loss = - elbo / n_pixels

        if return_diagnostic:
            diagnostic = Diagnostic({
                'loss': {
                    'bpd': loss,
                    'elbo': elbo,
                    'nll': -log_px,
                    'kl': kl
                }
            })
        else:
            diagnostic = Diagnostic()

        return loss, diagnostic, output
