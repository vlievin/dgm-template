from typing import *

import torch
from torch import Tensor

from lib.utils import Diagnostic, preprocess, infer_device
from .base import Analysis
from .utils import RunningMean, RunningVariance


class GradientAnalysis(Analysis):
    EPS = 1e-12

    def __init__(self, args: Dict = None, **kwargs):
        super().__init__(**kwargs)
        # unpack arguments
        self.n_samples = args['grad_samples']
        self.key_filters = args.get('grad_key_filter', '').split(',')

    def get_individual_gradients(self, x: Tensor, **config) -> Iterable[Tensor]:

        # get the list of parameters
        params = self.get_parameters()

        # evaluate the forward pass
        loss, diagnostics, output = self.estimator(self.model, x, **config)

        # compute the individual gradients
        for l in loss:
            yield self.compute_grads(l, params)

    def compute_grads(self, loss: Tensor, params: List[Tensor]):
        """
        Return the gradients for the parameters matching the `key_filter`

        :param model: VAE model
        :param params: list of parameters
        :return:  Tensor of shape [D,] where `D` is the number of parameters
        """
        self.model.zero_grad()
        # backward individual gradients \nabla L[i]
        loss.mean().backward(create_graph=True, retain_graph=True)
        # gather gradients for each parameter and concat such that each element across the dim 1 is a parameter
        grads = [p.grad.view(-1) for p in params if p.grad is not None]
        return torch.cat(grads, 0)  # size [D,]

    def get_parameters(self) -> List[Tensor]:
        params = [p for k, p in self.model.named_parameters() if any([(_key in k) for _key in self.key_filters])]
        assert len(params) > 0, f"No parameters matching filter = `{self.key_filters}`"
        return params

    def __call__(self, **kwargs) -> None:
        # initialize statistics
        device = infer_device(self.model)
        mean = RunningMean()
        variance = RunningVariance()

        while mean.n < self.n_samples:

            # get a batch of data
            data = next(iter(self.loader))
            x, *_ = preprocess(data, device)

            # iterate through individual gradients
            for g in self.get_individual_gradients(x, **kwargs):
                if mean.n >= self.n_samples:
                    break

                # update statistics
                mean.update(g)
                variance.update(g)

        # compute the final statistics
        magnitude = mean().abs()
        std = variance().sqrt()
        snr = magnitude / (self.EPS + std)

        # log the data
        data = {'snr': snr.mean().item(),
                'magnitude': magnitude.mean().item(),
                'variance': variance().mean().item()}
        diagnostic = Diagnostic({'grads': data})
        diagnostic.log(writer=self.writer, global_step=self.session.global_step)
