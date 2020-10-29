from functools import wraps
from time import time

import torch


def zero_grad(params):
    for p in params: p.grad = None


def append_elapsed_time(func):
    """append the elapsed time to the diagnostics"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        diagnostics = func(*args, **kwargs)
        diagnostics['info']['elapsed-time'] = time() - start_time
        return diagnostics

    return wrapper


@append_elapsed_time
def training_step(x, model, estimator, optimizer, grad_clip=1e18, **config):
    """Perform one optimization step of the `model` given the mini-batch of observations `x`,
    the gradient estimator/evaluator `estimator` and the optimizer `optimizer`"""
    loss, diagnostics, output = estimator(model, x, **config)

    # backward pass
    loss.mean().backward()

    # gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    diagnostics.update({'info': {'grad_norm': grad_norm}})

    # optimization step
    optimizer.step()

    # zeroing the gradients
    [zero_grad(group['params']) for group in optimizer.param_groups]

    return diagnostics


@torch.no_grad()
@append_elapsed_time
def test_step(x, model, estimator, **config):
    """Test the `model` given the mini-batch of observations `x` and the gradient estimator/evaluator `estimator`"""
    loss, diagnostics, output = estimator(model, x, backward=False, **config)
    return diagnostics
