from itertools import chain
from typing import *

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer, SGD, Adam, Adamax, RMSprop
from torch.utils.data import DataLoader

from lib.analysis import Analysis, GradientAnalysis, PriorSamplingImage
from lib.models import Vae
from lib.utils import ManualSeed, preprocess, Header


def set_manual_seed(seed: int) -> None:
    """set the initial random seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_fused_adam(*args, **kwargs):
    try:
        import apex
        return apex.optimizers.FusedAdam(*args, **kwargs)
    except ImportError as e:
        message = """
        You need first to install Apex to use FusedAdam: [https://nvidia.github.io/apex/optimizers.html]
        \n\t >> pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
        """
        print(message)
        raise


def init_optimizer(args, model, estimator) -> Optimizer:
    """
    Initialize the optimizer for the model and estimator parameters.
    """
    assert 'optimizer' in args.keys()
    assert 'lr' in args.keys()

    Optimizer = {'sgd': SGD,
                 'adam': Adam,
                 'fusedadam': get_fused_adam,
                 'adamax': Adamax,
                 'rmsprop': RMSprop}[args['optimizer']]

    return Optimizer(chain(model.parameters(), estimator.parameters()), lr=args['lr'])


def init_model(args: Dict, loader: DataLoader) -> Tuple[nn.Module, Dict]:
    # infer the input shape
    x, *_ = preprocess(next(iter(loader)), device=torch.device('cpu'))
    input_shape = x[0].shape

    # print the shape of the data
    with Header("Training batch:"):
        print(f"x.shape = {x.shape}, x.min = {x.min():.1f}, x.max = {x.max():.1f}, x.dtype = {x.dtype}")

    # define the hyper parameters
    hyperparams = {
        'input_shape': input_shape,
        **args
    }

    # get the right constructor
    Model = {'vae': Vae,
             }[args['model']]

    # initialize the model with a random seed
    with ManualSeed(seed=args['seed']):
        model = Model(**hyperparams)

    return model, hyperparams


def init_analyses(args: Dict, **kwargs) -> List[Analysis]:
    analysis_id = args['analysis'].split(',')

    Analyses = [
        {
            'prior_sampling_image': PriorSamplingImage,
            'gradient': GradientAnalysis
        }[id]
        for id in analysis_id]

    return [Analysis(args=args, **kwargs) for Analysis in Analyses]
