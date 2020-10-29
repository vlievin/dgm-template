import logging
from typing import *

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.estimators import GradientEstimator
from lib.utils.experiment import Experiment
from lib.utils.session import Session


class Analysis():
    """
    A utility class to implement model analysis (gradient statistics, prior sampling).
    The object `sessions`, `experiment`, `writer` and `logger` are passed to enable easy logging
    """

    def __init__(self, model: torch.nn.Module = None,
                 estimator: GradientEstimator = None,
                 loader: Optional[DataLoader] = None,
                 session: Session = None,
                 experiment: Experiment = None,
                 writer: Optional[SummaryWriter] = None,
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        self.model = model
        self.estimator = estimator
        self.loader = loader
        self.experiment = experiment
        self.session = session
        self.writer = writer
        self.logger = logger

    def __call__(self) -> None:
        raise NotImplementedError
