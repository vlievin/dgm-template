import math
import os

import matplotlib.image
import torch
from torchvision.utils import make_grid

from .base import Analysis


class PriorSamplingImage(Analysis):
    def __init__(self, *args, N=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N

    @torch.no_grad()
    def __call__(self) -> None:
        output = self.model.sample_from_prior(batch_size=torch.Size([self.N]))
        px = output.get('px', None)
        if px is None:
            return
        sample = px.sample()
        sample = sample.to('cpu')

        # make grid
        nrow = math.floor(math.sqrt(self.N))
        grid = make_grid(sample, nrow=nrow)

        # normalize
        grid -= grid.min()
        grid /= grid.max()

        # log to tensorboard
        if self.writer is not None:
            self.writer.add_image("test", grid, self.session.global_step)

        # save the raw image
        img = grid.data.permute(1, 2, 0).cpu().numpy()
        matplotlib.image.imsave(os.path.join(self.experiment.logdir, f"prior-sample.png"), img)
