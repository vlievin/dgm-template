from typing import *

from torch import nn, Size, zeros, Tensor
from torch.distributions import Normal, Bernoulli

from lib.utils import prod


class Vae(nn.Module):
    def __init__(self,
                 input_shape: Size = None,
                 num_layers: int = 2,
                 hidden_size: int = 64,
                 dropout: float = 0,
                 num_latents: int = 32,
                 **kwargs) -> None:
        super().__init__()

        self.input_shape = input_shape

        # prior parameters
        self.register_buffer('prior_params', zeros(2 * num_latents))

        # inference network
        layers = []
        xdim = prod(input_shape)
        for k in range(num_layers - 1):
            layers += [nn.Linear(xdim, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            xdim = hidden_size
        layers += [nn.Linear(xdim, 2 * num_latents)]
        self.inference_network = nn.Sequential(*layers)

        # generative model
        layers = []
        xdim = num_latents
        for k in range(num_layers - 1):
            layers += [nn.Linear(xdim, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            xdim = hidden_size
        layers += [nn.Linear(xdim, prod(input_shape))]
        self.generative_network = nn.Sequential(*layers)

    def infer(self, x: Tensor) -> Normal:
        x = x.view(x.size(0), -1)
        qlogits = self.inference_network(x)
        mu, log_sigma = qlogits.chunk(2, dim=-1)
        return Normal(loc=mu, scale=log_sigma.exp())

    def prior(self, batch_size: Size = Size([1])) -> Normal:
        plogits = self.prior_params.view(*(1 for _ in batch_size), *self.prior_params.shape)
        plogits = plogits.expand(*batch_size, *self.prior_params.shape)
        mu, log_sigma = plogits.chunk(2, dim=-1)
        return Normal(loc=mu, scale=log_sigma.exp())

    def generate(self, z: Tensor) -> Bernoulli:
        batch_size = z.shape[:-1]
        qlogits = self.generative_network(z)
        qlogits = qlogits.view(*batch_size, *self.input_shape)
        return Bernoulli(logits=qlogits)

    def forward(self, x: Tensor, **kwargs) -> Dict:
        # q(z|x)
        qz = self.infer(x)
        # p(z)
        pz = self.prior()
        # z ~q(z|x)
        z = qz.rsample()
        # p(x|z)
        px = self.generate(z)
        return {'px': px, 'qz': qz, 'pz': pz, 'z': z}

    def sample_from_prior(self, batch_size: Size = Size([1]), **kwargs) -> Dict:
        # p(z)
        pz = self.prior(batch_size=batch_size)
        # z ~p(z|x)
        z = pz.rsample()
        # p(x|z)
        px = self.generate(z)
        return {'px': px, 'pz': pz, 'z': z}
