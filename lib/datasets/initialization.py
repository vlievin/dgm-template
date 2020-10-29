from typing import *

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from .binmnist import get_binmnist_datasets
from .mini_celeba import get_mini_celeba


def init_dataloaders(args: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Initialize the optimizer for the model and estimator parameters.
    """

    # unpack arguments
    dataset_id = args['dataset']
    data_root = args['data_root']
    num_workers = args['num_workers']
    batch_size = args['batch_size']
    eval_batch_size = args['eval_batch_size']

    # torch.vision transform
    transform = ToTensor()

    # get the datasets
    dset_train, dset_valid, dset_test = {
        'binmnist': get_binmnist_datasets,
        'mini_celeba': get_mini_celeba,
    }[dataset_id](data_root, transform=transform)

    # initialize the dataloaders
    loader_train = DataLoader(dset_train, batch_size=batch_size, num_workers=num_workers)
    loader_valid = DataLoader(dset_valid, batch_size=eval_batch_size, num_workers=num_workers)
    loader_test = DataLoader(dset_test, batch_size=eval_batch_size, num_workers=num_workers)

    return loader_train, loader_valid, loader_test
