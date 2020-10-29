import os
import pickle

import numpy as np
from torch.utils.data import Dataset
from PIL import Image


def load_pickled_data(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    train_data, test_data = data['train'], data['test']
    train_data = train_data[:, :, :, [2, 1, 0]]
    test_data = test_data[:, :, :, [2, 1, 0]]
    return train_data, test_data


class MiniCeleba(Dataset):
    """Celeba dataset"""

    def __init__(self, data, transform=None):
        h, w, c = 32, 32, 3

        # binarize data
        data = data > 1

        self.data = np.ascontiguousarray(data.reshape(-1, h, w, 3), dtype=np.ubyte)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        sample = Image.fromarray(255 * sample)  # cannot read bytes directly: https://github.com/numpy/numpy/issues/5861

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_mini_celeba(root, transform=None):
    """credits to https://github.com/rll/deepul"""
    data_dir = 'raw_data/'
    train_data, test_data = load_pickled_data(os.path.join(data_dir, 'celeb.pkl'))
    dset_train = MiniCeleba(train_data, transform=transform)
    dset_test = MiniCeleba(test_data, transform=transform)
    return dset_train, dset_test, dset_test
