from torch.utils.data.dataset import Dataset

import os
import torch
import torch.nn.functional as F
import fnmatch
import numpy as np

class NYUv2(Dataset):
    def __init__(self, root, train=True, augmentation=False, sample=False):
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        # read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        if sample:
            self.data_len = 30
        else:
            self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

    def __getitem__(self, index):
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0))

        return image.float(), semantic.float(), depth.float(), normal.float()

    def __len__(self):
        return self.data_len
