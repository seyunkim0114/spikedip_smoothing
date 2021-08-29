"""
Tensor Forecast Using Smoothing Techniques
Authors: Seyun Kim(seyun0114kim@gmail.com), U Kang (ukang@snu.ac.kr)
Data Mining Lab., Seoul National University

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.
"""

from tensorly import decomposition

import numpy as np
import tensorly as tl
import torch

from torch.utils.data import Dataset

import sys

def get_dataset(PATH):
    """
    Get dataset from PATH and return in tensor format

    @param PATH
        path to COO format data
    """
    sys.path.append(PATH)
    train_idxs = np.load(rf'{PATH}\train_idxs.npy')
    train_vals = np.load(rf'{PATH}\train_vals.npy')
    l = train_idxs.shape[0]

    val_idxs = train_idxs[-(int(l / 5)):-1, :]
    val_vals = train_vals[-(int(l / 5)):-1]
    test_idxs = np.load(rf'{PATH}\test_idxs.npy')
    test_vals = np.load(rf'{PATH}\test_vals.npy')

    i = torch.LongTensor(np.transpose(train_idxs))
    v = torch.FloatTensor(train_vals)
    bstensor = torch.sparse_coo_tensor(i, v).coalesce()
    bstensor_tltrain = tl.tensor(bstensor.to_dense())

    ii = torch.LongTensor(np.transpose(test_idxs))
    vv = torch.FloatTensor(test_vals)
    bstensor_test = torch.sparse_coo_tensor(ii, vv).coalesce()
    bstensor_tltest = tl.tensor(bstensor_test.to_dense())

    iii = torch.LongTensor(np.transpose(val_idxs))
    vvv = torch.FloatTensor(val_vals)
    bstensor_val = torch.sparse_coo_tensor(iii, vvv).coalesce()
    bstensor_tlval = tl.tensor(bstensor_val.to_dense())

    return bstensor_tltrain, bstensor_tlval, bstensor_tltest

class dataset_holtwinter(Dataset):
    def __init__(self, tfactor, horizon, f_window):
        """
        Custom dataloader for learning holt-winters exponential smoothing

        @param tfactor
            temporal factor of decomposed tensor
        @param horizon
            number of time steps in the future to forecast
        @param f_window
            width of time steps of each row of temporal factor to learn to forecast
        """
        self.tfactor = tfactor
        self.horizon = horizon
        self.f_window = f_window
        self.tlength = self.tfactor.shape[0]
        self.num_samples = int(self.tlength // self.f_window)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        sample = self.tfactor[item:item + self.f_window, 0]
        sample = sample.reshape(self.f_window)
        val = self.tfactor[:, item + self.horizon]

        return sample, val





def get_CPfac(X, rank):
    """
    Compute CP Decomposition on X and return factor matrices

    @param X
        target tensor
    @param rank
        tensor rank
    """
    weights, factors = tl.decomposition.parafac(tensor=X, rank=rank, normalize_factors=True)
    tmode = factors[0]
    temfactor = torch.tensor(tl.transpose(tmode))
    ntmode1 = torch.tensor(factors[1])
    ntmode2 = torch.tensor(factors[2])
    return temfactor, ntmode1, ntmode2
