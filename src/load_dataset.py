from tensorly import decomposition
import torch.optim as optim

import numpy as np
import tensorly as tl
import torch

import sys


def get_dataset(PATH):
    """Get dataset from PATH and return in tensor format"""
    sys.path.append(PATH)
    print(rf'{PATH}\train_idxs.npy')
    train_idxs = np.load(rf'{PATH}\train_idxs.npy')
    train_vals = np.load(rf'{PATH}\train_vals.npy')
    l = train_idxs.shape[0]

    val_idxs = train_idxs[:, -(int(l/5))]
    val_vals = train_vals[:, -(int(l/5))]
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


def get_CPfac(X, rank):
    """Compute CP Decomposition on X and return factor matrices"""
    weights, factors = tl.decomposition.parafac(tensor=X, rank=rank, normalize_factors=True)
    tmode = factors[0]
    temfactor = tl.transpose(tmode)
    ntmode1 = factors[1]
    ntmode2 = factors[2]
    return temfactor, ntmode1, ntmode2
