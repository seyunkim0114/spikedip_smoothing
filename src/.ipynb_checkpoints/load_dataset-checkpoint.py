import sys

from tensorly import decomposition
import torch.optim as optim

import numpy as np
import tensorly as tl
import torch

def get_dataset(PATH):
    '''Get dataset from PATH and return in tensor format'''
    PATH = "/home/seyunkim/TATD/data/beijing"
    train_idxs = np.load(f'{PATH}/train_idxs.npy')
    train_vals = np.load(f'{PATH}/train_vals.npy')
    
    i = torch.LongTensor(np.transpose(train_idxs))
    v = torch.FloatTensor(train_vals)
    bstensor = torch.sparse_coo_tensor(i,v).coalesce()
    bstensor_tl = tl.tensor(bstensor.to_dense())
    return bs_tensor

def get_CPfac(X,rank):
    '''Compute CP Decomposition on X and return factor matrices'''
    weights,factors = tl.decomposition.parafac(X,rank,normalize_factors=True)
    tmode = factors[0]
    temfactor = tl.transpose(tmode)
    ntmode1 = factors[1] 
    ntmode2 = factors[2] 
    return tmode