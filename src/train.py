import numpy as np
import tensorly as tl
import torch
import torch.nn as nn
from torch.nn import functional as F

from tensorly import decomposition
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from smoothing_spikedips import *

def create_model(X, tmode, nmode1, nmode2, nmode, window, rank, tlength):
    """Create time factor smoothing model"""
    model = smooth_tfactor(X, tmode, nmode1, nmode2, nmode, window, rank, tlength)
    opt = torch.optim.SGD(model.parameters(),lr=0.001)
    return model, opt

def train_model(epochs, model, opt, temp, tempval, nmode, n):
    """Train the model"""
    for epoch in range(epochs):
        opt.zero_grad()
        loss = get_loss(model, nmode, temp)
        loss.backward()
        opt.step()

        lossval = get_loss(model, nmode, tempval)

        print(f'Epoch: {epoch} \tLoss: {loss}\tVal: {lossval}')

    return model(n)
