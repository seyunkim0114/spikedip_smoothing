import numpy as np
import tensorly as tl
import torch
import torch.nn as nn
from torch.nn import functional as F

from tensorly import decomposition
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import sys

def create_model(tmode, nmode, window, rank, tlength):
    '''Create time factor smoothing model'''
    model = smooth_tfactor(tmode, nmode, window, rank, tlength)
    opt = torch.optim.SGD(model.parameters(),lr=0.001)
    return model, opt

def train_model(epochs, model, loss, opt, temp)
    for epoch in range(epochs):
        opt.zero_grad()
        loss = get_loss(model,temp)
        loss.backward()
        opt.step()
        print(f'Epoch: {epoch}, Loss: {loss}')
    
