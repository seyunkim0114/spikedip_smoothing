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

from smoothing_spikeddips import *
from hwes import *
from train import *

# Parameters
PATH = "/home/seyunkim/TATD/data/beijing"
rank = 4
tmode = 0
nmode = 3
window = 5
ndim = 3
temp = torch.FloatTensor(temfactor[0])
tlength = temp.shape[0] 
epochs = 100
sigma = 1.5
train_idnx = int(tlength/3*2)

X = get_dataset(PATH)

tmode = get_CPfac(X,rank)

model,opt = create_model(tmode, nmode, window, rank, tlength)

train_model(epochs, model, loss, opt, temp)

holtwinter_forecast = HWES(sigma, window, density)

holtwinter_forecast.buildModelHWES(temp,train_idnx,seasonp,n,zoom)

