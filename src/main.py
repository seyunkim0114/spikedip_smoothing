# import numpy as np
# import tensorly as tl
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
#
# from tensorly import decomposition
# import torch.optim as optim
#
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
#
import sys
#
# from smoothing_spikedips import *
from hwes import *
from train import *

# Parameters
data = "radar"
PATH = rf'C:\Users\seyun\Google 드라이브\DMLab\DLab\forecast\spike-dip-reducing\spikedip_smoothing\data\{data}'

rank = 4
tmode = 0
nmode = 3
window = 5
ndim = 3

# Hyperparameters
epochs = 10
sigma = 1.5
seasonp = 9
n = 5
zoom = 30

data_train, data_val, data_test = get_dataset(PATH)
temp_train, nmode1_train, nmode2_train = get_CPfac(data_train, rank)
temp_test, nmode1_test, nmode2_test = get_CPfac(data_test, rank)
temp_val, nmode1_val, nmode2_val = get_CPfac(data_val, rank)

tlength = temp_train.shape[1]
model, opt = create_model(data_train, tmode, nmode1_train, nmode2_train, nmode, window, rank, tlength)
train_model(epochs, model, opt, temp_train, temp_val, nmode,n)



# Holt Winters Exponential Smoothing
# holtwinter_forecast = HWES(n, temp, sigma, window, seasonp, nmode, zoom)
# holtwinter_forecast.buildModelHWES()
# holtwinter_forecast.printHWESresults()

