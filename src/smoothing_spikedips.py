import numpy as np
import tensorly as tl
import torch
import torch.nn as nn
from torch.nn import functional as F

from statsmodels.tsa.holtwinters import ExponentialSmoothing as HOLTWINTER

from tensorly import decomposition
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from load_dataset import *

def gen_random(n,m):
    """Generate random tensor with size n x m"""
    temp = np.random.rand(n,m)
    return torch.FloatTensor(temp)

class sp_reduce(nn.Module):
    """Implement spike and dips reducing smoothing on temporal factor"""

    def __init__(self, window, rank, tlength):
        super().__init__()

        self.window = window
        # one row of temporal factor
        self.rank = rank
        self.tlength = tlength

    def spike_dips(self, factor):
        """Return rate of change of consecutive time steps"""
        w = torch.ones(1, 1, self.window)

        moving_average = F.conv1d(factor.view(1, 1, self.tlength), w, padding=self.window // 2) / self.window
        moving_average = moving_average.view(moving_average.shape[2])
        change = []
        change.append(0)

        for n in range(moving_average.shape[0] - 1):
            change.append((moving_average[n] - moving_average[n + 1]) ** 2)

        change = torch.FloatTensor(change)
        change = change / torch.sum(change)
        return change

    def forward(self, factor):
        """Implement spike and dips reducing smoothing"""
        change_rate = self.spike_dips(factor)
        return (1 - change_rate).pow(5) * factor


# Implement Gaussian smoothing regularization
class gaussian_smoothing(nn.Module):
    """Implement Gaussian smoothing on temporal factor"""

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.weight = self.gaussian()

    def gaussian(self):
        """Construct Gaussian kernel and return Gaussian weights"""
        x = np.arange(-self.window, self.window + 1)
        phi_g = np.exp(-x / 0.5)
        phi_g = phi_g / phi_g.sum()
        #         phi_gc = torch.FloatTensor(phi_g.reshape(1,1,window*2,1))
        phi_g = torch.FloatTensor(phi_g)
        return phi_g.view(1, 1, x.shape[0])

    def forward(self, tfactor):
        """Implement Gaussian smoothing"""
        #         row,col = tmode.shape
        #         tmode_c = torch.FloatTensor(tmode.reshape(1,1,row,col))
        tlength = tfactor.shape[0]
        factor = tfactor.reshape(1, 1, tlength)
        factor = torch.FloatTensor(factor)
        smoothed = F.conv1d(factor, self.weight, padding=self.window)
        return smoothed.view(tlength)


class holtwinter_forecasting(nn.Module):
    """Implement holt-winters forecasting"""

    def __init__(self, temp):
        super().__init__()
        self.temp = temp

    def initialize_hw(self):
        """Initialize Holt-Winters Exponential Smoothing parameters"""
        model = HOLTWINTER(self.train, seasonal_periods=self.seasonp, trend='add', seasonal='mul')
        L0, b0, S0 = model.initial_values()
        return L0, b0, S0

    def get_mod(self, i, s):
        """Compute modulus"""
        return i % s

    def forward(self, alpha, beta, gamma, seasonp):
        """Implement HWES (Compute HWES parameters to optimize smoothing parameters"""
        L0, b0, S0 = self.initialize_hw()
        L = torch.ones(self.temp.shape[1])
        b = torch.ones(self.temp.shape[1])
        S = torch.ones(self.temp.shape[1])

        L[0] = L0
        b[0] = b0
        S = S0
        for i in range(1, self.temp.shape[1]):
            L[i] = alpha * (self.temp[i] - S[self.get_mod(i, self.seasonp)]) \
                   + (1 - alpha) * (L[i - 1] + b[i - 1])
            b[i] = beta(L[i] - L[i - 1]) + (1 - beta) * b[i - 1]
            if i < self.seasonp:
                S[i] = gamma * (self.temp[i] - L[i]) + (1 - gamma) * S[i]
            else:
                S[i] = gamma * (self.temp[i] - L[i]) + (1 - gamma) * S[i - self.seasonp]

        f = (L+b)*S

        return f




