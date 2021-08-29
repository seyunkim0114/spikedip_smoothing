"""
Tensor Forecast Using Smoothing Techniques
Authors: Seyun Kim(seyun0114kim@gmail.com), U Kang (ukang@snu.ac.kr)
Data Mining Lab., Seoul National University

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.
"""

import torch.nn as nn
from torch.nn import functional as F

from load_dataset import *


def gen_random(n, m):
    """
    Generate random tensor with size n x m
    @param n
        size of dimension 0
    @param m
        size of dimension 1
    """
    return torch.tensor(np.random.rand(n, m))


class sp_reduce(nn.Module):
    """Implement spike and dips reducing smoothing on temporal factor"""

    def __init__(self, window, rank, tlength):
        """
        Initialize spike/dip reducing loss object

        @param window
            length of spike/dip reducing kernel
        @param rank
            tensor rank
        @param tlength
            length of time mode
        """
        super().__init__()

        self.window = window
        # one row of temporal factor
        self.rank = rank
        self.tlength = tlength

    def spike_dips(self, tfactor):
        """
        Return rate of change of consecutive time steps

        @param tfactor
            temporal factor of size KxR
        """

        w = torch.ones(1, 1, self.window, 1)
        tfactor = tfactor.float()

        w_temp = torch.tensor([1, -1]).float()
        w_change = w_temp.view(1, 1, 2, 1)

        tfactor = tfactor.view(1, 1, self.tlength, self.rank)

        if self.window % 2 == 0:
            pad = nn.ConstantPad2d((0, 0, (self.window - 1) // 2, (self.window - 1) // 2 + 1),
                                   0)  # VALUE NEEDS BE MORE REALISTIC
        else:
            pad = nn.ConstantPad2d((0, 0, (self.window - 1) // 2, (self.window - 1) // 2), 0)

        moving_average = F.conv2d(pad(tfactor), w) / self.window

        change_pad = nn.ConstantPad2d((0, 0, 0, 1), 0)
        change = F.conv2d(change_pad(moving_average), w_change) ** 2
        change = change / torch.sum(change)

        return change

    def forward(self, factor):
        """
        Get rate of change matrix from spike_dips function and
        compute weights inversely proportional to the magnitude of spike and dips

        @param factor
            raw temporal factor needs to be spike-dip reduced

        Return matrix K X R
        """

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
        """
        Construct Gaussian kernel
        Return Gaussian weights of size 1x1x(2*window-1)x1
        """
        x = np.arange(-self.window, self.window + 1)
        phi_g = np.exp(-x / 0.5)
        phi_g = phi_g / phi_g.sum()
        #         phi_gc = torch.FloatTensor(phi_g.reshape(1,1,window*2,1))
        phi_g = torch.tensor(phi_g)

        return phi_g.view(1, 1, x.shape[0], 1)

    def forward(self, tfactor, rank):
        """
        Implement Gaussian smoothing using conv2d
        Uses gaussian weight of shape (window*2-1, 1)

        @param tfactor
            temporal factor getting smoothed. Size is K X R
        @ param rank
            rank of target tensor

        Returns gaussian smoothed temporal factor of size K X R
        """
        tlength = tfactor.shape[0]

        factor = tfactor.reshape(1, 1, tlength, rank)
        factor = torch.tensor(factor)

        smoothed = F.conv2d(factor, self.weight, padding=(self.window, 0))
        return smoothed.view(tlength, rank)


class holtwinter_forecasting(nn.Module):
    """
    Implement holt-winters forecasting

    @param temp
        temporal factor
    @param seasonp
        estimated seasonality period of temporal factor

    This method is fully based on and inspired by [1]

    References
    ----------
    ... Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.
    """

    def __init__(self, alpha, beta, gamma, tfactor, seasonp, horizon, f_window):
        super().__init__()
        self.seasonp = seasonp
        self.horizon = horizon
        self.tfactor = tfactor
        self.f_window = f_window
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_mod(self, i, s):
        """
        Compute modulus

        @param i
            Integer dividend
        @param s
            Integer divisor
        """

        return int(i % s)

    def forward(self, data, alpha, beta, gamma, initial_values):
        """
        Implement HWES (Compute HWES parameters to optimize smoothing parameters

        @ param data
            data(train, val, or test) to learn Holt-Winters forecasting
        @ param alpha
            Weighing coefficient for level
        @ param beta
            Weighing coefficient for trend
        @ param gamma
            Weighing coefficient for seasonality
        @ param seasonp
            Seasonality period

        Returns a sequence forecast values of length equal to that of
            temporal dimension of target tensor
        """
        loss = 0

        hw_dataloader = dataset_holtwinter(data, self.horizon, self.f_window)

        L = torch.empty(self.f_window)
        b = torch.empty(self.f_window)
        S = torch.empty(self.seasonp)
        for data in range(len(hw_dataloader)):
            x, y = hw_dataloader[data]
            for i in range(1,self.f_window):
                L[i] = self.alpha*(x[i]-S[self.get_mod(i, self.seasonp)]) \
                       + (1-self.alpha)*(L[i-1]+b[i-1])
                b[i] = self.beta*(L[i]-L[i-1]) + (1-self.beta)*b[i-1]
                S[self.get_mod(i,self.seasonp)] = self.gamma*(x[i]-L[i]) + \
                       (1-self.gamma)*S[self.get_mod(i, self.seasonp)]
            f = (L[-1] + self.horizon*b[-1])*\
                S[self.get_mod(i+self.horizon, self.sesasonp)]
            loss = loss + (f-y)**2

        return loss


class tensor_reconstruction(nn.Module):
    """
    Compute reconstruction error from given factor matrices
    """

    def __init__(self, X):
        """
        Initialize tensor reconstruction loss object

        @param X
            target tensor
        """
        super().__init__()
        self.X = X

    def to_tl_tensor(self, x):
        """
        Returns tensor in tensorly Tensor format

        @param x
            Tensor not in tensorly tensor format
        """
        return tl.tensor(x, dtype=tl.float32)

    def to_tensor(self):
        """
        Reconstruct tensor from updated temporal factor and non-temporal factors
        """
        factors = [self.tfactor, self.nmode1, self.nmode2]
        recon_tensor = tl.cp_tensor.cp_to_tensor((None, factors), None)
        return recon_tensor

    def forward(self, data, nmode1, nmode2):
        """
        Compute loss between the original tensor and reconstructed tensor

        @param data
            train, validation or test data
        @param nmode1
            first non temporal factor
        @param nmode2
            second non temporal factor
        """
        data = self.to_tl_tensor(data.detach().numpy())
        nmode1 = self.to_tl_tensor(nmode1)
        nmode2 = self.to_tl_tensor(nmode2)

        factors = [data, nmode1, nmode2]
        recon_tensor = tl.cp_tensor.cp_to_tensor((None, factors), None)

        tensor_loss = (self.X - recon_tensor) ** 2
        return tensor_loss.sum()
