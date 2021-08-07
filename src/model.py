import torch.nn as nn
from torch.nn import functional as F

from smoothing_spikedips import *

class smooth_tfactor(nn.Module):
    """Implement smoothing on temporal factor"""

    def __init__(self, X, tmode, nmode1, nmode2, nmode, window, rank, tlength):
        super().__init__()

        self.X = X
        self.nmode1 = nmode1
        self.nmode2 = nmode2
        # self.tmode = tmode  # temp factor
        self.window = window
        self.nmode = nmode
        self.rank = rank
        self.tlength = tlength

        self.tfactor = nn.Parameter(gen_random(self.tlength, self.rank))
        self.alpha = nn.Parameter()
        self.beta = nn.Parameter()
        self.gamma = nn.Parameter()

        self.smooth = gaussian_smoothing(self.window)
        self.reduce = sp_reduce(self.window, self.rank, self.tlength)
        self.holtwinter = holtwinter_forecasting(self.tmode)

    def smooth_gaussian(self, tmode):
        """Call Gaussian smoothing class and implement it"""
        smoothed = self.smooth(self.tfactor)
        sloss = (smoothed - self.tfactor).pow(2)
        return sloss.sum()

    def spike_dips(self):
        """Call spike and dips reducing smoothing class and implement it"""
        reduced = self.reduce(self.tfactor)
        rloss = (reduced - self.tfactor).pow(2)
        return rloss.sum()

    def HoltWinters(self):
        """Learn Holt-Winters Exponential Smoothing parameters"""
        holtwinter_forecast = self.holtwinter(self.alpha, self.beta, self.gamma)
        floss = (holtwinter_forecast - self.tfactor).pow(2)
        return floss.sum()

    def tensordecomp(self):
        """Compute tensor reconstruction error"""
        facList = [self.tfactor, self.nmode1, self.nmode2]
        recon_tensor = tl.cp_tensor.cp_to_tensor((None, facList), None)
        tloss = (recon_tensor - self.X).pow(2)
        return tloss.sum()

    def l2_reg(self, mode):
        """Implement a L2 regularization"""

        return torch.norm(self.factors[mode]).pow(2)

    def forward(self,K):
        """Forecast using optimized parameterss"""
        forecast = []
        for k in range(K):
            f = (self.L[-1]+k*self.b[-1])*self.S[-1]
            forecast.append(f)

        return forecast


def get_loss(model, nmode):
    """Compute total loss"""

    tloss = model.tensordecomp()
    gloss = model.smooth_gaussian()
    sloss = model.spike_dips()
    floss = model.HoltWinters()
    loss = tloss + gloss + sloss + floss
    for mode in range(nmode):
        loss = loss + model.l2_reg(mode)
    return loss