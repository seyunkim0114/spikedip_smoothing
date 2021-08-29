"""
Tensor Forecast Using Smoothing Techniques
Authors: Seyun Kim(seyun0114kim@gmail.com), U Kang (ukang@snu.ac.kr)
Data Mining Lab., Seoul National University

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.
"""

from smoothing_spikedips import *

class smooth_tfactor(nn.Module):
    """Implement smoothing on temporal factor"""

    def __init__(self, X, tmode, nmode1, nmode2, nmode, window, rank, tlength, seasonp, horizon, f_window):
        """
        Initialize tensor forecasting model

        @param X
            target tensor
        @param nmode1
            first non temporal mode
        @param nmode2
            second non temporal mode
        @param nmode
            number of modes
        @param window
            length of Gaussian kernel and spike/dip reduce
        @param rank
            rank of tensor
        @param tlength
            length of time mode
        @param seasonp
            seasonality of temporal factor rows
        @param horizon
            number of time steps from the end of data to the forecast point
        @param f_window
            length of data
        """
        super().__init__()

        self.X = X
        self.nmode1 = tl.tensor(nmode1)
        self.nmode2 = tl.tensor(nmode2)
        self.window = window
        self.nmode = nmode
        self.rank = rank
        self.tlength = tlength
        self.seasonp = seasonp
        self.f_window = f_window
        self.horizon = horizon

        self.tfactor = nn.Parameter(gen_random(self.tlength, self.rank))

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))

        level = nn.Parameter(torch.tensor(0.5))
        trend = nn.Parameter(torch.tensor(0.5))
        season = nn.Parameter(torch.tensor(0.5))
        self.initial_values = nn.ParameterList([level, trend, season])
        self.initial_values.requires_grad_(False)

        self.smooth = gaussian_smoothing(self.window)
        self.reduce = sp_reduce(self.window, self.rank, self.tlength)
        self.holtwinter = holtwinter_forecasting(self.alpha, self.beta, self.gamma, self.tfactor, self.seasonp, self.horizon, self.f_window)

    def smooth_gaussian(self):
        """Call Gaussian smoothing class and implement it"""
        smoothed = self.smooth(self.tfactor, self.rank)
        sloss = (smoothed - self.tfactor).pow(2)
        return sloss.sum()

    def spike_dips(self):
        """Call spike and dips reducing smoothing class and implement it"""
        reduced = self.reduce(self.tfactor)
        rloss = (reduced - self.tfactor).pow(2)
        return rloss.sum()

    def HoltWinters(self):
        """Learn Holt-Winters Exponential Smoothing parameters"""
        forecast_loss = self.holtwinter(self.alpha, self.beta, self.gamma, self.initial_values)

        return forecast_loss

    def tensordecomp(self, data):
        """
        Compute tensor reconstruction error

        @param data
            train, validation, or test data
        """
        tensor_recon = tensor_reconstruction()
        tloss = tensor_recon(self.tfactor, self.nmode1, self.nmode2)
        return tloss

    def l2_reg(self, data):
        """
        Implement a L2 regularization

        @param data
            train, validation, or test data
        """

        return torch.norm(data).pow(2)

    def forward(self, data):
        """
        Forecast using optimized parameters

        @param data
            data to forecast. Can be train, validation, or test data

        Returns forecast value horizon time steps away
        """

        L = torch.empty(self.f_window)
        b = torch.empty(self.f_window)
        S = torch.empty(self.seasonp)
        for i in range(1, self.f_window):
            L[i] = self.alpha * (data[i] - S[self.get_mod(i, self.seasonp)]) \
                   + (1 - self.alpha) * (L[i - 1] + b[i - 1])
            b[i] = self.beta * (L[i] - L[i - 1]) + (1 - self.beta) * b[i - 1]
            S[self.get_mod(i, self.seasonp)] = self.gamma * (data[i] - L[i]) + \
                                               (1 - self.gamma) * S[self.get_mod(i, self.seasonp)]
        f = (L[-1] + self.horizon * b[-1]) * \
            S[self.get_mod(i + self.horizon, self.sesasonp)]

        return f

def get_loss(model, data, nmode1, nmode2):
    """
    Compute total loss

    @param model
        tensor forecasting model
    @param data
        train, validation, or test data
    @param nmode1
        first non temporal factor
    @param nmode2
        second non temporal factor
    """

    print(f'Computing tensor reconstruction loss...')
    tloss = model.tensordecomp(data, nmode1, nmode2)
    print(f'Tensor reconstruction loss done...')
    gloss = model.smooth_gaussian(data)
    print(f'Gaussian kernel smoothing loss done...')
    sloss = model.spike_dips(data)
    print(f'Spike-dip reducing loss done...')
    print(f'Computing Holt-Winters exponential smoothing loss...')
    floss = model.HoltWinters(data)
    print(f'Holt-Winters exponential smoothing loss done...')

    loss = tloss + gloss + sloss + floss

    return torch.tensor(tloss)



