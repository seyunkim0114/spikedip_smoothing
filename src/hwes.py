from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HOLTWINTER
from matplotlib.pyplot import figure

import torch
import torch.nn as nn


class HWES(nn.Module):
    def __init__(self, n, temp, sigma, window, seasonp, nmode, zoom):
        super().__init__()

        self.sigma = sigma
        self.temp = temp # temporal factor
        self.window = window
        self.seasonp = seasonp
        self.nmode = nmode
        self.zoom = zoom # amount of original data to appear
        self.n = n # number of time steps to forecast

        self.split_idx = int(temp.shape[1] / 3 * 2)
        self.train = self.temp[0][:self.split_idx]
        self.test = self.temp[0][self.split_idx:]


    def getDataset(self, csv_file):
        """Read CSV file using pandas"""
        T = pd.read_csv(csv_file, index_col=False)
        return T

    def printFigure(self, fac, w, h, original=True):
        """Print plots of temporal factor"""
        plt.figure()
        figure(figsize=(w,h), dpi=80)
        for i in range(0, self.nmode):
            plt.subplot(1, self.nmode, i+1)
            plt.plot(fac[i])
            if original:
                plt.title(f'{i}th row of temporal factor')
            else:
                plt.title(f'{i}th row of temporal factor after optimizing')

    def buildModelHWES(self):
        """Implement HWES and show results in graphs"""
        minval = abs(min(self.train))
        self.train = self.train + minval + 0.01
        model = HOLTWINTER(self.train, seasonal_periods=self.seasonp, trend='add',seasonal='mul')
        fit = model.fit(optimized=True, use_brute=True)
        print(fit.summary())
        forecast = fit.forecast(steps=self.n)

        return forecast

    def printHWESresults(self):
        """Print results of HWES forecasting"""
        forecast = self.buildModelHWES()

        ticks = range(len(self.train) + len(self.test))
        fig = plt.figure()
        past, = plt.plot(ticks[-self.zoom:-self.n+1], self.train[-self.zoom:-self.n+1], 'b.-', label='past')
        future, = plt.plot(ticks[-self.n:], self.train[-self.n:], 'r.-', label='original')
        predicted, = plt.plot(ticks[-self.n:], forecast, 'g.-', label='predicted')
        plt.legend()
        fig.show()
        plt.show()
