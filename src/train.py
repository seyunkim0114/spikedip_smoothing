"""
Tensor Forecast Using Smoothing Techniques
Authors: Seyun Kim(seyun0114kim@gmail.com), U Kang (ukang@snu.ac.kr)
Data Mining Lab., Seoul National University

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.
"""

from model import *

def create_model(X, tmode, nmode1, nmode2, nmode, window, rank, tlength, seasonp, horizon, f_window):
    """
    Create time factor smoothing model

    @param X
        target tensor
    @param tmode
        temporal factor obtained by decomposing a target tensor
    @param nmode1
        first non-temporal factor
    @param nmode2
        second non-temporal factor
    @param window
        width of Gaussian kernel and spike/dip reduce kernel
    @param rank
        rank of tensor decomposition
    @param tlength
        length of time dimension (K)
    @param seasonp
        seasonality of temporal factor rows
    @param horizon
        number of time steps from the end of dataset to the forecast
    @param f_window
        length of dataset to learn Holt-Winters exponential smoothing

    Returns tensor forecast model
    """
    model = smooth_tfactor(X, tmode, nmode1, nmode2, nmode, window, rank, tlength, seasonp, horizon, f_window)
    opt = torch.optim.SGD(model.parameters(),lr=0.001)
    return model, opt

def train_model(train, val, epochs, model, opt, nmode, n, val_nmode1, val_nmode2):
    """
    Train the model

    @param epochs
        number of epochs to train
    @param model
        tensor forecasting model
    @param opt
        optimization method
    @param nmode
        number of total modes
    @param n
    """
    for epoch in range(epochs):
        opt.zero_grad()
        loss = get_loss(model, train)
        loss.requires_grad = True
        loss.backward()
        opt.step()

        if epoch%10 == 0:
            model.eval()
            lossval = get_loss(model, val)
            print(f'Epoch: {epoch} \tTrain loss: {loss}\tVal loss: {lossval}\tVal acc:')

    return model

