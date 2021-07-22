import numpy as np
import tensorly as tl
import torch
import torch.nn as nn
from torch.nn import functional as F

from tensorly import decomposition
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from load_dataset import *

def gen_random(n):
    '''Generate random 1D tensor with length n'''
    temp = np.random.rand(n)
    return torch.FloatTensor(temp)


class sp_reduce(nn.Module):
    '''Implement spike and dips reducing smoothing on temporal factor'''
    def __init__(self,window,rank,tlength):
        super().__init__()
        self.window = window 
  # one row of temporal factor
        self.rank = rank
        self.tlength = tlength
        
        
    def spike_dips(self,factor):
        '''Return rate of change of consecutive time steps'''
        w = torch.ones(1,1,self.window)
        
        moving_average = F.conv1d(factor.view(1,1,tlength),w,padding=self.window//2)/self.window
        moving_average = moving_average.view(moving_average.shape[2])
        change = []
        change.append(0)
        
        for n in range(moving_average.shape[0]-1):
            change.append((moving_average[n]-moving_average[n+1])**2)
            
        change = torch.FloatTensor(change)
        change = change/torch.sum(change)
        return change
    
    def forward(self,factor):
        '''Implement spike and dips reducing smoothing'''
        change_rate = self.spike_dips(factor)
        return (1-change_rate).pow(5)*factor
    

# Implement Gaussian smoothing regularization
class gaussian_smoothing(nn.Module):
    '''Implement Gaussian smoothing on temporal factor'''
    def __init__(self,window):
        super().__init__()
        self.window = window
        self.weight = self.gaussian()
#         self.tfactor = self.tfactor()
        
    def gaussian(self):
        '''Construct Gaussian kernel and return Gaussian weights'''
        x = np.arange(-self.window,self.window+1)
        phi_g = np.exp(-x/0.5)
        phi_g = phi_g/phi_g.sum()
#         phi_gc = torch.FloatTensor(phi_g.reshape(1,1,window*2,1))
        phi_g = torch.FloatTensor(phi_g)
        return phi_g.view(1,1,x.shape[0])
    
    def forward(self,tfactor):
        '''Implement Gaussian smoothing'''
#         row,col = tmode.shape
#         tmode_c = torch.FloatTensor(tmode.reshape(1,1,row,col))
        tlength = tfactor.shape[0]
        factor = tfactor.reshape(1,1,tlength)
        factor = torch.FloatTensor(factor)
        smoothed = F.conv1d(factor, self.weight, padding=self.window)
        return smoothed.view(tlength)

class smooth_tfactor(nn.Module):
    '''Implement smoothing on temporal factor'''
    def __init__(self,tmode,nmode,window,rank,tlength):
        super().__init__()
        
        self.tmode = tmode # temp factor
        self.window = window
        self.nmode = nmode
        self.rank = rank
        self.tlength = tlength
        
        self.tfactor = nn.Parameter(gen_random(self.tlength))
        self.smooth = gaussian_smoothing(self.window)
        self.reduce = sp_reduce(self.window, self.rank,self.tlength)  
        
    def smooth_gaussian(self):
        '''Call Gaussian smoothing class and implement it'''
        smoothed = self.smooth(self.tfactor)
        sloss = (smoothed-self.tfactor).pow(2)
        return sloss.sum()

    def spike_dips(self):
        '''Call spike and dips reducing smoothing class and implement it'''
        reduced = self.reduce(self.tfactor)
        rloss = (reduced-self.tfactor).pow(2)
        return rloss.sum()

    def forward(self):
        '''Reconstruct the tensor with nonzero's indices'''

        return self.tfactor

def get_loss(model,temp):
    '''Compute total loss'''
    gloss = model.smooth_gaussian()
    sloss = model.spike_dips()
    loss = (temp - model()).sum().pow(2) + gloss 
    return gloss

