"""
Tensor Forecast Using Smoothing Techniques
Authors: Seyun Kim(seyun0114kim@gmail.com), U Kang (ukang@snu.ac.kr)
Data Mining Lab., Seoul National University

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.
"""

from train import *

# Parameters
data = "radar"
PATH = rf'C:\Users\seyun\Google 드라이브\DMLab\DLab\forecast\spike-dip-reducing\tensor_smoothing_2\data\{data}'

rank = 5
tmode = 0
nmode = 3
window = 6
ndim = 3

# Hyperparameters
epochs = 10
sigma = 1.5
seasonp = 9
n = 5
zoom = 30
horizon = 3
f_window = 500

data_train, data_val, data_test = get_dataset(PATH)

temp_train, nmode1_train, nmode2_train = get_CPfac(data_train, rank)
temp_test, nmode1_test, nmode2_test = get_CPfac(data_test, rank)
temp_val, nmode1_val, nmode2_val = get_CPfac(data_val, rank)

tlength = temp_train.shape[1]
model, opt = create_model(data_train, tmode, nmode1_train, nmode2_train, nmode,
                          window, rank, tlength, seasonp, horizon, f_window)
train_model(temp_train, temp_val, epochs, model, opt, nmode, n)

