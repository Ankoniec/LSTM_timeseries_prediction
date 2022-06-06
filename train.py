import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

from dataset_manager import MeteoDataset
from taylor_diagram import TaylorDiagram
from model import LSTM


def create_sequences(data, seq_length):
    X_train = []
    y_pred = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        X_train.append(_x)
        y_pred.append(_y)

    return np.array(X_train), np.array(y_pred)


dataset = MeteoDataset("C:\\Users\\AnnaMaria\\Desktop\\Fizyka Systemu Ziemi\\data_pm10.csv")
data = dataset.df.values
sc = MinMaxScaler() 
training_data = sc.fit_transform(data)

seq_length = 16
x, y = create_sequences(training_data, seq_length)
ratio = 0.8
N_train = (int)(ratio*len(x))

X_train = x[:N_train]
X_test = x[N_train:]
y_train = y[:N_train]
y_test = y[N_train:]

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))
trainX = Variable(torch.Tensor(np.array(X_train)))
trainY = Variable(torch.Tensor(np.array(y_train)))
testX = Variable(torch.Tensor(np.array(X_test)))
testY = Variable(torch.Tensor(np.array(y_test)))


num_epochs = 10000
learning_rate = 0.01

num_classes = 1
input_size = 1
hidden_size = 7
num_layers = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)