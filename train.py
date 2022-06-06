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



dataset = MeteoDataset("C:\\Users\\AnnaMaria\\Desktop\\Fizyka Systemu Ziemi\\data_pm10.csv")
data = dataset.df.values