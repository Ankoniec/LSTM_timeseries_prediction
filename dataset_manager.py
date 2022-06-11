import pandas as pd
import numpy as np

    

class MeteoDataset:

    def __init__(self, filename:str) -> object:
        self.df = self.read_data(filename)
        self.N = len(self.df)


    def read_data(self, filename:str) -> pd.DataFrame:
        meteo = pd.read_csv(filename, sep=';')
        meteo = meteo.dropna(axis=0)
        meteo = meteo.set_index('time',drop=True)
        return meteo


    def __len__(self) -> int:
        return self.N



def create_sequences(data:np.ndarray, seq_length:int):
    X_train = []
    y_pred = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        X_train.append(_x)
        y_pred.append(_y)

    return np.array(X_train), np.array(y_pred)