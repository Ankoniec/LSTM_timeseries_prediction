import pandas as pd

    

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
