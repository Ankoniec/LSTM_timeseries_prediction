import requests
import datetime
import pandas as pd


class OpenWeatherDataset:

    def __init__(self, lat:float, lon:float, start_date, end_date):
        self.data = self.read_data(lat,lon,start_date,end_date)
        self.N = len(self.data['list'])
        self.data_df = self.json_to_dataframe(self.data)
        self.columns = list(self.data_df.columns)

    
    def read_data(self, lat, lon, start_date, end_date):
        app_key='733d7f6f0e407d1a6e6278fc26ff16df'
        unix_start_date = (int)(datetime.datetime.timestamp(start_date))
        unix_end_date = (int)(datetime.datetime.timestamp(end_date))
        url = 'http://api.openweathermap.org/data/2.5/air_pollution/history?lat={}&lon={}&start={}&end={}&appid={}'.format(
            lat, lon, unix_start_date, unix_end_date, app_key)
        result = requests.get(url)
        return result.json()

    
    def json_to_dataframe(self, data):
        data_dict = {datetime.datetime.fromtimestamp(data['list'][i]['dt']): data['list'][i]['components'] for i in range(self.N)}
        df = pd.DataFrame.from_dict(data_dict)
        df = df.transpose()
        return df

    
    def plot(self):
        self.data_df.plot(figsize=(20,10))

    
    def __len__(self):
        return self.N

    

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
