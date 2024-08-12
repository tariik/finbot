import pandas as pd
import numpy as np
from datetime import datetime

import talib

from core.data_process.scaler import Scaler


class DataProcess:
    def __init__(
            self,
            path: str,
            symbol: str,
            tech_indicator_list

    ):
        self.scaler = None
        self.dataframe = None
        self.path = path
        self.symbol = symbol,
        self.tech_indicator_list = tech_indicator_list

    def load_dataset(self) -> pd.DataFrame:
        """
        load csv dataset from path
        :return: (df) pandas dataframe
        """
        # _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
        self.dataframe = pd.read_csv(self.path)

    def run(self):
        self.load_dataset()
        self.clean_data()
        self.add_technical_indicator()


        price_array, tech_array = self.df_to_array()
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        return price_array, tech_array

    def clean_data(self):
        if "date" in self.dataframe.columns.values.tolist():
            self.dataframe.rename(columns={"date": "time"}, inplace=True)
        if "datetime" in self.dataframe.columns.values.tolist():
            self.dataframe.rename(columns={"datetime": "time"}, inplace=True)
        # self.dataframe.dropna(inplace=True)
        self.dataframe = self.dataframe.ffill().bfill()

        # self.dataframe["time"] = [datetime.fromtimestamp(float(time) / 1000) for time in df["time"]]
        self.dataframe["open"] = self.dataframe["open"].astype(np.float64)
        self.dataframe["high"] = self.dataframe["high"].astype(np.float64)
        self.dataframe["low"] = self.dataframe["low"].astype(np.float64)
        self.dataframe["close"] = self.dataframe["close"].astype(np.float64)
        self.dataframe["volume"] = self.dataframe["volume"].astype(np.float64)
        # adjusted_close: adjusted close price
        if "adjusted_close" not in self.dataframe.columns.values.tolist():
            self.dataframe["adjusted_close"] = self.dataframe["close"]
        self.dataframe.sort_values(by=["time"], inplace=True)
        self.dataframe = self.dataframe[
            [
                "time",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
            ]
        ]

    def add_technical_indicator(self):
        """
        Add technical indicators to the dataframe using the talib library.
        """
        # Calculate indicators for each technical indicator in the list
        for indicator in self.tech_indicator_list:
            if indicator == 'macd':
                self.dataframe['macd'], self.dataframe['macd_signal'], self.dataframe['macd_hist'] = talib.MACD(
                    self.dataframe['close'],
                    fastperiod=12,
                    slowperiod=26,
                    signalperiod=9
                )
            elif indicator == 'rsi':
                self.dataframe['rsi'] = talib.RSI(self.dataframe['close'], timeperiod=14)
            elif indicator == 'cci':
                self.dataframe['cci'] = talib.CCI(
                    self.dataframe['high'],
                    self.dataframe['low'],
                    self.dataframe['close'],
                    timeperiod=14
                )
            elif indicator == 'dx':
                self.dataframe['dx'] = talib.DX(
                    self.dataframe['high'],
                    self.dataframe['low'],
                    self.dataframe['close'],
                    timeperiod=14
                )

        self.dataframe.dropna(inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

    def df_to_array(self):
        """
        Convert the processed DataFrame into NumPy arrays for price and technical indicators.
        :return: Tuple of NumPy arrays (price_arr, tech_arr).
        """
        # List of columns to include in the technical indicators array
        common_tech_indicator_list = [
            i for i in self.tech_indicator_list if i in self.dataframe.columns.values.tolist()
        ]

        # Ensure that 'open', 'close', 'high', 'low', and 'volume' are included in the tech_array
        basic_columns = ['open', 'high', 'low', 'close', 'volume']
        tech_indicator_columns = basic_columns + common_tech_indicator_list

        # Create the price array using the 'close' prices
        price_arr = self.dataframe['close'].values.reshape(-1, 1)

        # Normalize the data (includes normalization of the basic columns and technical indicators)
        self.normalize_data()

        # Create the tech array including open, high, low, close, volume, and technical indicators
        tech_arr = self.dataframe[tech_indicator_columns].values

        print("Successfully transformed into array")
        return price_arr, tech_arr

    def normalize_data(self):
        """Normalize the data using the Scaler class."""
        self.scaler = Scaler(columns=self.dataframe.columns.difference(['time']))
        self.dataframe = self.scaler.fit_transform(self.dataframe)


if __name__ == "__main__":
    path = 'D:\\Lab\\quant\\code\\finbot\\data\\btcusd_2022-01-02_to_2022-07-03.csv'
    technical_indicator_list = ['macd', 'rsi', 'cci','dx']
    process = DataProcess(path, 'BTCUSDT', technical_indicator_list)
    price_array, tech_array = process.run()
    print(tech_array)
