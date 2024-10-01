import pandas as pd

class DataManager:
    def __init__(self, symbols):
        self.symbols = symbols
        self.data = {symbol: pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close']) for symbol in symbols}
        self.max_data_points = 1000

    def update(self, symbol, tick):
        new_data = pd.DataFrame({
            'timestamp': [tick['epoch']],
            'open': [tick['quote']],
            'high': [tick['quote']],
            'low': [tick['quote']],
            'close': [tick['quote']]
        })
        self.data[symbol] = pd.concat([self.data[symbol], new_data]).tail(self.max_data_points)

    def get_close_prices(self, symbol):
        return self.data[symbol]['close'].tolist()

    def get_ohlc_data(self, symbol):
        return self.data[symbol]

