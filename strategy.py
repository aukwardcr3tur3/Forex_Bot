import talib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class StrategyManager:
    def __init__(self):
        self.strategies = [
            RSIStrategy(),
            MACDStrategy(),
            MLStrategy()
        ]

    def should_enter_trade(self, symbol, data_manager):
        return any(strategy.should_enter_trade(symbol, data_manager) for strategy in self.strategies)

class RSIStrategy:
    def __init__(self):
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30

    def should_enter_trade(self, symbol, data_manager):
        close_prices = np.array(data_manager.get_close_prices(symbol))
        
        if len(close_prices) < self.rsi_period:
            return False

        rsi = talib.RSI(close_prices, timeperiod=self.rsi_period)
        
        return rsi[-1] < self.rsi_oversold

class MACDStrategy:
    def __init__(self):
        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9

    def should_enter_trade(self, symbol, data_manager):
        close_prices = np.array(data_manager.get_close_prices(symbol))
        
        if len(close_prices) < self.slow_period:
            return False

        macd, signal, _ = talib.MACD(close_prices, fastperiod=self.fast_period, slowperiod=self.slow_period, signalperiod=self.signal_period)
        
        return macd[-1] > signal[-1] and macd[-2] <= signal[-2]

class MLStrategy:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.trained = False

    def should_enter_trade(self, symbol, data_manager):
        if not self.trained:
            self.train(data_manager)

        features = self.extract_features(symbol, data_manager)
        prediction = self.model.predict([features])[0]
        current_price = data_manager.get_close_prices(symbol)[-1]
        
        return prediction > current_price * 1.001  # Predict 0.1% increase

    def train(self, data_manager):
        X, y = self.prepare_training_data(data_manager)
        self.model.fit(X, y)
        self.trained = True

    def prepare_training_data(self, data_manager):
        X, y = [], []
        for symbol in data_manager.symbols:
            prices = data_manager.get_close_prices(symbol)
            for i in range(len(prices) - 11):
                features = self.extract_features(symbol, data_manager, i)
                target = (prices[i+10] - prices[i+9]) / prices[i+9]  # Next candle's return
                X.append(features)
                y.append(target)
        return np.array(X), np.array(y)

    def extract_features(self, symbol, data_manager, offset=0):
        prices = data_manager.get_close_prices(symbol)[offset:offset+10]
        returns = np.diff(prices) / prices[:-1]
        return [
            np.mean(returns),
            np.std(returns),
            (prices[-1] - prices[0]) / prices[0],  # 10-period return
            talib.RSI(np.array(prices), timeperiod=9)[-1],
            *talib.BBANDS(np.array(prices), timeperiod=5, nbdevup=2, nbdevdn=2)[0]
        ]

