import pandas as pd
import numpy as np
from deriv_api import DerivAPI

class Backtester:
    def __init__(self, config, strategy_manager):
        self.config = config
        self.strategy_manager = strategy_manager
        self.api = DerivAPI(app_id=1089)

    async def run(self):
        results = {}
        for symbol in self.config.SYMBOLS:
            historical_data = await self.fetch_historical_data(symbol)
            results[symbol] = self.backtest_symbol(symbol, historical_data)
        return results

    async def fetch_historical_data(self, symbol):
        candles = await self.api.ticks_history({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": 5000,
            "end": self.config.BACKTEST_END_DATE,
            "start": self.config.BACKTEST_START_DATE,
            "style": "candles"
        })
        return pd.DataFrame(candles['candles'])

    def backtest_symbol(self, symbol, data):
        balance = 10000
        trades = []

        for i in range(len(data)):
            if self.strategy_manager.should_enter_trade(symbol, data[:i+1]):
                entry_price = data.iloc[i]['close']
                stop_loss = entry_price - (self.config.STOP_LOSS_PIPS / 10000)
                take_profit = entry_price + (self.config.TAKE_PROFIT_PIPS / 10000)
                
                for j in range(i+1, len(data)):
                    current_price = data.iloc[j]['close']
                    if current_price <= stop_loss:
                        profit = (stop_loss - entry_price) * 10000
                        trades.append(profit)
                        balance += profit
                        break
                    elif current_price >= take_profit:
                        profit = (take_profit - entry_price) * 10000
                        trades.append(profit)
                        balance += profit
                        break

