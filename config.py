import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        self.API_TOKEN = os.getenv('DERIV_API_TOKEN')
        self.SYMBOLS = ['frxEURUSD', 'frxGBPUSD', 'frxUSDJPY']
        self.TIMEFRAME = '5m'
        self.RISK_PERCENTAGE = 0.01
        self.MAX_TRADES_PER_SYMBOL = 2
        self.STOP_LOSS_PIPS = 20
        self.TAKE_PROFIT_PIPS = 40
        self.TRAILING_STOP_PIPS = 15
        self.BACKTEST_START_DATE = '2023-01-01'
        self.BACKTEST_END_DATE = '2023-06-30'

