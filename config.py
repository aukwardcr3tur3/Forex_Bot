import os, time, sys
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        
        # Check for APP_ID after loading environment variables
        if os.getenv("APP_ID") is None or len(os.getenv("APP_ID")) == 0:
            print("APP_ID environment variable is not set")
            time.sleep(2)
            sys.exit('Exiting...')

        # Load other configuration variables
        self.API_TOKEN = os.getenv('DERIV_API_TOKEN')
        self.EndPoint = f'wss://frontend.binaryws.com/websockets/v3?l=EN&app_id={os.getenv("APP_ID")}'
        self.SYMBOLS = ['frxEURUSD', 'frxGBPUSD', 'frxUSDJPY']
        self.TIMEFRAME = '5m'
        self.RISK_PERCENTAGE = 0.01
        self.MAX_TRADES_PER_SYMBOL = 2
        self.STOP_LOSS_PIPS = 20
        self.TAKE_PROFIT_PIPS = 40
        self.TRAILING_STOP_PIPS = 15
        self.BACKTEST_START_DATE = '2023-01-01'
        self.BACKTEST_END_DATE = '2023-06-30'

