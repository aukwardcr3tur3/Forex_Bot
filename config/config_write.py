#config.add_section('Settings')
#config.set('Settings', 'SYMBOLS', 'frxAUDCAD,frxEURUSD,frxGBPUSD') #, 'frxAUDCHF', 'frxAUDJPY', 'frxAUDNZD', 'frxAUDUSD', 'frxEURAUD', 'frxEURCAD', 'frxEURCHF', 'frxEURGBP', 'frxEURJPY', 'frxEURNZD', 'frxEURUSD', 'frxGBPAUD', 'frxGBPCAD', 'frxGBPCHF', 'frxGBPJPY', 'frxGBPUSD', 'frxNZDUSD', 'frxUSDCAD', 'frxUSDCHF', 'frxUSDJPY', 'frxUSDMXN', 'frxUSDNOK', 'frxUSDPLN', 'frxUSDSEK', 'frxXAGUSD', 'frxXAUUSD', 'frxXPDUSD', 'frxXPTUSD']
#config.set('Settings', 'TIMEFRAME', '5m')
#config.set('Settings', 'RISK_PERCENTAGE', '0.01')
#config.set('Settings', 'MAX_TRADES_PER_SYMBOL', '2')
#config.set('Settings', 'STOP_LOSS_PIPS', '20')
#config.set('Settings', 'TAKE_PROFIT_PIPS', '40')
#config.set('Settings', 'TRAILING_STOP_PIPS', '15')
#config.set('Settings', 'HISTORICAL_DAYS', '90')
#config.set('Settings', 'BACKTEST_START_DATE', '2023-01-01')
#config.set('Settings', 'BACKTEST_END_DATE', '2023-06-30')

import os
import sys
import time
import logging
from configparser import ConfigParser
from pathlib import Path
from typing import List, Optional, Any

# Set up logging for config debugging
logger = logging.getLogger("Config")

def create_default_config(config_file: Path) -> None:
    """Create a default config.ini file with predefined settings."""
    config = ConfigParser()

    # Set DEFAULT section values
    config['DEFAULT'] = {
        'DERIV_API_TOKEN': '',
        'APP_ID': '',
        'EndPoint': 'wss://ws.derivws.com/websockets/v3?app_id='
    }

    # Add Settings section
    config['Settings'] = {
        'SYMBOLS': 'R_50',
        'TIMEFRAME': '15m',
        'RISK_PERCENTAGE': '0.01',
        'MAX_TRADES_PER_SYMBOL': '2',
        'STOP_LOSS_PIPS': '20',
        'TAKE_PROFIT_PIPS': '40',
        'TRAILING_STOP_PIPS': '15',
        'HISTORICAL_DAYS': '30',
        'MAX_DATA_POINTS': '5000',
        'BACKTEST_START_DATE': '2023-01-01',
        'BACKTEST_END_DATE': '2023-06-30'
    }

    # Add Runtime section
    config['Runtime'] = {
        'MAX_RUNTIME': '3600',
        'STARVATION_THRESHOLD': '0.5',
        'WATCHDOG_INTERVAL': '5'
    }

    # Add ML section
    config['ML'] = {
        'MODEL_DIR': 'models/',
        'ML_N_ESTIMATORS': '200',
        'ML_MAX_DEPTH': '10',
        'ML_MIN_SAMPLES': '100',
        'ML_WINDOW_SIZE': '20',
        'ML_THRESHOLD': '0.001',
        'SEED': '42'
    }

    # Add RSI section
    config['RSI'] = {
        'RSI_PERIOD': '14',
        'RSI_OVERBOUGHT': '70',
        'RSI_OVERSOLD': '30'
    }

    # Add MACD section
    config['MACD'] = {
        'MACD_FAST_PERIOD': '12',
        'MACD_SLOW_PERIOD': '26',
        'MACD_SIGNAL_PERIOD': '9'
    }

    # Ensure the config directory exists
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Write the config file
    with config_file.open('w') as f:
        config.write(f)
    logger.info(f'{config_file} was created successfully.')

class Config:
    """Configuration class for the Forex trading bot."""

    def __init__(self, config_file: str = 'config.ini') -> None:
        """Initialize the Config class by reading settings from config.ini."""
        self.config_file = Path(config_file)
        logger.debug(f"Loading config from {self.config_file}")

        # Create config file if it doesn't exist
        if not self.config_file.exists():
            logger.warning(f"Config file {self.config_file} does not exist, creating default")
            create_default_config(self.config_file)

        # Read the config file
        self.config = ConfigParser()
        if not self.config.read(self.config_file):
            logger.error(f"Failed to read configuration file: {self.config_file}")
            raise FileNotFoundError(f"Failed to read configuration file: {self.config_file}")

        # Log all settings for debugging
        for section in self.config.sections():
            logger.debug(f"Section [{section}]: {dict(self.config[section])}")

        # Validate APP_ID
        self.APP_ID = self.config['DEFAULT'].get('APP_ID', '')
        if not self.APP_ID:
            logger.error("APP_ID is not set in config.ini")
            print("Error: APP_ID is not set in config.ini")
            time.sleep(2)
            sys.exit(1)

        # Set attributes from DEFAULT section
        self.DERIV_API_TOKEN = self.config['DEFAULT'].get('DERIV_API_TOKEN', '')
        self.ENDPOINT = self.config['DEFAULT'].get('EndPoint', f'wss://ws.derivws.com/websockets/v3?app_id={self.APP_ID}')

        # Set attributes from Settings section
        symbols_str = self.config['Settings'].get('SYMBOLS', '')
        self.SYMBOLS: List[str] = [s.strip() for s in symbols_str.split(',') if s.strip()]
        self.TIMEFRAME: str = self.config['Settings'].get('TIMEFRAME', '15m')
        self.RISK_PERCENTAGE: float = self.config['Settings'].getfloat('RISK_PERCENTAGE', 0.01)
        self.MAX_TRADES_PER_SYMBOL: int = self.config['Settings'].getint('MAX_TRADES_PER_SYMBOL', 2)
        self.STOP_LOSS_PIPS: int = self.config['Settings'].getint('STOP_LOSS_PIPS', 20)
        self.TAKE_PROFIT_PIPS: int = self.config['Settings'].getint('TAKE_PROFIT_PIPS', 40)
        self.TRAILING_STOP_PIPS: int = self.config['Settings'].getint('TRAILING_STOP_PIPS', 15)
        self.HISTORICAL_DAYS: int = self.config['Settings'].getint('HISTORICAL_DAYS', 30)
        self.MAX_DATA_POINTS: int = self.config['Settings'].getint('MAX_DATA_POINTS', 5000)
        self.BACKTEST_START_DATE: str = self.config['Settings'].get('BACKTEST_START_DATE', '2023-01-01')
        self.BACKTEST_END_DATE: str = self.config['Settings'].get('BACKTEST_END_DATE', '2023-06-30')

        # Set attributes from Runtime section
        self.MAX_RUNTIME: int = self.config['Runtime'].getint('MAX_RUNTIME', 3600)
        self.STARVATION_THRESHOLD: float = self.config['Runtime'].getfloat('STARVATION_THRESHOLD', 0.5)
        self.WATCHDOG_INTERVAL: int = self.config['Runtime'].getint('WATCHDOG_INTERVAL', 5)

        # Set attributes from ML section
        self.MODEL_DIR: str = self.config['ML'].get('MODEL_DIR', 'models/')
        self.ML_N_ESTIMATORS: int = self.config['ML'].getint('ML_N_ESTIMATORS', 200)
        self.ML_MAX_DEPTH: int = self.config['ML'].getint('ML_MAX_DEPTH', 10)
        self.ML_MIN_SAMPLES: int = self.config['ML'].getint('ML_MIN_SAMPLES', 100)
        self.ML_WINDOW_SIZE: int = self.config['ML'].getint('ML_WINDOW_SIZE', 20)
        self.ML_THRESHOLD: float = self.config['ML'].getfloat('ML_THRESHOLD', 0.001)
        self.SEED: int = self.config['ML'].getint('SEED', 42)

        # Set attributes from RSI section
        self.RSI_PERIOD: int = self.config['RSI'].getint('RSI_PERIOD', 14)
        self.RSI_OVERBOUGHT: int = self.config['RSI'].getint('RSI_OVERBOUGHT', 70)
        self.RSI_OVERSOLD: int = self.config['RSI'].getint('RSI_OVERSOLD', 30)

        # Set attributes from MACD section
        self.MACD_FAST_PERIOD: int = self.config['MACD'].getint('MACD_FAST_PERIOD', 12)
        self.MACD_SLOW_PERIOD: int = self.config['MACD'].getint('MACD_SLOW_PERIOD', 26)
        self.MACD_SIGNAL_PERIOD: int = self.config['MACD'].getint('MACD_SIGNAL_PERIOD', 9)

        # Log MAX_DATA_POINTS to confirm
        logger.info(f"MAX_DATA_POINTS set to {self.MAX_DATA_POINTS}")

        # Validate configuration
        self.validate()

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value by key with an optional default."""
        return getattr(self, key, default)

    def validate(self) -> None:
        """Validate required configuration attributes."""
        required = [
            'APP_ID', 'DERIV_API_TOKEN', 'SYMBOLS', 'MAX_DATA_POINTS',
            'RSI_PERIOD', 'RSI_OVERBOUGHT', 'RSI_OVERSOLD',
            'MACD_FAST_PERIOD', 'MACD_SLOW_PERIOD', 'MACD_SIGNAL_PERIOD'
        ]
        for attr in required:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                logger.error(f"Missing required config atributo: {attr}")
                raise ValueError(f"Missing required config atributo: {attr}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    config = Config()
    print(f"Loaded configuration from {config.config_file}")
    print(f"Symbols: {config.SYMBOLS}")
    print(f"API Endpoint: {config.ENDPOINT}")
    print(f"MAX_DATA_POINTS: {config.MAX_DATA_POINTS}")
    print(f"Runtime Settings: MAX_RUNTIME={config.MAX_RUNTIME}, STARVATION_THRESHOLD={config.STARVATION_THRESHOLD}")
    print(f"ML Settings: MODEL_DIR={config.MODEL_DIR}, ML_N_ESTIMATORS={config.ML_N_ESTIMATORS}")
    print(f"RSI Settings: RSI_PERIOD={config.RSI_PERIOD}, RSI_OVERBOUGHT={config.RSI_OVERBOUGHT}")
    print(f"MACD Settings: MACD_FAST_PERIOD={config.MACD_FAST_PERIOD}, MACD_SLOW_PERIOD={config.MACD_SLOW_PERIOD}")
