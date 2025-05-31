import os, time, sys, logging
import pandas as pd
from configparser import ConfigParser
from pathlib import Path
from typing import Any
#Dict, List, Tuple, Optional

logger = logging.getLogger("Config")

class Config:
    def __init__(self, config_file: str = "config/config.ini"):
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
        self.app_id = self.config['DEFAULT'].get('APP_ID', '')
        if not self.app_id:
            logger.error("APP_ID is not set in config.ini")
            print("Error: APP_ID is not set in config.ini")
            time.sleep(2)
            sys.exit(1)

        # Set attributes from DEFAULT section
        self.deriv_api_token = self.config['DEFAULT'].get('DERIV_API_TOKEN', '')
        self.endpoint = self.config['DEFAULT'].get('EndPoint', f'wss://ws.derivws.com/websockets/v3?app_id={self.app_id}')

        # Set attributes from Settings section
        symbols_str = self.config["Settings"].get("SYMBOLS")
        self.symbols = [s.strip() for s in symbols_str.split(",")]
        self.timeframe: str = self.config['Settings'].get('TIMEFRAME', '15m')
        self.risk_percentage: float = self.config['Settings'].getfloat('RISK_PERCENTAGE', 0.01)
        self.max_trades_per_symbol: int = self.config['Settings'].getint('MAX_TRADES_PER_SYMBOL', 2)
        self.stop_loss_pips: int = self.config['Settings'].getint('STOP_LOSS_PIPS', 20)
        self.take_profit_pips: int = self.config['Settings'].getint('TAKE_PROFIT_PIPS', 40)
        self.trailing_stop_pips: int = self.config['Settings'].getint('TRAILING_STOP_PIPS', 15)
        self.historical_days: int = self.config['Settings'].getint('HISTORICAL_DAYS', 30)
        self.max_data_points: int = self.config['Settings'].getint('MAX_DATA_POINTS', 5000)
        self.backtest_start_date = pd.to_datetime(self.config['Settings'].get('BACKTEST_START_DATE', '2023-01-01'), utc=True)
        self.backtest_end_date = pd.to_datetime(self.config['Settings'].get('BACKTEST_END_DATE', '2023-06-30'),utc=True)



        # Set attributes from Runtime section
        self.max_runtime: int = self.config['Runtime'].getint('MAX_RUNTIME', 3600)
        self.starvation_threshold: float = self.config['Runtime'].getfloat('STARVATION_THRESHOLD', 0.5)
        self.watchdog_interval: int = self.config['Runtime'].getint('WATCHDOG_INTERVAL', 5)

        # Set attributes from ML section
        self.model_dir: str = self.config['ML'].get('MODEL_DIR', 'models/')
        self.ml_n_estimators: int = self.config['ML'].getint('ML_N_ESTIMATORS', 200)
        self.ml_max_depth: int = self.config['ML'].getint('ML_MAX_DEPTH', 10)
        self.ml_min_samples: int = self.config['ML'].getint('ML_MIN_SAMPLES', 100)
        self.ml_window_size: int = self.config['ML'].getint('ML_WINDOW_SIZE', 20)
        self.ml_threshold: float = self.config['ML'].getfloat('ML_THRESHOLD', 0.001)
        self.seed: int = self.config['ML'].getint('SEED', 42)

        # Set attributes from RSI section
        self.rsi_period: int = self.config['RSI'].getint('RSI_PERIOD', 14)
        self.rsi_overbought: int = self.config['RSI'].getint('RSI_OVERBOUGHT', 70)
        self.rsi_oversold: int = self.config['RSI'].getint('RSI_OVERSOLD', 30)

        # Set attributes from MACD section
        self.macd_fast_period: int = self.config['MACD'].getint('MACD_FAST_PERIOD', 12)
        self.macd_slow_period: int = self.config['MACD'].getint('MACD_SLOW_PERIOD', 26)
        self.macd_signal_period: int = self.config['MACD'].getint('MACD_SIGNAL_PERIOD', 9)

        # Log MAX_DATA_POINTS to confirm
        logger.info(f"MAX_DATA_POINTS set to {self.max_data_points}")

        # Validate configuration
        self.validate()

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value by key with an optional default."""
        return getattr(self, key, default)

    def validate(self) -> None:
        """Validate required configuration attributes."""
        required = [
            'app_id', 'deriv_api_token', 'symbols', 'max_data_points',
            'rsi_period', 'rsi_overbought', 'rsi_oversold',
            'macd_fast_period', 'macd_slow_period', 'macd_signal_period'
        ]
        for attr in required:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                logger.error(f"Missing required config attribute: {attr}")
                raise ValueError(f"Missing required config attribute: {attr}")
