import asyncio
import logging
import pandas as pd
import numpy as np
import talib
from typing import Optional, Dict, List
from config import Config
from data_manager import DataManager
from portfolio_manager import PortfolioManager
from MLStrat import MLStrategy
from base_strategy import BaseStrategy
from stratestic.strategies._mixin import StrategyMixin
from stratestic.backtesting.helpers.evaluation import SIDE

class RSIStrategy(StrategyMixin):
    def __init__(self, config: Config, data_manager: DataManager, logger: logging.Logger, symbol: str, **kwargs):
        super().__init__()
        self.config = config
        self.logger = logger
        self.data_manager = data_manager
        self.symbol = symbol
        self.rsi_period = config.rsi_period
        self.rsi_overbought = config.rsi_overbought
        self.rsi_oversold = config.rsi_oversold
        self._validate_config()
        
        # Get the DataFrame for the specific symbol
        data = data_manager.get_snapshot(symbol)
        if data is None:
            self.logger.warning(f"No data available for symbol {symbol}, initializing with empty DataFrame")
            data = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
            data.index.name = 'timestamp'

        self.logger.debug(f"RSIStrategy initialized for {symbol}")
        StrategyMixin.__init__(self, data, **kwargs)

    def _validate_config(self):
        required = ['rsi_period', 'rsi_overbought', 'rsi_oversold']
        for param in required:
            if not hasattr(self.config, param):
                raise ValueError(f"Missing config parameter: {param}")

    async def should_enter_trade(self, symbol: str, data_manager: DataManager) -> bool:
        try:
            if symbol != self.symbol:
                self.logger.warning(f"Symbol mismatch: strategy for {self.symbol}, called with {symbol}")
                return False
            snapshot = data_manager.get_snapshot(symbol)
            if snapshot is None or len(snapshot) < self.rsi_period:
                self.logger.debug(f"Insufficient data for RSI on {symbol}")
                return False
            close_prices = snapshot['close'].values
            rsi = talib.RSI(close_prices, timeperiod=self.rsi_period)[-1]
            self.logger.debug(f"RSI for {symbol}: {rsi:.2f}")
            return rsi < self.rsi_oversold
        except Exception as e:
            self.logger.error(f"Error in RSI enter trade for {symbol}: {e}", exc_info=True)
            return False

    async def should_exit_trade(self, symbol: str, data_manager: DataManager) -> bool:
        try:
            if symbol != self.symbol:
                self.logger.warning(f"Symbol mismatch: strategy for {self.symbol}, called with {symbol}")
                return False
            snapshot = data_manager.get_snapshot(symbol)
            if snapshot is None or len(snapshot) < self.rsi_period:
                self.logger.debug(f"Insufficient data for RSI on {symbol}")
                return False
            close_prices = snapshot['close'].values
            rsi = talib.RSI(close_prices, timeperiod=self.rsi_period)[-1]
            self.logger.debug(f"RSI for {symbol}: {rsi:.2f}")
            return rsi > self.rsi_overbought
        except Exception as e:
            self.logger.error(f"Error in RSI exit trade for {symbol}: {e}", exc_info=True)
            return False

    def calculate_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        data[SIDE] = 0
        rsi = talib.RSI(data['close'], timeperiod=self.rsi_period)
        data[SIDE] = np.where(rsi < self.rsi_oversold, 1, data[SIDE])
        data[SIDE] = np.where(rsi > self.rsi_overbought, -1, data[SIDE])
        self.logger.debug(f"RSI signals for {self.symbol}: {data[SIDE].value_counts().to_dict()}")
        return data

class MACDStrategy(StrategyMixin):
    def __init__(self, config: Config, data_manager: DataManager, logger: logging.Logger, symbol: str, **kwargs):
        super().__init__()
        self.config = config
        self.logger = logger
        self.data_manager = data_manager
        self.symbol = symbol
        self.macd_fast = config.macd_fast_period
        self.macd_slow = config.macd_slow_period
        self.macd_signal = config.macd_signal_period
        self._validate_config()
        
        # Get the DataFrame for the specific symbol
        data = data_manager.get_snapshot(symbol)
        if data is None:
            self.logger.warning(f"No data available for symbol {symbol}, initializing with empty DataFrame")
            data = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
            data.index.name = 'timestamp'

        self.logger.debug(f"MACDStrategy initialized for {symbol}")
        StrategyMixin.__init__(self, data, **kwargs)

    def _validate_config(self):
        required = ['macd_fast_period', 'macd_slow_period', 'macd_signal_period']
        for param in required:
            if not hasattr(self.config, param):
                raise ValueError(f"Missing config parameter: {param}")

    async def should_enter_trade(self, symbol: str, data_manager: DataManager) -> bool:
        try:
            if symbol != self.symbol:
                self.logger.warning(f"Symbol mismatch: strategy for {self.symbol}, called with {symbol}")
                return False
            snapshot = data_manager.get_snapshot(symbol)
            if snapshot is None or len(snapshot) < self.macd_slow:
                self.logger.debug(f"Insufficient data for MACD on {symbol}")
                return False
            close_prices = snapshot['close'].values
            macd, signal, _ = talib.MACD(
                close_prices,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            if len(macd) < 2 or len(signal) < 2:
                return False
            self.logger.debug(f"MACD for {symbol}: {macd[-1]:.6f}, Signal: {signal[-1]:.6f}")
            return macd[-1] > signal[-1] and macd[-2] <= signal[-2]
        except Exception as e:
            self.logger.error(f"Error in MACD enter trade for {symbol}: {e}", exc_info=True)
            return False

    async def should_exit_trade(self, symbol: str, data_manager: DataManager) -> bool:
        try:
            if symbol != self.symbol:
                self.logger.warning(f"Symbol mismatch: strategy for {self.symbol}, called with {symbol}")
                return False
            snapshot = data_manager.get_snapshot(symbol)
            if snapshot is None or len(snapshot) < self.macd_slow:
                self.logger.debug(f"Insufficient data for MACD on {symbol}")
                return False
            close_prices = snapshot['close'].values
            macd, signal, _ = talib.MACD(
                close_prices,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            if len(macd) < 2 or len(signal) < 2:
                return False
            self.logger.debug(f"MACD for {symbol}: {macd[-1]:.6f}, Signal: {signal[-1]:.6f}")
            return macd[-1] < signal[-1] and macd[-2] >= signal[-2]
        except Exception as e:
            self.logger.error(f"Error in MACD exit trade for {symbol}: {e}", exc_info=True)
            return False

    def calculate_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        data[SIDE] = 0
        macd, signal, _ = talib.MACD(
            data['close'],
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        data['macd_diff'] = macd - signal
        data['macd_diff_prev'] = data['macd_diff'].shift(1)
        data[SIDE] = np.where(
            (data['macd_diff'] > 0) & (data['macd_diff_prev'] <= 0),
            1,
            data[SIDE]
        )
        data[SIDE] = np.where(
            (data['macd_diff'] < 0) & (data['macd_diff_prev'] >= 0),
            -1,
            data[SIDE]
        )
        data = data.drop(['macd_diff', 'macd_diff_prev'], axis=1)
        self.logger.debug(f"MACD signals for {self.symbol}: {data[SIDE].value_counts().to_dict()}")
        return data

class StrategyManager:
    def __init__(
        self,
        data_manager: DataManager,
        api,
        logger: logging.Logger,
        portfolio_manager: Optional[PortfolioManager] = None
    ):
        self.logger = logger
        self.logger.debug("Starting StrategyManager initialization")
        self.data_manager = data_manager
        self.api = api
        self.portfolio_manager = portfolio_manager
        self.config = Config()
        self._validate_config()
        self._price_groups = data_manager._symbol_groups
        self.logger.debug(f"Initialized _price_groups: {list(self._price_groups.keys())}")
        self.logger.debug("Initializing strategies")

        # Initialize strategies for each symbol
        self.strategies = {}
        for symbol in self.config.symbols:
            self.strategies[symbol] = {
                'rsi': RSIStrategy(self.config, data_manager, logger, symbol),
                'macd': MACDStrategy(self.config, data_manager, logger, symbol),
                'ml': MLStrategy(self.config, logger, data_manager, portfolio_manager)
            }
            self.logger.debug(f"Strategies initialized for {symbol}")

        self.logger.debug("StrategyManager initialization completed")

    def _validate_config(self):
        required = [
            'symbols', 'stop_loss_pips', 'take_profit_pips', 'trailing_stop_pips',
            'risk_percentage', 'ml_n_estimators', 'ml_max_depth', 'ml_min_samples',
            'ml_window_size', 'ml_threshold', 'seed', 'model_dir'
        ]
        for param in required:
            if not hasattr(self.config, param):
                raise ValueError(f"Missing config parameter: {param}")

    async def should_enter_trade(self, symbol: str) -> bool:
        try:
            if symbol not in self.strategies:
                self.logger.error(f"No strategies defined for symbol {symbol}")
                return False
            rsi_result = await self.strategies[symbol]['rsi'].should_enter_trade(symbol, self.data_manager)
            macd_result = await self.strategies[symbol]['macd'].should_enter_trade(symbol, self.data_manager)
            ml_result = await self.strategies[symbol]['ml'].should_enter_trade(symbol, self.data_manager)
            self.logger.debug(f"Enter signals for {symbol}: RSI={rsi_result}, MACD={macd_result}, ML={ml_result}")
            return any([rsi_result, macd_result, ml_result])
        except Exception as e:
            self.logger.error(f"Error checking enter trade for {symbol}: {e}", exc_info=True)
            return False

    async def should_exit_trade(self, symbol: str) -> bool:
        try:
            if symbol not in self.strategies:
                self.logger.error(f"No strategies defined for symbol {symbol}")
                return False
            rsi_result = await self.strategies[symbol]['rsi'].should_exit_trade(symbol, self.data_manager)
            macd_result = await self.strategies[symbol]['macd'].should_exit_trade(symbol, self.data_manager)
            ml_result = await self.strategies[symbol]['ml'].should_exit_trade(symbol, self.data_manager)
            self.logger.debug(f"Exit signals for {symbol}: RSI={rsi_result}, MACD={macd_result}, ML={ml_result}")
            return any([rsi_result, macd_result, ml_result])
        except Exception as e:
            self.logger.error(f"Error checking exit trade for {symbol}: {e}", exc_info=True)
            return False
