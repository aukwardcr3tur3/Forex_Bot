from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import talib
import asyncio
import joblib
from statsmodels.tsa.stattools import adfuller, kpss
from config import Config
from typing import Optional, Tuple, List
import logging
import os
from sklearn.model_selection import GridSearchCV
from data_manager import DataManager
from portfolio_manager import PortfolioManager
from base_strategy import BaseStrategy
from stratestic.strategies._mixin import StrategyMixin
from stratestic.backtesting.helpers.evaluation import SIDE

class MLStrategy(StrategyMixin):
    def __init__(
        self,
        config: Config,
        logger: logging.Logger,
        data_manager: DataManager,
        portfolio_manager: Optional[PortfolioManager] = None
    ):
        super().__init__()
        if not isinstance(config, Config):
            raise TypeError("Requires valid Config instance")
        if not isinstance(data_manager, DataManager):
            raise TypeError("Requires initialized DataManager instance")
        self.config = config
        self.logger = logger
        self.data_manager = data_manager
        self.portfolio_manager = portfolio_manager
        self._validate_config()
        self.model = RandomForestRegressor(
            n_estimators=config.ml_n_estimators,
            max_depth=config.ml_max_depth,
            random_state=config.seed,
        )
        self.trained = False
        self.min_training_samples = config.ml_min_samples
        self.model_path = os.path.join(config.model_dir, "random_forest.joblib")
        self.stop_loss = self._pips_to_percentage(config.stop_loss_pips)
        self.take_profit = self._pips_to_percentage(config.take_profit_pips)
        self.trailing_stop = self._pips_to_percentage(config.trailing_stop_pips)
        self.risk_percentage = config.risk_percentage
        self.performance_metrics = {'trades': 0, 'wins': 0, 'total_return': 0.0, 'total_pips': 0.0}
        self.last_trade_price = {}
        self.highest_price = {}
        self.lag_features = [1, 2]  # Fixed lags for consistency
        os.makedirs(config.model_dir, exist_ok=True)
        self.load_model()

    def _validate_config(self) -> None:
        required = ['symbols',    'model_dir', 'ml_n_estimators',    'ml_max_depth',    'ml_min_samples',    'ml_window_size',
            'ml_threshold',    'seed',    'stop_loss_pips',    'take_profit_pips',  'trailing_stop_pips',    'risk_percentage'
        ]
        for param in required:
            if not hasattr(self.config, param):
                raise ValueError(f"Missing config parameter: {param}")

    def _pips_to_percentage(self, pips: int) -> float:
        return pips * 0.0001

    async def should_enter_trade(self, symbol: str, data_manager: DataManager) -> bool:
        try:
            if not await self._validate_symbol_data(symbol):
                return False
            if not self.trained and not await self._attempt_training(symbol):
                return False
            features = await self._extract_features(symbol)
            if features is None or len(features) != 9:
                self.logger.warning(f"Invalid feature count for {symbol}: {len(features) if features else 'None'}")
                return False
            self.logger.debug(f"Entry features for {symbol}: {features}")
            prediction = self.model.predict([features])[0]
            df = data_manager.get_snapshot(symbol)
            current_price = df['close'].iloc[-1]
            if not await self._validate_prediction(prediction, current_price):
                return False
            if current_price * (1 - self.stop_loss) < self._get_last_trade_price(symbol):
                self.logger.debug(f"Stop-loss triggered for {symbol}")
                return False
            self.last_trade_price[symbol] = current_price
            self.highest_price[symbol] = current_price
            self.performance_metrics['trades'] += 1
            return True
        except Exception as e:
            self.logger.error(f"Prediction error for {symbol}: {e}", exc_info=True)
            return False

    async def should_exit_trade(self, symbol: str, data_manager: DataManager) -> bool:
        try:
            if not await self._validate_symbol_data(symbol):
                return False
            df = data_manager.get_snapshot(symbol)
            features = await self._extract_features(symbol)
            if features is None or len(features) != 9:
                self.logger.warning(f"Invalid feature count for {symbol}: {len(features) if features else 'None'}")
                return False
            self.logger.debug(f"Exit features for {symbol}: {features}")
            prediction = self.model.predict([features])[0]
            current_price = df['close'].iloc[-1]
            entry_price = self._get_last_trade_price(symbol)
            self.highest_price[symbol] = max(self.highest_price.get(symbol, current_price), current_price)
            if current_price >= entry_price * (1 + self.take_profit):
                self._record_trade(symbol, current_price, entry_price)
                return True
            if current_price <= self.highest_price[symbol] * (1 - self.trailing_stop):
                self._record_trade(symbol, current_price, entry_price)
                return True
            threshold = current_price * (1 - self.config.ml_threshold)
            if prediction < threshold:
                self._record_trade(symbol, current_price, entry_price)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Exit prediction error for {symbol}: {e}", exc_info=True)
            return False

    def calculate_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        data[SIDE] = 0
        try:
            if not self.trained:
                self.logger.warning("Model not trained, skipping signal generation")
                return data
            window_size = self.config.ml_window_size
            if len(data) < window_size:
                self.logger.warning(f"Insufficient data: {len(data)}/{window_size}")
                return data
            symbol = self.config.symbols[0]  # Assume single symbol
            self.data_manager._snapshots[symbol] = data
            for idx in range(window_size, len(data)):
                self.data_manager._snapshots[symbol] = data.iloc[:idx + 1]
                features = asyncio.run(self._extract_features(symbol))
                if features is None or len(features) != 9:
                    continue
                prediction = self.model.predict([features])[0]
                current_price = data['close'].iloc[idx]
                threshold = current_price * (1 + self.config.ml_threshold)
                if prediction > threshold:
                    data.iloc[idx, data.columns.get_loc(SIDE)] = 1
                threshold_exit = current_price * (1 - self.config.ml_threshold)
                if prediction < threshold_exit:
                    data.iloc[idx, data.columns.get_loc(SIDE)] = -1
            self.logger.debug(f"ML signals: {data[SIDE].value_counts().to_dict()}")
        except Exception as e:
            self.logger.error(f"Error generating ML signals: {e}", exc_info=True)
        return data

    def _record_trade(self, symbol: str, exit_price: float, entry_price: float) -> None:
        profit = (exit_price - entry_price) / entry_price
        pips = (exit_price - entry_price) / 0.0001
        self.performance_metrics['total_return'] += profit
        self.performance_metrics['total_pips'] += pips
        self.performance_metrics['wins'] += 1 if profit > 0 else 0
        self._log_performance(symbol)

    async def _attempt_training(self, symbol: str, retries: int = 3):
        for attempt in range(retries):
            try:
                await self.train(symbol)
                if self.trained:
                    self.logger.info(f"Training successful for {symbol}")
                    return True
                await asyncio.sleep(2**attempt)
            except Exception as e:
                self.logger.warning(f"Training attempt {attempt+1} failed for {symbol}: {e}")
        self.logger.error(f"All training attempts failed for {symbol}")
        return False

    async def train(self, symbol: str) -> None:
        try:
            df = self.data_manager.get_snapshot(symbol)
            if df is None or df.empty:
                raise ValueError(f"No data for {symbol}")
            if not isinstance(df.index, pd.DataFrame):
                raise ValueError(f"Invalid index for {symbol}: {df.index}")
            prices = df['close'].values()
            stationarity = await self._check_stationarity(prices)
            if not stationarity["is_stationary"]:
                prices = await self._make_stationary(prices)
            X, y = await self._prepare_training_data(prices)
            if len(X) < self.min_training_samples:
                raise ValueError(f"Insufficient training samples: {len(X)}/{self.min_training_samples}")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
            }
            grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            self.trained = True
            joblib.dump(self.model, self.model_path)
            self.logger.info(f"Best parameters for {symbol}: {grid_search.best_params_}")
            self.logger.info(f"Trained model with {X.shape[1]} features")
        except Exception as e:
            self.trained = False
            self.logger.error(f"Training failed for {symbol}: {str(e)}")
            raise

    async def _prepare_training_data(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        window_size = self.config.ml_window_size
        for i in range(len(prices) - window_size - 1):
            window = prices[i:i + window_size]
            features = await self._extract_features_from_window(window)
            if len(features) != 9:
                self.logger.warning(f"Skipping sample with {len(features)} features")
                continue
            target = await self._calculate_target(prices, i)
            X.append(features)
            y.append(target)
        X = np.array(X)
        y = np.array(y)
        if len(X) == 0:
            raise ValueError("No valid training samples")
        return X, y

    async def _extract_features(self, symbol: str) -> np.ndarray:
        try:
            df = self.data_manager.get_snapshot(symbol)
            if df is None or df.empty or len(df) < self.config.ml_window_size:
                self.logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 'None'}")
                return None
            prices = df['close'].values[-self.config.ml_window_size:]
            if not isinstance(prices, np.ndarray) or len(prices) < 2:
                self.logger.error(f"Invalid prices for {symbol}: {prices}")
                return None
            return await self._extract_features_from_window(prices)
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {symbol}: {e}", exc_info=True)
            return None

    async def _extract_features_from_window(self, prices: np.ndarray) -> np.ndarray:
        try:
            returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([0.0])
            features = [
                np.mean(returns) if len(returns) > 0 else 0.0,
                np.std(returns) if len(returns) > 0 else 0.0,
                (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0,
            ]
            if len(prices) >= 9:
                rsi = talib.RSI(prices, timeperiod=9)[-1]
                features.append(rsi if not np.isnan(rsi) else 0.0)
            else:
                features.append(0.0)
            if len(prices) >= 5:
                upper, middle, lower = talib.BBANDS(prices, timeperiod=5)
                features.extend([
                    upper[-1] if not np.isnan(upper[-1]) else 0.0,
                    middle[-1] if not np.isnan(middle[-1]) else 0.0,
                    lower[-1] if not np.isnan(lower[-1]) else 0.0,
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
            if len(prices) >= 3:
                lag_1 = prices[-2] / prices[-1] if prices[-1] != 0 else 0.0
                lag_2 = prices[-3] / prices[-1] if prices[-1] != 0 else 0.0
                features.extend([lag_1, lag_2])
            else:
                features.extend([0.0, 0.0])
            self.logger.debug(f"Feature breakdown: base=3, rsi=1, bb=3, lags=2, total={len(features)}")
            return np.array(features)
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}", exc_info=True)
            return np.array([])

    async def _check_stationarity(self, series: np.ndarray) -> dict:
        try:
            series = np.array(series)
            if len(series) < 2:
                return {"is_stationary": False, "adf_pvalue": 1.0, "kpss_pvalue": 0.0}
            adf_result = adfuller(series, maxlag=10, regression='c')
            kpss_result = kpss(series, regression='c', nlags='auto')
            return {
                "is_stationary": adf_result[1] < 0.05 and kpss_result[1] > 0.05,
                "adf_pvalue": adf_result[1],
                "kpss_pvalue": kpss_result[1],
            }
        except Exception as e:
            self.logger.warning(f"Stationarity check failed: {e}")
            return {"is_stationary": False, "adf_pvalue": 1.0, "kpss_pvalue": 0.0}

    async def _make_stationary(self, series: np.ndarray, max_diff: int = 2) -> np.ndarray:
        series = np.array(series)
        for d in range(1, max_diff + 1):
            diff_series = np.diff(series, n=d)
            if (await self._check_stationarity(diff_series))["is_stationary"]:
                return diff_series
        return series

    async def _calculate_target(self, prices: np.ndarray, index: int) -> float:
        try:
            if index + self.config.ml_window_size + 1 >= len(prices):
                return 0.0
            current_price = prices[index + self.config.ml_window_size - 1]
            future_price = prices[index + self.config.ml_window_size]
            return (future_price - current_price) / current_price if current_price != 0 else 0.0
        except Exception as e:
            self.logger.warning(f"Target calculation failed: {e}")
            return 0.0

    async def _validate_symbol_data(self, symbol: str) -> bool:
        if symbol not in self.config.symbols:
            self.logger.warning(f"Symbol {symbol} not in configured symbols")
            return False
        df = self.data_manager.get_snapshot(symbol)
        if df is None or len(df) < self.config.ml_window_size:
            self.logger.warning(f"Insufficient data for {symbol}: {len(df) if df else 'None'}")
            return False
        if 'close' not in df.columns:
            self.logger.error(f"Missing 'close' column in snapshot for {symbol}")
            return False
        return True

    async def _validate_prediction(self, prediction: float, current_price: float) -> bool:
        if np.isnan(prediction) or not np.isfinite(prediction):
            self.logger.warning("Invalid prediction value")
            return False
        threshold = current_price * (1 + self.config.ml_threshold)
        self.logger.debug(f"Prediction: {prediction:.6f}, Threshold: {threshold:.6f}")
        return prediction > threshold

    def load_model(self) -> None:
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.trained = True
                self.logger.info("Loaded pre-trained model")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                self.trained = False
        else:
            self.logger.warning(f"No pre-trained model found at {self.model_path}")

    def _get_last_trade(self, symbol: str) -> float:
        return self.last_trade_price.get(symbol, 1.0)

    def _log_performance(self, symbol: str) -> None:
        win_rate = self.performance_metrics['wins'] / self.performance_metrics['trades'] if self.performance_metrics['trades'] > 0 else 0
        avg_return = self.performance_metrics['total_return'] / self.performance_metrics['trades'] if self.performance_metrics['trades'] > 0 else 0
        avg_pips = self.performance_metrics['total_pips'] / self.performance_metrics['trades'] if self.performance_metrics['trades'] > 0 else 0
        self.logger.info(f"MLStrategy {symbol} - Win Rate: {win_rate:.2f}, Avg Return: {avg_return:.2f}, Avg Pips: {avg_pips:.2f}")
        with open('performance_metrics.csv', 'a') as f:
            f.write(f"MLStrategy,{symbol},{win_rate:.2f},{avg_return:.2f},{avg_pips:.2f}\n")
        if win_rate < 0.3:
            self.logger.warning(f"Low win rate for MLStrategy on {symbol}: {win_rate:.2f}")

    async def optimize(self, data_manager: DataManager, symbol: str) -> dict:
        try:
            df = data_manager.get_snapshot(symbol)
            if df is None or df.empty:
                return {}
            if 'close' not in df.columns:
                self.logger.error(f"Missing 'close' column in snapshot for {symbol}")
                return {}
            prices = df['close'].values
            X, y = await self._prepare_training_data(prices)
            if len(X) < self.min_training_samples:
                return {}
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
            }
            grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X, y)
            return grid_search.best_params_
        except Exception as e:
            self.logger.error(f"Parameter optimization failed: {e}")
            return {}
