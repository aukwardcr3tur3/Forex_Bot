import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import talib  # TA-Lib for technical indicators

from config import Config
from data_manager import DataManager
from deriv_api import DerivAPI

class PortfolioManager:
    def __init__(
        self,
        config: Config,
        data_manager: DataManager,
        api: DerivAPI,
        logger: logging.Logger,
        account_number: Optional[str] = None
    ):
        """Initialize a PortfolioManager for forex trading and AI-based optimization.

        Args:
            config: Configuration object.
            data_manager: DataManager for price data.
            api: DerivAPI instance for platform interaction.
            logger: Logger for debugging and monitoring.
            account_number: Optional account number for the portfolio.
        """
        self.config = config
        self.data_manager = data_manager
        self.api = api
        self.logger = logger
        self.account_number = account_number
        self.positions: Dict[str, List[Dict]] = {symbol: [] for symbol in config.symbols}
        self.profit_loss: float = 0.0
        self.positions_count: int = 0

    def add_trade(self, symbol: str, contract: Dict):
        """Add a trade contract to the portfolio.

        Args:
            symbol: Symbol of the instrument (e.g., 'frxEURUSD').
            contract: Contract details from DerivAPI.
        """
        if symbol not in self.positions:
            self.logger.warning(f"Symbol {symbol} not in config.symbols")
            self.positions[symbol] = []
        self.positions[symbol].append(contract)
        self.positions_count += 1
        self.logger.info(f"Added trade for {symbol}: contract_id={contract.get('contract_id')}")

    def close_trade(self, symbol: str, contract_id: str):
        """Close a trade by removing its contract.

        Args:
            symbol: Symbol of the instrument.
            contract_id: ID of the contract to close.
        """
        if symbol in self.positions:
            initial_count = len(self.positions[symbol])
            self.positions[symbol] = [
                c for c in self.positions[symbol] if c.get("contract_id") != contract_id
            ]
            if len(self.positions[symbol]) < initial_count:
                self.positions_count -= 1
                self.logger.info(f"Closed trade for {symbol}: contract_id={contract_id}")
            else:
                self.logger.warning(f"Contract {contract_id} not found for {symbol}")
        else:
            self.logger.warning(f"Symbol {symbol} not in portfolio")

    def get_open_positions(self) -> Dict[str, int]:
        """Get the number of open positions per symbol.

        Returns:
            Dict mapping symbols to the number of open contracts.
        """
        return {symbol: len(positions) for symbol, positions in self.positions.items()}

    def get_total_exposure(self) -> int:
        """Get the total number of open contracts.

        Returns:
            Total number of open contracts across all symbols.
        """
        return sum(len(positions) for positions in self.positions.values())

    async def get_current_value(self) -> float:
        """Calculate the current market value of the portfolio.

        Returns:
            Total market value based on current prices.
        """
        total_value = 0.0
        for symbol, contracts in self.positions.items():
            snapshot = self.data_manager.get_snapshot(symbol)
            if snapshot is None or snapshot.empty:
                self.logger.warning(f"No snapshot for {symbol}")
                continue
            current_price = snapshot['close'].iloc[-1]
            for contract in contracts:
                quantity = contract.get('amount', 0)
                total_value += current_price * quantity
        self.logger.debug(f"Portfolio value: {total_value:.2f}")
        return total_value

    async def get_historical_returns(self, lookback: int = 252) -> np.ndarray:
        """Compute historical returns for all symbols over a lookback period.

        Args:
            lookback: Number of periods (e.g., days) to consider.

        Returns:
            NumPy array of shape (lookback, n_assets) with daily returns.
        """
        snapshots = {s: self.data_manager.get_snapshot(s) for s in self.config.symbols}
        valid_snapshots = {s: df for s, df in snapshots.items() if df is not None and not df.empty}
        if not valid_snapshots:
            self.logger.warning("No valid snapshots for returns calculation")
            return np.zeros((lookback, len(self.config.symbols)))

        returns = pd.DataFrame({
            s: df['close'].pct_change().dropna().tail(lookback) for s, df in valid_snapshots.items()
        }).fillna(0)
        returns_array = np.zeros((lookback, len(self.config.symbols)))
        for i, symbol in enumerate(self.config.symbols):
            if symbol in returns:
                returns_array[:, i] = returns[symbol].values[-lookback:] if len(returns[symbol]) >= lookback else np.pad(
                    returns[symbol].values, (lookback - len(returns[symbol]), 0), mode='constant'
                )
        self.logger.debug(f"Historical returns shape: {returns_array.shape}")
        return returns_array

    async def get_covariance_matrix(self, lookback: int = 252) -> np.ndarray:
        """Compute the covariance matrix of asset returns.

        Args:
            lookback: Number of periods for returns calculation.

        Returns:
            NumPy array of shape (n_assets, n_assets) with covariance matrix.
        """
        returns = await self.get_historical_returns(lookback)
        cov_matrix = np.cov(returns.T)
        self.logger.debug(f"Covariance matrix shape: {cov_matrix.shape}")
        return cov_matrix

    async def get_feature_tensors(self, lookback: int = 252, features: List[str] = None) -> torch.Tensor:
        """Generate feature tensors for AI models using TA-Lib indicators.

        Args:
            lookback: Number of periods for feature calculation.
            features: List of features to include (e.g., ['rsi', 'sma', 'volatility']). Defaults to a standard set.

        Returns:
            PyTorch tensor of shape (n_channels, lookback, n_assets).
        """
        if features is None:
            features = ['rsi', 'sma', 'volatility']
        snapshots = {s: self.data_manager.get_snapshot(s) for s in self.config.symbols}
        valid_snapshots = {s: df for s, df in snapshots.items() if df is not None and not df.empty}
        if not valid_snapshots:
            self.logger.warning("No valid snapshots for feature calculation")
            return torch.zeros((len(features), lookback, len(self.config.symbols)))

        feature_data = []
        for feature in features:
            feature_array = np.zeros((lookback, len(self.config.symbols)))
            for i, symbol in enumerate(self.config.symbols):
                if symbol not in valid_snapshots:
                    continue
                df = valid_snapshots[symbol].tail(lookback + 20)  # Extra data for indicator calculations
                close_prices = df['close'].values
                try:
                    if feature == 'rsi':
                        if len(close_prices) >= 14:
                            rsi = talib.RSI(close_prices, timeperiod=14)
                            feature_array[:, i] = rsi[-lookback:] if len(rsi) >= lookback else np.pad(
                                rsi, (lookback - len(rsi), 0), mode='constant'
                            )
                    elif feature == 'sma':
                        if len(close_prices) >= 20:
                            sma = talib.SMA(close_prices, timeperiod=20)
                            feature_array[:, i] = sma[-lookback:] if len(sma) >= lookback else np.pad(
                                sma, (lookback - len(sma), 0), mode='constant'
                            )
                    elif feature == 'volatility':
                        if len(close_prices) >= 20:
                            returns = np.diff(close_prices) / close_prices[:-1]
                            volatility = talib.STDDEV(returns, timeperiod=20)
                            feature_array[:, i] = volatility[-lookback:] if len(volatility) >= lookback else np.pad(
                                volatility, (lookback - len(volatility), 0), mode='constant'
                            )
                except Exception as e:
                    self.logger.error(f"Error calculating {feature} for {symbol}: {e}")
                    feature_array[:, i] = np.zeros(lookback)
            feature_data.append(feature_array)
        tensor = torch.tensor(feature_data, dtype=torch.float32)
        self.logger.debug(f"Feature tensor shape: {tensor.shape}")
        return tensor

    async def optimize_allocation(self, method: str = 'risk_parity', predicted_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Optimize portfolio weights using an AI-inspired method.

        Args:
            method: Optimization method ('risk_parity' or 'max_sharpe').
            predicted_returns: Optional array of predicted returns for max_sharpe method.

        Returns:
            Dictionary mapping symbols to optimized weights.
        """
        cov_matrix = await self.get_covariance_matrix()
        n_assets = len(self.config.symbols)
        weights = np.ones(n_assets) / n_assets  # Equal weights as fallback

        if method == 'risk_parity':
            volatilities = np.sqrt(np.diag(cov_matrix))
            weights = (1 / volatilities) / np.sum(1 / volatilities)
        elif method == 'max_sharpe' and predicted_returns is not None:
            risk_free_rate = 0.02 / 252  # Annualized, daily
            excess_returns = predicted_returns - risk_free_rate
            inv_cov = np.linalg.inv(cov_matrix)
            weights = inv_cov @ excess_returns
            weights = weights / np.sum(np.abs(weights))

        weights_dict = {symbol: max(0, w) for symbol, w in zip(self.config.symbols, weights)}
        total = sum(weights_dict.values())
        if total > 0:
            weights_dict = {s: w / total for s, w in weights_dict.items()}
        self.logger.debug(f"Optimized weights: {weights_dict}")
        return weights_dict

    async def export_deepdow_dataset(self, lookback: int = 252, features: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Export data in deepdow-compatible format.

        Args:
            lookback: Number of periods for data.
            features: List of features for feature tensor.

        Returns:
            Tuple of (returns_tensor, feature_tensor) with shapes (lookback, n_assets) and (n_channels, lookback, n_assets).
        """
        returns = await self.get_historical_returns(lookback)
        features = await self.get_feature_tensors(lookback, features)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        self.logger.debug(f"Deepdow dataset: returns_shape={returns_tensor.shape}, features_shape={features.shape}")
        return returns_tensor, features

    async def predict_returns(self, lookback: int = 252) -> np.ndarray:
        """Placeholder for predicting asset returns using a deep learning model.

        Args:
            lookback: Number of periods for feature data.

        Returns:
            NumPy array of predicted returns for each asset.
        """
        self.logger.warning("Return prediction not implemented. Returning zero predictions.")
        return np.zeros(len(self.config.symbols))

    async def portfolio_metrics(self) -> Dict:
        """Calculate portfolio risk and performance metrics.

        Returns:
            Dictionary with metrics like variance, returns, and weights.
        """
        metrics = {}
        weights = await self.portfolio_weights()
        snapshots = {s: self.data_manager.get_snapshot(s) for s in self.config.symbols}
        valid_snapshots = {s: df for s, df in snapshots.items() if df is not None and not df.empty}

        if not valid_snapshots:
            self.logger.warning("No valid snapshots for metrics calculation")
            return {"portfolio": {"variance": 0.0}}

        daily_returns = pd.DataFrame({
            s: df['close'].pct_change().dropna() for s, df in valid_snapshots.items()
        })
        if daily_returns.empty:
            self.logger.warning("No daily returns data")
            return {"portfolio": {"variance": 0.0}}

        cov_matrix = daily_returns.cov()
        portfolio_variance = np.sqrt(np.dot(
            np.array([weights.get(s, 0) for s in daily_returns.columns]).T,
            np.dot(cov_matrix, np.array([weights.get(s, 0) for s in daily_returns.columns]))
        ))

        for symbol in valid_snapshots:
            metrics[symbol] = {
                "weight": weights.get(symbol, 0),
                "average_returns": daily_returns[symbol].mean(),
                "std_returns": daily_returns[symbol].std(),
                "variance_returns": daily_returns[symbol].std() ** 2
            }
        metrics["portfolio"] = {"variance": portfolio_variance}
        self.logger.debug(f"Portfolio metrics: {metrics}")
        return metrics

    async def portfolio_weights(self) -> Dict[str, float]:
        """Calculate the weight of each symbol in the portfolio.

        Returns:
            Dictionary mapping symbols to their portfolio weights.
        """
        weights = {}
        total_value = 0.0
        for symbol in self.positions:
            snapshot = self.data_manager.get_snapshot(symbol)
            if snapshot is None or snapshot.empty:
                self.logger.warning(f"No snapshot for {symbol}")
                continue
            current_price = snapshot['close'].iloc[-1]
            symbol_value = sum(c.get('amount', 0) * current_price for c in self.positions[symbol])
            weights[symbol] = symbol_value
            total_value += symbol_value
        weights = {s: v / total_value if total_value > 0 else 0 for s, v in weights.items()}
        self.logger.debug(f"Portfolio weights: {weights}")
        return weights
