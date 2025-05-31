import pandas as pd
from datetime import datetime, timezone
import logging
import asyncio
from typing import Optional, Tuple, Dict
from stratestic.backtesting import VectorizedBacktester
from stratestic.backtesting.combining import StrategyCombiner
from stratestic.strategies import MovingAverageCrossover, Momentum, BollingerBands
from backtesting import Backtest, Strategy
from config import Config
from data_manager import DataManager
from strategy import StrategyManager, RSIStrategy, MACDStrategy
from MLStrat import MLStrategy
from deriv_api import DerivAPI
from portfolio_manager import PortfolioManager

class BacktestStrategy(Strategy):
    """Backtesting Strategy for backtesting library."""
    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.config = params['config']
        self.strategy_manager = params['strategy_manager']
        self.data_manager = params['data_manager']
        self.symbol = params['symbol']
        self.logger = params['logger']
        # Update DataManager with backtest data
        df = self.data.df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'timestamp'
        # Store data in DataManager
        with self.data_manager._data_lock:
            self.data_manager._data[self.symbol] = df
            self.data_manager._last_update[self.symbol] = datetime.now(tz=timezone.utc).timestamp()

    def init(self):
        self.position_size = self.config.initial_capital * self.config.risk_percentage
        self.logger.debug(f"Initialized backtest strategy for {self.symbol}")

    def next(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                should_enter = asyncio.run_coroutine_threadsafe(
                    self.strategy_manager.should_enter_trade(self.symbol), loop
                ).result()
                should_exit = asyncio.run_coroutine_threadsafe(
                    self.strategy_manager.should_exit_trade(self.symbol), loop
                ).result()
            else:
                should_enter = asyncio.run(self.strategy_manager.should_enter_trade(self.symbol))
                should_exit = asyncio.run(self.strategy_manager.should_exit_trade(self.symbol))
            self.logger.debug(f"{self.symbol}: enter={should_enter}, exit={should_exit}, price={self.data.Close[-1]}")
            if should_enter and not self.position:
                sl = self.data.Close[-1] * (1 - self.config.stop_loss_pips / 10000)
                tp = self.data.Close[-1] * (1 + self.config.take_profit_pips / 10000)
                self.buy(size=self.position_size, sl=sl, tp=tp)
                self.logger.info(f"Backtesting buy order for {self.symbol} at {self.data.Close[-1]}")
            elif should_exit and self.position:
                self.position.close()
                self.logger.info(f"Backtesting sell order for {self.symbol} at {self.data.Close[-1]}")
        except Exception as e:
            self.logger.error(f"Backtesting step failed for {self.symbol}: {e}", exc_info=True)

class Backtester:
    def __init__(self, config: Config, api: DerivAPI, data_manager: DataManager, strategy_manager: StrategyManager, logger: Optional[logging.Logger] = None):
        self.config = config
        self.api = api
        self.data_manager = data_manager
        self.strategy_manager = strategy_manager
        self.logger = logger or logging.getLogger(__name__)
        # Initialize stratestic strategies
        self.mov_avg = MovingAverageCrossover(sma_s=50, sma_l=200, moving_av='sma')
        self.momentum = Momentum(window=100)
        self.boll_bands = BollingerBands(ma=20, sd=2)

    async def fetch_historical_data(self, symbol: str) -> Tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Retrieves historical data for a symbol from DataManager, fetching only if necessary.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (symbol, stratestic DataFrame, backtesting DataFrame)
        """
        try:
            # Check subscription status
            status = await self.data_manager.get_subscription_status(symbol)
            if status.get(symbol, {}).get('status') == 'skipped':
                self.logger.info(f"Skipping {symbol} backtest (market closed)")
                return symbol, None, None

            # Define date range for backtesting (align with trading data)
            start_date = pd.to_datetime(self.config.backtest_start_date, utc=True)
            end_date = pd.to_datetime(self.config.backtest_end_date, utc=True)
            granularity = 60  # 1-minute candles

            # Check if DataManager has sufficient data
            df = self.data_manager.get_snapshot(symbol)
            data_sufficient = False

            if df is not None and not df.empty:
                # Ensure index is timezone-aware
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                data_start = df.index.min()
                data_end = df.index.max()
                if (data_start <= start_date) and (data_end >= end_date):
                    data_sufficient = True
                    self.logger.info(f"Using existing historical data for {symbol} ({len(df)} candles, {data_start} to {data_end})")
                else:
                    self.logger.info(f"Existing data for {symbol} insufficient (covers {data_start} to {data_end}, need {start_date} to {end_date})")

            # Fetch data if necessary
            if not data_sufficient:
                self.logger.info(f"Fetching historical data for {symbol}")
                success = await self.data_manager.grab_historical_data(
                    symbol=symbol,
                    granularity=granularity,
                    start_time=start_date,
                    end_time=end_date
                )
                if not success:
                    self.logger.warning(f"Failed to fetch historical data for {symbol}")
                    return symbol, None, None

                # Retrieve updated data
                df = self.data_manager.get_snapshot(symbol)
                if df is None or df.empty:
                    self.logger.warning(f"No data available for {symbol} after fetch")
                    return symbol, None, None

            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            if df.empty:
                self.logger.warning(f"No data within date range {start_date} to {end_date} for {symbol}")
                return symbol, None, None

            # Create DataFrames for stratestic and backtesting
            df_stratestic = df[['open', 'high', 'low', 'close']].copy()
            df_backtesting = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close'
            })
            # Add volume column if not present
            if 'volume' not in df_backtesting.columns:
                df_backtesting['Volume'] = 0.0

            self.logger.info(f"Prepared {len(df)} candles for {symbol} backtesting")
            return symbol, df_stratestic, df_backtesting

        except Exception as e:
            self.logger.error(f"Error preparing historical data for {symbol}: {e}", exc_info=True)
            return symbol, None, None

    async def run(self) -> Dict[str, Dict[str, Dict]]:
        """
        Runs backtests for all configured symbols using both stratestic and backtesting libraries.

        Returns:
            Dictionary of symbol -> backtest results
        """
        results = {}
        valid_symbols = 0
        for symbol in self.config.symbols:
            self.logger.info(f"Starting backtest for {symbol}")
            symbol, df_stratestic, df_backtesting = await self.fetch_historical_data(symbol)
            if df_stratestic is None or df_backtesting is None or df_stratestic.empty or df_backtesting.empty:
                self.logger.warning(f"Skipping backtest for {symbol} due to insufficient data")
                continue

            valid_symbols += 1
            results[symbol] = {'stratestic': {}, 'backtest': {}}

            # Initialize StrategyCombiner for this symbol
            combiner = StrategyCombiner(
                [
                    self.strategy_manager.strategies[symbol]['rsi'],
                    self.strategy_manager.strategies[symbol]['macd'],
                    self.strategy_manager.strategies[symbol]['ml'],
                    self.mov_avg,
                    self.momentum,
                    self.boll_bands
                ],
                method='Majority'
            )

            try:
                # Stratestic backtest
                backtester = VectorizedBacktester(
                    strategy=combiner,
                    data=df_stratestic,
                    trading_symbol=symbol,
                    capital=self.config.initial_capital,
                    trading_costs=self.config.trading_costs
                )
                stats = backtester.run()
                results[symbol]['stratestic'] = {
                    'Start Date': stats.get('StartTime', df_stratestic.index[0]),
                    'End Date': stats.get('EndTime', df_stratestic.index[-1]),
                    'Duration': stats.get('Duration', df_stratestic.index[-1] - df_stratestic.index[0]),
                    'Exposure (%)': stats.get('ExposureTime', 0.0) * 100,
                    'Final Equity ($)': stats.get('EquityFinal', self.config.initial_capital),
                    'Peak Equity ($)': stats.get('EquityPeak', self.config.initial_capital),
                    'Return (%)': stats.get('Return', 0.0) * 100,
                    'Buy & Hold Return (%)': stats.get('BuyAndHoldReturn', 0.0) * 100,
                    'Annualized Return (%)': stats.get('AnnualizedReturn', 0.0) * 100,
                    'Annualized Volatility (%)': stats.get('AnnualizedVolatility', 0.0) * 100,
                    'CAGR (%)': stats.get('CAGR', 0.0) * 100,
                    'Sharpe Ratio': stats.get('SharpeRatio', 0.0),
                    'Sortino Ratio': stats.get('SortinoRatio', 0.0),
                    'Calmar Ratio': stats.get('CalmarRatio', 0.0),
                    'Max Drawdown (%)': stats.get('MaxDrawdown', 0.0) * 100,
                    'Avg Drawdown (%)': stats.get('AvgDrawdown', 0.0) * 100,
                    'Max Drawdown Duration': stats.get('MaxDrawdownDuration', pd.Timedelta(0)),
                    'Avg Drawdown Duration': stats.get('AvgDrawdownDuration', pd.Timedelta(0)),
                    'Number of Trades': stats.get('TotalTrades', 0),
                    'Win Rate (%)': stats.get('WinRate', 0.0) * 100,
                    'Best Trade (%)': stats.get('BestTrade', 0.0) * 100,
                    'Worst Trade (%)': stats.get('WorstTrade', 0.0) * 100,
                    'Avg Trade (%)': stats.get('AvgTrade', 0.0) * 100,
                    'Max Trade Duration': stats.get('MaxTradeDuration', pd.Timedelta(0)),
                    'Avg Trade Duration': stats.get('AvgTradeDuration', pd.Timedelta(0)),
                    'Profit Factor': stats.get('ProfitFactor', 0.0),
                    'Expectancy (%)': stats.get('Expectancy', 0.0) * 100,
                    'SQN': stats.get('SQN', 0.0),
                    'Kelly Criterion': stats.get('KellyCriterion', 0.0),
                    'Strategy': f"Combiner({combiner.method.name})"
                }
                self.logger.info(f"Stratestic backtest results for {symbol}: {results[symbol]['stratestic']}")
            except Exception as e:
                self.logger.error(f"Stratestic backtest failed for {symbol}: {e}")
                results[symbol]['stratestic'] = {}

            try:
                # Backtesting library backtest
                bt = Backtest(
                    df_backtesting,
                    BacktestStrategy,
                    cash=self.config.initial_capital,
                    commission=self.config.trading_costs,
                    exclusive_orders=True
                )
                stats = bt.run(
                    config=self.config,
                    strategy_manager=self.strategy_manager,
                    data_manager=self.data_manager,
                    symbol=symbol,
                    logger=self.logger
                )
                results[symbol]['backtest'] = {
                    'Start Date': stats['_equity_curve'].index[0],
                    'End Date': stats['_equity_curve'].index[-1],
                    'Duration': stats['Duration'],
                    'Exposure (%)': stats['ExposureProbability'],
                    'Final Equity ($)': stats['EquityFinal'],
                    'Peak Equity ($)': stats['EquityPeak'],
                    'Return (%)': stats['Return'],
                    'Buy & Hold Return (%)': stats['BuyHoldReturn'],
                    'Annualized Return (%)': stats['AnnualizedReturn'],
                    'Annualized Volatility (%)': stats['AnnualizedVolatility'],
                    'Sharpe Ratio': stats['SharpeRatio'],
                    'Sortino Ratio': stats['SortinoRatio'],
                    'Calmar Ratio': stats['CalmarRatio'],
                    'Max Drawdown (%)': stats['MaxDrawdown'],
                    'Avg Drawdown (%)': stats['AvgDrawdown'],
                    'Max Drawdown Duration': stats['MaxDrawdownDuration'],
                    'Avg Drawdown Duration': stats['AvgDrawdownDuration'],
                    'Number of Trades': stats['TotalTrades'],
                    'Win Rate (%)': stats['WinRate'],
                    'Best Trade (%)': stats['BestTrade'],
                    'Worst Trade (%)': stats['WorstTrade'],
                    'Avg Trade (%)': stats['AvgTrade'],
                    'Max Trade Duration': stats['MaxTradeDuration'],
                    'Avg Trade Duration': stats['AvgTradeDuration'],
                    'Profit Factor': stats['ProfitFactor'],
                    'Expectancy (%)': stats['Expectancy'],
                    'Strategy': 'BacktestStrategy'
                }
                self.logger.info(f"Backtesting backtest results for {symbol}: {results[symbol]['backtest']}")
            except Exception as e:
                self.logger.error(f"Backtesting backtest failed for {symbol}: {e}")
                results[symbol]['backtest'] = {}

        if valid_symbols == 0:
            self.logger.error("No symbols had valid data for backtesting")
        else:
            self.logger.info(f"Backtesting completed for {valid_symbols}/{len(self.config.symbols)} symbols")

        return results
