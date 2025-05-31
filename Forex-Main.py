import asyncio
import logging
import sys
import os
import tracemalloc
import time
import traceback
import tenacity
from pandas import DataFrame as df
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Union, Tuple, Optional
from config import Config
from data_manager import DataManager
from deriv_api import DerivAPI
from bot import ForexBot
from utils import TimestampUtils
from strategy import StrategyManager
from portfolio_manager import PortfolioManager
from web_interface import start_web_interface

# Configure logging with process ID
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - PID:%(process)d - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot_debug.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("Main")

async def main() -> Tuple[Dict[str, Dict], Optional["ForexBot"], Optional[DerivAPI]]:
    """Main function to start the bot with multiple tasks and return historical data."""
    logger.info("Starting main() in process %d", os.getpid())
    tracemalloc.start()
    config = None
    bot = None
    api = None
    tasks = []
    historical_data = {}

    try:
        # Initialize Config
        logger.debug("Initializing Config...")
        config = Config()
        if not config.symbols:
            raise ValueError("No symbols defined in config.symbols")

        # Validate symbols
        invalid_symbols = [s for s in config.symbols if not s.startswith('frx')]
        if invalid_symbols:
            raise ValueError(f"Invalid symbols in config.symbols: {invalid_symbols}")

        historical_data = {symbol: {"candles": []} for symbol in config.symbols}
        logger.info("Initialized config with symbols: %s, app_id: %s", config.symbols, config.app_id)

        # Initialize DerivAPI
        logger.debug("Initializing DerivAPI with app_id: %s", config.app_id)
        api = DerivAPI(app_id=config.app_id)

        # Retry ping with exponential backoff
        @tenacity.retry(
            stop=tenacity.stop_after_attempt(3),
            wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
            retry=tenacity.retry_if_exception_type((TimeoutError, ConnectionError)),
            before_sleep=lambda retry_state: logger.error(
                f"Retrying API ping: attempt {retry_state.attempt_number}, error: {retry_state.outcome.exception()}"
            )
        )
        async def ping_api():
            response = await asyncio.wait_for(api.ping({"ping": 1}), timeout=15)
            if response.get("ping") != "pong":
                raise ConnectionError(f"API ping failed: {response}")
            return response

        try:
            response = await ping_api()
            logger.info("API ping successful: %s", response)
        except Exception as e:
            logger.error(f"Failed to ping API after retries: {type(e).__name__}: {str(e)}")
            raise

        # Initialize DataManager
        logger.debug("Creating DataManager...")
        data_manager = DataManager(config=config, api=api, logger=logger)

        # Start subscriptions with retries (no historical fetch)
        @tenacity.retry(
            stop=tenacity.stop_after_attempt(3),
            wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
            retry=tenacity.retry_if_result(lambda result: not any(status['success'] for status in result.values())),
            before_sleep=lambda retry_state: logger.error(
                f"Retrying subscriptions: attempt {retry_state.attempt_number}"
            )
        )
        async def start_subscriptions_with_retry():
            return await data_manager.start_subscriptions(config.symbols, fetch_historical=False)

        logger.info("Starting data subscriptions for symbols: %s", config.symbols)
        try:
            subscription_results = await start_subscriptions_with_retry()
        except Exception as e:
            logger.error(f"Subscription retries failed: {type(e).__name__}: {str(e)}")
            raise RuntimeError("Failed to start subscriptions after retries")

        # Check subscription results
        successful_subscriptions = sum(1 for status in subscription_results.values() if status['success'])
        if successful_subscriptions == 0:
            logger.error(f"No successful subscriptions: {subscription_results}")
            raise RuntimeError("No symbols could be subscribed")

        for symbol, status in subscription_results.items():
            if status['success']:
                logger.info(f"Subscription successful for {symbol}")
            elif status.get('skipped'):
                logger.info(f"Subscription skipped for {symbol}: {status['error']}")
            else:
                logger.warning(f"Subscription failed for {symbol}: {status['error']}")

        # Wait for initial data
        await asyncio.sleep(10)

        # Fetch historical data with retries
        logger.debug("Calculating historical data timestamps...")
        timestamp_utils = TimestampUtils()
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - timedelta(days=config.historical_days)

        logger.info("Fetching historical prices for %s...", config.symbols)
        @tenacity.retry(
            stop=tenacity.stop_after_attempt(3),
            wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
            retry=tenacity.retry_if_result(lambda success: not success),
            before_sleep=lambda retry_state: logger.error(
                f"Retrying historical data fetch for {symbol}: attempt {retry_state.attempt_number}"
            )
        )
        async def fetch_historical_with_retry(symbol, start_time, end_time):
            return await data_manager.grab_historical_data(
                symbol=symbol,
                granularity=60,
                start_time=start_time,
                end_time=end_time
            )

        for symbol in config.symbols:
            if subscription_results[symbol].get('skipped'):
                logger.info(f"Skipping historical data fetch for {symbol} (market closed)")
                continue
            try:
                success = await fetch_historical_with_retry(symbol, start_time, end_time)
                if success:
                    df = data_manager.get_snapshot(symbol)
                    if df is not None and not df.empty:
                        first_candle = {
                            "epoch": df.index.min().timestamp(),
                            "datetime": df.index.min().isoformat(),
                            "close": df['close'].iloc[0]
                        }
                        historical_data[symbol]["candles"].append(first_candle)
                        logger.info(
                            "First candle datetime for %s: %s",
                            symbol,
                            first_candle["datetime"],
                        )
                    else:
                        logger.info("No historical data for %s", symbol)
                else:
                    logger.warning("Failed to fetch historical data for %s", symbol)
            except Exception as e:
                logger.error("Error fetching historical data for %s: %s", symbol, str(e))

        # Validate data
        logger.debug("Validating data...")
        valid_symbols = 0
        for symbol in config.symbols:
            df = data_manager.get_snapshot(symbol)
            if df is None or df.empty:
                if subscription_results[symbol].get('skipped'):
                    logger.info(f"No data for {symbol} (market closed, skipped)")
                    continue
                logger.warning(f"No data for {symbol}, but market is open")
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.error(f"Invalid index for {symbol}: {df.index}")
                raise RuntimeError(f"Invalid index for {symbol}")
            logger.debug(f"Data validated for {symbol}: shape={df.shape}")
            valid_symbols += 1

        if valid_symbols == 0:
            logger.error("No symbols have valid data. Cannot proceed.")
            raise RuntimeError("No valid data for any symbols")

        # Check DataManager health
        logger.debug("Checking DataManager health...")
        is_healthy, status = data_manager.is_healthy(freshness_threshold=30)
        if not is_healthy:
            logger.error(f"DataManager not healthy: {status}")
            raise RuntimeError("DataManager initialization failed")

        # Initialize PortfolioManager
        logger.debug("Creating PortfolioManager...")
        portfolio_manager = PortfolioManager(
            config=config,
            data_manager=data_manager,
            api=api,
            logger=logger
        )

        # Initialize StrategyManager
        logger.debug("Creating StrategyManager...")
        strategy_manager = StrategyManager(
            data_manager=data_manager,
            api=api,
            logger=logger,
            portfolio_manager=portfolio_manager
        )

        # Initialize ForexBot
        logger.debug("Creating ForexBot...")
        bot = ForexBot(
            config=config,
            data_manager=data_manager,
            strategy_manager=strategy_manager,
            logger=logger,
            api=api,
        )

        # Authorization
        logger.debug("Authorizing API with token...")
        try:
            auth_response = await asyncio.wait_for(
                api.authorize({"authorize": config.deriv_api_token}), timeout=15
            )
            if not auth_response.get("authorize", {}).get("loginid"):
                raise ConnectionError(f"Authorization failed: {auth_response}")
            logger.info("Authorization successful")
        except asyncio.TimeoutError:
            logger.error("Authorization timed out")
            raise
        except Exception as e:
            logger.error(f"Authorization failed: {type(e).__name__}: {str(e)}")
            raise

        logger.info("DataManager initialization completed.")

        # Start background tasks
        logger.info("Starting ForexBot with multiple background tasks...")
        tasks = [
            asyncio.create_task(bot.run(), name="bot_run"),
            asyncio.create_task(start_web_interface(bot), name="web_interface"),
            asyncio.create_task(run_blocking_tasks(bot, config), name="blocking_tasks"),
            asyncio.create_task(event_loop_watchdog(config, data_manager), name="watchdog"),
            asyncio.create_task(monitor_data_health(data_manager), name="data_health"),
        ]
        await asyncio.sleep(2)

        logger.info("All tasks scheduled")
        return historical_data, bot, api

    except Exception as e:
        logger.critical(
            f"Critical error in main: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        )
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.debug("Task %s cancelled during error cleanup", task.get_name())
        if bot:
            await bot.stop()
        if api:
            await api.clear()
        return historical_data, bot, api

    finally:
        tracemalloc.stop()

async def event_loop_watchdog(config: Config, data_manager: DataManager):
    """Monitors event loop health and DataManager status."""
    logger.info("Watchdog started in process %d", os.getpid())
    while True:
        start_time = time.monotonic()
        await asyncio.sleep(config.watchdog_interval)
        delay = time.monotonic() - start_time - config.watchdog_interval
        if delay > config.starvation_threshold:
            logger.warning("Event loop starvation detected! Delay: %.2fs", delay)
        current_tasks = [
            t for t in asyncio.all_tasks() if t is not asyncio.current_task()
        ]
        logger.debug("Active tasks: %d", len(current_tasks))
        is_healthy, status = data_manager.is_healthy(freshness_threshold=30)
        if not is_healthy:
            logger.warning("DataManager unhealthy in watchdog: %s", status)

async def monitor_data_health(data_manager: DataManager):
    """Monitors DataManager health and retries closed-market symbols."""
    logger.info("Data health monitor started")
    while True:
        is_healthy, status = data_manager.is_healthy(freshness_threshold=30)
        if not is_healthy:
            logger.warning("DataManager unhealthy, attempting to restart subscriptions: %s", status)
            await data_manager.stop_subscriptions()
            await data_manager.start_subscriptions(data_manager.config.symbols)
        else:
            # Retry closed-market symbols
            closed_symbols = [
                s for s, info in status['symbols'].items()
                if info.get('is_skipped')
            ]
            if closed_symbols:
                logger.info(f"Retrying subscriptions for closed markets: {closed_symbols}")
                await data_manager.start_subscriptions(closed_symbols)
        await asyncio.sleep(3600)  # Check hourly

async def run_blocking_tasks(bot: ForexBot, config: Config):
    """Run CPU-intensive tasks in executor."""
    logger.info("Blocking tasks started in process %d", os.getpid())
    while True:
        try:
            await asyncio.sleep(60)
        except Exception as e:
            logger.error("Blocking task error: %s", str(e))
            await asyncio.sleep(5)

async def stop_bot(bot: ForexBot, api: DerivAPI):
    """Stop the bot and clean up."""
    logger.info("Stopping bot")
    await bot.stop()
    await api.clear()
    logger.info("Bot stopped")

async def run_bot():
    """Run the bot with proper event loop management."""
    logger.info("Main process started with PID %d", os.getpid())
    try:
        result = await main()
        logger.debug(f"Main returned: {result}")
        if result is None or len(result) != 3:
            logger.critical(
                "Unexpected result from main function. Expected 3 values, got: %s",
                result,
            )
            sys.exit(1)

        historical_data, bot, api = result
        logger.info("Main completed")
        if historical_data and any(historical_data.values()):
            for symbol, data in historical_data.items():
                if data and "candles" in data and data["candles"]:
                    logger.info("First candle for %s: %s", symbol, data["candles"][0])
        else:
            logger.warning("No historical data returned from main()")

        logger.info("Bot running in background. Press Ctrl+C to stop.")
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Received interrupt, stopping...")
            await stop_bot(bot, api)

    except Exception as e:
        logger.critical(
            f"Application bootstrap failed: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        )
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_bot())
