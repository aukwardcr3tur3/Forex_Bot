import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from websockets.exceptions import ConnectionClosed, WebSocketException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from deriv_api import DerivAPI
from config import Config

class DataManager:
    """
    Manages real-time and historical financial data for trading symbols using the Deriv API.

    Handles Observable responses from modern deriv-api versions and provides enhanced
    error handling and diagnostics for subscription management.
    """

    def __init__(self, config: Config, api: DerivAPI, logger: Optional[logging.Logger] = None, initial_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize the DataManager.

        Args:
            config: Configuration object containing settings
            api: DerivAPI instance for WebSocket communication
            logger: Logger instance (optional, creates default if None)
            initial_data: Optional initial data dictionary
        """
        self.config = config
        self.api = api
        self.logger = logger or logging.getLogger(__name__)

        # Thread-safe data storage
        self._data_lock = threading.RLock()
        self._data: Dict[str, pd.DataFrame] = {}

        # Subscription management
        self._subscriptions: Dict[str, str] = {}  # symbol -> subscription_id
        self._observables: Dict[str, Any] = {}    # symbol -> observable object
        self._subscription_status: Dict[str, Dict[str, Any]] = {}  # symbol -> status info
        self._last_update: Dict[str, float] = {}  # symbol -> timestamp

        # Connection status
        self._is_connected = False

        # Initialize with provided data
        if initial_data:
            self._initialize_data(initial_data)

        self.logger.info("DataManager initialized successfully")

    def copy(self):
        """
        Returns a copy of the DataManager's data dictionary.
        This method is required for compatibility with stratestic library.
        
        Returns:
            Dictionary of symbol -> DataFrame copies
        """
        with self._data_lock:
            return {symbol: df.copy() for symbol, df in self._data.items()}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    async def _make_api_request(self, request: Dict[str, Any]) -> Optional[Union[Dict[str, Any], Any]]:
        """
        Makes an API request with proper error handling.
        Handles both traditional responses and Observable objects from modern deriv-api.

        Args:
            request: Request dictionary

        Returns:
            Response dictionary, Observable object, or None if failed
        """
        self.logger.debug(f"Making API request: {request}")

        try:
            if 'ticks_history' in request:
                response = await self.api.ticks_history(request)
            elif 'ticks' in request and 'subscribe' in request:
                response = await self.api.subscribe(request)
                self.logger.debug(f"Subscription response type: {type(response)}, content: {response}")
                return response  # Return Observable directly for subscriptions
            elif 'forget' in request:
                response = await self.api.forget(request)
            else:
                response = await self.api.send(request)

            self.logger.debug(f"API response type: {type(response)}, content: {response}")

            # Only check for errors if response is a dictionary
            if isinstance(response, dict) and response.get('error'):
                error_msg = response['error'].get('message', str(response['error']))
                self.logger.error(f"API error: {error_msg}")
                return None

            return response

        except (ConnectionClosed, WebSocketException) as e:
            self.logger.error(f"WebSocket error during API request: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during API request: {e}")
            return None

    async def get_tradable_symbols(self, symbols: List[str]) -> List[str]:
        """
        Returns a list of symbols that are currently tradable (market open).

        Args:
            symbols: List of symbols to check

        Returns:
            List of tradable symbols
        """
        try:
            request = {'active_symbols': 'brief'}
            response = await self._make_api_request(request)
            if response is None or response.get('error'):
                self.logger.error(f"Failed to fetch active symbols: {response.get('error', 'No response')}")
                return symbols  # Fallback to trying all symbols

            tradable_symbols = []
            for symbol_info in response.get('active_symbols', []):
                symbol = symbol_info.get('symbol')
                market_status = symbol_info.get('market_status', 'open').lower()
                if symbol in symbols and market_status == 'open':
                    tradable_symbols.append(symbol)

            self.logger.info(f"Tradable symbols: {tradable_symbols}")
            return tradable_symbols
        except Exception as e:
            self.logger.error(f"Error checking tradable symbols: {e}")
            return symbols  # Fallback

    def _initialize_data(self, initial_data: Dict[str, pd.DataFrame]) -> None:
        """
        Initialize internal data structures with provided historical data.

        Args:
            initial_data: Dictionary mapping symbols to their DataFrames
        """
        with self._data_lock:
            for symbol, df in initial_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Ensure datetime index
                    if not isinstance(df.index, pd.DatetimeIndex):
                        if 'timestamp' in df.columns:
                            df.set_index('timestamp', inplace=True)
                        df.index = pd.to_datetime(df.index)

                    # Sort by timestamp
                    df.sort_index(inplace=True)

                    self._data[symbol] = df.copy()
                    self._last_update[symbol] = time.time()

                    self.logger.info(f"Initialized {symbol} with {len(df)} historical records")

    @property
    def data(self) -> Dict[str, pd.DataFrame]:
        """
        Thread-safe access to stored data.

        Returns:
            Dictionary of symbol -> DataFrame copies
        """
        with self._data_lock:
            return {symbol: df.copy() for symbol, df in self._data.items()}

    @property
    def _symbol_groups(self) -> Dict[str, pd.DataFrame]:
        """
        Groups DataFrames by symbol for indicator calculations.

        Returns:
            Dictionary of symbol -> DataFrame
        """
        return self.data

    def get_snapshot(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Returns a thread-safe copy of the DataFrame for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            DataFrame copy or None if symbol not found
        """
        with self._data_lock:
            if symbol in self._data:
                return self._data[symbol].copy()
            return None

    def get_close_prices(self, symbol: str, periods: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Returns an array of closing prices for a symbol.

        Args:
            symbol: Trading symbol
            periods: Number of recent periods to return (None for all)

        Returns:
            Numpy array of close prices or None if symbol not found
        """
        df = self.get_snapshot(symbol)
        if df is None or df.empty:
            return None

        close_prices = df['close'].values
        if periods is not None and periods > 0:
            close_prices = close_prices[-periods:]

        return close_prices

    def is_healthy(self, max_age_seconds: int = 60, freshness_threshold: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Checks if data subscriptions are active and data is recent.

        Args:
            max_age_seconds: Maximum age of data in seconds
            freshness_threshold: Alternative name for max_age_seconds (for backward compatibility)

        Returns:
            Tuple of (is_healthy, status_details)
        """
        if freshness_threshold is not None:
            max_age_seconds = freshness_threshold
        current_time = time.time()
        status = {
            'connected': self._is_connected,
            'active_subscriptions': len(self._subscriptions),
            'symbols': {},
            'stale_symbols': []
        }

        with self._data_lock:
            for symbol in self._data.keys():
                last_update = self._last_update.get(symbol, 0)
                age = current_time - last_update
                is_recent = age <= max_age_seconds
                is_skipped = self._subscription_status.get(symbol, {}).get('status') == 'skipped'

                status['symbols'][symbol] = {
                    'last_update': datetime.fromtimestamp(last_update),
                    'age_seconds': age,
                    'is_recent': is_recent,
                    'has_subscription': symbol in self._subscriptions,
                    'is_skipped': is_skipped
                }

                if not is_recent and not is_skipped:
                    status['stale_symbols'].append(symbol)
                    self.logger.warning(f"Stale data for {symbol}: {age:.1f}s old")

        is_healthy = (
            self._is_connected and
            len(self._subscriptions) > 0 and
            len(status['stale_symbols']) == 0
        )

        status['is_healthy'] = is_healthy
        return is_healthy, status

    async def _validate_api_connectivity(self) -> Tuple[bool, str]:
        """
        Validates API connectivity and authentication.

        Returns:
            Tuple of (is_connected, status_message)
        """
        try:
            # Test basic connectivity
            if hasattr(self.api, 'ping'):
                response = await self.api.ping()
                if response.get('error'):
                    return False, f"Ping failed: {response['error']}"

            # Test authentication if available
            if hasattr(self.api, 'authorize') and hasattr(self.config, 'api_token'):
                try:
                    auth_response = await self.api.authorize(self.config.api_token)
                    if auth_response.get('error'):
                        return False, f"Authentication failed: {auth_response['error']}"
                except Exception as e:
                    self.logger.warning(f"Auth check failed (may be normal): {e}")

            # Test time endpoint as fallback
            if hasattr(self.api, 'time'):
                time_response = await self.api.time()
                if time_response.get('error'):
                    return False, f"Time endpoint failed: {time_response['error']}"

            self._is_connected = True
            return True, "API connectivity validated successfully"

        except (ConnectionClosed, WebSocketException) as e:
            self._is_connected = False
            return False, f"WebSocket connection error: {e}"
        except Exception as e:
            self._is_connected = False
            return False, f"Unexpected connectivity error: {e}"

    async def check_connection(self) -> bool:
        """
        Verifies the WebSocket connection to the Deriv API.

        Returns:
            True if connected, False otherwise
        """
        is_connected, message = await self._validate_api_connectivity()
        if is_connected:
            self.logger.debug(f"Connection check successful: {message}")
        else:
            self.logger.warning(f"Connection check failed: {message}")
        return is_connected

    async def _handle_observable_subscription(self, symbol: str, observable: Any) -> bool:
        """
        Handles Observable-based subscription from modern deriv-api.

        Args:
            symbol: Trading symbol
            observable: Observable object returned by API

        Returns:
            True if subscription setup successful, False otherwise
        """
        try:
            # Store the observable for later cleanup
            self._observables[symbol] = observable

            # Define handlers for the observable
            def on_next(response):
                """Handle incoming tick data."""
                try:
                    if response.get('tick'):
                        asyncio.create_task(self.tick_callback(symbol, response['tick']))
                    else:
                        self.logger.debug(f"Non-tick response for {symbol}: {response}")
                except Exception as e:
                    self.logger.error(f"Error in tick handler for {symbol}: {e}")

            def on_error(error):
                """Handle subscription errors."""
                self.logger.error(f"Subscription error for {symbol}: {error}")
                # Update subscription status
                if symbol in self._subscription_status:
                    self._subscription_status[symbol]['status'] = 'error'
                    self._subscription_status[symbol]['error'] = str(error)
                    self._subscription_status[symbol]['last_error_time'] = time.time()

            def on_complete():
                """Handle subscription completion."""
                self.logger.info(f"Subscription completed for {symbol}")
                if symbol in self._subscription_status:
                    self._subscription_status[symbol]['status'] = 'completed'

            # Subscribe to the observable
            observable.subscribe(
                on_next=on_next,
                on_error=on_error,
                on_completed=on_complete
            )

            self.logger.info(f"Observable subscription established for {symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to setup observable subscription for {symbol}: {e}")
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionClosed, WebSocketException, asyncio.TimeoutError))
    )

    async def grab_historical_data(
        self,
        symbol: str,
        granularity: int = 60,  # seconds
        count: int = 1000,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> bool:
        """
        Fetches historical candlestick data for a symbol with pagination support.

        Args:
            symbol: Trading symbol
            granularity: Timeframe in seconds (60=1min, 3600=1hour, etc.)
            count: Number of candles to fetch (used only if start_time is None)
            start_time: Start time for historical data (None to use config.historical_days)
            end_time: End time for historical data (None for latest)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Fetching historical data for {symbol} (granularity: {granularity}s)")

            # Set default end_time if not provided
            if end_time is None:
                end_time = datetime.now(tz=timezone.utc)

            # Set start_time if not provided
            if start_time is None:
                start_time = end_time - timedelta(days=self.config.historical_days)

            # Ensure times are timezone-aware
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)

            # Calculate required candles
            total_seconds = (end_time - start_time).total_seconds()
            required_candles = int(total_seconds // granularity) + 1
            max_candles_per_request = 5000  # Deriv API limit

            # Initialize DataFrame to store all candles
            all_candles = []

            # Paginate if required_candles exceeds max_candles_per_request
            current_end_time = end_time
            candles_fetched = 0

            while candles_fetched < required_candles:
                candles_to_fetch = min(max_candles_per_request, required_candles - candles_fetched)
                self.logger.debug(f"Fetching {candles_to_fetch} candles for {symbol}, from {start_time} to {current_end_time}")

                # Prepare request parameters
                request = {
                    'ticks_history': symbol,
                    'granularity': granularity,
                    'style': 'candles',
                    'start': int(start_time.timestamp()),
                    'end': int(current_end_time.timestamp())
                }

                # Make API request
                response = await self._make_api_request(request)

                if response is None:
                    self.logger.error(f"Failed to get response for {symbol}")
                    return False

                if response.get('error'):
                    error_msg = response['error'].get('message', str(response['error']))
                    self.logger.error(f"API error for {symbol}: {error_msg}")
                    return False

                # Extract candles data
                candles = response.get('candles', [])
                if not candles:
                    self.logger.warning(f"No candles data received for {symbol}")
                    break

                all_candles.extend(candles)
                candles_fetched += len(candles)

                # Update end_time for the next request
                if candles:
                    earliest_epoch = candles[0]['epoch']
                    current_end_time = datetime.fromtimestamp(earliest_epoch, tz=timezone.utc) - timedelta(seconds=granularity)
                    start_time = current_end_time - timedelta(seconds=candles_to_fetch * granularity)
                else:
                    break

                # Avoid infinite loop
                if len(candles) < candles_to_fetch:
                    self.logger.info(f"Reached end of available data for {symbol}")
                    break

            if not all_candles:
                self.logger.warning(f"No historical data fetched for {symbol}")
                return False

            # Convert to DataFrame
            df_data = []
            for candle in all_candles:
                df_data.append({
                    'timestamp': pd.to_datetime(candle['epoch'], unit='s', utc=True),
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close'])
                })

            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            # Store data thread-safely
            with self._data_lock:
                if symbol in self._data:
                    existing_df = self._data[symbol]
                    combined_df = pd.concat([existing_df, df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df.sort_index(inplace=True)
                    self._data[symbol] = combined_df
                else:
                    self._data[symbol] = df

                self._last_update[symbol] = time.time()

            self.logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            self.logger.debug(f"Fetched DataFrame for {symbol}: columns={df.columns}, index={df.index}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return False




    async def tick_callback(self, symbol: str, tick_data: Dict[str, Any]) -> None:
        """
        Processes incoming tick data and updates the DataFrame.

        Args:
            symbol: Trading symbol
            tick_data: Tick data from API
        """
        try:
            # Extract relevant information from tick
            timestamp = pd.to_datetime(tick_data.get('epoch', time.time()), unit='s', utc=True)
            price = float(tick_data.get('quote', tick_data.get('bid', 0)))

            if price <= 0:
                self.logger.warning(f"Invalid price received for {symbol}: {price}")
                return

            with self._data_lock:
                if symbol not in self._data:
                    # Initialize empty DataFrame if not exists
                    self._data[symbol] = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
                    self._data[symbol].index.name = 'timestamp'

                df = self._data[symbol]

                # Update or create candle based on granularity
                if df.empty or timestamp not in df.index:
                    # Create new row
                    df.loc[timestamp] = [price, price, price, price]
                else:
                    # Update existing candle
                    df.loc[timestamp, 'high'] = max(df.loc[timestamp, 'high'], price)
                    df.loc[timestamp, 'low'] = min(df.loc[timestamp, 'low'], price)
                    df.loc[timestamp, 'close'] = price

                # Keep only recent data to prevent memory issues
                if len(df) > self.config.get('max_candles', 10000):
                    df = df.tail(self.config.get('max_candles', 10000))
                    self._data[symbol] = df

                self._last_update[symbol] = time.time()

            self.logger.debug(f"Processed tick for {symbol}: {tick_data}")

        except Exception as e:
            self.logger.error(f"Error processing tick for {symbol}: {e}")

    async def update(self, symbol: str, tick_data: Dict[str, Any]) -> None:
        """
        Public interface to trigger the tick callback asynchronously.

        Args:
            symbol: Trading symbol
            tick_data: Tick data to process
        """
        await self.tick_callback(symbol, tick_data)

    async def start_subscriptions(self, symbols: List[str], fetch_historical: bool = True, retry_attempts: int = 3) -> Dict[str, Dict[str, Any]]:
        """
        Starts subscriptions for all configured symbols with enhanced error handling.

        Args:
            symbols: List of symbols to subscribe to
            fetch_historical: Whether to fetch initial historical data
            retry_attempts: Number of retry attempts per symbol

        Returns:
            Dictionary of symbol -> detailed status information
        """
        # Validate API connectivity first
        is_connected, connectivity_message = await self._validate_api_connectivity()
        if not is_connected:
            self.logger.error(f"API connectivity validation failed: {connectivity_message}")
            return {symbol: {
                'success': False,
                'error': f"API not connected: {connectivity_message}",
                'timestamp': time.time()
            } for symbol in symbols}

        # Filter tradable symbols
        tradable_symbols = await self.get_tradable_symbols(symbols)
        if not tradable_symbols:
            self.logger.error("No tradable symbols available")
            return {symbol: {
                'success': False,
                'error': "No tradable symbols available",
                'timestamp': time.time()
            } for symbol in symbols}

        results = {}
        skipped_symbols = set(symbols) - set(tradable_symbols)
        for symbol in skipped_symbols:
            results[symbol] = {
                'success': False,
                'subscription_id': None,
                'historical_data': False,
                'error': "Market closed",
                'attempts': 0,
                'skipped': True,
                'timestamp': time.time()
            }
            self._subscription_status[symbol] = {
                'status': 'skipped',
                'start_time': time.time(),
                'attempts': 0,
                'last_error': None,
                'last_error_time': None
            }
            self.logger.warning(f"Skipping {symbol} as market is closed")

        for symbol in tradable_symbols:
            symbol_status = {
                'success': False,
                'subscription_id': None,
                'historical_data': False,
                'error': None,
                'attempts': 0,
                'skipped': False,
                'timestamp': time.time()
            }

            # Initialize subscription status tracking
            self._subscription_status[symbol] = {
                'status': 'attempting',
                'start_time': time.time(),
                'attempts': 0,
                'last_error': None,
                'last_error_time': None
            }

            # Retry logic for each symbol
            for attempt in range(retry_attempts):
                symbol_status['attempts'] = attempt + 1
                self._subscription_status[symbol]['attempts'] = attempt + 1

                try:
                    self.logger.info(f"Attempting subscription for {symbol} (attempt {attempt + 1}/{retry_attempts})")

                    # Fetch historical data if requested
                    if fetch_historical:
                        historical_success = await self.grab_historical_data(symbol)
                        symbol_status['historical_data'] = historical_success
                        if historical_success:
                            self.logger.info(f"Historical data fetched successfully for {symbol}")
                        else:
                            self.logger.warning(f"Failed to fetch historical data for {symbol}")

                    # Start tick subscription
                    subscription_request = {
                        'ticks': symbol,
                        'subscribe': 1
                    }

                    response = await self._make_api_request(subscription_request)

                    if response is None:
                        symbol_status['error'] = f"No response received (attempt {attempt + 1})"
                        self.logger.error(f"No response received for {symbol} subscription (attempt {attempt + 1})")
                        continue

                    # Handle Observable response (modern deriv-api)
                    if hasattr(response, 'subscribe') and callable(getattr(response, 'subscribe')):
                        self.logger.info(f"Received Observable response for {symbol}")
                        observable_success = await self._handle_observable_subscription(symbol, response)
                        if observable_success:
                            # For observables, we don't get a traditional subscription ID
                            self._subscriptions[symbol] = f"observable_{symbol}_{int(time.time())}"
                            symbol_status['success'] = True
                            symbol_status['subscription_id'] = self._subscriptions[symbol]
                            self._subscription_status[symbol]['status'] = 'active'
                            self.logger.info(f"Successfully subscribed to {symbol} via Observable")
                            break
                        else:
                            symbol_status['error'] = f"Failed to setup Observable subscription (attempt {attempt + 1})"
                            if symbol in self._subscriptions:
                                del self._subscriptions[symbol]
                            continue

                    # Handle traditional dictionary response
                    elif isinstance(response, dict):
                        if response.get('error'):
                            error_msg = response['error'].get('message', str(response['error']))
                            if "market is presently closed" in error_msg.lower():
                                symbol_status['error'] = f"Market closed: {error_msg} (attempt {attempt + 1})"
                                self.logger.warning(f"Skipping {symbol} due to closed market: {error_msg}")
                                symbol_status['success'] = False
                                symbol_status['skipped'] = True
                                self._subscription_status[symbol]['status'] = 'skipped'
                                break
                            else:
                                symbol_status['error'] = f"API error: {error_msg} (attempt {attempt + 1})"
                                self.logger.error(f"Subscription error for {symbol}: {error_msg} (attempt {attempt + 1})")
                                continue

                        # Extract subscription ID from traditional response
                        subscription_id = response.get('subscription', {}).get('id')
                        if subscription_id:
                            self._subscriptions[symbol] = subscription_id
                            symbol_status['success'] = True
                            symbol_status['subscription_id'] = subscription_id
                            self._subscription_status[symbol]['status'] = 'active'
                            self.logger.info(f"Successfully subscribed to {symbol} (ID: {subscription_id})")
                            break
                        else:
                            symbol_status['error'] = f"No subscription ID in response (attempt {attempt + 1})"
                            self.logger.error(f"No subscription ID received for {symbol} (attempt {attempt + 1})")
                            continue

                    else:
                        symbol_status['error'] = f"Unexpected response type: {type(response)} (attempt {attempt + 1})"
                        self.logger.error(f"Unexpected response type for {symbol}: {type(response)} (attempt {attempt + 1})")
                        continue

                except Exception as e:
                    symbol_status['error'] = f"Exception: {str(e)} (attempt {attempt + 1})"
                    self.logger.error(f"Failed to start subscription for {symbol} (attempt {attempt + 1}): {e}")
                    self._subscription_status[symbol]['last_error'] = str(e)
                    self._subscription_status[symbol]['last_error_time'] = time.time()

                    # Wait before retry (except on last attempt)
                    if attempt < retry_attempts - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        self.logger.info(f"Waiting {wait_time}s before retry for {symbol}")
                        await asyncio.sleep(wait_time)

            # Update final status
            if not symbol_status['success'] and not symbol_status['skipped']:
                self._subscription_status[symbol]['status'] = 'failed'

            results[symbol] = symbol_status

        # Update connection status
        successful_subscriptions = sum(1 for status in results.values() if status['success'])
        self._is_connected = successful_subscriptions > 0

        self.logger.info(f"Subscription results: {successful_subscriptions}/{len(symbols)} successful")

        # Log detailed results
        for symbol, status in results.items():
            if status['success']:
                self.logger.info(f"✓ {symbol}: subscribed successfully (ID: {status['subscription_id']})")
            elif status['skipped']:
                self.logger.info(f"↷ {symbol}: skipped (market closed)")
            else:
                self.logger.error(f"✗ {symbol}: failed after {status['attempts']} attempts - {status['error']}")

        return results

    async def stop_subscriptions(self) -> Dict[str, Dict[str, Any]]:
        """
        Stops all active subscriptions with enhanced error handling.

        Returns:
            Dictionary of symbol -> detailed status information
        """
        results = {}

        # Handle Observable subscriptions
        for symbol, observable in self._observables.items():
            try:
                if hasattr(observable, 'dispose'):
                    observable.dispose()
                elif hasattr(observable, 'unsubscribe'):
                    observable.unsubscribe()

                results[symbol] = {
                    'success': True,
                    'method': 'observable_disposal',
                    'timestamp': time.time()
                }
                self.logger.info(f"Successfully disposed Observable subscription for {symbol}")

            except Exception as e:
                results[symbol] = {
                    'success': False,
                    'error': str(e),
                    'method': 'observable_disposal',
                    'timestamp': time.time()
                }
                self.logger.error(f"Failed to dispose Observable subscription for {symbol}: {e}")

        # Clear observables
        self._observables.clear()

        # Handle traditional subscriptions
        subscriptions_copy = dict(self._subscriptions)

        for symbol, subscription_id in subscriptions_copy.items():
            # Skip if already handled as observable
            if symbol in results:
                continue

            try:
                # Send unsubscribe request
                unsubscribe_request = {
                    'forget': subscription_id
                }

                response = await self._make_api_request(unsubscribe_request)

                if response is None:
                    results[symbol] = {
                        'success': False,
                        'error': 'No response received',
                        'method': 'traditional_unsubscribe',
                        'timestamp': time.time()
                    }
                    self.logger.error(f"No response received for {symbol} unsubscribe")
                elif response.get('error'):
                    error_msg = response['error'].get('message', str(response['error']))
                    results[symbol] = {
                        'success': False,
                        'error': error_msg,
                        'method': 'traditional_unsubscribe',
                        'timestamp': time.time()
                    }
                    self.logger.error(f"Unsubscribe error for {symbol}: {error_msg}")
                else:
                    results[symbol] = {
                        'success': True,
                        'method': 'traditional_unsubscribe',
                        'timestamp': time.time()
                    }
                    self.logger.info(f"Successfully unsubscribed from {symbol}")

                # Remove from active subscriptions regardless of result
                del self._subscriptions[symbol]

            except Exception as e:
                results[symbol] = {
                    'success': False,
                    'error': str(e),
                    'method': 'traditional_unsubscribe',
                    'timestamp': time.time()
                }
                self.logger.error(f"Failed to stop subscription for {symbol}: {e}")

        # Clear subscription status
        self._subscription_status.clear()

        # Update connection status
        if not self._subscriptions and not self._observables:
            self._is_connected = False

        successful_stops = sum(1 for status in results.values() if status['success'])
        self.logger.info(f"Stopped {successful_stops}/{len(subscriptions_copy) + len(self._observables)} subscriptions")
        return results

    async def get_subscription_status(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves the current subscription status for a specific symbol or all symbols.

        Args:
            symbol: Trading symbol to check (None for all symbols)

        Returns:
            Dictionary with subscription status information
        """
        with self._data_lock:
            if symbol:
                if symbol in self._subscription_status:
                    return {symbol: self._subscription_status[symbol].copy()}
                else:
                    return {symbol: {
                        'status': 'not_subscribed',
                        'start_time': None,
                        'attempts': 0,
                        'last_error': None,
                        'last_error_time': None
                    }}
            else:
                return {s: status.copy() for s, status in self._subscription_status.items()}

    async def cleanup_old_data(self, max_age_days: float = 7.0) -> Dict[str, int]:
        """
        Removes data older than the specified age to manage memory usage.

        Args:
            max_age_days: Maximum age of data to keep in days

        Returns:
            Dictionary of symbol -> number of removed rows
        """
        results = {}
        cutoff_time = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=max_age_days)

        with self._data_lock:
            for symbol, df in self._data.items():
                if df.empty:
                    results[symbol] = 0
                    continue

                # Ensure index is UTC-aware
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')

                # Filter out old data
                initial_rows = len(df)
                df = df[df.index >= cutoff_time]
                removed_rows = initial_rows - len(df)
                self._data[symbol] = df
                results[symbol] = removed_rows

                if removed_rows > 0:
                    self.logger.info(f"Cleaned up {removed_rows} old rows for {symbol}")

        return results

    async def grab_historical_data_multiple(
        self,
        symbol: str,
        granularities: List[int],
        count: int = 1000,
        end_time: Optional[datetime] = None
    ) -> Dict[int, bool]:
        """
        Fetches historical candlestick data for a symbol across multiple granularities.

        Args:
            symbol: Trading symbol
            granularities: List of timeframes in seconds (e.g., [60, 300, 3600])
            count: Number of candles to fetch per granularity (overridden by date range)
            end_time: End time for historical data (None for latest)

        Returns:
            Dictionary of granularity -> success status
        """
        results = {}
        if end_time is None:
            end_time = datetime.now(tz=timezone.utc)

        for granularity in granularities:
            success = await self.grab_historical_data(symbol, granularity=granularity, count=count, end_time=end_time)
            results[granularity] = success
            self.logger.info(f"Historical data fetch for {symbol} at {granularity}s: {'success' if success else 'failed'}")
        return results

    async def restart_failed_subscriptions(self, max_age_seconds: int = 60) -> Dict[str, Dict[str, Any]]:
        """
        Attempts to restart subscriptions for symbols with failed or stale data.

        Args:
            max_age_seconds: Maximum age of data in seconds to consider stale

        Returns:
            Dictionary of symbol -> restart status information
        """
        is_healthy, status = self.is_healthy(max_age_seconds=max_age_seconds)
        results = {}

        if is_healthy:
            self.logger.info("All subscriptions are healthy, no restarts needed")
            return results

        # Identify symbols needing restart
        symbols_to_restart = status['stale_symbols'] + [
            s for s, info in status['symbols'].items()
            if info.get('has_subscription') and self._subscription_status.get(s, {}).get('status') in ['failed', 'error']
        ]

        if not symbols_to_restart:
            self.logger.info("No subscriptions need restarting")
            return results

        self.logger.info(f"Attempting to restart subscriptions for: {symbols_to_restart}")

        # Stop existing subscriptions for these symbols
        for symbol in symbols_to_restart:
            if symbol in self._subscriptions:
                unsubscribe_request = {'forget': self._subscriptions[symbol]}
                await self._make_api_request(unsubscribe_request)
                del self._subscriptions[symbol]
                self.logger.info(f"Stopped existing subscription for {symbol}")

            if symbol in self._observables:
                try:
                    observable = self._observables[symbol]
                    if hasattr(observable, 'dispose'):
                        observable.dispose()
                    elif hasattr(observable, 'unsubscribe'):
                        observable.unsubscribe()
                    del self._observables[symbol]
                    self.logger.info(f"Disposed Observable for {symbol}")
                except Exception as e:
                    self.logger.error(f"Failed to dispose Observable for {symbol}: {e}")

        # Start new subscriptions
        results = await self.start_subscriptions(symbols_to_restart, fetch_historical=True)
        return results
