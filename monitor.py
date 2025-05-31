import asyncio
import logging
import time
from typing import Dict

from config import Config
from portfolio_manager import PortfolioManager
from deriv_api import DerivAPI

class PerformanceMonitor:
    def __init__(self, portfolio_manager: PortfolioManager, logger: logging.Logger):
        """Initialize performance monitor for tracking bot metrics.

        Args:
            portfolio_manager: PortfolioManager instance for position data.
            logger: Logger for debugging and monitoring.
        """
        self.portfolio_manager = portfolio_manager
        self.logger = logger
        self.start_time = time.time()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0

    async def update(self):
        """Update and log performance metrics."""
        try:
            elapsed_time = time.time() - self.start_time
            open_positions = self.portfolio_manager.get_open_positions()
            total_exposure = self.portfolio_manager.get_total_exposure()
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            self.logger.info(
                f"Performance Update: "
                f"Elapsed time: {elapsed_time:.2f}s, "
                f"Open positions: {open_positions}, "
                f"Total exposure: {total_exposure}, "
                f"Total trades: {self.total_trades}, "
                f"Win rate: {win_rate:.2%}, "
                f"Total profit: ${self.total_profit:.2f}"
            )
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    def record_trade(self, profit: float):
        """Record a trade outcome.

        Args:
            profit: Profit or loss from the trade.
        """
        self.total_trades += 1
        if profit > 0:
            self.winning_trades += 1
        self.total_profit += profit
        self.logger.info(f"Recorded trade: profit=${profit:.2f}, total_trades={self.total_trades}")

class Monitor:
    def __init__(self, config: Config, portfolio_manager: PortfolioManager, api: DerivAPI, logger: logging.Logger):
        """Initialize the Monitor for managing open positions and performance tracking.

        Args:
            config: Configuration object.
            portfolio_manager: PortfolioManager instance for position tracking.
            api: DerivAPI instance for platform interaction.
            logger: Logger for debugging and monitoring.
        """
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.api = api
        self.logger = logger
        self.performance_monitor = PerformanceMonitor(portfolio_manager, logger)

    async def check_open_positions(self):
        """Check the status of all open positions and apply trailing stops."""
        for symbol, contracts in self.portfolio_manager.positions.items():
            for contract in contracts[:]:  # Copy to allow modification
                contract_id = contract.get("contract_id")
                if not contract_id:
                    self.logger.warning(f"Invalid contract for {symbol}: missing contract_id")
                    continue
                try:
                    updated_contract = await self.api.proposal_open_contract({"contract_id": contract_id})
                    self.logger.debug(f"Contract status for {symbol}/{contract_id}: {updated_contract.get('status')}")
                    if updated_contract.get("status") == "closed":
                        profit = updated_contract.get("profit", 0.0)
                        self.performance_monitor.record_trade(profit)
                        self.portfolio_manager.close_trade(symbol, contract_id)
                        self.logger.info(f"Closed contract {contract_id} for {symbol}, profit=${profit:.2f}")
                    elif self.should_apply_trailing_stop(updated_contract):
                        await self.apply_trailing_stop(updated_contract)
                except Exception as e:
                    self.logger.error(f"Error checking contract {contract_id} for {symbol}: {e}")

    def should_apply_trailing_stop(self, contract: Dict) -> bool:
        """Determine if a trailing stop should be applied.

        Args:
            contract: Contract details from DerivAPI.

        Returns:
            bool: True if trailing stop should be applied.
        """
        if not all(k in contract for k in ["current_spot", "entry_spot", "buy_price"]):
            self.logger.debug(f"Contract missing required fields: {contract}")
            return False
        try:
            profit_pips = (contract["current_spot"] - contract["entry_spot"]) * 10000
            self.logger.debug(f"Profit for contract {contract['contract_id']}: {profit_pips:.2f} pips")
            return profit_pips > self.config.TRAILING_STOP_PIPS
        except (TypeError, ValueError) as e:
            self.logger.error(f"Error calculating profit pips: {e}")
            return False

    async def apply_trailing_stop(self, contract: Dict):
        """Apply a trailing stop to an open contract.

        Args:
            contract: Contract details from DerivAPI.
        """
        contract_id = contract.get("contract_id")
        try:
            new_stop_loss = contract["current_spot"] - (self.config.TRAILING_STOP_PIPS / 10000)
            response = await self.api.sell({"contract_id": contract_id, "price": new_stop_loss})
            self.logger.info(f"Applied trailing stop to {contract_id}: new_stop_loss={new_stop_loss}, response={response}")
        except Exception as e:
            self.logger.error(f"Failed to apply trailing stop to {contract_id}: {e}")

    async def is_contract_active(self, contract_id: str) -> bool:
        """Check if a contract is still active.

        Args:
            contract_id: ID of the contract to check.

        Returns:
            bool: True if the contract is active, False otherwise.
        """
        try:
            contract = await self.api.proposal_open_contract({"contract_id": contract_id})
            status = contract.get("status", "unknown")
            self.logger.debug(f"Contract {contract_id} status: {status}")
            return status == "open"
        except Exception as e:
            self.logger.error(f"Error checking contract {contract_id}: {e}")
            return False
