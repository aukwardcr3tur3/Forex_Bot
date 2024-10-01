import asyncio
import logging
from deriv_api import DerivAPI
from strategy import StrategyManager
from risk_manager import RiskManager
from monitor import Monitor
from data_manager import DataManager
from portfolio_manager import PortfolioManager
from performance_monitor import PerformanceMonitor
from backtester import Backtester

class ForexBot:
    def __init__(self, config):
        self.config = config
        self.api = DerivAPI(app_id=1089)
        self.strategy_manager = StrategyManager()
        self.risk_manager = RiskManager(config)
        self.monitor = Monitor(config)
        self.data_manager = DataManager(config.SYMBOLS)
        self.portfolio_manager = PortfolioManager(config)
        self.performance_monitor = PerformanceMonitor()
        self.backtester = Backtester(config, self.strategy_manager)
        self.logger = logging.getLogger(__name__)
        self.running = False

    async def run(self):
        self.running = True
        await self.api.connect()
        await self.api.authorize(self.config.API_TOKEN)
        
        while self.running:
            try:
                for symbol in self.config.SYMBOLS:
                    tick = await self.api.ticks(symbol)
                    self.data_manager.update(symbol, tick)
                    
                    if self.strategy_manager.should_enter_trade(symbol, self.data_manager):
                        await self.enter_trade(symbol)
                
                await self.monitor.check_open_positions(self.api)
                await self.performance_monitor.update(self.portfolio_manager)
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)

    async def enter_trade(self, symbol):
        if not self.risk_manager.can_enter_trade(symbol):
            return

        try:
            position_size = self.risk_manager.calculate_position_size(symbol)
            contract = await self.api.buy({
                "contract_type": "CALL",
                "amount": position_size,
                "symbol": symbol,
                "duration": 5,
                "duration_unit": "m"
            })
            self.monitor.add_position(contract)
            self.portfolio_manager.add_trade(symbol, contract)
            self.logger.info(f"Entered trade: {contract}")
        except Exception as e:
            self.logger.error(f"Error entering trade: {e}")

    def stop(self):
        self.running = False

    async def run_backtest(self):
        results = await self.backtester.run()
        self.logger.info(f"Backtest results: {results}")
        return results

