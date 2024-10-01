import time

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0

    async def update(self, portfolio_manager):
        # This method would be called periodically to update performance metrics
        # For now, we'll just print some basic info
        elapsed_time = time.time() - self.start_time
        open_positions = portfolio_manager.get_open_positions()
        total_exposure = portfolio_manager.get_total_exposure()

        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Open positions: {open_positions}")
        print(f"Total exposure: {total_exposure}")
        print(f"Total trades: {self.total_trades}")
        print(f"Win rate: {self.winning_trades / self.total_trades if self.total_trades > 0 else 0:.2%}")
        print(f"Total profit: ${self.total_profit:.2f}")

    def record_trade(self, profit):
        self.total_trades += 1
        if profit > 0:
            self.winning_trades += 1
        self.total_profit += profit

