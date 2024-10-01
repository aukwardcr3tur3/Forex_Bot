class PortfolioManager:
    def __init__(self, config):
        self.config = config
        self.positions = {symbol: [] for symbol in config.SYMBOLS}

    def add_trade(self, symbol, contract):
        self.positions[symbol].append(contract)

    def close_trade(self, symbol, contract_id):
        self.positions[symbol] = [c for c in self.positions[symbol] if c['id'] != contract_id]

    def get_open_positions(self):
        return {symbol: len(positions) for symbol, positions in self.positions.items()}

    def get_total_exposure(self):
        return sum(len(positions) for positions in self.positions.values())

