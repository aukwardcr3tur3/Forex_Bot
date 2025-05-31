from abc import ABC, abstractmethod
from data_manager import DataManager

class BaseStrategy(ABC):
    @abstractmethod
    async def should_enter_trade(self, symbol: str, data_manager: DataManager) -> bool:
        pass

    @abstractmethod
    async def should_exit_trade(self, symbol: str, data_manager: DataManager) -> bool:
        pass
