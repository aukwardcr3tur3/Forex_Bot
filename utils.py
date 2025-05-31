from datetime import datetime, timezone
from typing import Union
import time



class TimestampUtils:
    """Utility class for timestamp conversions."""

    @staticmethod
    def to_seconds(dt: datetime) -> int:
        """Convert datetime to Unix timestamp in seconds."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    @staticmethod
    def to_milliseconds(dt: datetime) -> int:
        """Convert datetime to Unix timestamp in milliseconds."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    @staticmethod
    def from_seconds(ts: Union[int, float]) -> datetime:
        """Convert Unix timestamp in seconds to UTC datetime."""
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    @staticmethod
    def from_milliseconds(ts: Union[int, float]) -> datetime:
        """Convert Unix timestamp in milliseconds to UTC datetime."""
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)


