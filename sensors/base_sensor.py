"""Base sensor abstraction for future sensor expansions (DO, Ammonia, Water Level, Conductivity)."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any


class BaseSensor(ABC):
    """Abstract base class for aquarium sensors."""

    def __init__(self, name: str, unit: str):
        self.name = name
        self.unit = unit

    @abstractmethod
    def read_raw(self) -> Any:
        """Perform hardware-specific sensor read."""
        pass

    def read(self) -> Dict[str, Any]:
        """Return standardized sensor reading format."""
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            val = self.read_raw()
            return {"value": val, "unit": self.unit, "timestamp": timestamp}
        except Exception as exc:
            return {"value": None, "unit": self.unit, "timestamp": timestamp, "error": str(exc)}
