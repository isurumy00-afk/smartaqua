"""Watchdog module for tracking system health, module execution stats, and failures.

Ensures fault-tolerant operation on Raspberry Pi by monitoring task runtimes,
exception counts, restart counts, and status without crashing the master scheduler.
"""

import time
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Dict, Any
from config import DATA_DIR
from storage.json_store import load_json, save_json
from utils.logger import get_logger

LOG = get_logger(__name__)


class Watchdog:
    """Monitors task execution stats and writes health records to watchdog.json."""

    def __init__(self, watchdog_path=None):
        self.path = watchdog_path or (DATA_DIR / "watchdog.json")
        self.log = LOG
        self.slow_threshold_seconds = 10.0  # Alert if task takes longer than 10 seconds

    @contextmanager
    def monitor(self, task_name: str):
        """Context manager to measure runtime and catch failures for any task."""
        started_time = time.monotonic()
        start_iso = datetime.now(timezone.utc).isoformat()
        error_msg = None
        success = False

        try:
            yield
            success = True
        except Exception as exc:
            error_msg = str(exc)
            self.log.exception("Task '%s' encountered an exception", task_name)
        finally:
            runtime = round(time.monotonic() - started_time, 3)
            self._record_execution(
                task=task_name,
                success=success,
                runtime=runtime,
                start_iso=start_iso,
                error=error_msg,
            )

    def _record_execution(
        self, task: str, success: bool, runtime: float, start_iso: str, error: str = None
    ) -> None:
        """Update watchdog state in watchdog.json atomically."""
        state: Dict[str, Any] = load_json(self.path, default={}) or {}
        previous = state.get(task, {})

        exception_count = previous.get("exception_count", 0) + (0 if success else 1)
        restart_count = previous.get("restart_count", 0) + (1 if not success else 0)
        is_slow = runtime > self.slow_threshold_seconds

        if is_slow:
            self.log.warning("Task '%s' executed slowly (%.3fs)", task, runtime)

        state[task] = {
            "task_name": task,
            "last_execution_time": start_iso,
            "success": success,
            "runtime_seconds": runtime,
            "exception_count": exception_count,
            "restart_count": restart_count,
            "is_slow": is_slow,
            "last_error": error or previous.get("last_error", ""),
            "status": "HEALTHY" if success and not is_slow else ("SLOW" if success else "FAILED"),
        }

        save_json(self.path, state)

    def check_health(self) -> Dict[str, Any]:
        """Check all tasks status, reporting failed or degraded modules."""
        state = load_json(self.path, default={}) or {}
        summary = {"healthy": [], "failed": [], "slow": []}

        for task, data in state.items():
            if not isinstance(data, dict):
                continue
            status = data.get("status", "UNKNOWN")
            if status == "FAILED":
                summary["failed"].append(task)
            elif status == "SLOW":
                summary["slow"].append(task)
            else:
                summary["healthy"].append(task)

        if summary["failed"]:
            self.log.error("Watchdog health alert - Failed modules: %s", summary["failed"])
        if summary["slow"]:
            self.log.warning("Watchdog health alert - Slow modules: %s", summary["slow"])

        return summary
