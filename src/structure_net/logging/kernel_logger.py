"""
Kernel logging service.

``KernelLogger`` is the kernel's core logging facility: a thin wrapper over
the standard library logger that additionally keeps a bounded structured
buffer of recent records so services (health monitor, scorecards, tests) can
inspect what happened without parsing text output.

``KernelComponentLogger`` is the per-component adapter injected by
``StructureNetKernel.create_component``. It accepts keyword context on every
call and remains call-compatible with the plain ``logging.Logger`` methods
components may already use (including ``log(level, message)`` with an integer
level), so injection never breaks an existing component.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
import logging
import traceback
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _level_name(level: Any) -> str:
    if isinstance(level, int):
        return logging.getLevelName(level)
    name = str(level).upper()
    return name if name in _LEVELS else "INFO"


class KernelLogger:
    """Core logging service with a structured recent-record buffer."""

    def __init__(
        self,
        name: str = "structure_net",
        level: str = "INFO",
        handlers: Optional[List[str]] = None,
        log_file: str = "structure_net.log",
        buffer_size: int = 1000,
    ):
        self.name = name
        self.handlers = list(handlers or ["console"])
        self.log_file = Path(log_file)
        self.recent_logs: Deque[Dict[str, Any]] = deque(maxlen=buffer_size)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, _level_name(level)))
        self.logger.propagate = False
        if not self.logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            if "console" in self.handlers:
                console = logging.StreamHandler()
                console.setFormatter(formatter)
                self.logger.addHandler(console)
            if "file" in self.handlers:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def _emit(
        self,
        level: Any,
        message: str,
        component: Optional[str],
        context: Dict[str, Any],
    ) -> None:
        """Record and forward one structured log entry.

        Context arrives as a plain dict so caller-supplied keys can never
        collide with this method's own parameters (components legitimately
        log keys like ``level`` or ``component``).
        """
        level_name = _level_name(level)
        self.recent_logs.append(
            {
                "timestamp": datetime.now(),
                "level": level_name,
                "message": message,
                "component": component or "kernel",
                "context": dict(context),
            }
        )
        rendered = message
        if component:
            rendered = f"[{component}] {rendered}"
        if context:
            details = ", ".join(f"{key}={value!r}" for key, value in context.items())
            rendered = f"{rendered} ({details})"
        self.logger.log(getattr(logging, level_name), rendered)

    def log(
        self,
        level: Any,
        message: str,
        component: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Log ``message`` with structured ``context`` at ``level``."""
        self._emit(level, message, component, context)

    def debug(self, message: str, **context: Any) -> None:
        self._emit("DEBUG", message, None, context)

    def info(self, message: str, **context: Any) -> None:
        self._emit("INFO", message, None, context)

    def warning(self, message: str, **context: Any) -> None:
        self._emit("WARNING", message, None, context)

    def error(self, message: str, **context: Any) -> None:
        self._emit("ERROR", message, None, context)

    def records_for(self, component: str) -> List[Dict[str, Any]]:
        """Return buffered records emitted by one component."""
        return [item for item in self.recent_logs if item["component"] == component]

    def close(self) -> None:
        """Close and detach all handlers (used by tests and shutdown)."""
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)


class KernelComponentLogger:
    """Per-component logging adapter injected by the kernel."""

    def __init__(self, kernel_logger: KernelLogger, component_name: str):
        self.kernel_logger = kernel_logger
        self.component_name = component_name

    def log(self, level: Any, message: str, **context: Any) -> None:
        self.kernel_logger._emit(level, message, self.component_name, context)

    def debug(self, message: str, **context: Any) -> None:
        self.kernel_logger._emit("DEBUG", message, self.component_name, context)

    def info(self, message: str, **context: Any) -> None:
        self.kernel_logger._emit("INFO", message, self.component_name, context)

    def warning(self, message: str, **context: Any) -> None:
        self.kernel_logger._emit("WARNING", message, self.component_name, context)

    def error(
        self, message: str, exception: Optional[Exception] = None, **context: Any
    ) -> None:
        if exception is not None:
            context["exception"] = str(exception)
            context["traceback"] = traceback.format_exc()
        self.kernel_logger._emit("ERROR", message, self.component_name, context)
