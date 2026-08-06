"""Small POSIX wall-time guard for potentially unbounded diagnostics."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


class WalltimeLimitExceeded(TimeoutError):
    """Raised when a registered diagnostic exceeds its per-call wall time."""


def run_with_walltime_limit(function: Callable[[], T], seconds: float) -> T:
    """Run ``function`` on the main POSIX thread under ``ITIMER_REAL``.

    The registered server runtime is Linux. Unsupported hosts/threads fail
    closed so callers can apply their documented conservative metric policy;
    they never silently fall back to an unbounded call.
    """

    if seconds <= 0:
        raise ValueError("wall-time limit must be positive")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("wall-time guard requires the main Python thread")
    if not all(hasattr(signal, value) for value in ("SIGALRM", "ITIMER_REAL")):
        raise RuntimeError("POSIX real-time signal guard is unavailable")
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    if previous_delay > 0 or previous_interval > 0:
        raise RuntimeError("refusing to replace an active process wall-time timer")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(signum: int, frame: object) -> None:
        del signum, frame
        raise WalltimeLimitExceeded(
            f"diagnostic exceeded the registered {float(seconds):g}s wall-time limit"
        )

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        return function()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
