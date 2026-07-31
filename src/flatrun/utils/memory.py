"""Memory-accounting helpers.

The runtime tracks process RSS to validate the "constant RAM
regardless of model size" promise. This module isolates the
platform-specific bits behind a small interface so the rest of the
code can be unit-tested without touching ``/proc``.
"""

from __future__ import annotations

import os
from typing import Callable, Protocol, TypeAlias


try:  # pragma: no cover - psutil is optional
    import psutil

    _HAS_PSUTIL = True
except Exception:  # pragma: no cover - import failures are non-fatal
    _HAS_PSUTIL = False


class MemoryProbe(Protocol):
    """Strategy interface for sampling the process's resident set size."""

    def rss(self) -> int:
        """Return current RSS in bytes."""

    def peak_rss(self) -> int:
        """Return peak RSS observed so far in bytes."""


class _PsutilProbe:
    """``psutil``-backed implementation (preferred)."""

    def __init__(self) -> None:
        if not _HAS_PSUTIL:  # pragma: no cover - guarded at call site
            raise RuntimeError("psutil is not available")
        self._proc = psutil.Process(os.getpid())
        self._peak = self.rss()

    def rss(self) -> int:
        return int(self._proc.memory_info().rss)

    def peak_rss(self) -> int:
        cur = self.rss()
        if cur > self._peak:
            self._peak = cur
        return int(self._peak)


class _ProcStatProbe:
    """Fallback probe: ``/proc/self/status`` on Linux, ``resource`` elsewhere."""

    _LABEL = b"VmRSS:"

    def __init__(self) -> None:
        self._peak = self.rss()

    def rss(self) -> int:
        try:
            with open("/proc/self/status", "rb") as fh:
                for line in fh:
                    if line.startswith(self._LABEL):
                        # "VmRSS:     12345 kB"
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024
        except OSError:
            pass
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KB on Linux, bytes on macOS.
        return int(usage.ru_maxrss) * (1024 if usage.ru_maxrss < 2**31 else 1)

    def peak_rss(self) -> int:
        cur = self.rss()
        if cur > self._peak:
            self._peak = cur
        return int(self._peak)


def default_probe() -> MemoryProbe:
    """Return the best available :class:`MemoryProbe` for this platform."""
    if _HAS_PSUTIL:
        try:
            return _PsutilProbe()
        except Exception:
            pass
    return _ProcStatProbe()


# Sentinel for "user passed a callable to wrap as a probe".
ProbeFactory: TypeAlias = Callable[[], "MemoryProbe | int"]


__all__ = ["MemoryProbe", "default_probe"]
