"""Temporal scheduling state for inexpensive and full-frame detection passes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ScanKind(str, Enum):
    SKIP = "skip"
    FAST = "fast"
    FULL = "full"


@dataclass(frozen=True)
class ScanDecision:
    kind: ScanKind
    reason: str


@dataclass(frozen=True)
class SchedulerSnapshot:
    last_fast_ns: Optional[int]
    last_full_ns: Optional[int]
    consecutive_misses: int
    full_scan_pending: bool


class DetectionScheduler:
    """Schedule cheap tracking frequently and an exhaustive scan periodically.

    ``decide`` is side-effect free with respect to completed work.  Call
    ``record`` after the selected pass finishes.  Repeated fast-pass misses make
    the next decision a full scan, while a successful full pass returns to the
    inexpensive cadence.
    """

    def __init__(
        self,
        *,
        fast_interval_ns: int = 33_333_333,
        full_interval_ns: int = 1_000_000_000,
        misses_before_full: int = 2,
    ) -> None:
        if fast_interval_ns < 1 or full_interval_ns < 1:
            raise ValueError("scan intervals must be positive")
        if full_interval_ns < fast_interval_ns:
            raise ValueError("full interval cannot be shorter than fast interval")
        if misses_before_full < 1:
            raise ValueError("misses_before_full must be at least 1")
        self.fast_interval_ns = fast_interval_ns
        self.full_interval_ns = full_interval_ns
        self.misses_before_full = misses_before_full
        self._last_fast_ns: Optional[int] = None
        self._last_full_ns: Optional[int] = None
        self._last_seen_ns: Optional[int] = None
        self._consecutive_misses = 0
        self._full_scan_pending = True

    def _check_timestamp(self, timestamp_ns: int) -> None:
        if timestamp_ns < 0:
            raise ValueError("timestamp cannot be negative")
        if self._last_seen_ns is not None and timestamp_ns < self._last_seen_ns:
            raise ValueError("timestamps must be monotonic")
        self._last_seen_ns = timestamp_ns

    def decide(self, timestamp_ns: int, *, force_full: bool = False) -> ScanDecision:
        self._check_timestamp(timestamp_ns)
        if force_full:
            return ScanDecision(ScanKind.FULL, "forced")
        if self._full_scan_pending or self._last_full_ns is None:
            reason = "tracking_missed" if self._consecutive_misses else "initial"
            return ScanDecision(ScanKind.FULL, reason)
        if timestamp_ns - self._last_full_ns >= self.full_interval_ns:
            return ScanDecision(ScanKind.FULL, "periodic_refresh")
        if self._last_fast_ns is None or timestamp_ns - self._last_fast_ns >= self.fast_interval_ns:
            return ScanDecision(ScanKind.FAST, "tracking_due")
        return ScanDecision(ScanKind.SKIP, "not_due")

    def record(self, timestamp_ns: int, kind: ScanKind, *, detections_present: bool) -> None:
        self._check_timestamp(timestamp_ns)
        if kind is ScanKind.SKIP:
            return
        self._last_fast_ns = timestamp_ns
        if kind is ScanKind.FULL:
            self._last_full_ns = timestamp_ns
            self._full_scan_pending = False
        if detections_present:
            self._consecutive_misses = 0
            return
        self._consecutive_misses += 1
        if self._consecutive_misses >= self.misses_before_full:
            self._full_scan_pending = True

    def reset(self) -> None:
        """Forget tracking state, for example after monitor or ROI changes."""
        self._last_fast_ns = None
        self._last_full_ns = None
        self._last_seen_ns = None
        self._consecutive_misses = 0
        self._full_scan_pending = True

    def snapshot(self) -> SchedulerSnapshot:
        return SchedulerSnapshot(
            self._last_fast_ns,
            self._last_full_ns,
            self._consecutive_misses,
            self._full_scan_pending,
        )
