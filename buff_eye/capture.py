"""Capture primitives for a future Windows screen source.

This module deliberately knows nothing about OpenCV, NumPy, or a game process.
A source may return a NumPy array, but frames are treated as opaque Python
objects so the pacing and queue behavior can be tested with the standard
library alone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Optional, Protocol


class FrameSource(Protocol):
    """Public contract for monitor/window capture implementations.

    A Windows implementation can later use Desktop Duplication or a supported
    screen-capture API behind this boundary.  It must only return pixels; this
    interface provides no game memory, input, injection, or rendering hook.
    """

    def open(self) -> None: ...

    def read(self) -> Any: ...

    def close(self) -> None: ...


class CaptureTargetKind(str, Enum):
    MONITOR = "monitor"
    WINDOW = "window"


@dataclass(frozen=True)
class CaptureRegion:
    """Pixel ROI relative to the selected monitor or window client area."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0 or self.width < 1 or self.height < 1:
            raise ValueError("capture region must have non-negative origin and positive size")


@dataclass(frozen=True)
class ScreenCaptureRequest:
    """Platform-neutral description consumed by a future Windows backend."""

    target_kind: CaptureTargetKind
    target_id: str
    region: CaptureRegion

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id cannot be empty")


class ScreenCaptureProvider(Protocol):
    """Factory boundary for a monitor/window pixel-capture implementation."""

    def create_source(self, request: ScreenCaptureRequest) -> FrameSource: ...


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    captured_at_ns: int
    frame: Any


@dataclass(frozen=True)
class QueueStats:
    accepted: int
    dropped: int
    pending: int


class LatestFrameQueue:
    """Small thread-safe FIFO that discards the oldest frame on overflow."""

    def __init__(self, capacity: int = 2) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._items: deque[FramePacket] = deque()
        self._accepted = 0
        self._dropped = 0
        self._condition = threading.Condition()

    def put(self, packet: FramePacket) -> Optional[FramePacket]:
        """Append a packet and return the dropped oldest packet, if any."""
        with self._condition:
            dropped = None
            if len(self._items) == self.capacity:
                dropped = self._items.popleft()
                self._dropped += 1
            self._items.append(packet)
            self._accepted += 1
            self._condition.notify()
            return dropped

    def get(self, timeout: Optional[float] = None) -> FramePacket:
        """Remove the oldest available packet, waiting up to ``timeout``."""
        with self._condition:
            if timeout is not None and timeout < 0:
                raise ValueError("timeout cannot be negative")
            ready = self._condition.wait_for(lambda: bool(self._items), timeout)
            if not ready:
                raise TimeoutError("no frame available")
            return self._items.popleft()

    def get_latest(self) -> FramePacket:
        """Return the newest packet and discard any older pending packets."""
        with self._condition:
            if not self._items:
                raise TimeoutError("no frame available")
            packet = self._items.pop()
            skipped = len(self._items)
            self._items.clear()
            self._dropped += skipped
            return packet

    def stats(self) -> QueueStats:
        with self._condition:
            return QueueStats(self._accepted, self._dropped, len(self._items))

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)


class StaticFrameSource:
    """Finite deterministic source for still frames, including NumPy arrays."""

    def __init__(self, frames: Iterable[Any], *, repeat: bool = False) -> None:
        if repeat:
            materialized = tuple(frames)
            if not materialized:
                raise ValueError("a repeating source needs at least one frame")
            self._frames: Iterable[Any] = materialized
        else:
            self._frames = frames
        self.repeat = repeat
        self._iterator: Optional[Iterator[Any]] = None
        self._opened = False

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("source is already open")
        self._iterator = iter(self._frames)
        self._opened = True

    def read(self) -> Any:
        if not self._opened or self._iterator is None:
            raise RuntimeError("source is not open")
        try:
            return next(self._iterator)
        except StopIteration:
            if not self.repeat:
                raise
            self._iterator = iter(self._frames)
            return next(self._iterator)

    def close(self) -> None:
        self._opened = False
        self._iterator = None


class CaptureLoop:
    """Paced producer that timestamps frames and writes them to a bounded queue."""

    def __init__(
        self,
        source: FrameSource,
        output: LatestFrameQueue,
        *,
        target_fps: float = 30.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        self.source = source
        self.output = output
        self.period_ns = max(1, round(1_000_000_000 / target_fps))
        self._clock_ns = clock_ns
        self._sleep = sleep
        self._last_timestamp_ns = -1

    def _timestamp(self) -> int:
        # Preserve the public monotonic guarantee even with a coarse or faulty
        # platform clock.  Normally monotonic_ns already satisfies this.
        observed = self._clock_ns()
        timestamp = max(observed, self._last_timestamp_ns + 1)
        self._last_timestamp_ns = timestamp
        return timestamp

    def run(
        self,
        *,
        stop_event: Optional[threading.Event] = None,
        max_frames: Optional[int] = None,
    ) -> int:
        """Capture until stopped, exhausted, or ``max_frames`` is reached.

        The source is always closed.  A finite source signals normal completion
        with ``StopIteration``.  Other source errors propagate after cleanup.
        """
        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames cannot be negative")
        stop_event = stop_event or threading.Event()
        produced = 0
        deadline_ns = self._clock_ns()
        self.source.open()
        try:
            while not stop_event.is_set() and (max_frames is None or produced < max_frames):
                now_ns = self._clock_ns()
                remaining_ns = deadline_ns - now_ns
                if remaining_ns > 0:
                    self._sleep(remaining_ns / 1_000_000_000)
                try:
                    frame = self.source.read()
                except StopIteration:
                    break
                packet = FramePacket(produced, self._timestamp(), frame)
                self.output.put(packet)
                produced += 1
                # Advance from the intended deadline.  If capture fell behind,
                # skip missed slots instead of emitting a burst of stale frames.
                deadline_ns += self.period_ns
                after_capture_ns = self._clock_ns()
                if deadline_ns < after_capture_ns:
                    missed = (after_capture_ns - deadline_ns) // self.period_ns
                    deadline_ns += (missed + 1) * self.period_ns
        finally:
            self.source.close()
        return produced
