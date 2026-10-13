"""Glue capture packets, scan scheduling, and the shared recognition engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capture import FramePacket
from .state import DetectionScheduler, ScanKind


@dataclass(frozen=True)
class RecognitionResult:
    sequence: int
    captured_at_ns: int
    scan_kind: ScanKind
    reason: str
    detections: tuple[dict[str, Any], ...]
    reused_previous: bool


class RecognitionRuntime:
    """Run adjacent-scale scans frequently and exhaustive recovery scans periodically."""

    def __init__(self, detector, scale: int, scheduler: DetectionScheduler | None = None) -> None:
        if scale not in (80,90,100,110,120):
            raise ValueError('unsupported buff scale')
        self.detector=detector
        self.scale=scale
        self.scheduler=scheduler or DetectionScheduler()
        self._latest: tuple[dict[str,Any], ...]=()

    def process(self, packet: FramePacket, *, force_full: bool = False) -> RecognitionResult:
        decision=self.scheduler.decide(packet.captured_at_ns,force_full=force_full)
        if decision.kind is ScanKind.SKIP:
            return RecognitionResult(packet.sequence,packet.captured_at_ns,decision.kind,
                                     decision.reason,self._latest,True)
        search_mode='exhaustive' if decision.kind is ScanKind.FULL else 'slots'
        detections=tuple(self.detector.detect(packet.frame,self.scale,search_mode=search_mode))
        self.scheduler.record(packet.captured_at_ns,decision.kind,
                              detections_present=bool(detections))
        self._latest=detections
        return RecognitionResult(packet.sequence,packet.captured_at_ns,decision.kind,
                                 decision.reason,detections,False)

    def reset(self, *, scale: int | None = None) -> None:
        if scale is not None:
            if scale not in (80,90,100,110,120):
                raise ValueError('unsupported buff scale')
            self.scale=scale
        self.scheduler.reset()
        self._latest=()
