"""One-shot physical-pixel monitor capture for the ROI editor."""
from dataclasses import dataclass
from typing import Any

from .capture import CaptureRegion, CaptureTargetKind, ScreenCaptureRequest


@dataclass(frozen=True)
class MonitorSnapshot:
    monitor_id: str
    frame: Any


def capture_monitor_snapshot(monitor_id: str | None, provider=None) -> MonitorSnapshot:
    if provider is None:
        from .win_capture import WindowsScreenCaptureProvider
        provider = WindowsScreenCaptureProvider()
    monitors = provider.enumerate_monitors()
    monitor = next((m for m in monitors if m.stable_id == monitor_id), None)
    if monitor is None and monitor_id is None:
        monitor = next((m for m in monitors if m.primary), None)
    if monitor is None:
        raise ValueError("선택한 모니터를 찾을 수 없습니다. 모니터 설정을 확인하세요.")
    source = provider.create_source(ScreenCaptureRequest(
        CaptureTargetKind.MONITOR, monitor.stable_id,
        CaptureRegion(0, 0, monitor.bounds.width, monitor.bounds.height)))
    try:
        source.open()
        frame = source.read()
        if frame.shape != (monitor.bounds.height, monitor.bounds.width, 3):
            raise ValueError("전체 모니터 캡처 크기가 일치하지 않습니다.")
        return MonitorSnapshot(monitor.stable_id, frame.copy())
    finally:
        source.close()
