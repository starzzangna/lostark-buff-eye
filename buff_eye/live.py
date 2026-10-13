"""Non-blocking live capture and cropped-region recognition worker."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable

from .roi_normalization import normalize_roi, restore_detection_coordinates
from .temporal import TemporalDetectionTracker, TemporalEvent


@dataclass(frozen=True)
class LiveRecognitionUpdate:
    sequence: int
    captured_at_ns: int
    frame: Any | None
    detections: tuple[dict[str, Any], ...]
    status: str = "인식 중"
    error: str | None = None
    events: tuple[TemporalEvent, ...] = ()
    capture_started_ns: int | None = None
    processed_at_ns: int | None = None
    row_search: str | None = None
    target_update: Any | None = None


class LiveRecognitionWorker:
    """Own a capture source and detector on one background thread.

    Only the most recent result is retained.  The Qt thread can therefore fall
    behind temporarily without building an unbounded queue of stale frames.
    """

    def __init__(
        self,
        session: Any,
        detector: Any,
        scale: int,
        *,
        target_fps: float = 20.0,
        scale_provider: Callable[[], int] | None = None,
        auto_scale: bool = False,
        tracker: TemporalDetectionTracker | None = None,
        recognition_size: tuple[int, int] | None = None,
        region_id: str = "self_buff",
        native_roi: bool = False,
    ) -> None:
        if scale not in (80, 90, 100, 110, 120):
            raise ValueError("unsupported buff scale")
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        self.session = session
        self.detector = detector
        self.scale = scale
        self.scale_provider = scale_provider
        from .auto_scale import AutoScale
        self.auto_scale = AutoScale() if auto_scale else None
        self.tracker = tracker or TemporalDetectionTracker(stabilize_display=True)
        if recognition_size is not None and (
            len(recognition_size) != 2 or min(recognition_size) < 1
        ):
            raise ValueError("recognition_size must be positive width and height")
        self.recognition_size = recognition_size
        self.region_id = region_id
        self.native_roi = native_roi
        self.period_seconds = 1.0 / target_fps
        self._stop_event = threading.Event()
        if hasattr(self.detector,'set_cancel_check'):
            self.detector.set_cancel_check(self._stop_event.is_set)
        self._lock = threading.Lock()
        self._latest: LiveRecognitionUpdate | None = None
        self._pending_templates = None
        self._capture_context = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        if self._thread is not None:
            raise RuntimeError("live recognition worker cannot be restarted")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="buff-eye-live", daemon=False)
        self._thread.start()

    def _publish(self, update: LiveRecognitionUpdate) -> None:
        with self._lock:
            self._latest = update

    def take_latest(self) -> LiveRecognitionUpdate | None:
        with self._lock:
            update, self._latest = self._latest, None
            return update

    def request_template_update(self, specs) -> None:
        # Latest complete snapshot wins; the Qt thread never mutates Detector.
        with self._lock:
            self._pending_templates = tuple(tuple(spec) for spec in specs)

    def _apply_template_update(self) -> None:
        with self._lock:
            specs, self._pending_templates = self._pending_templates, None
        if specs is not None:
            self.detector.replace_user_templates(specs)
            self.tracker = TemporalDetectionTracker(stabilize_display=True)

    def _run(self) -> None:
        sequence = 0
        opened = False
        try:
            self.session.open()
            opened = True
            while not self._stop_event.is_set():
                started = time.monotonic()
                capture_started_ns = time.perf_counter_ns()
                poll = self.session.poll()
                if self._stop_event.is_set():break
                captured_at_ns = time.perf_counter_ns()
                frame = poll.frame
                if frame is None:
                    self._reset_capture_context()
                    session_status = poll.status
                    detail = session_status.last_error or session_status.warning
                    label = {
                        "backoff": "캡처 재연결 대기",
                        "connecting": "캡처 연결 중",
                        "closed": "캡처 종료",
                    }.get(session_status.state, "캡처 대기")
                    self._publish(
                        LiveRecognitionUpdate(sequence, captured_at_ns, None, (), label, detail)
                    )
                    remaining = self.period_seconds - (time.monotonic() - started)
                    if remaining > 0:
                        self._stop_event.wait(remaining)
                    continue
                scale = self.scale if self.scale_provider is None else self.scale_provider()
                if self.auto_scale is not None:
                    scale = self.auto_scale.resolve(self.detector, frame,
                        (getattr(poll.status, 'binding', None), frame.shape))
                    if scale is None:
                        self._publish(LiveRecognitionUpdate(sequence, captured_at_ns, frame, (), '버프 크기 자동 확인 중'))
                        sequence += 1
                        self._stop_event.wait(self.period_seconds)
                        continue
                if scale not in (80, 90, 100, 110, 120):
                    raise ValueError(f"unsupported live buff scale: {scale}")
                try:
                    context = (getattr(poll.status, 'binding', None), frame.shape, scale)
                    if context != self._capture_context:
                        self._reset_capture_context()
                        self._capture_context = context
                    self._apply_template_update()
                    recognition_frame = frame
                    transform = None
                    if self.recognition_size is not None and not self.native_roi:
                        recognition_frame, transform = normalize_roi(
                            frame,
                            self.recognition_size,
                            region_id=self.region_id,
                            expected_region_id=self.region_id,
                        )
                    fresh_detections = tuple(
                        self.detector.detect_region(recognition_frame, scale,
                            search_mode="slots_tracked" if self.native_roi else "slots")
                    )
                    if self._stop_event.is_set():break
                    if self.native_roi and hasattr(self.session,'observe_detections'):
                        self.session.observe_detections(fresh_detections)
                    if transform is not None:
                        fresh_detections = restore_detection_coordinates(
                            fresh_detections, transform
                        )
                    prepared = []
                    frame_height, frame_width = frame.shape[:2]
                    for source in fresh_detections:
                        detection = dict(source)
                        bbox = detection.get("bbox")
                        if (
                            isinstance(bbox, (list, tuple))
                            and len(bbox) == 4
                            and all(type(value) is int for value in bbox)
                        ):
                            x, y, width, height = bbox
                            if (
                                x >= 0 and y >= 0 and width > 0 and height > 0
                                and x + width <= frame_width and y + height <= frame_height
                            ):
                                detection["_icon_frame"] = frame[y:y + height, x:x + width].copy()
                        canonical = getattr(self.detector, 'overlay_icons', {}).get(detection['effect_id'])
                        if canonical is not None:
                            detection['_icon_frame'] = canonical.copy()
                        prepared.append(detection)
                    stabilized = self.tracker.update(captured_at_ns, prepared)
                except Exception as exc:
                    self._reset_capture_context()
                    self._publish(
                        LiveRecognitionUpdate(
                            sequence,
                            captured_at_ns,
                            frame,
                            (),
                            "인식 오류",
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
                    sequence += 1
                    remaining = self.period_seconds - (time.monotonic() - started)
                    if remaining > 0:
                        self._stop_event.wait(remaining)
                    continue
                self._publish(
                    LiveRecognitionUpdate(
                        sequence,
                        captured_at_ns,
                        frame,
                        stabilized.detections,
                        "인식 중" if not poll.status.warning else f"인식 중 · {poll.status.warning}",
                        events=stabilized.events,
                        capture_started_ns=capture_started_ns,
                        processed_at_ns=time.perf_counter_ns(),
                        row_search=getattr(self.detector, 'last_row_search', None),
                    )
                )
                sequence += 1
                remaining = self.period_seconds - (time.monotonic() - started)
                if remaining > 0:
                    self._stop_event.wait(remaining)
        except StopIteration:
            self._publish(
                LiveRecognitionUpdate(sequence, time.perf_counter_ns(), None, (), "캡처 종료")
            )
        except Exception as exc:
            self._publish(
                LiveRecognitionUpdate(
                    sequence,
                    time.perf_counter_ns(),
                    None,
                    (),
                    "캡처 오류",
                    f"{type(exc).__name__}: {exc}",
                )
            )
        finally:
            if opened:
                try:
                    self.session.close()
                except Exception:
                    pass

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop(self, timeout: float = 3.0) -> None:
        if timeout < 0:
            raise ValueError("timeout cannot be negative")
        self.request_stop()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout)
        if thread.is_alive():
            raise RuntimeError("live recognition worker did not stop before timeout")

    def _reset_capture_context(self) -> None:
        self._capture_context = None
        self.tracker = TemporalDetectionTracker(stabilize_display=True)
        if hasattr(self.detector, 'reset_row_tracking'):
            self.detector.reset_row_tracking()

    close = stop


