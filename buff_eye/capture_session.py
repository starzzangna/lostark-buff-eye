"""Non-blocking bridge from persisted settings to a Windows capture source.

The UI can call :meth:`ManagedCaptureSession.poll` from a worker or timer.  A
missing monitor or a transient GDI failure returns a status instead of killing
the application; retries use bounded exponential backoff.  No game process or
input API is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping, Sequence

from .capture import CaptureRegion, CaptureTargetKind, FrameSource, ScreenCaptureRequest
from .overlay_model import CaptureRegion as StoredCaptureRegion
from .win_capture import CaptureError, MonitorInfo, WindowsScreenCaptureProvider


@dataclass(frozen=True)
class CaptureBinding:
    request: ScreenCaptureRequest
    monitor: MonitorInfo
    used_monitor_fallback: bool
    used_normalized_roi: bool


@dataclass(frozen=True)
class CaptureSessionStatus:
    state: str
    binding: CaptureBinding | None
    consecutive_failures: int
    retry_at_ns: int | None
    last_error: str | None
    overlay_capture_excluded: bool | None
    warning: str | None


@dataclass(frozen=True)
class CapturePoll:
    frame: Any | None
    status: CaptureSessionStatus


def _selected_profile(document: Mapping[str, Any]) -> Mapping[str, Any]:
    profile_id = document.get("selected_profile_id")
    profiles = document.get("profiles")
    if not isinstance(profile_id, str) or not isinstance(profiles, Mapping):
        raise CaptureError("settings do not contain a selected profile")
    profile = profiles.get(profile_id)
    if not isinstance(profile, Mapping):
        raise CaptureError(f"capture profile is unavailable: {profile_id}")
    return profile


def _integer_region(
    region: StoredCaptureRegion,
    monitor: MonitorInfo,
    *,
    prefer_normalized: bool = False,
) -> tuple[CaptureRegion, bool]:
    """Resolve a stored ROI using the requested pixel/normalized preference."""
    # The HUD is anchored to the horizontal screen center. At unchanged height,
    # switching 21:9 to 16:9 translates the HUD rather than stretching its icons.
    # Derive the calibration canvas from the saved pixel/normalized pair, without
    # rewriting that pair (so returning to the original mode restores exactly).
    pixel, fraction = region.pixel_rect, region.normalized_rect
    if (not prefer_normalized and region.region_id in ('self_buff', 'target_debuff')
            and pixel is not None and fraction is not None
            and fraction.width > 0 and fraction.height > 0):
        reference_width = pixel.width / fraction.width
        reference_height = pixel.height / fraction.height
        if (abs(reference_height - monitor.bounds.height) < 1
                and abs(reference_width - monitor.bounds.width) >= 1):
            left = round(pixel.x + (monitor.bounds.width - reference_width) / 2)
            top = round(pixel.y)
            right = min(monitor.bounds.width, round(left + pixel.width))
            bottom = min(monitor.bounds.height, round(top + pixel.height))
            left, top = max(0, left), max(0, top)
            if right > left and bottom > top:
                return CaptureRegion(left, top, right-left, bottom-top), False
    candidates = (
        ((region.normalized_rect, True), (region.pixel_rect, False))
        if prefer_normalized
        else ((region.pixel_rect, False), (region.normalized_rect, True))
    )
    for rectangle, normalized in candidates:
        if rectangle is None:
            continue
        if normalized:
            x = round(rectangle.x * monitor.bounds.width)
            y = round(rectangle.y * monitor.bounds.height)
            width = max(1, round(rectangle.width * monitor.bounds.width))
            height = max(1, round(rectangle.height * monitor.bounds.height))
        else:
            x, y = round(rectangle.x), round(rectangle.y)
            width, height = max(1, round(rectangle.width)), max(1, round(rectangle.height))
        if x < 0 or y < 0:
            continue
        width = min(width, monitor.bounds.width - x)
        height = min(height, monitor.bounds.height - y)
        if width > 0 and height > 0:
            return CaptureRegion(x, y, width, height), normalized
    raise CaptureError(f"ROI {region.region_id} is not configured for monitor {monitor.stable_id}")


def resolve_capture_binding(
    document: Mapping[str, Any],
    roi_id: str,
    monitors: Sequence[MonitorInfo],
) -> CaptureBinding:
    """Map persisted monitor/ROI settings to a concrete capture request.

    If the saved monitor disappeared, the current primary monitor is used for
    this session without modifying the saved setting.  This lets the session
    move back automatically if the monitor reconnects.
    """
    if not monitors:
        raise CaptureError("Windows reported no active monitors")
    display = document.get("display")
    if not isinstance(display, Mapping):
        raise CaptureError("settings do not contain display configuration")
    requested_id = display.get("capture_monitor_id")
    if requested_id is not None and not isinstance(requested_id, str):
        raise CaptureError("capture_monitor_id must be a string or null")
    selected = None
    if requested_id:
        selected = next(
            (monitor for monitor in monitors if monitor.stable_id.casefold() == requested_id.casefold()),
            None,
        )
    fallback = selected is None
    if selected is None:
        selected = next((monitor for monitor in monitors if monitor.primary), monitors[0])

    profile = _selected_profile(document)
    raw_regions = profile.get("rois")
    if not isinstance(raw_regions, list):
        raise CaptureError("selected profile does not contain capture ROIs")
    try:
        regions = [StoredCaptureRegion.from_dict(value) for value in raw_regions]
    except (KeyError, TypeError, ValueError) as exc:
        raise CaptureError(f"invalid capture ROI settings: {exc}") from exc
    stored = next((region for region in regions if region.region_id == roi_id), None)
    if stored is None:
        raise CaptureError(f"capture ROI is unavailable: {roi_id}")
    from .auto_roi import automatic, initial_region
    if automatic(stored):
        region=CaptureRegion(*initial_region((0,0,selected.bounds.width,selected.bounds.height),roi_id))
        normalized=False
    else:
        region, normalized = _integer_region(stored, selected, prefer_normalized=fallback)
    request = ScreenCaptureRequest(CaptureTargetKind.MONITOR, selected.stable_id, region)
    return CaptureBinding(request, selected, fallback, normalized)


class ManagedCaptureSession:
    """Own and reconnect one capture source without blocking the UI thread."""

    def __init__(
        self,
        provider: WindowsScreenCaptureProvider,
        settings: Callable[[], Mapping[str, Any]],
        roi_id: str,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        overlay_capture_excluded: Callable[[], bool | None] | None = None,
        initial_backoff_ms: int = 100,
        maximum_backoff_ms: int = 2_000,
        viewport_provider=None,
        auto_cache_directory=None,
    ) -> None:
        if initial_backoff_ms < 1 or maximum_backoff_ms < initial_backoff_ms:
            raise ValueError("invalid capture retry backoff")
        self.provider = provider
        self.viewport_provider = viewport_provider
        self._viewport_anchor = None
        from .auto_roi import AutoRow
        self._auto_row = AutoRow(auto_cache_directory)
        import threading
        self._auto_reset_requested=threading.Event()
        self._auto_initial = None
        self.settings = settings
        self.roi_id = roi_id
        self._clock_ns = clock_ns
        self._overlay_capture_excluded = overlay_capture_excluded
        self._initial_backoff_ns = initial_backoff_ms * 1_000_000
        self._maximum_backoff_ns = maximum_backoff_ms * 1_000_000
        self._source: FrameSource | None = None
        self._binding: CaptureBinding | None = None
        self._failures = 0
        self._retry_at_ns: int | None = None
        self._last_error: str | None = None
        self._opened = False

    @property
    def status(self) -> CaptureSessionStatus:
        if not self._opened:
            state = "closed"
        elif self._source is not None:
            state = "capturing"
        elif self._retry_at_ns is not None:
            state = "backoff"
        else:
            state = "connecting"
        excluded: bool | None = None
        warning: str | None = None
        if self._overlay_capture_excluded is not None:
            try:
                requested = self._overlay_capture_excluded()
                excluded = None if requested is None else bool(requested)
            except Exception as exc:
                warning = f"cannot verify overlay capture exclusion: {exc}"
            else:
                if excluded is False:
                    warning = (
                        "overlay capture exclusion is inactive; preview or overlay pixels "
                        "may be recaptured"
                    )
        return CaptureSessionStatus(
            state,
            self._binding,
            self._failures,
            self._retry_at_ns,
            self._last_error,
            excluded,
            warning,
        )

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("capture session is already open")
        self._opened = True

    def close(self) -> None:
        self._close_source()
        self._opened = False
        self._binding = None
        self._failures = 0
        self._retry_at_ns = None
        self._last_error = None

    def _close_source(self) -> None:
        source, self._source = self._source, None
        if source is not None:
            try:
                source.close()
            except Exception:
                pass

    def _failure(self, error: Exception, now_ns: int) -> CapturePoll:
        self._close_source()
        self._failures += 1
        delay = min(
            self._maximum_backoff_ns,
            self._initial_backoff_ns * (2 ** min(self._failures - 1, 30)),
        )
        self._retry_at_ns = now_ns + delay
        self._last_error = f"{type(error).__name__}: {error}"
        return CapturePoll(None, self.status)

    def observe_detections(self, detections):
        if self._auto_initial is None or self._binding is None:return
        r=self._binding.request.region
        self._auto_row.observe(detections,(r.x,r.y,r.width,r.height),
                               self._auto_initial,self._clock_ns()/1e9)

    def request_auto_reset(self):
        self._auto_reset_requested.set()

    def poll(self) -> CapturePoll:
        if not self._opened:
            raise RuntimeError("capture session is not open")
        if self._auto_reset_requested.is_set():
            self._auto_reset_requested.clear()
            self._auto_row.forget()
        now_ns = self._clock_ns()
        if self._retry_at_ns is not None and now_ns < self._retry_at_ns:
            return CapturePoll(None, self.status)
        try:
            binding = resolve_capture_binding(
                self.settings(), self.roi_id, self.provider.enumerate_monitors()
            )
            from .auto_roi import automatic, initial_region
            from dataclasses import replace
            profile = _selected_profile(self.settings())
            stored = StoredCaptureRegion.from_dict(next(r for r in profile['rois'] if r['region_id']==self.roi_id))
            is_auto=automatic(stored)
            viewport=None
            if self.viewport_provider is not None:
                viewport = self.viewport_provider(binding.monitor)
                if viewport is not None and not is_auto:
                    from dataclasses import replace
                    from .game_viewport import transform_roi
                    profile = _selected_profile(self.settings())
                    raw = next(r for r in profile['rois'] if r['region_id']==self.roi_id)
                    saved = StoredCaptureRegion.from_dict(raw)
                    p=saved.pixel_rect
                    if p is not None:
                        rect=(p.x,p.y,p.width,p.height)
                        reference=profile.get('roi_viewports',{}).get(self.roi_id)
                        if reference and reference.get('rect')==list(rect):
                            anchor=reference['viewport']
                        else:
                            key=(self.roi_id,rect,binding.monitor.stable_id)
                            if self._viewport_anchor is None or self._viewport_anchor[0]!=key:
                                vx,vy,vw,vh=viewport
                                inside=(p.x>=vx and p.y>=vy and p.x+p.width<=vx+vw and p.y+p.height<=vy+vh)
                                n=saved.normalized_rect
                                anchor=viewport if inside else (0,0,
                                    p.width/n.width if n and n.width else binding.monitor.bounds.width,
                                    p.height/n.height if n and n.height else binding.monitor.bounds.height)
                                self._viewport_anchor=(key,anchor)
                            anchor=self._viewport_anchor[1]
                        x,y,w,h=transform_roi(rect,anchor,viewport,self.roi_id)
                        vx,vy,vw,vh=viewport
                        right,bottom=min(x+w,vx+vw),min(y+h,vy+vh)
                        x,y=max(x,vx),max(y,vy)
                        if right>x and bottom>y:
                            request=replace(binding.request,region=CaptureRegion(x,y,right-x,bottom-y))
                            binding=replace(binding,request=request)
            if is_auto:
                viewport=viewport or (0,0,binding.monitor.bounds.width,binding.monitor.bounds.height)
                self._auto_initial=initial_region(viewport,self.roi_id)
                rect=self._auto_row.region((self.settings().get('selected_profile_id'),self.roi_id,binding.monitor,viewport),self._auto_initial,now_ns/1e9)
                binding=replace(binding,request=replace(binding.request,region=CaptureRegion(*rect)))
            else:
                self._auto_initial=None
                self._auto_row.reset()
            # Monitor bounds/DPI may change while the device name stays the
            # same, so compare the complete binding rather than only request.
            if binding != self._binding:
                self._close_source()
                self._binding = binding
            else:
                self._binding = binding
            if self._source is None:
                source = self.provider.create_source(binding.request)
                source.open()
                self._source = source
            frame = self._source.read()
        except Exception as exc:
            return self._failure(exc, now_ns)
        self._failures = 0
        self._retry_at_ns = None
        self._last_error = None
        return CapturePoll(frame, self.status)
