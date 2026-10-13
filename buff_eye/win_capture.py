"""Windows monitor enumeration and GDI pixel capture.

Only public desktop APIs are used.  This module does not access a game process,
send input, inject code, or install a rendering hook.  Importing it is safe on
non-Windows hosts; constructing the native backend there raises a clear error.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Protocol, Sequence

import numpy as np

from .capture import CaptureTargetKind, FrameSource, ScreenCaptureRequest


class CaptureError(RuntimeError):
    """A monitor lookup or desktop capture operation failed."""


class UnsupportedPlatformError(CaptureError):
    """The requested capture facility is unavailable on this platform."""


@dataclass(frozen=True)
class PixelRect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class MonitorInfo:
    """Physical-pixel monitor metadata from the Windows virtual desktop."""

    stable_id: str
    device_name: str
    bounds: PixelRect
    work_area: PixelRect
    primary: bool
    dpi_x: int = 96
    dpi_y: int = 96

    @property
    def scale_percent(self) -> int:
        return round(self.dpi_x * 100 / 96)


class CaptureBackend(Protocol):
    def enumerate_monitors(self) -> Sequence[MonitorInfo]: ...

    def capture_bgra(self, left: int, top: int, width: int, height: int) -> np.ndarray: ...


class WindowsScreenCaptureSource(FrameSource):
    """Frame source for one ROI relative to a selected monitor."""

    def __init__(self, request: ScreenCaptureRequest, backend: CaptureBackend) -> None:
        if request.target_kind is not CaptureTargetKind.MONITOR:
            raise UnsupportedPlatformError("the GDI provider supports monitor capture only")
        self.request = request
        self.backend = backend
        self._monitor: MonitorInfo | None = None

    @property
    def monitor(self) -> MonitorInfo | None:
        return self._monitor

    def open(self) -> None:
        if self._monitor is not None:
            raise RuntimeError("source is already open")
        wanted = self.request.target_id.casefold()
        monitor = next(
            (
                item
                for item in self.backend.enumerate_monitors()
                if item.stable_id.casefold() == wanted
            ),
            None,
        )
        if monitor is None:
            raise CaptureError(f"capture monitor is unavailable: {self.request.target_id}")
        region = self.request.region
        if region.x + region.width > monitor.bounds.width or region.y + region.height > monitor.bounds.height:
            raise CaptureError(
                f"capture region {region.width}x{region.height}+{region.x}+{region.y} "
                f"does not fit monitor {monitor.bounds.width}x{monitor.bounds.height}"
            )
        self._monitor = monitor

    def read(self) -> np.ndarray:
        if self._monitor is None:
            raise RuntimeError("source is not open")
        region = self.request.region
        bgra = self.backend.capture_bgra(
            self._monitor.bounds.left + region.x,
            self._monitor.bounds.top + region.y,
            region.width,
            region.height,
        )
        expected = (region.height, region.width, 4)
        if not isinstance(bgra, np.ndarray) or bgra.dtype != np.uint8 or bgra.shape != expected:
            raise CaptureError(f"backend returned invalid BGRA frame; expected uint8 {expected}")
        # Copy makes the BGR result contiguous and independent of native storage.
        return np.ascontiguousarray(bgra[:, :, :3])

    def close(self) -> None:
        self._monitor = None


class WindowsScreenCaptureProvider:
    """Provider implementing the platform-neutral ScreenCaptureProvider contract."""

    def __init__(self, backend: CaptureBackend | None = None) -> None:
        self.backend = WinGDIBackend() if backend is None else backend

    def enumerate_monitors(self) -> tuple[MonitorInfo, ...]:
        return tuple(self.backend.enumerate_monitors())

    def create_source(self, request: ScreenCaptureRequest) -> WindowsScreenCaptureSource:
        return WindowsScreenCaptureSource(request, self.backend)


class _GdiApi(Protocol):
    def enumerate_monitors(self) -> Sequence[MonitorInfo]: ...
    def get_screen_dc(self) -> Any: ...
    def create_compatible_dc(self, screen_dc: Any) -> Any: ...
    def create_compatible_bitmap(self, screen_dc: Any, width: int, height: int) -> Any: ...
    def select_object(self, dc: Any, obj: Any) -> Any: ...
    def bit_blt(self, destination_dc: Any, screen_dc: Any, left: int, top: int, width: int, height: int) -> None: ...
    def read_bgra(self, dc: Any, bitmap: Any, width: int, height: int) -> bytes: ...
    def delete_object(self, obj: Any) -> None: ...
    def delete_dc(self, dc: Any) -> None: ...
    def release_screen_dc(self, dc: Any) -> None: ...


class WinGDIBackend:
    """GDI resource orchestration with an injectable native-call adapter."""

    def __init__(self, api: _GdiApi | None = None) -> None:
        if api is None:
            if os.name != "nt":
                raise UnsupportedPlatformError("Windows GDI capture is available only on Windows")
            api = _NativeGdiApi()
        self._api = api

    def enumerate_monitors(self) -> tuple[MonitorInfo, ...]:
        monitors = tuple(self._api.enumerate_monitors())
        if not monitors:
            raise CaptureError("Windows reported no active monitors")
        return monitors

    def capture_bgra(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        if width < 1 or height < 1:
            raise ValueError("capture width and height must be positive")
        screen_dc = memory_dc = bitmap = old_object = None
        try:
            screen_dc = self._api.get_screen_dc()
            memory_dc = self._api.create_compatible_dc(screen_dc)
            bitmap = self._api.create_compatible_bitmap(screen_dc, width, height)
            old_object = self._api.select_object(memory_dc, bitmap)
            self._api.bit_blt(memory_dc, screen_dc, left, top, width, height)
            pixels = self._api.read_bgra(memory_dc, bitmap, width, height)
            expected = width * height * 4
            if len(pixels) != expected:
                raise CaptureError(f"GetDIBits returned {len(pixels)} bytes; expected {expected}")
            return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4).copy()
        except CaptureError:
            raise
        except Exception as exc:
            raise CaptureError(f"Windows desktop capture failed: {exc}") from exc
        finally:
            # A bitmap cannot be deleted while selected into a DC.
            if memory_dc is not None and old_object is not None:
                try:
                    self._api.select_object(memory_dc, old_object)
                except Exception:
                    pass
            if bitmap is not None:
                try:
                    self._api.delete_object(bitmap)
                except Exception:
                    pass
            if memory_dc is not None:
                try:
                    self._api.delete_dc(memory_dc)
                except Exception:
                    pass
            if screen_dc is not None:
                try:
                    self._api.release_screen_dc(screen_dc)
                except Exception:
                    pass


class _NativeGdiApi:
    """Thin ctypes adapter.  Kept private so tests can replace every WinAPI call."""

    SRCCOPY = 0x00CC0020
    CAPTUREBLT = 0x40000000
    MONITORINFOF_PRIMARY = 1
    DIB_RGB_COLORS = 0

    def __init__(self) -> None:
        if os.name != "nt":
            raise UnsupportedPlatformError("Windows GDI capture is available only on Windows")
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.wintypes = wintypes
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        try:
            self.shcore = ctypes.WinDLL("shcore", use_last_error=True)
        except OSError:
            self.shcore = None
        self._define_types()
        self._declare_signatures()
        # Per-monitor awareness keeps coordinates and captured pixels in the same
        # physical coordinate system. ERROR_ACCESS_DENIED means it was set earlier.
        try:
            self.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            pass

    def _define_types(self) -> None:
        ctypes, wt = self.ctypes, self.wintypes

        class MONITORINFOEXW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wt.DWORD),
                ("rcMonitor", wt.RECT),
                ("rcWork", wt.RECT),
                ("dwFlags", wt.DWORD),
                ("szDevice", wt.WCHAR * 32),
            ]

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG),
                ("biPlanes", wt.WORD), ("biBitCount", wt.WORD), ("biCompression", wt.DWORD),
                ("biSizeImage", wt.DWORD), ("biXPelsPerMeter", wt.LONG),
                ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD),
                ("biClrImportant", wt.DWORD),
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wt.DWORD * 3)]

        self.MONITORINFOEXW = MONITORINFOEXW
        self.BITMAPINFOHEADER = BITMAPINFOHEADER
        self.BITMAPINFO = BITMAPINFO
        self.MONITORENUMPROC = ctypes.WINFUNCTYPE(
            wt.BOOL, wt.HMONITOR, wt.HDC, ctypes.POINTER(wt.RECT), wt.LPARAM
        )

    def _declare_signatures(self) -> None:
        c, wt = self.ctypes, self.wintypes
        self.user32.EnumDisplayMonitors.argtypes = [wt.HDC, c.POINTER(wt.RECT), self.MONITORENUMPROC, wt.LPARAM]
        self.user32.EnumDisplayMonitors.restype = wt.BOOL
        self.user32.GetMonitorInfoW.argtypes = [wt.HMONITOR, c.POINTER(self.MONITORINFOEXW)]
        self.user32.GetMonitorInfoW.restype = wt.BOOL
        self.user32.GetDC.argtypes = [wt.HWND]
        self.user32.GetDC.restype = wt.HDC
        self.user32.ReleaseDC.argtypes = [wt.HWND, wt.HDC]
        self.user32.ReleaseDC.restype = c.c_int
        self.gdi32.CreateCompatibleDC.argtypes = [wt.HDC]
        self.gdi32.CreateCompatibleDC.restype = wt.HDC
        self.gdi32.CreateCompatibleBitmap.argtypes = [wt.HDC, c.c_int, c.c_int]
        self.gdi32.CreateCompatibleBitmap.restype = wt.HBITMAP
        self.gdi32.SelectObject.argtypes = [wt.HDC, wt.HGDIOBJ]
        self.gdi32.SelectObject.restype = wt.HGDIOBJ
        self.gdi32.BitBlt.argtypes = [wt.HDC, c.c_int, c.c_int, c.c_int, c.c_int, wt.HDC, c.c_int, c.c_int, wt.DWORD]
        self.gdi32.BitBlt.restype = wt.BOOL
        self.gdi32.GetDIBits.argtypes = [wt.HDC, wt.HBITMAP, wt.UINT, wt.UINT, wt.LPVOID, c.POINTER(self.BITMAPINFO), wt.UINT]
        self.gdi32.GetDIBits.restype = c.c_int
        self.gdi32.DeleteObject.argtypes = [wt.HGDIOBJ]
        self.gdi32.DeleteObject.restype = wt.BOOL
        self.gdi32.DeleteDC.argtypes = [wt.HDC]
        self.gdi32.DeleteDC.restype = wt.BOOL

    def _error(self, operation: str) -> CaptureError:
        code = self.ctypes.get_last_error()
        return CaptureError(f"{operation} failed (WinError {code})")

    @staticmethod
    def _rect(value: Any) -> PixelRect:
        return PixelRect(value.left, value.top, value.right, value.bottom)

    def _dpi(self, monitor: Any) -> tuple[int, int]:
        if self.shcore is None or not hasattr(self.shcore, "GetDpiForMonitor"):
            return (96, 96)
        x, y = self.wintypes.UINT(), self.wintypes.UINT()
        try:
            result = self.shcore.GetDpiForMonitor(monitor, 0, self.ctypes.byref(x), self.ctypes.byref(y))
        except (AttributeError, OSError):
            return (96, 96)
        return (int(x.value), int(y.value)) if result == 0 else (96, 96)

    def enumerate_monitors(self) -> tuple[MonitorInfo, ...]:
        results: list[MonitorInfo] = []

        def callback(handle: Any, _dc: Any, _rect: Any, _data: int) -> bool:
            info = self.MONITORINFOEXW()
            info.cbSize = self.ctypes.sizeof(info)
            if not self.user32.GetMonitorInfoW(handle, self.ctypes.byref(info)):
                return False
            dpi_x, dpi_y = self._dpi(handle)
            device = str(info.szDevice)
            results.append(
                MonitorInfo(
                    stable_id=device,
                    device_name=device,
                    bounds=self._rect(info.rcMonitor),
                    work_area=self._rect(info.rcWork),
                    primary=bool(info.dwFlags & self.MONITORINFOF_PRIMARY),
                    dpi_x=dpi_x,
                    dpi_y=dpi_y,
                )
            )
            return True

        proc = self.MONITORENUMPROC(callback)
        if not self.user32.EnumDisplayMonitors(None, None, proc, 0):
            raise self._error("EnumDisplayMonitors")
        return tuple(results)

    def get_screen_dc(self) -> Any:
        value = self.user32.GetDC(None)
        if not value:
            raise self._error("GetDC")
        return value

    def create_compatible_dc(self, screen_dc: Any) -> Any:
        value = self.gdi32.CreateCompatibleDC(screen_dc)
        if not value:
            raise self._error("CreateCompatibleDC")
        return value

    def create_compatible_bitmap(self, screen_dc: Any, width: int, height: int) -> Any:
        value = self.gdi32.CreateCompatibleBitmap(screen_dc, width, height)
        if not value:
            raise self._error("CreateCompatibleBitmap")
        return value

    def select_object(self, dc: Any, obj: Any) -> Any:
        value = self.gdi32.SelectObject(dc, obj)
        if not value or int(value) == -1:
            raise self._error("SelectObject")
        return value

    def bit_blt(self, destination_dc: Any, screen_dc: Any, left: int, top: int, width: int, height: int) -> None:
        # CAPTUREBLT includes layered overlay windows, but some display drivers
        # reject that flag even though an ordinary desktop copy is available.
        # Retry with SRCCOPY so capture still works on those systems.  Clear and
        # record last-error around each call; otherwise an earlier DPI-awareness
        # call can leave ERROR_ACCESS_DENIED behind and produce a misleading
        # diagnosis here.
        attempts: list[tuple[str, int]] = []
        for mode, raster_operation in (
            ("SRCCOPY|CAPTUREBLT", self.SRCCOPY | self.CAPTUREBLT),
            ("SRCCOPY", self.SRCCOPY),
        ):
            self.ctypes.set_last_error(0)
            if self.gdi32.BitBlt(
                destination_dc,
                0,
                0,
                width,
                height,
                screen_dc,
                left,
                top,
                raster_operation,
            ):
                return
            attempts.append((mode, int(self.ctypes.get_last_error())))

        details = "; ".join(f"{mode} WinError {code}" for mode, code in attempts)
        hint = ""
        if any(code in (5, 6) for _, code in attempts):
            hint = (
                "; the process may not have access to the interactive desktop "
                "session (run it as the signed-in user in that session)"
            )
        raise CaptureError(f"BitBlt failed ({details}){hint}")

    def read_bgra(self, dc: Any, bitmap: Any, width: int, height: int) -> bytes:
        info = self.BITMAPINFO()
        info.bmiHeader.biSize = self.ctypes.sizeof(self.BITMAPINFOHEADER)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = -height  # top-down, matching screen coordinates
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        buffer = self.ctypes.create_string_buffer(width * height * 4)
        lines = self.gdi32.GetDIBits(dc, bitmap, 0, height, buffer, self.ctypes.byref(info), self.DIB_RGB_COLORS)
        if lines != height:
            raise self._error("GetDIBits")
        return buffer.raw

    def delete_object(self, obj: Any) -> None:
        if not self.gdi32.DeleteObject(obj):
            raise self._error("DeleteObject")

    def delete_dc(self, dc: Any) -> None:
        if not self.gdi32.DeleteDC(dc):
            raise self._error("DeleteDC")

    def release_screen_dc(self, dc: Any) -> None:
        if not self.user32.ReleaseDC(None, dc):
            raise self._error("ReleaseDC")
