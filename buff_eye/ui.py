"""PySide6 UI boundary and GUI-independent settings controller.

PySide6 is imported only by :func:`create_qt_application`.  The recognition
engine and its tests therefore remain usable on headless machines.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .overlay_model import ROI_IDS, CaptureRegion, IconPlacement, OverlayLayout, Rect
from .presentation import compose_overlay_items, display_slot
from .effect_groups import grouped_order, CLASS_ATTACK, SUPPORT_EFFECTS as MERGED_COMMON, SUPPORT_LABELS as COMMON_LABELS, OVERLAY_LABELS, GREEN_EFFECTS, GREEN_ASSETS, enabled_green
from .target_buffs import TARGET_SLOTS, TARGET_LABELS, target_items
from .settings import SettingsStore
from .roi_snapshot import capture_monitor_snapshot


class UiDependencyError(RuntimeError):
    """Raised when the optional desktop UI dependency is unavailable."""


# 96px icon + two 36px text bands + 16px gap per row.
TARGET_DEFAULT_Y = 24
SUPPORT_DEFAULT_Y = 208
GREEN_DEFAULT_Y = 392


@dataclass(frozen=True)
class MonitorChoice:
    monitor_id: str
    label: str
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    primary: bool = False


def resolve_qt_screen(monitor_id, monitors, screens):
    """Match Windows device IDs to Qt product names using native screen bounds.

    On Windows Qt retains native screen origins but scales sizes to DIPs.
    Do not use enumeration order: the two providers may order screens differently.
    """
    direct = next((screen for screen in screens if screen.name() == monitor_id), None)
    if direct is not None:
        return direct
    monitor = next((m for m in monitors if m.monitor_id == monitor_id), None)
    if monitor is None:
        return None
    matches = []
    for screen in screens:
        rect = screen.geometry()
        ratio = screen.devicePixelRatio()
        if (abs(rect.x() - monitor.x) <= 1 and abs(rect.y() - monitor.y) <= 1
                and abs(rect.width() * ratio - monitor.width) <= 2
                and abs(rect.height() * ratio - monitor.height) <= 2):
            matches.append(screen)
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class PreviewItem:
    effect_id: str
    confidence: float | None = None
    remaining_text: str | None = None
    stack_text: str | None = None
    recognized: bool = True
    icon_frame: Any | None = field(default=None, compare=False, repr=False)
    border_color: Any | None = None


@dataclass
class RecognitionPreview:
    """Latest recognition output shown by settings and overlay windows."""

    frame: Any | None = None
    region_id: str = "self_buff"
    items: tuple[PreviewItem, ...] = ()
    status: str = "대기 중"
    sequence: int = 0


class SettingsController:
    """Edit persistent settings while keeping Qt out of the data boundary."""

    def __init__(self, store: SettingsStore | None = None) -> None:
        self.store = store or SettingsStore()
        self.document = self.store.load()
        self.monitors: tuple[MonitorChoice, ...] = ()
        self.preview = RecognitionPreview()
        self.overlay_capture_excluded: bool | None = None

    @property
    def profile(self) -> dict[str, Any]:
        return self.document["profiles"][self.document["selected_profile_id"]]

    @property
    def overlay(self) -> OverlayLayout:
        return OverlayLayout.from_dict(self.profile["overlay"])

    @property
    def rois(self) -> dict[str, CaptureRegion]:
        return {
            region.region_id: region
            for region in (CaptureRegion.from_dict(item) for item in self.profile["rois"])
        }

    def set_monitors(self, monitors: Iterable[MonitorChoice]) -> None:
        choices = tuple(monitors)
        if len({item.monitor_id for item in choices}) != len(choices):
            raise ValueError("monitor ids must be unique")
        self.monitors = choices

    def select_monitor(self, role: str, monitor_id: str | None) -> None:
        key = f"{role}_monitor_id"
        if key not in ("capture_monitor_id", "overlay_monitor_id", "settings_monitor_id"):
            raise ValueError(f"unknown monitor role: {role}")
        if monitor_id is not None and self.monitors and monitor_id not in {
            item.monitor_id for item in self.monitors
        }:
            raise ValueError(f"unknown monitor id: {monitor_id}")
        self.document["display"][key] = monitor_id
        if role == "overlay":
            layout = self.overlay
            layout.monitor_id = monitor_id
            self.profile["overlay"] = layout.to_dict()

    def set_roi(self, region_id: str, pixel_rect: Rect | None) -> None:
        if region_id not in ROI_IDS:
            raise ValueError(f"unknown ROI: {region_id}")
        regions = self.rois
        old = regions[region_id]
        normalized = old.normalized_rect if pixel_rect is not None else None
        if pixel_rect is not None:
            monitor_id = self.document["display"].get("capture_monitor_id")
            monitor = next((item for item in self.monitors if item.monitor_id == monitor_id), None)
            if monitor is None:
                monitor = next((item for item in self.monitors if item.primary), None)
            if (
                monitor is not None and monitor.width > 0 and monitor.height > 0
                and pixel_rect.x >= 0 and pixel_rect.y >= 0
                and pixel_rect.x + pixel_rect.width <= monitor.width
                and pixel_rect.y + pixel_rect.height <= monitor.height
            ):
                normalized = Rect(
                    pixel_rect.x / monitor.width,
                    pixel_rect.y / monitor.height,
                    pixel_rect.width / monitor.width,
                    pixel_rect.height / monitor.height,
                )
        regions[region_id] = CaptureRegion(region_id, pixel_rect, normalized)
        self.profile["rois"] = [regions[key].to_dict() for key in ROI_IDS]
        self._remember_roi_viewport(region_id,pixel_rect)

    def _remember_roi_viewport(self,region_id,pixel_rect):
        from .game_viewport import cached
        viewport=cached(self.document['display'].get('capture_monitor_id'))
        references=self.profile.setdefault('roi_viewports',{})
        if viewport and pixel_rect is not None:
            references[region_id]={'viewport':list(viewport),
                'rect':[pixel_rect.x,pixel_rect.y,pixel_rect.width,pixel_rect.height]}
        else:
            references.pop(region_id,None)

    def set_roi_from_preview(
        self,
        region_id: str,
        pixel_rect: Rect,
        *,
        monitor_width: int,
        monitor_height: int,
    ) -> None:
        """Persist one visual selection in pixel and normalized coordinates."""
        if region_id not in ROI_IDS:
            raise ValueError(f"unknown ROI: {region_id}")
        if monitor_width < 1 or monitor_height < 1:
            raise ValueError("monitor dimensions must be positive")
        normalized = Rect(
            pixel_rect.x / monitor_width,
            pixel_rect.y / monitor_height,
            pixel_rect.width / monitor_width,
            pixel_rect.height / monitor_height,
        )
        region = CaptureRegion(region_id, pixel_rect, normalized)
        regions = self.rois
        regions[region_id] = region
        self.profile["rois"] = [regions[key].to_dict() for key in ROI_IDS]
        self._remember_roi_viewport(region_id,pixel_rect)

    def set_game_ui(self, key: str, value: Any) -> None:
        """Binding point for HUD/buff/target scales and in-game display options."""
        allowed = {
            "hud_scale_pct",
            "buff_scale_pct",
            "target_scale_pct",
            "buff_display_count",
            "target_display_count",
            "stack_merge",
        }
        if key not in allowed:
            raise ValueError(f"unknown game UI setting: {key}")
        self.profile.setdefault("game_ui", {})[key] = value

    def ensure_icon(self, effect_id: str) -> IconPlacement:
        layout = self.overlay
        if effect_id not in layout.icons:
            index = len(layout.icons)
            x,y,size=24+index*112,SUPPORT_DEFAULT_Y,96
            if effect_id=='support.major_code' and 'support.attack.01' in layout.icons:
                anchor=layout.icons['support.attack.01'];size=anchor.size
                x,y=anchor.x+size*7/6,anchor.y
            elif effect_id in GREEN_EFFECTS:
                anchor=layout.icons.get('support.hyper.01')
                if anchor is not None:
                    size=anchor.size;x=anchor.x+GREEN_EFFECTS.index(effect_id)*size*7/6
                    y=anchor.y+size+2*(max(20,round(size*.32))+max(3,round(size*.05)))+size/6
            layout.add_icon(IconPlacement(effect_id, x,y,size,0))
            self.profile["overlay"] = layout.to_dict()
        return self.overlay.icons[effect_id]

    def set_position_locked(self, locked: bool) -> None:
        layout = self.overlay
        layout.position_locked = bool(locked)
        self.profile["overlay"] = layout.to_dict()

    def compact_green_icons(self) -> None:
        """Reflow checked slots only; recognition never changes layout."""
        for effect in GREEN_EFFECTS:
            self.ensure_icon(effect)
        layout=self.overlay
        anchor=layout.icons[GREEN_EFFECTS[0]]
        x,y=anchor.x,anchor.y
        for effect in enabled_green(self.profile):
            icon=layout.icons[effect]
            icon.x,icon.y=x,y
            x+=icon.size*7/6
        self.profile['overlay']=layout.to_dict()

    def place_target_defaults(self, *, reset: bool = False) -> None:
        """Initialize target slots; upgrade untouched old default coordinates."""
        layout = self.overlay
        legacy = any(all(
            effect in layout.icons
            and (layout.icons[effect].x, layout.icons[effect].y, layout.icons[effect].size)
                == (24 + 112 * index, old_y, 96)
            for index, effect in enumerate(('target.dark', 'target.support'))
        ) for old_y in (250, 392))
        # Move untouched former support rows once; custom coordinates remain saved.
        for effect, icon in layout.icons.items():
            if self.profile.get('overlay_rows_version') == 2 or icon.size != 96:
                continue
            if effect in GREEN_EFFECTS and icon.y == 208:
                icon.y = GREEN_DEFAULT_Y
            elif effect in MERGED_COMMON and effect not in GREEN_EFFECTS and icon.y == 24:
                icon.y = SUPPORT_DEFAULT_Y
        self.profile['overlay'] = layout.to_dict()
        self.profile['overlay_rows_version'] = 2
        missing = set(TARGET_SLOTS) - set(layout.icons)
        for effect in TARGET_SLOTS:
            self.ensure_icon(effect)
        layout = self.overlay
        for index, effect in enumerate(TARGET_SLOTS):
            if reset or legacy or effect in missing:
                layout.icons[effect].x = 24 + 112 * index
                layout.icons[effect].y = TARGET_DEFAULT_Y
        self.profile['overlay'] = layout.to_dict()

    def reset_overlay_positions(self) -> bool:
        layout = self.overlay
        if layout.position_locked:
            return False
        for index, effect_id in enumerate(grouped_order(set(layout.icons)-set(TARGET_SLOTS))):
            layout.icons[effect_id].x = 24 + (GREEN_EFFECTS.index(effect_id) if effect_id in GREEN_EFFECTS else index) * 112
            layout.icons[effect_id].y = GREEN_DEFAULT_Y if effect_id in GREEN_EFFECTS else SUPPORT_DEFAULT_Y
        self.profile['overlay'] = layout.to_dict()
        return True

    def move_icon(self, effect_id: str, x: float, y: float) -> bool:
        layout = self.overlay
        if effect_id not in layout.icons:
            layout.add_icon(IconPlacement(effect_id))
        accepted = layout.request_move(effect_id, x, y)
        if accepted:
            self.profile["overlay"] = layout.to_dict()
        return accepted

    def move_common_group(self, dx: float, dy: float) -> bool:
        layout = self.overlay
        if layout.position_locked:
            return False
        for effect_id in MERGED_COMMON:
            if effect_id in layout.icons:
                layout.icons[effect_id].x += dx
                layout.icons[effect_id].y += dy
        self.profile['overlay'] = layout.to_dict()
        return True

    def resize_common_group(self, size: float) -> bool:
        layout = self.overlay
        if layout.position_locked:
            return False
        icons = [layout.icons[e] for e in MERGED_COMMON if e in layout.icons]
        if not icons:
            return False
        size = max(24, min(512, size))
        anchor_x, anchor_y = icons[0].x, icons[0].y
        factor = size / icons[0].size
        # Preserve the layout direction and proportions, including vertical cards.
        # Also repair spacing left by older versions that resized only the images.
        padding = size / 6
        card_width = size + padding
        card_height = size + 2 * (max(20, round(size * .32)) + max(3, round(size * .05))) + padding
        for index, first in enumerate(icons):
            for second in icons[index + 1:]:
                dx, dy = abs(second.x-first.x), abs(second.y-first.y)
                if dx or dy:
                    factor = max(factor, min(card_width/dx if dx else float('inf'),
                                             card_height/dy if dy else float('inf')))
        for icon in icons:
            icon.x = anchor_x + (icon.x-anchor_x) * factor
            icon.y = anchor_y + (icon.y-anchor_y) * factor
            icon.size = size
        self.profile['overlay'] = layout.to_dict()
        return True

    def resize_icon(self, effect_id: str, size: float) -> bool:
        layout = self.overlay
        if effect_id not in layout.icons:
            layout.add_icon(IconPlacement(effect_id))
        accepted = layout.request_resize(effect_id, size)
        if accepted:
            self.profile["overlay"] = layout.to_dict()
        return accepted

    def set_icon_transparency(self, effect_id: str, percent: float) -> None:
        layout = self.overlay
        if effect_id not in layout.icons:
            layout.add_icon(IconPlacement(effect_id))
        layout.set_transparency(effect_id, percent)
        self.profile["overlay"] = layout.to_dict()

    def update_preview(
        self,
        *,
        frame: Any | None,
        region_id: str,
        detections: Sequence[PreviewItem | Mapping[str, Any]],
        status: str = "인식 중",
        sequence: int | None = None,
    ) -> RecognitionPreview:
        if region_id not in ROI_IDS:
            raise ValueError(f"unknown ROI: {region_id}")
        items = tuple(
            item if isinstance(item, PreviewItem) else PreviewItem(**dict(item))
            for item in detections
        )
        self.preview = RecognitionPreview(
            frame=frame,
            region_id=region_id,
            items=items,
            status=status,
            sequence=self.preview.sequence + 1 if sequence is None else sequence,
        )
        return self.preview

    def save(self) -> None:
        self.store.save(self.document)


@dataclass
class QtDesktop:
    application: Any
    settings_window: Any
    overlay_window: Any
    live_worker: Any | None = None
    live_timer: Any | None = None

    def show(self) -> None:
        self.settings_window.show()
        self.overlay_window.show()

    def exec(self) -> int:
        self.show()
        return int(self.application.exec())

    def pump_live_once(self) -> bool:
        """Transfer at most one latest worker result onto the Qt thread."""
        if self.live_worker is None:
            return False
        update = self.live_worker.take_latest()
        if update is None:
            return False
        if update.error and getattr(update,'target_update',None) is None:
            self.settings_window.set_live_error(update.status, update.error)
        else:
            self.settings_window.apply_live_update(update)
        return True

    def shutdown(self) -> None:
        if getattr(self.settings_window, 'dev_timer', None) is not None:
            self.settings_window.dev_timer.stop()
        if self.live_timer is not None:
            self.live_timer.stop()
        if self.live_worker is not None:
            self.live_worker.stop()


def _load_qt() -> tuple[Any, Any, Any]:
    try:
        return tuple(importlib.import_module(f"PySide6.{name}") for name in ("QtCore", "QtGui", "QtWidgets"))  # type: ignore[return-value]
    except (ImportError, ModuleNotFoundError) as exc:
        raise UiDependencyError(
            "데스크톱 UI를 실행하려면 PySide6가 필요합니다. "
            "프로젝트 실행 환경에 'PySide6'를 설치한 뒤 다시 실행하세요."
        ) from exc


def refresh_worker_monitors(controller: SettingsController, worker: Any) -> bool:
    """Refresh physical geometry without changing the user's monitor/ROI choices."""
    source = getattr(worker, 'support', worker)
    provider = getattr(getattr(source, 'session', None), 'provider', None)
    if provider is None:
        return False
    monitors = tuple(provider.enumerate_monitors())
    if not monitors:
        return False
    controller.set_monitors(MonitorChoice(
        m.stable_id, f'{m.device_name} ({m.bounds.width}×{m.bounds.height})',
        m.bounds.left, m.bounds.top, m.bounds.width, m.bounds.height, m.primary)
        for m in monitors)
    return True


def _monitor_choices(application: Any) -> tuple[MonitorChoice, ...]:
    choices = []
    primary = application.primaryScreen()
    for index, screen in enumerate(application.screens()):
        geometry = screen.geometry()
        # Qt screen name is stable enough for the current session. Native display
        # discovery may replace it with an EDID/device-path id without UI changes.
        monitor_id = screen.name() or f"screen-{index}"
        choices.append(
            MonitorChoice(
                monitor_id,
                f"{monitor_id} ({geometry.width()}×{geometry.height()})",
                geometry.x(), geometry.y(), geometry.width(), geometry.height(),
                screen is primary,
            )
        )
    return tuple(choices)


def _apply_windows_click_through(widget: Any, locked: bool) -> None:
    """Mirror Qt's input-transparent flags at the Win32 extended-style layer."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    getter = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
    setter = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
    long_ptr = ctypes.c_ssize_t
    getter.argtypes = [wintypes.HWND, ctypes.c_int]
    getter.restype = long_ptr
    setter.argtypes = [wintypes.HWND, ctypes.c_int, long_ptr]
    setter.restype = long_ptr
    hwnd = int(widget.winId())
    # Headless Qt platform plugins expose an opaque WId which is not an HWND.
    # Qt's own input-transparent flags are still testable in that environment;
    # native styles are applied only when Windows confirms the handle.
    if not user32.IsWindow(hwnd):
        return
    gwl_exstyle = -20
    ws_ex_layered = 0x00080000
    ws_ex_transparent = 0x00000020
    ws_ex_noactivate = 0x08000000
    style = int(getter(hwnd, gwl_exstyle)) | ws_ex_layered
    if locked:
        style |= ws_ex_transparent | ws_ex_noactivate
    else:
        style &= ~(ws_ex_transparent | ws_ex_noactivate)
    ctypes.set_last_error(0)
    previous = setter(hwnd, gwl_exstyle, style)
    error = ctypes.get_last_error()
    if previous == 0 and error:
        raise OSError(error, "SetWindowLongPtrW failed for the overlay window")


def _apply_windows_capture_exclusion(widget: Any, excluded: bool = True) -> bool | None:
    """Ask Windows desktop capture APIs to omit the overlay itself.

    ``None`` means that no native HWND exists (for example, Qt offscreen).
    ``False`` means Windows rejected or does not support the affinity; the
    overlay remains usable and callers can surface that diagnostic.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
    hwnd = int(widget.winId())
    if not user32.IsWindow(hwnd):
        return None
    affinity = 0x00000011 if excluded else 0
    ctypes.set_last_error(0)
    applied = bool(user32.SetWindowDisplayAffinity(hwnd, affinity))
    return applied if excluded else (False if applied else None)


def create_qt_application(
    controller: SettingsController | None = None,
    argv: Sequence[str] | None = None,
    *,
    live_worker: Any | None = None,
    preview_interval_ms: int = 16,
    snapshot_provider: Any = capture_monitor_snapshot,
    default_preview_roi: Rect | None = None,
) -> QtDesktop:
    """Create the settings and overlay windows; performs no game input or hooking."""
    QtCore, QtGui, QtWidgets = _load_qt()
    controller = controller or SettingsController()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(list(argv or []))
    overlay_font_family = ""
    if sys.platform == "win32" and not QtGui.QFontDatabase.families():
        # Qt's offscreen Windows plugin may not enumerate system fonts. Load a
        # known Korean-capable system font so evidence renders real glyphs.
        for font_path in (Path(r"C:\Windows\Fonts\malgun.ttf"), Path(r"C:\Windows\Fonts\arial.ttf")):
            if font_path.exists():
                font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path))
                families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
                if families and not overlay_font_family:
                    overlay_font_family = families[0]
    if not controller.monitors:
        controller.set_monitors(_monitor_choices(app))
    if preview_interval_ms < 1:
        raise ValueError("preview_interval_ms must be positive")
    settings_window = None
    dev_overlay = None
    # Load the idle/editing artwork once, also when capture is disabled.
    default_support_icons = dict(getattr(getattr(live_worker, 'detector', None), 'overlay_icons', {}))
    if any(effect not in default_support_icons for effect in MERGED_COMMON):
        from .app import DEFAULT_CONFIG
        from .core import image_read, read_json, safe_path
        for effect, path in read_json(DEFAULT_CONFIG).get('overlay_icons', {}).items():
            if effect in MERGED_COMMON and effect not in default_support_icons:
                default_support_icons[effect] = image_read(safe_path(DEFAULT_CONFIG.parent, path))

    def rendered_layout():
        base = controller.overlay
        layout = base if dev_overlay is None else dev_overlay.layout(base)
        for effect in CLASS_ATTACK:
            slot = display_slot(effect, controller.profile)
            if slot != effect and slot in layout.icons:
                values = layout.icons[slot].to_dict()
                values['effect_id'] = effect
                layout.icons[effect] = IconPlacement.from_dict(values)
        return layout

    def editing_allowed(effect_id=None) -> bool:
        unlocked = (not controller.profile.get('target_locked', True) if effect_id in TARGET_SLOTS
                    else not controller.overlay.position_locked)
        if effect_id is None:
            unlocked = unlocked or not controller.profile.get('target_locked', True)
        return (
            unlocked
            and settings_window is not None
            and settings_window.isVisible()
            and not settings_window.isMinimized()
            and not (getattr(settings_window, "roi_dialog", None) is not None
                     and settings_window.roi_dialog.isVisible())
        )

    class OverlayIconWidget(QtWidgets.QWidget):
        def __init__(self, effect_id: str, parent: Any) -> None:
            super().__init__(parent)
            self.effect_id = effect_id
            self._origin = None
            self._resizing = False
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            shadow = QtWidgets.QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(6)
            shadow.setOffset(0, 2)
            shadow.setColor(QtGui.QColor(0, 0, 0, 85))
            self.setGraphicsEffect(shadow)

        def paintEvent(self, event: Any) -> None:
            placement = rendered_layout().icons[self.effect_id]
            painter = QtGui.QPainter(self)
            painter.setOpacity(max(0.0, 1.0 - placement.transparency_pct / 100.0))
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            preview_item = next(
                (item for item in controller.preview.items if item.effect_id == self.effect_id),
                None,
            )
            icon_frame = None if preview_item is None else preview_item.icon_frame
            if icon_frame is None and self.effect_id in MERGED_COMMON:
                icon_frame = default_support_icons.get(self.effect_id)
            if icon_frame is None and self.effect_id in TARGET_SLOTS and hasattr(live_worker,'target'):
                source='target.dark' if self.effect_id=='target.dark' else 'target.bard'
                icon_frame=live_worker.target.detector.overlay_icons.get(source)
            if dev_overlay is not None:
                icon_frame = dev_overlay.images.get(self.effect_id, icon_frame)
            if icon_frame is not None and hasattr(icon_frame, "shape") and len(icon_frame.shape) == 3:
                height, width = icon_frame.shape[:2]
                image = QtGui.QImage(
                    icon_frame.data,
                    width,
                    height,
                    icon_frame.strides[0],
                    QtGui.QImage.Format.Format_BGR888,
                ).copy()
                fitted_size = image.size().scaled(self.icon_rect.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                target = QtCore.QRect(QtCore.QPoint(0, 0), fitted_size)
                target.moveCenter(self.icon_rect.center())
                painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
                painter.drawImage(target, image)
            else:
                painter.fillRect(self.icon_rect, QtGui.QColor(17, 23, 28, 210))
                painter.setPen(QtGui.QColor("white"))
                painter.drawText(
                    self.icon_rect,
                    QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap,
                    OVERLAY_LABELS.get(self.effect_id, self.effect_id)
                )
            pen_width = max(1, round(self.icon_rect.width() * 2 / 37))
            border_color = (QtGui.QColor(255,65,65) if self.effect_id in TARGET_SLOTS else
                            QtGui.QColor(80,220,100) if self.effect_id in GREEN_EFFECTS else QtGui.QColor(255,215,70))
            border_pen = QtGui.QPen(border_color, pen_width)
            border_pen.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)
            painter.setPen(border_pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            inset = pen_width / 2
            painter.drawRect(QtCore.QRectF(self.icon_rect).adjusted(inset,inset,-inset,-inset))
            # Icon, border and text panels share the configured opacity.
            if preview_item is not None:
                font = painter.font()
                if overlay_font_family:
                    font.setFamily(overlay_font_family)
                font.setBold(True)
                font.setPixelSize(max(11, round(self.width() * 0.24)))
                painter.setFont(font)
                for rect, text, color in (
                    (self.stack_rect, preview_item.stack_text, QtGui.QColor(185, 255, 190)),
                    (self.time_rect, preview_item.remaining_text, QtGui.QColor(255, 255, 255)),
                ):
                    if not text:
                        continue
                    if rect == self.time_rect and controller.profile.get('target_time_centered' if self.effect_id in TARGET_SLOTS else 'time_text_centered', False):
                        text = text.replace('초', '')
                    if rect == self.stack_rect and text == '▼':
                        # Game-style overlap mark: a bar over a downward triangle.
                        # Draw geometry so font substitutions cannot alter its shape.
                        mark_width = min(rect.height() * 0.85, self.icon_rect.width() * 0.34)
                        center_x = rect.center().x()
                        top = rect.bottom() - mark_width * 0.9
                        painter.save()
                        painter.setOpacity(max(0.0, 1.0 - placement.transparency_pct / 100.0))
                        painter.setPen(QtCore.Qt.PenStyle.NoPen)
                        triangle = QtGui.QPolygonF([
                            QtCore.QPointF(center_x - mark_width / 2, top + mark_width * 0.3),
                            QtCore.QPointF(center_x + mark_width / 2, top + mark_width * 0.3),
                            QtCore.QPointF(center_x, top + mark_width * 0.9),
                        ])
                        painter.setBrush(QtGui.QColor(75, 255, 95, 90))
                        painter.setPen(QtGui.QPen(QtGui.QColor(60, 255, 80, 100), max(2, mark_width * 0.24)))
                        painter.drawPolygon(triangle)
                        painter.drawLine(QtCore.QPointF(center_x - mark_width / 2, top),
                                         QtCore.QPointF(center_x + mark_width / 2, top))
                        painter.setPen(QtCore.Qt.PenStyle.NoPen)
                        painter.setBrush(QtGui.QColor(225, 255, 228))
                        painter.drawPolygon(triangle)
                        painter.fillRect(QtCore.QRectF(center_x - mark_width / 2, top - 1,
                                                      mark_width, max(2, mark_width * 0.09)),
                                         QtGui.QColor(225, 255, 228))
                        painter.restore()
                        continue
                    fitted = QtGui.QFont(font)
                    green_stack = rect == self.stack_rect and self.effect_id in GREEN_EFFECTS
                    if rect == self.time_rect or green_stack:
                        fitted.setWeight(QtGui.QFont.Weight.Black)
                        if green_stack or controller.profile.get('target_time_centered' if self.effect_id in TARGET_SLOTS else 'time_text_centered', False):
                            fitted.setPixelSize(round(fitted.pixelSize() * 1.25))
                    fit_text = text
                    while fitted.pixelSize() > 8 and QtGui.QFontMetrics(fitted).horizontalAdvance(fit_text) > rect.width() - 4:
                        fitted.setPixelSize(fitted.pixelSize() - 1)
                    painter.setFont(fitted)
                    if rect == self.time_rect and controller.profile.get('target_time_centered' if self.effect_id in TARGET_SLOTS else 'time_text_centered', False):
                        path = QtGui.QPainterPath()
                        metrics = QtGui.QFontMetricsF(fitted)
                        x = rect.x() + (rect.width() - metrics.horizontalAdvance(text)) / 2
                        baseline = rect.y() + (rect.height() - metrics.height()) / 2 + metrics.ascent()
                        path.addText(x, baseline, fitted, text)
                        outline = QtGui.QPen(QtGui.QColor(0, 0, 0, 230),
                                            max(2.0, fitted.pixelSize() * 0.14))
                        outline.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
                        painter.setPen(outline)
                        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                        painter.drawPath(path)
                        # Paint the same text glyphs as the below-icon mode over
                        # the outline, so its stroke cannot thin the white fill.
                        painter.setPen(color)
                        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)
                        continue
                    painter.fillRect(rect, QtGui.QColor(10, 14, 18, 235))
                    painter.setPen(color)
                    painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)

        def set_card_geometry(self, x: int, y: int, size: int) -> None:
            band = max(20, round(size * 0.32))
            gap = max(3, round(size * 0.05))
            self.stack_rect = QtCore.QRect(0, 0, size, band)
            self.icon_rect = QtCore.QRect(0, band + gap, size, size)
            self.time_rect = QtCore.QRect(0, band + gap + size + gap, size, band)
            if controller.profile.get('target_time_centered' if self.effect_id in TARGET_SLOTS else 'time_text_centered', False):
                self.time_rect.setHeight(max(band, round(size * 0.5)))
                self.time_rect.moveCenter(self.icon_rect.center())
            self.setGeometry(x, y, size, size + 2 * (band + gap))

        def mousePressEvent(self, event: Any) -> None:
            if not editing_allowed(self.effect_id) or event.button() != QtCore.Qt.MouseButton.LeftButton:
                return
            self._origin = event.globalPosition()
            margin = 14
            self._resizing = event.position().x() >= self.width() - margin and event.position().y() >= self.height() - margin
            self.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor if self._resizing else QtCore.Qt.CursorShape.ClosedHandCursor)

        def mouseMoveEvent(self, event: Any) -> None:
            if self._origin is None or not editing_allowed(self.effect_id):
                return
            delta = event.globalPosition() - self._origin
            slot = display_slot(self.effect_id, controller.profile)
            placement = controller.overlay.icons[slot]
            if self.effect_id in TARGET_SLOTS:
                layout=controller.overlay
                effects=TARGET_SLOTS if controller.profile.get('target_edit_mode','common')=='common' else (self.effect_id,)
                if self._resizing:
                    layout.icons[self.effect_id].size=max(24,min(512,placement.size+max(delta.x(),delta.y())))
                else:
                    for effect in effects:
                        layout.icons[effect].x+=delta.x()
                        layout.icons[effect].y+=delta.y()
                controller.profile['overlay']=layout.to_dict()
            elif self._resizing:
                controller.resize_icon(slot, max(24, placement.size + max(delta.x(), delta.y())))
            elif controller.profile.get('icon_edit_mode', 'common') != 'single' and slot in MERGED_COMMON:
                controller.move_common_group(delta.x(), delta.y())
            else:
                controller.move_icon(slot, placement.x + delta.x(), placement.y + delta.y())
            self._origin = event.globalPosition()
            overlay.sync_layout()

        def mouseReleaseEvent(self, event: Any) -> None:
            if self._origin is not None:
                self._origin = None
                self._resizing = False
                self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
                controller.save()

    class OverlayWindow(QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.icons: dict[str, Any] = {}
            self.capture_exclusion_active: bool | None = None
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint, True)
            self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
            self.setWindowFlag(QtCore.Qt.WindowType.Tool, True)
            self.sync_layout()

        def _set_locked_flags(self, locked: bool) -> None:
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, locked)
            self.setWindowFlag(QtCore.Qt.WindowType.WindowTransparentForInput, locked)
            self.setWindowFlag(QtCore.Qt.WindowType.WindowDoesNotAcceptFocus, locked)
            if locked:
                self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            else:
                self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            _apply_windows_click_through(self, locked)

        def sync_layout(self) -> None:
            layout = rendered_layout()
            editable = editing_allowed()
            self._set_locked_flags(not editable)
            monitor_id = controller.document["display"].get("overlay_monitor_id")
            monitor = next((m for m in controller.monitors if m.monitor_id == monitor_id), None)
            if monitor is None:
                monitor = next((m for m in controller.monitors if m.primary), None)
            if monitor:
                self.setGeometry(monitor.x, monitor.y, monitor.width, monitor.height)
            active = {item.effect_id for item in controller.preview.items if item.recognized}
            # While unlocked, configured icons remain visible as placement
            # handles. During play, only current detections are rendered.
            if editable or controller.preview.status == "대기 중":
                active.update(effect for effect in layout.icons
                              if editing_allowed(effect) or controller.preview.status == '대기 중')
            if controller.profile.get('overlay_scope') == 'support_common':
                active = {effect for effect in active if effect in MERGED_COMMON + TARGET_SLOTS}
            active.difference_update(set(GREEN_EFFECTS)-set(enabled_green(controller.profile)))
            if controller.profile.get('game_ui', {}).get('stack_merge', True):
                active.difference_update((*CLASS_ATTACK, 'support.extra.01'))
            if not controller.profile.get('game_ui', {}).get('stack_merge', True):
                recognized_classes = {item.effect_id for item in controller.preview.items
                                      if item.recognized and item.effect_id in CLASS_ATTACK}
                active.difference_update(set(CLASS_ATTACK) - recognized_classes)
                if recognized_classes:
                    active.discard('support.attack.additional')
            for effect_id in active:
                if effect_id not in layout.icons:
                    controller.ensure_icon(effect_id)
                    layout = rendered_layout()
                widget = self.icons.get(effect_id)
                if widget is None:
                    widget = self.icons[effect_id] = OverlayIconWidget(effect_id, self)
                item = layout.icons[effect_id]
                size = max(24, round(item.size))
                widget.set_card_geometry(round(item.x), round(item.y), size)
                widget.setVisible(effect_id in active)
                widget.update()
            for effect_id, widget in list(self.icons.items()):
                if not editable:
                    widget._origin = None
                    widget._resizing = False
                if effect_id not in active:
                    widget.hide()
                    if controller.profile.get('overlay_scope') == 'support_common' and effect_id not in MERGED_COMMON + TARGET_SLOTS:
                        widget.deleteLater()
                        del self.icons[effect_id]
            self.show()
            self.capture_exclusion_active = _apply_windows_capture_exclusion(
                self, excluded=False)
            controller.overlay_capture_excluded = self.capture_exclusion_active

    class RoiPreviewLabel(QtWidgets.QLabel):
        """Settings-only drag surface mapped to monitor pixel coordinates."""

        def __init__(self, on_commit: Any) -> None:
            super().__init__("캡처 프레임 대기 중")
            self._on_commit = on_commit
            self._edit_enabled = False
            self._drag_start = None
            self._drag_end = None
            self._source_rect = None
            self._monitor_size = None
            self._original_pixmap = None
            self.setMinimumSize(480, 120)
            self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.setStyleSheet("background:#080808; color:#aaa")

        def set_edit_enabled(self, enabled: bool) -> None:
            self._edit_enabled = bool(enabled)
            self.setCursor(
                QtCore.Qt.CursorShape.CrossCursor if enabled else QtCore.Qt.CursorShape.ArrowCursor
            )
            if not enabled:
                self._drag_start = self._drag_end = None
                self.update()

        def set_preview(self, pixmap: Any, source_rect: Rect | None, monitor_size: tuple[int, int] | None) -> None:
            self._source_rect = source_rect
            self._monitor_size = monitor_size
            self._original_pixmap = pixmap
            self.setPixmap(
                pixmap.scaled(
                    self.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )

        def resizeEvent(self, event: Any) -> None:
            super().resizeEvent(event)
            if self._original_pixmap is not None:
                self.set_preview(self._original_pixmap, self._source_rect, self._monitor_size)

        def _content_rect(self) -> Any:
            pixmap = self.pixmap()
            if pixmap is None or pixmap.isNull():
                return QtCore.QRectF()
            return QtCore.QRectF(
                (self.width() - pixmap.width()) / 2,
                (self.height() - pixmap.height()) / 2,
                pixmap.width(),
                pixmap.height(),
            )

        def mousePressEvent(self, event: Any) -> None:
            if (not self._edit_enabled or self._source_rect is None or
                    event.button() != QtCore.Qt.MouseButton.LeftButton):
                return super().mousePressEvent(event)
            if self._content_rect().contains(event.position()):
                self._drag_start = event.position()
                self._drag_end = event.position()
                self.update()

        def mouseMoveEvent(self, event: Any) -> None:
            if self._drag_start is not None:
                self._drag_end = event.position()
                self.update()

        def mouseReleaseEvent(self, event: Any) -> None:
            if self._drag_start is None or event.button() != QtCore.Qt.MouseButton.LeftButton:
                return super().mouseReleaseEvent(event)
            self._drag_end = event.position()
            content = self._content_rect()
            selection = QtCore.QRectF(self._drag_start, self._drag_end).normalized().intersected(content)
            self._drag_start = self._drag_end = None
            self.update()
            if selection.width() < 3 or selection.height() < 3 or self._monitor_size is None:
                return
            source = self._source_rect
            pixel = Rect(
                source.x + (selection.left() - content.left()) / content.width() * source.width,
                source.y + (selection.top() - content.top()) / content.height() * source.height,
                selection.width() / content.width() * source.width,
                selection.height() / content.height() * source.height,
            )
            self._on_commit(pixel, *self._monitor_size)

        def paintEvent(self, event: Any) -> None:
            super().paintEvent(event)
            if self._drag_start is None or self._drag_end is None:
                return
            painter = QtGui.QPainter(self)
            painter.fillRect(
                QtCore.QRectF(self._drag_start, self._drag_end).normalized(),
                QtGui.QColor(40, 180, 255, 55),
            )
            painter.setPen(QtGui.QPen(QtGui.QColor(40, 200, 255), 2))
            painter.drawRect(QtCore.QRectF(self._drag_start, self._drag_end).normalized())

    class SettingsWindow(QtWidgets.QMainWindow):
        ROI_LABELS = {"self_buff": "캐릭터 버프", "target_debuff": "타겟 UI 버프"}

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("LostArk Buff Eye 설정")
            self.save_timer = QtCore.QTimer(self)
            self.save_timer.setSingleShot(True)
            self.save_timer.setInterval(300)
            self.save_timer.timeout.connect(self._save_settings)
            self.resize(820, 720)
            root = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(root)
            self.monitor_boxes: dict[str, Any] = {}
            self._settings_monitor_applied = False
            for role, label in (("capture", "게임 화면 모니터 (취득·오버레이)"), ("settings", "설정창 모니터")):
                box = QtWidgets.QComboBox()
                for monitor in controller.monitors:
                    box.addItem(monitor.label, monitor.monitor_id)
                current = controller.document["display"].get(f"{role}_monitor_id")
                index = box.findData(current)
                box.setCurrentIndex(index if index >= 0 else 0)
                box.currentIndexChanged.connect(lambda _i, r=role, b=box: self._monitor_changed(r, b.currentData()))
                self.monitor_boxes[role] = box
                # Retain legacy values for settings compatibility, not user input.
                box.setParent(self)
                box.hide()
            game_monitor = controller.document['display'].get('capture_monitor_id')
            if game_monitor is not None and any(m.monitor_id == game_monitor for m in controller.monitors):
                controller.select_monitor('overlay', game_monitor)

            game_ui = controller.profile.get("game_ui", {})
            self.game_ui_widgets: dict[str, Any] = {}
            game_specs = (
                ("hud_scale_pct", "HUD 크기", (70, 90, 100, 105, 110)),
                ("buff_scale_pct", "버프 크기", (80, 90, 100, 110, 120)),
                ("buff_display_count", "버프 표시 개수", tuple(range(10, 18))),
            )
            for key, label, values in game_specs:
                box = QtWidgets.QComboBox()
                for value in values:
                    box.addItem(str(value), value)
                wanted = game_ui.get(key, 100 if "scale" in key else 15)
                box.setCurrentIndex(max(0, box.findData(wanted)))
                box.currentIndexChanged.connect(lambda _i, k=key, b=box: self._game_ui_changed(k, b.currentData()))
                self.game_ui_widgets[key] = box
                box.setParent(self)
                box.hide()
            root_form = form
            share_help = QtWidgets.QLabel('공유: 모니터 전체 선택 · 표시: 창모드/전체 창모드 지원')
            share_help.setWordWrap(True)
            root_form.addRow(share_help)
            self.buff_settings_group = QtWidgets.QGroupBox("서포터 버프 설정")
            form = QtWidgets.QFormLayout(self.buff_settings_group)
            form.setContentsMargins(12, 18, 12, 12)
            root_form.addRow(self.buff_settings_group)
            self.green_group = QtWidgets.QGroupBox('녹색 버프 감지 (선택한 항목만 인식)')
            green_form=QtWidgets.QFormLayout(self.green_group)
            self.green_checks={}
            for effect_id,(label,_asset) in GREEN_ASSETS.items():
                check=QtWidgets.QCheckBox(label)
                check.setChecked(effect_id in enabled_green(controller.profile))
                check.toggled.connect(lambda checked,k=effect_id:self._green_changed(k,checked))
                green_form.addRow(check);self.green_checks[effect_id]=check
            root_form.addRow(self.green_group)
            self.target_settings_group = QtWidgets.QGroupBox("타겟 UI 버프 설정")
            target_form = QtWidgets.QFormLayout(self.target_settings_group)
            target_form.setContentsMargins(12, 18, 12, 12)
            for key, label, values in (
                ('target_scale_pct', '버프 크기 (%)', (80, 90, 100, 110, 120)),
                ('target_display_count', '버프 표시 개수', tuple(range(10, 18))),
            ):
                box = QtWidgets.QComboBox()
                for value in values:
                    box.addItem(str(value), value)
                box.setCurrentIndex(max(0, box.findData(game_ui.get(key, 100 if 'scale' in key else 17))))
                box.currentIndexChanged.connect(lambda _i, k=key, b=box: self._game_ui_changed(k, b.currentData()))
                self.game_ui_widgets[key] = box
                box.setParent(self)
                box.hide()
            self.roi_edits: dict[str, list[Any]] = {}
            for region_id in self.ROI_LABELS:
                row = QtWidgets.QWidget()
                layout = QtWidgets.QHBoxLayout(row)
                layout.setContentsMargins(0, 0, 0, 0)
                region = controller.rois[region_id]
                values = (0, 0, 1, 1) if region.pixel_rect is None else (
                    region.pixel_rect.x, region.pixel_rect.y, region.pixel_rect.width, region.pixel_rect.height
                )
                fields = []
                for name, value in zip(("X", "Y", "W", "H"), values):
                    edit = QtWidgets.QSpinBox()
                    edit.setRange(-32768 if name in ("X", "Y") else 1, 65535)
                    edit.setValue(round(value))
                    edit.setPrefix(f"{name}: ")
                    edit.editingFinished.connect(lambda rid=region_id: self._roi_changed(rid))
                    fields.append(edit)
                    layout.addWidget(edit)
                self.roi_edits[region_id] = fields
                auto_button = QtWidgets.QPushButton('자동 배치')
                auto_button.setToolTip('게임 화면 비율로 영역을 배치하고 버프 줄이 보이면 보정합니다.')
                auto_button.clicked.connect(lambda _checked=False, rid=region_id: self._auto_roi(rid))
                layout.addWidget(auto_button)
                if region_id == 'target_debuff':
                    self.target_roi_row = row
                else:
                    self.buff_roi_row = row

            self.lock = QtWidgets.QCheckBox("오버레이 위치 고정")
            self.lock.setChecked(controller.overlay.position_locked)
            self.lock.toggled.connect(self._lock_changed)
            self.reset_positions_button = QtWidgets.QPushButton("오버레이 위치 초기화")
            self.reset_positions_button.setToolTip("위치 고정을 해제한 뒤 사용하세요. 모든 아이콘의 위치만 기본 배치로 되돌립니다.")
            self.reset_positions_button.setEnabled(not controller.overlay.position_locked)
            self.reset_positions_button.clicked.connect(self._reset_overlay_positions)
            position_row = QtWidgets.QWidget()
            position_layout = QtWidgets.QHBoxLayout(position_row)
            position_layout.setContentsMargins(0, 0, 0, 0)
            position_layout.addWidget(self.lock)
            position_layout.addWidget(self.reset_positions_button)
            form.addRow("플레이 보호", position_row)
            self.mode_tabs = QtWidgets.QTabBar()
            self.mode_tabs.addTab('공통')
            self.mode_tabs.addTab('단일')
            self.mode_tabs.setCurrentIndex(1 if controller.profile.get('icon_edit_mode') == 'single' else 0)
            form.addRow(self.mode_tabs)
            self.effect_label = QtWidgets.QLabel('효과')
            self.effect_box = QtWidgets.QComboBox()
            for effect_id, label in zip(MERGED_COMMON, COMMON_LABELS):
                self.effect_box.addItem(label, effect_id)
            form.addRow(self.effect_label, self.effect_box)
            self.size = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.size.setRange(24, 512)
            self.size.setValue(96)
            self.size_value = QtWidgets.QSpinBox()
            self.size_value.setRange(24, 512)
            self.size_value.setSuffix(' px')
            self.size_value.setValue(96)
            self.size.valueChanged.connect(self.size_value.setValue)
            self.size_value.valueChanged.connect(self.size.setValue)
            self.reset_size_button = QtWidgets.QPushButton('기본값')
            self.reset_size_button.setToolTip('선택한 아이콘 크기를 96px로 복원합니다.')
            self.reset_size_button.clicked.connect(self._reset_icon_size)
            icon_row = QtWidgets.QWidget()
            icon_layout = QtWidgets.QHBoxLayout(icon_row)
            icon_layout.setContentsMargins(0, 0, 0, 0)
            icon_layout.addWidget(self.size, 1)
            icon_layout.addWidget(self.size_value)
            icon_layout.addWidget(self.reset_size_button)
            self.opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.opacity.setRange(0, 100)
            self.opacity_value = QtWidgets.QLabel(f'{self.opacity.value()}%')
            self.opacity_value.setMinimumWidth(42)
            self.opacity_value.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.opacity.valueChanged.connect(lambda value: self.opacity_value.setText(f'{value}%'))
            self.size.valueChanged.connect(lambda value: self._effect_changed('size', value))
            self.opacity.valueChanged.connect(lambda value: self._effect_changed('opacity', value))
            self.effect_box.currentTextChanged.connect(self._load_effect_controls)
            form.addRow("아이콘", icon_row)
            opacity_row = QtWidgets.QWidget()
            opacity_layout = QtWidgets.QHBoxLayout(opacity_row)
            opacity_layout.setContentsMargins(0, 0, 0, 0)
            opacity_layout.addWidget(self.opacity, 1)
            opacity_layout.addWidget(self.opacity_value)
            self.reset_opacity_button = QtWidgets.QPushButton('기본값')
            self.reset_opacity_button.setToolTip('현재 모드에 해당하는 아이콘 투명도를 0%로 복원합니다.')
            self.reset_opacity_button.clicked.connect(self._reset_icon_opacity)
            opacity_layout.addWidget(self.reset_opacity_button)
            form.addRow("투명도 (%)", opacity_row)
            self.center_time = QtWidgets.QCheckBox('시간을 이미지 중앙에 표시')
            self.center_time.setChecked(controller.profile.get('time_text_centered', False))
            self.center_time.setToolTip('체크: 중앙에 0.7처럼 숫자만 표시 / 해제: 이미지 아래에 0.7초로 표시')
            self.center_time.toggled.connect(self._center_time_changed)
            form.addRow('시간 텍스트', self.center_time)
            self.screen_area_group = QtWidgets.QGroupBox('화면영역')
            form.addRow(self.screen_area_group)
            form = QtWidgets.QFormLayout(self.screen_area_group)
            form.setContentsMargins(12, 18, 12, 12)
            self.roi_drag_edit = QtWidgets.QCheckBox("미리보기에서 캐릭터 버프 ROI 편집")
            self.full_roi_button = QtWidgets.QPushButton("전체 모니터에서 인식 영역 설정")
            self.full_roi_button.clicked.connect(self._begin_full_roi)
            form.addRow("전체 화면 영역 설정", self.full_roi_button)
            self.roi_drag_edit.setToolTip("설정창 안의 미리보기만 사용하며 게임 입력에는 영향을 주지 않습니다.")
            self.reset_preview_button = QtWidgets.QPushButton("영역 미리보기 초기화")
            self.reset_preview_button.setToolTip("캐릭터 버프 인식 영역을 현재 인식 설정의 기본 좌표로 복원합니다.")
            self.reset_preview_button.setEnabled(default_preview_roi is not None)
            self.reset_preview_button.clicked.connect(self._reset_preview_roi)
            self.preview_image = RoiPreviewLabel(self._preview_roi_committed)
            self.roi_drag_edit.toggled.connect(self.preview_image.set_edit_enabled)
            self.preview_toggle = QtWidgets.QPushButton()
            self.preview_toggle.setCheckable(True)
            self.preview_toggle.setChecked(controller.document.get('ui', {}).get('preview_visible', True))
            self.preview_toggle.toggled.connect(self._toggle_preview)
            form.addRow("영역 미리보기", self.preview_toggle)
            form.addRow(self.preview_image)
            self.buff_roi_label = QtWidgets.QLabel(self.ROI_LABELS['self_buff'])
            form.addRow(self.buff_roi_label, self.buff_roi_row)
            self.roi_drag_label = QtWidgets.QLabel('시각적 ROI 설정')
            self.roi_controls = QtWidgets.QWidget()
            roi_controls_layout = QtWidgets.QHBoxLayout(self.roi_controls)
            roi_controls_layout.setContentsMargins(0, 0, 0, 0)
            roi_controls_layout.addWidget(self.roi_drag_edit)
            roi_controls_layout.addWidget(self.reset_preview_button)
            form.addRow(self.roi_drag_label, self.roi_controls)
            self._set_preview_visibility(self.preview_toggle.isChecked())
            form = root_form
            form.addRow(self.target_settings_group)
            self.target_roi_button = QtWidgets.QPushButton('타겟 UI 버프 영역 설정')
            self.target_roi_button.clicked.connect(lambda: self._begin_target_roi())
            self.target_lock=QtWidgets.QCheckBox('오버레이 위치 고정')
            self.target_lock.setChecked(controller.profile.get('target_locked',True))
            self.target_lock.toggled.connect(lambda value:self._target_option('target_locked',value))
            self.target_reset=QtWidgets.QPushButton('오버레이 위치 초기화')
            self.target_reset.clicked.connect(self._reset_target_positions)
            row=QtWidgets.QWidget(); row_layout=QtWidgets.QHBoxLayout(row); row_layout.setContentsMargins(0,0,0,0)
            row_layout.addWidget(self.target_lock); row_layout.addWidget(self.target_reset)
            target_form.addRow('플레이 보호',row)
            self.target_tabs=QtWidgets.QTabBar(); self.target_tabs.addTab('공통'); self.target_tabs.addTab('단일')
            self.target_tabs.setCurrentIndex(1 if controller.profile.get('target_edit_mode')=='single' else 0)
            self.target_tabs.currentChanged.connect(lambda index:self._target_option('target_edit_mode','single' if index else 'common'))
            target_form.addRow(self.target_tabs)
            self.target_effect=QtWidgets.QComboBox()
            for effect,label in zip(TARGET_SLOTS,TARGET_LABELS): self.target_effect.addItem(label,effect)
            self.target_effect_label=QtWidgets.QLabel('효과')
            target_form.addRow(self.target_effect_label,self.target_effect)
            self.target_effect.currentIndexChanged.connect(self._load_target_controls)
            self.target_size=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.target_size.setRange(24,512)
            self.target_size_value=QtWidgets.QSpinBox(); self.target_size_value.setRange(24,512); self.target_size_value.setSuffix(' px')
            self.target_size.valueChanged.connect(self.target_size_value.setValue)
            self.target_size_value.valueChanged.connect(self.target_size.setValue)
            self.target_size.valueChanged.connect(lambda value:self._target_style('size',value))
            self.target_opacity=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.target_opacity.setRange(0,100)
            self.target_opacity_value=QtWidgets.QLabel('0%')
            self.target_opacity.valueChanged.connect(lambda value:self._target_style('transparency_pct',value))
            for label,slider,value,field,default in (
                ('아이콘',self.target_size,self.target_size_value,'size',96),
                ('투명도',self.target_opacity,self.target_opacity_value,'transparency_pct',0)):
                row=QtWidgets.QWidget(); line=QtWidgets.QHBoxLayout(row); line.setContentsMargins(0,0,0,0)
                line.addWidget(slider,1); line.addWidget(value)
                button=QtWidgets.QPushButton('기본값'); button.clicked.connect(lambda _checked=False,f=field,v=default:self._target_style(f,v))
                line.addWidget(button); target_form.addRow(label,row)
            self.target_center=QtWidgets.QCheckBox('시간을 이미지 중앙에 표시')
            self.target_center.setChecked(controller.profile.get('target_time_centered',False))
            self.target_center.toggled.connect(lambda value:self._target_option('target_time_centered',value))
            target_form.addRow('시간 텍스트',self.target_center)
            self.target_area_group=QtWidgets.QGroupBox('화면영역')
            target_form.addRow(self.target_area_group)
            target_form=QtWidgets.QFormLayout(self.target_area_group)
            target_form.setContentsMargins(12,18,12,12)
            target_form.addRow('전체 화면 영역 설정',self.target_roi_button)
            self.target_preview_toggle=QtWidgets.QPushButton('미리보기 보기'); self.target_preview_toggle.setCheckable(True)
            self.target_preview_toggle.setChecked(controller.profile.get('target_preview_visible',False))
            self.target_preview_toggle.toggled.connect(lambda value:self._target_option('target_preview_visible',value))
            target_form.addRow('영역 미리보기',self.target_preview_toggle)
            self.target_preview=RoiPreviewLabel(lambda rect,w,h:self._target_roi_committed(rect,w,h))
            target_form.addRow(self.target_preview)
            self.target_roi_label=QtWidgets.QLabel('타겟 UI 버프 좌표')
            target_form.addRow(self.target_roi_label,self.target_roi_row)
            self.target_roi_edit=QtWidgets.QCheckBox('미리보기에서 타겟 영역 편집')
            self.target_roi_edit.toggled.connect(self.target_preview.set_edit_enabled)
            self.target_roi_reset=QtWidgets.QPushButton('영역 미리보기 초기화')
            self.target_roi_reset.clicked.connect(self._reset_target_roi)
            self.target_roi_controls=QtWidgets.QWidget(); line=QtWidgets.QHBoxLayout(self.target_roi_controls); line.setContentsMargins(0,0,0,0)
            line.addWidget(self.target_roi_edit); line.addWidget(self.target_roi_reset)
            target_form.addRow(self.target_roi_controls)
            self.target_capture_status=QtWidgets.QLabel('타겟 캡처 대기 · 위치 고정을 해제하면 배치용 아이콘이 표시됩니다.')
            self.target_capture_status.setWordWrap(True)
            target_form.addRow(self.target_capture_status)
            self._latest_target_update=None
            self._load_target_controls()
            save = QtWidgets.QPushButton("설정 저장")
            save.clicked.connect(self._save_settings)
            form.addRow(save)
            self.save_status = QtWidgets.QLabel("설정 변경 시 자동 저장")
            form.addRow(self.save_status)
            scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(root)
            self.setCentralWidget(scroll)
            self.mode_tabs.currentChanged.connect(self._mode_changed)
            self._show_effect_selector()
            self._load_effect_controls()

        def _schedule_save(self) -> None:
            self.save_status.setText("저장 대기 중")
            self.save_timer.start()

        def _target_option(self, key, value):
            controller.profile[key]=value
            self._load_target_controls()
            if self._latest_target_update is not None:
                self._refresh_target_preview(self._latest_target_update)
            overlay.sync_layout()
            self._schedule_save()

        def _load_target_controls(self, *_):
            if not hasattr(self,'target_capture_status'): return
            single=self.target_tabs.currentIndex()==1
            self.target_effect.setVisible(single); self.target_effect_label.setVisible(single)
            placement=controller.overlay.icons.get(self.target_effect.currentData() if single else TARGET_SLOTS[0])
            if placement:
                for widget,value in ((self.target_size,placement.size),(self.target_size_value,placement.size),(self.target_opacity,placement.transparency_pct)):
                    widget.blockSignals(True); widget.setValue(round(value)); widget.blockSignals(False)
                self.target_opacity_value.setText(f'{round(placement.transparency_pct)}%')
            locked=controller.profile.get('target_locked',True)
            self.target_size.setEnabled(not locked); self.target_size_value.setEnabled(not locked); self.target_reset.setEnabled(not locked)
            visible=self.target_preview_toggle.isChecked()
            self.target_preview_toggle.setText('미리보기 숨기기' if visible else '미리보기 보기')
            for widget in (self.target_preview,self.target_roi_row,self.target_roi_label,self.target_roi_controls): widget.setVisible(visible)
            if not visible:
                self.target_roi_edit.setChecked(False); self.target_preview.clear(); self.target_preview._original_pixmap=None

        def _target_style(self, field, value):
            if field=='size' and controller.profile.get('target_locked',True): return
            for effect in TARGET_SLOTS: controller.ensure_icon(effect)
            layout=controller.overlay
            effects=TARGET_SLOTS if self.target_tabs.currentIndex()==0 else (self.target_effect.currentData(),)
            if field=='size' and len(effects)>1:
                anchor=layout.icons[effects[0]]; x,y=anchor.x,anchor.y
                ratio=value/anchor.size
                for effect in effects:
                    icon=layout.icons[effect]; icon.x=x+(icon.x-x)*ratio; icon.y=y+(icon.y-y)*ratio
            for effect in effects: setattr(layout.icons[effect],field,value)
            controller.profile['overlay']=layout.to_dict()
            self._load_target_controls(); overlay.sync_layout(); self._schedule_save()

        def _reset_target_positions(self):
            if controller.profile.get('target_locked',True): return
            controller.place_target_defaults(reset=True)
            overlay.sync_layout(); self._schedule_save()

        def _target_roi_committed(self, rect, width, height):
            controller.set_roi_from_preview('target_debuff',rect,monitor_width=width,monitor_height=height)
            for field,value in zip(self.roi_edits['target_debuff'],(rect.x,rect.y,rect.width,rect.height)): field.setValue(round(value))
            self._schedule_save()

        def _reset_target_roi(self):
            monitor=next((m for m in controller.monitors if m.monitor_id==controller.document['display'].get('capture_monitor_id')),None)
            if monitor is None: return
            rect=Rect(round(monitor.width*1600/3440),round(monitor.height*180/1440),
                      round(monitor.width*650/3440),round(monitor.height*100/1440))
            self._target_roi_committed(rect,monitor.width,monitor.height)

        def _refresh_target_preview(self, update):
            self._latest_target_update=update
            shape = None if update.frame is None else update.frame.shape
            if shape is not None and (shape[1] < 26 or shape[0] < 40):
                status='타겟 영역이 너무 작습니다. 영역 미리보기 초기화 또는 영역 설정을 사용하세요.'
            elif update.error:
                status=f'{update.status}: {update.error}'
            else:
                status=f'{update.status} · 타겟 버프 {len(target_items(update.detections,controller.profile))}개 표시'
            if self.target_capture_status.text()!=status:
                self.target_capture_status.setText(status)
            frame=update.frame
            if self.target_preview_toggle.isChecked() and frame is not None:
                height,width=frame.shape[:2]
                im=QtGui.QImage(frame.data,width,height,frame.strides[0],QtGui.QImage.Format.Format_BGR888).copy()
                monitor=next((m for m in controller.monitors if m.monitor_id==controller.document['display'].get('capture_monitor_id')),None)
                self.target_preview.set_preview(QtGui.QPixmap.fromImage(im),controller.rois['target_debuff'].pixel_rect,
                    (monitor.width,monitor.height) if monitor else None)

        def _begin_target_roi(self) -> None:
            self._requested_roi = 'target_debuff'
            self._begin_full_roi()

        def _begin_full_roi(self) -> None:
            self.full_roi_button.setEnabled(False)
            self.hide()
            # Give the compositor time to remove settings from the captured image.
            QtCore.QTimer.singleShot(200, self._capture_full_roi)

        def _capture_full_roi(self) -> None:
            try:
                snapshot = snapshot_provider(controller.document["display"].get("capture_monitor_id"))
                self.open_roi_snapshot(snapshot)
            except (OSError, RuntimeError, ValueError) as exc:
                self.save_status.setText(f"전체 화면 캡처 실패: {exc}")
            finally:
                self.full_roi_button.setEnabled(True)
                self.show()

        def open_roi_snapshot(self, snapshot: Any) -> Any:
            dialog = QtWidgets.QDialog(self)
            dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
            dialog.setWindowTitle("인식 영역 설정 · 정지 화면")
            dialog.resize(1100, 650)
            layout = QtWidgets.QVBoxLayout(dialog)
            region_box = QtWidgets.QComboBox()
            for region_id, label in self.ROI_LABELS.items():
                region_box.addItem(label, region_id)
            region_box.setCurrentIndex(max(0, region_box.findData(getattr(self, '_requested_roi', 'self_buff'))))
            self._requested_roi = 'self_buff'
            layout.addWidget(region_box)
            status = QtWidgets.QLabel("영역 종류를 선택하고 드래그하세요. 선택 즉시 저장됩니다.")
            status.setWordWrap(True)
            layout.addWidget(status)
            height, width = snapshot.frame.shape[:2]

            def commit(rect: Rect, _width: int, _height: int) -> None:
                current = controller.document["display"].get("capture_monitor_id")
                if current not in (None, snapshot.monitor_id):
                    status.setText("모니터가 변경되었습니다. 창을 닫고 다시 캡처하세요.")
                    return
                import copy
                before = copy.deepcopy(controller.document)
                region_id = region_box.currentData()
                try:
                    controller.document["display"]["capture_monitor_id"] = snapshot.monitor_id
                    pixel = Rect(round(rect.x), round(rect.y), max(1, round(rect.width)), max(1, round(rect.height)))
                    controller.set_roi_from_preview(region_id, pixel, monitor_width=width, monitor_height=height)
                    controller.save()
                except (OSError, RuntimeError, ValueError) as exc:
                    controller.document = before
                    status.setText(f"영역 저장 실패: {exc}")
                    return
                for field, value in zip(self.roi_edits[region_id], (pixel.x, pixel.y, pixel.width, pixel.height)):
                    field.setValue(round(value))
                status.setText(f"{self.ROI_LABELS[region_id]} 저장됨 · X {pixel.x}, Y {pixel.y}, {pixel.width} × {pixel.height}")

            preview = RoiPreviewLabel(commit)
            preview.setMinimumSize(640, 360)
            preview.set_edit_enabled(True)
            layout.addWidget(preview, 1)
            frame = snapshot.frame
            qimage = QtGui.QImage(frame.data, width, height, frame.strides[0], QtGui.QImage.Format.Format_BGR888).copy()
            preview.set_preview(QtGui.QPixmap.fromImage(qimage), Rect(0, 0, width, height), (width, height))
            close = QtWidgets.QPushButton("완료")
            close.clicked.connect(dialog.accept)
            layout.addWidget(close)
            dialog.region_box, dialog.preview, dialog.status_label = region_box, preview, status
            dialog.setModal(True)
            self.roi_dialog = dialog
            def release_snapshot(_result):
                if self.roi_dialog is dialog:
                    self.roi_dialog = None
                overlay.sync_layout()
            dialog.finished.connect(release_snapshot)
            dialog.show()
            overlay.sync_layout()
            return dialog

        def _save_settings(self) -> bool:
            self.save_timer.stop()
            try:
                controller.save()
            except (OSError, RuntimeError, ValueError) as exc:
                self.save_status.setText(f"설정 저장 실패: {exc}")
                return False
            self.save_status.setText("설정 저장됨")
            return True

        def _load_effect_controls(self, *_args: Any) -> None:
            placement = controller.overlay.icons.get(MERGED_COMMON[0] if self.mode_tabs.currentIndex() == 0 else self.effect_box.currentData())
            if placement is None:
                return
            for widget, value in ((self.size, placement.size), (self.size_value, placement.size), (self.opacity, placement.transparency_pct)):
                blocked = widget.blockSignals(True)
                widget.setValue(round(value))
                widget.blockSignals(blocked)
            self.opacity_value.setText(f'{self.opacity.value()}%')
            self.size.setEnabled(not controller.overlay.position_locked)
            self.size_value.setEnabled(not controller.overlay.position_locked)
            self.reset_size_button.setEnabled(not controller.overlay.position_locked)

        def showEvent(self, event: Any) -> None:
            super().showEvent(event)
            if not self._settings_monitor_applied:
                self._settings_monitor_applied = True
                QtCore.QTimer.singleShot(0, self._move_settings_to_monitor)
            overlay.sync_layout()

        def hideEvent(self, event: Any) -> None:
            super().hideEvent(event)
            overlay.sync_layout()

        def changeEvent(self, event: Any) -> None:
            super().changeEvent(event)
            if event.type() == QtCore.QEvent.Type.WindowStateChange:
                overlay.sync_layout()

        def closeEvent(self, event: Any) -> None:
            if not self._save_settings():
                event.ignore()
                return
            super().closeEvent(event)

        def _monitor_changed(self, role: str, monitor_id: str) -> None:
            controller.select_monitor(role, monitor_id)
            if role == 'capture':
                controller.select_monitor('overlay', monitor_id)
            elif role == 'settings':
                self._move_settings_to_monitor()
            overlay.sync_layout()
            self._schedule_save()

        def _move_settings_to_monitor(self) -> None:
            monitor_id = controller.document['display'].get('settings_monitor_id')
            screen = resolve_qt_screen(monitor_id, controller.monitors, app.screens())
            if screen is None:
                screen = app.primaryScreen()
            if screen is None:
                return
            # QScreen uses Qt logical coordinates, unlike capture's physical pixels.
            area = screen.availableGeometry()
            if self.windowHandle() is not None:
                self.windowHandle().setScreen(screen)
            self.resize(min(self.width(), area.width()), min(self.height(), area.height()))
            self.move(area.x() + max(0, (area.width() - self.frameGeometry().width()) // 2),
                      area.y() + max(0, (area.height() - self.frameGeometry().height()) // 2))

        def _game_ui_changed(self, key: str, value: Any) -> None:
            controller.set_game_ui(key, value)
            self._schedule_save()

        def _auto_roi(self, region_id: str) -> None:
            previous = list(controller.profile['rois'])
            references = dict(controller.profile.get('roi_viewports', {}))
            controller.set_roi(region_id, None)
            if not self._save_settings():
                controller.profile['rois'] = previous
                controller.profile['roi_viewports'] = references
                return
            self.save_status.setText('자동 배치 적용 · 버프가 나타나면 위치를 확인합니다.')
            source=getattr(live_worker,'support',live_worker)
            worker=getattr(live_worker,'target',None) if region_id=='target_debuff' else source
            session=getattr(worker,'session',None)
            if hasattr(session,'request_auto_reset'):session.request_auto_reset()

        def _roi_changed(self, region_id: str) -> None:
            # A fullscreen resolution change can leave the initial UI monitor
            # dimensions stale, even though capture has already reconnected.
            try:
                refresh_worker_monitors(controller, live_worker)
            except (OSError, RuntimeError):
                self.save_status.setText('화면 크기를 확인할 수 없습니다. 잠시 후 다시 설정하세요.')
                return
            x, y, w, h = (field.value() for field in self.roi_edits[region_id])
            controller.set_roi(region_id, Rect(x, y, w, h))
            self._schedule_save()

        def _show_effect_selector(self) -> None:
            single = self.mode_tabs.currentIndex() == 1
            self.effect_box.setVisible(single)
            self.effect_label.setVisible(single)
            self.reset_size_button.setToolTip('현재 모드에 해당하는 아이콘 크기를 96px로 복원합니다.')

        def _mode_changed(self, index: int) -> None:
            controller.profile['icon_edit_mode'] = 'single' if index == 1 else 'common'
            self._show_effect_selector()
            self._load_effect_controls()
            self._schedule_save()

        def _reset_icon_size(self) -> None:
            self.size.setValue(96)
            self._effect_changed('size', 96)

        def _reset_icon_opacity(self) -> None:
            self.opacity.setValue(0)
            self._effect_changed('opacity', 0)

        def _center_time_changed(self, enabled: bool) -> None:
            controller.profile['time_text_centered'] = enabled
            overlay.sync_layout()
            self._schedule_save()

        def _lock_changed(self, locked: bool) -> None:
            controller.set_position_locked(locked)
            self.size.setEnabled(not locked)
            self.size_value.setEnabled(not locked)
            self.reset_size_button.setEnabled(not locked)
            self.reset_positions_button.setEnabled(not locked)
            self._schedule_save()
            overlay.sync_layout()
            # Changing native window flags may hide the widget on Windows.
            overlay.show()

        def _reset_overlay_positions(self) -> None:
            previous = controller.overlay.to_dict()
            if not controller.reset_overlay_positions():
                return
            if not self._save_settings():
                controller.profile['overlay'] = previous
            overlay.sync_layout()

        def _preview_roi_committed(self, pixel_rect: Rect, monitor_width: int, monitor_height: int) -> None:
            controller.set_roi_from_preview(
                "self_buff",
                pixel_rect,
                monitor_width=monitor_width,
                monitor_height=monitor_height,
            )
            region = controller.rois["self_buff"].pixel_rect
            if region is not None:
                for field, value in zip(self.roi_edits["self_buff"], region.to_dict().values()):
                    field.setValue(round(value))
            controller.save()

        def _reset_preview_roi(self) -> None:
            if default_preview_roi is None:
                return
            monitor_id = controller.document['display'].get('capture_monitor_id')
            monitor = next((m for m in controller.monitors if m.monitor_id == monitor_id), None)
            roi = default_preview_roi
            if monitor and (roi.x + roi.width > monitor.width or roi.y + roi.height > monitor.height):
                self.save_status.setText('기본 영역이 현재 모니터 밖에 있습니다. 전체 화면에서 영역을 설정하세요.')
                return
            previous = list(controller.profile['rois'])
            controller.set_roi('self_buff', roi)
            if not self._save_settings():
                controller.profile['rois'] = previous
                return
            for widget, value in zip(self.roi_edits['self_buff'], roi.to_dict().values()):
                widget.setValue(round(value))
            controller.update_preview(frame=None, region_id='self_buff', detections=(),
                                      status='기본 영역 적용 · 다음 캡처 대기')
            self.preview_image.clear()
            self.preview_image._original_pixmap = None
            self.preview_image._source_rect = None
            self.preview_image.setText('기본 영역 적용 · 다음 캡처 대기')
            self.refresh_preview()

        def _green_changed(self, effect_id, checked):
            controller.profile.setdefault('green_buffs',{})[effect_id]=bool(checked)
            controller.ensure_icon(effect_id)
            controller.compact_green_icons()
            self._schedule_save()
            overlay.sync_layout()

        def _effect_changed(self, field: str, value: int) -> None:
            effects = MERGED_COMMON if self.mode_tabs.currentIndex() == 0 else (self.effect_box.currentData(),)
            if field == 'size' and self.mode_tabs.currentIndex() == 0:
                for effect_id in MERGED_COMMON:
                    controller.ensure_icon(effect_id)
                controller.resize_common_group(value)
                overlay.sync_layout()
                self._schedule_save()
                return
            for effect_id in effects:
                if not effect_id:
                    continue
                controller.ensure_icon(effect_id)
                if field == 'size':
                    controller.resize_icon(effect_id, value)
                else:
                    controller.set_icon_transparency(effect_id, value)
            overlay.sync_layout()
            self._schedule_save()

        def _set_preview_visibility(self, visible: bool) -> None:
            self.preview_toggle.setText('미리보기 숨기기' if visible else '미리보기 보기')
            self.preview_image.setVisible(visible)
            self.buff_roi_label.setVisible(visible)
            self.buff_roi_row.setVisible(visible)
            self.roi_drag_label.setVisible(visible)
            self.roi_controls.setVisible(visible)
            self.roi_drag_edit.setVisible(visible)
            self.roi_drag_edit.setEnabled(visible)
            if not visible:
                self.roi_drag_edit.setChecked(False)
                self.preview_image.clear()
                self.preview_image._original_pixmap = None
                self.preview_image._source_rect = None

        def _toggle_preview(self, visible: bool) -> None:
            controller.document.setdefault('ui', {})['preview_visible'] = visible
            self._set_preview_visibility(visible)
            self.refresh_preview()
            self._schedule_save()

        def refresh_preview(self) -> None:
            preview = controller.preview
            frame = preview.frame
            if self.preview_toggle.isChecked() and frame is not None and hasattr(frame, "shape") and len(frame.shape) in (2, 3):
                height, width = frame.shape[:2]
                if len(frame.shape) == 2:
                    image = QtGui.QImage(frame.data, width, height, frame.strides[0], QtGui.QImage.Format.Format_Grayscale8)
                elif frame.shape[2] == 4:
                    image = QtGui.QImage(frame.data, width, height, frame.strides[0], QtGui.QImage.Format.Format_BGRA8888)
                else:
                    image = QtGui.QImage(frame.data, width, height, frame.strides[0], QtGui.QImage.Format.Format_BGR888)
                pixmap = QtGui.QPixmap.fromImage(image.copy())
                monitor_id = controller.document["display"].get("capture_monitor_id")
                monitor = next((item for item in controller.monitors if item.monitor_id == monitor_id), None)
                monitor = monitor or next((item for item in controller.monitors if item.primary), None)
                source_rect = None
                monitor_size = None
                if monitor is not None:
                    monitor_size = (monitor.width, monitor.height)
                    if (width, height) == monitor_size:
                        source_rect = Rect(0, 0, width, height)
                    else:
                        current = controller.rois["self_buff"].pixel_rect
                        if current is not None and (round(current.width), round(current.height)) == (width, height):
                            source_rect = current
                self.preview_image.set_preview(pixmap, source_rect, monitor_size)
            overlay.sync_layout()

        def apply_live_update(self, update: Any) -> None:
            items = []
            presented = compose_overlay_items(
                update.detections, controller.profile, "self_buff"
            )
            target_update = getattr(update, 'target_update', None)
            all_detections = update.detections
            if target_update is not None:
                all_detections += target_update.detections
                presented += target_items(target_update.detections, controller.profile)
                self._refresh_target_preview(target_update)
            for item in presented:
                detection = next(
                    (value for value in all_detections if value.get("effect_id") == (item.source_effect_id or item.effect_id)),
                    None,
                )
                icon_frame = None if detection is None else detection.get("_icon_frame")
                if icon_frame is None and update.frame is not None and item.source_bbox is not None:
                    x, y, width, height = item.source_bbox
                    frame_height, frame_width = update.frame.shape[:2]
                    if x >= 0 and y >= 0 and x + width <= frame_width and y + height <= frame_height:
                        icon_frame = update.frame[y:y + height, x:x + width].copy()
                items.append(
                    PreviewItem(
                        effect_id=item.effect_id,
                        confidence=item.confidence,
                        remaining_text=item.remaining_text,
                        stack_text=item.stack_text,
                        recognized=True,
                        icon_frame=icon_frame,
                        border_color=None if detection is None else detection.get('border_color'),
                    )
                )
            controller.update_preview(
                frame=update.frame,
                region_id="self_buff",
                detections=items,
                status=update.status,
                sequence=update.sequence,
            )
            self.refresh_preview()

        def enable_dev_overlay(self, watcher):
            nonlocal dev_overlay
            dev_overlay = watcher
            self.dev_timer = QtCore.QTimer(self)
            self.dev_timer.setInterval(300)
            def reload_display():
                if watcher.poll():
                    overlay.sync_layout()
                self.setWindowTitle('LostArk Buff Eye 설정 [개발: ' +
                                    ('오류 - ' + watcher.error if watcher.error else '즉시 반영') + ']')
            self.dev_timer.timeout.connect(reload_display)
            self.dev_timer.start()
            reload_display()

        def set_live_error(self, status: str, detail: str) -> None:
            controller.update_preview(
                frame=None,
                region_id="self_buff",
                detections=(),
                status=f"{status}: {detail}",
            )
            self.refresh_preview()

    overlay = OverlayWindow()
    settings_window = SettingsWindow()
    if live_worker is not None:
        monitor_timer = QtCore.QTimer(settings_window)
        monitor_timer.setSingleShot(True)
        monitor_timer.setInterval(100)
        def refresh_geometry():
            try:
                if refresh_worker_monitors(controller, live_worker):
                    from .capture_session import resolve_capture_binding
                    source = getattr(live_worker, 'support', live_worker)
                    monitors = source.session.provider.enumerate_monitors()
                    for region_id, fields in settings_window.roi_edits.items():
                        if any(field.hasFocus() for field in fields):
                            continue
                        try:
                            region = resolve_capture_binding(controller.document, region_id, monitors).request.region
                            region_worker=getattr(live_worker,'target',None) if region_id=='target_debuff' else source
                            binding=getattr(getattr(getattr(region_worker,'session',None),'status',None),'binding',None)
                            if binding is not None:
                                region=binding.request.region
                        except (OSError, RuntimeError):
                            continue
                        for field, value in zip(fields, (region.x, region.y, region.width, region.height)):
                            blocked = field.blockSignals(True)
                            field.setValue(value)
                            field.blockSignals(blocked)
                    overlay.sync_layout()
            except (OSError, RuntimeError):
                pass  # Capture owns retry/backoff; keep the last UI geometry.
        monitor_timer.timeout.connect(refresh_geometry)
        def watch_screen(screen):
            screen.geometryChanged.connect(lambda *_: monitor_timer.start())
            monitor_timer.start()
        for screen in app.screens():
            screen.geometryChanged.connect(lambda *_: monitor_timer.start())
        app.screenAdded.connect(watch_screen)
        app.screenRemoved.connect(lambda *_: monitor_timer.start())
        settings_window.monitor_refresh_timer = monitor_timer
        monitor_timer.start()
        viewport_timer=QtCore.QTimer(settings_window)
        viewport_timer.setInterval(1000)
        viewport_timer.timeout.connect(lambda:refresh_geometry() if settings_window.isVisible() else None)
        viewport_timer.start()
        settings_window.viewport_refresh_timer=viewport_timer
    desktop = QtDesktop(app, settings_window, overlay, live_worker=live_worker)
    if live_worker is not None:
        timer = QtCore.QTimer(settings_window)
        timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        timer.setInterval(preview_interval_ms)
        timer.timeout.connect(desktop.pump_live_once)
        desktop.live_timer = timer
        app.aboutToQuit.connect(desktop.shutdown)
        live_worker.start()
        timer.start()
    return desktop
