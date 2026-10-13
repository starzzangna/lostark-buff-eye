"""GUI-independent models for capture regions and overlay placement.

The UI may translate these models into native windows, but all edit permission and
range checks live here so a renderer cannot accidentally bypass position locking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


ROI_IDS = ("self_buff", "self_debuff", "target_debuff")


def _finite(value: float, name: str) -> float:
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class Rect:
    """Rectangle in pixels or normalized display coordinates."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite(self.x, "x"))
        object.__setattr__(self, "y", _finite(self.y, "y"))
        object.__setattr__(self, "width", _finite(self.width, "width"))
        object.__setattr__(self, "height", _finite(self.height, "height"))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("rectangle width and height must be positive")

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Rect":
        return cls(value["x"], value["y"], value["width"], value["height"])


@dataclass(frozen=True)
class CaptureRegion:
    """One of the three independently calibrated game UI regions."""

    region_id: str
    pixel_rect: Rect | None = None
    normalized_rect: Rect | None = None

    def __post_init__(self) -> None:
        if self.region_id not in ROI_IDS:
            raise ValueError(f"unknown ROI: {self.region_id}")
        if self.normalized_rect is not None:
            r = self.normalized_rect
            if r.x < 0 or r.y < 0 or r.x + r.width > 1 or r.y + r.height > 1:
                raise ValueError("normalized ROI must fit inside 0..1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "pixel_rect": None if self.pixel_rect is None else self.pixel_rect.to_dict(),
            "normalized_rect": (
                None if self.normalized_rect is None else self.normalized_rect.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CaptureRegion":
        pixel = value.get("pixel_rect")
        normalized = value.get("normalized_rect")
        return cls(
            str(value["region_id"]),
            None if pixel is None else Rect.from_dict(pixel),
            None if normalized is None else Rect.from_dict(normalized),
        )


@dataclass
class IconPlacement:
    effect_id: str
    x: float = 0
    y: float = 0
    size: float = 64
    transparency_pct: float = 0

    def __post_init__(self) -> None:
        self.effect_id = str(self.effect_id)
        if not self.effect_id:
            raise ValueError("effect_id must not be empty")
        self.x = _finite(self.x, "x")
        self.y = _finite(self.y, "y")
        self.size = _finite(self.size, "size")
        self.transparency_pct = _finite(self.transparency_pct, "transparency_pct")
        if self.size <= 0:
            raise ValueError("icon size must be positive")
        if not 0 <= self.transparency_pct <= 100:
            raise ValueError("transparency_pct must be between 0 and 100")

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect_id": self.effect_id,
            "x": self.x,
            "y": self.y,
            "size": self.size,
            "transparency_pct": self.transparency_pct,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IconPlacement":
        return cls(
            effect_id=value["effect_id"],
            x=value.get("x", 0),
            y=value.get("y", 0),
            size=value.get("size", 64),
            transparency_pct=value.get("transparency_pct", 0),
        )


@dataclass
class OverlayLayout:
    """Persistent overlay layout with lock-enforced edit operations."""

    monitor_id: str | None = None
    position_locked: bool = True
    icons: dict[str, IconPlacement] = field(default_factory=dict)

    def add_icon(self, icon: IconPlacement) -> None:
        self.icons[icon.effect_id] = icon

    def request_move(self, effect_id: str, x: float, y: float) -> bool:
        """Apply a move only while unlocked; return whether it was accepted."""
        if self.position_locked:
            return False
        icon = self.icons[effect_id]
        icon.x = _finite(x, "x")
        icon.y = _finite(y, "y")
        return True

    def request_resize(self, effect_id: str, size: float) -> bool:
        if self.position_locked:
            return False
        size = _finite(size, "size")
        if size <= 0:
            raise ValueError("icon size must be positive")
        self.icons[effect_id].size = size
        return True

    def set_transparency(self, effect_id: str, transparency_pct: float) -> None:
        value = _finite(transparency_pct, "transparency_pct")
        if not 0 <= value <= 100:
            raise ValueError("transparency_pct must be between 0 and 100")
        self.icons[effect_id].transparency_pct = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitor_id": self.monitor_id,
            "position_locked": self.position_locked,
            "icons": [self.icons[key].to_dict() for key in sorted(self.icons)],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OverlayLayout":
        icons = [IconPlacement.from_dict(item) for item in value.get("icons", [])]
        return cls(
            monitor_id=value.get("monitor_id"),
            position_locked=bool(value.get("position_locked", True)),
            icons={icon.effect_id: icon for icon in icons},
        )

