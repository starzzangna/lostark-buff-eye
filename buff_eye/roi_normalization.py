"""Validated ROI resizing and coordinate restoration for recognition.

This module is independent of capture threads and UI widgets.  It makes every
resize decision explicit so a frame from the wrong ROI or an incompatible
aspect ratio cannot be silently stretched to match a detector configuration.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


class RoiNormalizationError(ValueError):
    """The captured ROI is incompatible with the configured recognition ROI."""


@dataclass(frozen=True)
class RoiTransform:
    region_id: str
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    scale_x: float
    scale_y: float
    source_aspect_ratio: float
    target_aspect_ratio: float
    aspect_ratio_error: float
    aspect_ratio_tolerance: float
    interpolation: str

    @property
    def resized(self) -> bool:
        return (self.source_width, self.source_height) != (
            self.target_width,
            self.target_height,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["resized"] = self.resized
        return value


def _size(value: tuple[int, int], name: str) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(part) is not int or part < 1 for part in value)
    ):
        raise RoiNormalizationError(f"{name} must be a positive integer (width, height)")
    return value


def build_roi_transform(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    *,
    region_id: str,
    expected_region_id: str,
    aspect_ratio_tolerance: float = 0.02,
) -> RoiTransform:
    """Validate identity/geometry and describe the requested resize."""
    if not isinstance(region_id, str) or not region_id:
        raise RoiNormalizationError("region_id must be non-empty")
    if region_id != expected_region_id:
        raise RoiNormalizationError(
            f"ROI identity mismatch: received {region_id!r}, expected {expected_region_id!r}"
        )
    if (
        isinstance(aspect_ratio_tolerance, bool)
        or not isinstance(aspect_ratio_tolerance, (int, float))
        or not math.isfinite(float(aspect_ratio_tolerance))
        or not 0 <= float(aspect_ratio_tolerance) <= 0.25
    ):
        raise RoiNormalizationError("aspect_ratio_tolerance must be between 0 and 0.25")
    source_width, source_height = _size(source_size, "source_size")
    target_width, target_height = _size(target_size, "target_size")
    source_ratio = source_width / source_height
    target_ratio = target_width / target_height
    ratio_error = abs(source_ratio / target_ratio - 1.0)
    tolerance = float(aspect_ratio_tolerance)
    if ratio_error > tolerance:
        raise RoiNormalizationError(
            "ROI aspect ratio differs from recognition reference by "
            f"{ratio_error:.2%} (allowed {tolerance:.2%}); "
            f"source={source_width}x{source_height}, target={target_width}x{target_height}"
        )
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    if source_width == target_width and source_height == target_height:
        interpolation = "identity"
    elif scale_x < 1 or scale_y < 1:
        interpolation = "area"
    else:
        interpolation = "linear"
    return RoiTransform(
        region_id,
        source_width,
        source_height,
        target_width,
        target_height,
        scale_x,
        scale_y,
        source_ratio,
        target_ratio,
        ratio_error,
        tolerance,
        interpolation,
    )


def normalize_roi(
    frame: np.ndarray,
    target_size: tuple[int, int],
    *,
    region_id: str,
    expected_region_id: str,
    aspect_ratio_tolerance: float = 0.02,
) -> tuple[np.ndarray, RoiTransform]:
    """Return a recognition-sized frame and complete transform metadata."""
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        raise RoiNormalizationError("frame must be a 2D or 3D NumPy image")
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise RoiNormalizationError("frame cannot be empty")
    transform = build_roi_transform(
        (int(frame.shape[1]), int(frame.shape[0])),
        target_size,
        region_id=region_id,
        expected_region_id=expected_region_id,
        aspect_ratio_tolerance=aspect_ratio_tolerance,
    )
    if not transform.resized:
        return frame, transform
    interpolation = cv2.INTER_AREA if transform.interpolation == "area" else cv2.INTER_LINEAR
    normalized = cv2.resize(
        frame,
        (transform.target_width, transform.target_height),
        interpolation=interpolation,
    )
    return normalized, transform


def bbox_to_source(bbox: Any, transform: RoiTransform) -> list[int]:
    """Map a target-frame bbox back to source ROI pixels using edge rounding."""
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or any(type(value) is not int for value in bbox)
    ):
        raise RoiNormalizationError("bbox must contain four integers")
    x, y, width, height = bbox
    if (
        x < 0
        or y < 0
        or width < 1
        or height < 1
        or x + width > transform.target_width
        or y + height > transform.target_height
    ):
        raise RoiNormalizationError("bbox lies outside the normalized ROI")
    left = math.floor(x / transform.scale_x)
    top = math.floor(y / transform.scale_y)
    right = math.ceil((x + width) / transform.scale_x)
    bottom = math.ceil((y + height) / transform.scale_y)
    left = min(max(left, 0), transform.source_width - 1)
    top = min(max(top, 0), transform.source_height - 1)
    right = min(max(right, left + 1), transform.source_width)
    bottom = min(max(bottom, top + 1), transform.source_height)
    return [left, top, right - left, bottom - top]


def restore_detection_coordinates(
    detections: Iterable[Mapping[str, Any]], transform: RoiTransform
) -> tuple[dict[str, Any], ...]:
    """Deep-copy detections and restore icon/time/stack bboxes to source ROI."""
    restored: list[dict[str, Any]] = []
    metadata = transform.to_dict()
    for raw in detections:
        if not isinstance(raw, Mapping):
            raise RoiNormalizationError("detections must be mappings")
        detection = copy.deepcopy(dict(raw))
        if "bbox" in detection:
            detection["bbox"] = bbox_to_source(detection["bbox"], transform)
        for field in ("time", "stack"):
            payload = detection.get(field)
            if isinstance(payload, Mapping) and payload.get("bbox") is not None:
                converted = dict(payload)
                converted["bbox"] = bbox_to_source(converted["bbox"], transform)
                detection[field] = converted
        detection["roi_transform"] = copy.deepcopy(metadata)
        restored.append(detection)
    return tuple(restored)

