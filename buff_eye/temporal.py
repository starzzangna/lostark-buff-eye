"""Pure temporal stabilization for live recognition detections."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable, Mapping


EffectKey = tuple[str, str]


@dataclass(frozen=True)
class TemporalEvent:
    kind: str
    region_id: str
    effect_id: str
    captured_at_ns: int
    absent_for_ns: int = 0


@dataclass(frozen=True)
class TemporalFrame:
    captured_at_ns: int
    detections: tuple[dict[str, Any], ...]
    events: tuple[TemporalEvent, ...]


@dataclass
class _EffectState:
    detection: dict[str, Any]
    first_seen_ns: int
    last_seen_ns: int
    missing_since_ns: int | None
    missed_frames: int
    order: int
    last_time_value: float | None


def _score(detection: Mapping[str, Any]) -> float:
    value = detection.get("icon_score")
    return float(value) if isinstance(value, Real) and not isinstance(value, bool) else -1.0


def _key(detection: Mapping[str, Any]) -> EffectKey:
    effect_id = detection.get("effect_id")
    region_id = detection.get("region_id")
    if not isinstance(effect_id, str) or not effect_id:
        raise ValueError("detection needs a non-empty effect_id")
    if not isinstance(region_id, str) or not region_id:
        raise ValueError("detection needs a non-empty region_id")
    return region_id, effect_id


def _time_value(detection: Mapping[str, Any]) -> float | None:
    payload = detection.get("time")
    if not isinstance(payload, Mapping) or payload.get("status") not in {"read", "recognized", "ok"}:
        return None
    value = payload.get("value", payload.get("normalized_value"))
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if result >= 0 and result == result and result != float("inf") else None


def _held_detection(state: _EffectState, captured_at_ns: int) -> dict[str, Any]:
    """Keep icon identity briefly while explicitly invalidating dynamic text."""
    detection = copy.deepcopy(state.detection)
    detection["temporal"] = {
        "observed": False,
        "first_seen_ns": state.first_seen_ns,
        "last_seen_ns": state.last_seen_ns,
        "missed_frames": state.missed_frames,
        "age_since_observation_ns": captured_at_ns - state.last_seen_ns,
    }
    for field in ("time", "stack"):
        if field in detection:
            detection[field] = {
                "status": "unreadable",
                "raw_text": None,
                "value": None,
                "reason": "temporal_miss",
            }
    return detection


def _observed_detection(state: _EffectState) -> dict[str, Any]:
    detection = copy.deepcopy(state.detection)
    detection["temporal"] = {
        "observed": True,
        "first_seen_ns": state.first_seen_ns,
        "last_seen_ns": state.last_seen_ns,
        "missed_frames": 0,
        "age_since_observation_ns": 0,
    }
    return detection


class TemporalDetectionTracker:
    """Suppress one/two-frame icon flicker without retaining stale time text.

    Effects disappear when either the configured missed-frame grace is exceeded
    or the elapsed capture time reaches ``disappear_after_ns``.  The latter
    prevents a long dropped-frame gap from extending an effect indefinitely.
    """

    def __init__(
        self,
        *,
        miss_grace_frames: int = 2,
        disappear_after_ns: int = 250_000_000,
        reapply_time_increase: float = 0.5,
        stabilize_display: bool = False,
    ) -> None:
        if miss_grace_frames < 0:
            raise ValueError("miss_grace_frames cannot be negative")
        if disappear_after_ns < 1:
            raise ValueError("disappear_after_ns must be positive")
        if reapply_time_increase < 0:
            raise ValueError("reapply_time_increase cannot be negative")
        self.miss_grace_frames = miss_grace_frames
        self.disappear_after_ns = disappear_after_ns
        self.reapply_time_increase = reapply_time_increase
        self._states: dict[EffectKey, _EffectState] = {}
        self._last_timestamp_ns: int | None = None
        from .display_stability import DisplayStabilizer
        self._display = DisplayStabilizer() if stabilize_display else None

    def reset(self) -> None:
        self._states.clear()
        self._last_timestamp_ns = None
        if self._display is not None:
            self._display.groups.clear()

    def update(
        self,
        captured_at_ns: int,
        detections: Iterable[Mapping[str, Any]],
        *,
        scan_performed: bool = True,
    ) -> TemporalFrame:
        if captured_at_ns < 0:
            raise ValueError("captured_at_ns cannot be negative")
        if self._last_timestamp_ns is not None and captured_at_ns < self._last_timestamp_ns:
            raise ValueError("captured_at_ns must be monotonic")
        self._last_timestamp_ns = captured_at_ns

        detections = list(detections)
        selected: dict[EffectKey, tuple[int, dict[str, Any]]] = {}
        if scan_performed:
            for index, raw in enumerate(detections):
                if not isinstance(raw, Mapping):
                    raise ValueError("detections must be mappings")
                detection = copy.deepcopy(dict(raw))
                key = _key(detection)
                previous = selected.get(key)
                if previous is None or _score(detection) > _score(previous[1]):
                    selected[key] = index, detection
        else:
            # A skipped scan provides no fresh evidence.  Ignore any caller's
            # reused detections so OCR values cannot silently become stale.
            selected = {}

        events: list[TemporalEvent] = []
        observed_keys = set(selected)
        for key, (order, detection) in selected.items():
            region_id, effect_id = key
            current_time = _time_value(detection)
            state = self._states.get(key)
            if state is None:
                self._states[key] = _EffectState(
                    detection, captured_at_ns, captured_at_ns, None, 0, order, current_time
                )
                events.append(TemporalEvent("appeared", region_id, effect_id, captured_at_ns))
                continue
            if state.missed_frames:
                events.append(
                    TemporalEvent(
                        "reacquired",
                        region_id,
                        effect_id,
                        captured_at_ns,
                        captured_at_ns - state.last_seen_ns,
                    )
                )
            if (
                current_time is not None
                and state.last_time_value is not None
                and current_time > state.last_time_value + self.reapply_time_increase
            ):
                events.append(TemporalEvent("reapplied", region_id, effect_id, captured_at_ns))
            state.detection = detection
            state.last_seen_ns = captured_at_ns
            state.missing_since_ns = None
            state.missed_frames = 0
            state.order = order
            if current_time is not None:
                state.last_time_value = current_time

        for key, state in list(self._states.items()):
            if key in observed_keys:
                continue
            if scan_performed:
                state.missed_frames += 1
                if state.missing_since_ns is None:
                    state.missing_since_ns = captured_at_ns
            absent_for = captured_at_ns - state.last_seen_ns
            if state.missed_frames > self.miss_grace_frames or absent_for >= self.disappear_after_ns:
                region_id, effect_id = key
                events.append(
                    TemporalEvent("disappeared", region_id, effect_id, captured_at_ns, absent_for)
                )
                del self._states[key]

        current_order = {key: value[0] for key, value in selected.items()}
        ordered = sorted(
            self._states.items(),
            key=lambda pair: (
                0 if pair[0] in observed_keys else 1,
                current_order.get(pair[0], pair[1].order),
                pair[0],
            ),
        )
        output = tuple(
            _observed_detection(state) if key in observed_keys else _held_detection(state, captured_at_ns)
            for key, state in ordered
        )
        if self._display is not None:
            output = self._display.update(captured_at_ns, detections if scan_performed else [], output)
        return TemporalFrame(captured_at_ns, output, tuple(events))
