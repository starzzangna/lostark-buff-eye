"""Pure recognition-to-overlay presentation mapping.

This module owns no windows and mutates neither detections nor settings.  The
renderer receives immutable items whose placement has already been resolved by
``overlay_model`` and whose visibility has been filtered by the active profile.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
import re
from typing import Any, Iterable, Mapping

from .overlay_model import ROI_IDS, OverlayLayout
from .effect_groups import CLASS_ATTACK, SUPPORT_EFFECTS as MERGED_COMMON, GREEN_EFFECTS, enabled_green


def display_slot(effect_id, profile):
    if profile.get('overlay_scope') == 'support_common' and effect_id in (*CLASS_ATTACK, 'support.attack.additional'):
        return 'support.attack.01'
    if not profile.get('game_ui', {}).get('stack_merge', True) and effect_id in CLASS_ATTACK:
        return 'support.attack.additional'
    return effect_id


@dataclass(frozen=True)
class OverlayItem:
    effect_id: str
    region_id: str
    x: float
    y: float
    size: float
    opacity: float
    confidence: float | None
    source_bbox: tuple[int, int, int, int] | None
    icon_path: str | None
    remaining_text: str | None
    remaining_seconds: int | float | None
    time_status: str
    stack_text: str | None
    stack_value: int | float | None
    stack_status: str
    source_effect_id: str | None = None


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return int(result) if result.is_integer() else result


def _recognized_text(payload: Any) -> tuple[str | None, int | float | None, str]:
    if not isinstance(payload, Mapping):
        return None, None, "missing"
    status = str(payload.get("status", "missing"))
    raw = payload.get("raw_text")
    # Recognition abstention must remain an abstention in the overlay.  Never
    # display a stale or diagnostic raw string from an unreadable result.
    if status not in {"read", "recognized", "ok"} or not isinstance(raw, str) or not raw:
        return None, None, status
    value = payload.get("value", payload.get("normalized_value"))
    return raw, _number(value), status


def _source_bbox(value: Any) -> tuple[int, int, int, int] | None:
    if (not isinstance(value, (list, tuple)) or len(value) != 4 or
            any(type(part) is not int for part in value) or
            min(value[2:]) <= 0):
        return None
    return tuple(value)


def _display_timer(detection: Mapping[str, Any]) -> tuple[str | None, int | float | None, str]:
    """Compare the stabilized countdown, never the older raw OCR underneath it."""
    if 'display_time' in detection:
        payload = detection['display_time']
        text = payload.get('raw_text') if isinstance(payload, Mapping) else None
        match = re.fullmatch(r'(0(?:\.[0-9])?|[1-9][0-9]*|00)초', text or '')
        return text, float(match.group(1)) if match else None, 'display'
    return _recognized_text(detection.get('time'))


def compose_overlay_items(
    detections: Iterable[Mapping[str, Any]],
    profile: Mapping[str, Any],
    region_id: str = "self_buff",
    *,
    max_items: int | None = None,
) -> tuple[OverlayItem, ...]:
    """Resolve one ROI's detector output against persistent profile settings.

    Duplicate variants of one effect collapse to the highest-confidence
    detection.  ``tracked_effects`` order defines display priority; when that
    list is empty, confidence and effect ID provide a deterministic order.
    Effects without a configured ``IconPlacement`` are omitted because this
    pure layer must not silently mutate persistent layout.
    """
    if region_id not in ROI_IDS:
        raise ValueError(f"unknown ROI: {region_id}")
    if max_items is not None and (type(max_items) is not int or max_items < 0):
        raise ValueError("max_items must be a non-negative integer or None")

    layout = OverlayLayout.from_dict(profile.get("overlay", {}))
    game_ui = profile.get("game_ui", {})
    # The current single-attack overlay follows observed red detections in
    # either game UI mode. Legacy saved stack_merge is not a live mode switch.
    unified_attack = profile.get('overlay_scope') == 'support_common'
    merged = unified_attack or game_ui.get('stack_merge', True)
    configured_limit = game_ui.get(
        "target_display_count" if region_id == "target_debuff" else "buff_display_count",
        17 if region_id == "target_debuff" else 15,
    )
    if type(configured_limit) is not int or configured_limit < 0:
        raise ValueError("profile display count must be a non-negative integer")
    limit = configured_limit if max_items is None else min(configured_limit, max_items)

    tracked = profile.get("tracked_effects", [])
    if not isinstance(tracked, list):
        raise ValueError("tracked_effects must be a list")
    definitions: dict[str, Mapping[str, Any]] = {}
    priority: dict[str, int] = {}
    for index, definition in enumerate(tracked):
        if not isinstance(definition, Mapping):
            continue
        effect_id = definition.get("effect_id")
        if isinstance(effect_id, str) and effect_id:
            definitions[effect_id] = definition
            priority.setdefault(effect_id, index)
    filtering_definitions = {
        effect_id: definition
        for effect_id, definition in definitions.items()
        if definition.get("registration_only") is not True
    }

    selected: dict[str, Mapping[str, Any]] = {}
    for detection in detections:
        if not isinstance(detection, Mapping):
            continue
        effect_id = detection.get("effect_id")
        if not isinstance(effect_id, str) or not effect_id:
            continue
        slot = display_slot(effect_id, profile)
        if effect_id in GREEN_EFFECTS and effect_id not in enabled_green(profile):
            continue
        if profile.get('overlay_scope') == 'support_common' and slot not in MERGED_COMMON:
            continue
        if detection.get("region_id") != region_id or (slot not in layout.icons and effect_id not in layout.icons):
            continue
        if merged and effect_id in {
            'support.attack.bard', 'support.attack.paladin',
            'support.attack.artist', 'support.attack.valkyrie',
            'support.extra.01',
        }:
            continue
        if filtering_definitions:
            definition = filtering_definitions.get(effect_id)
            if definition is None or definition.get("enabled", True) is not True:
                continue
        elif effect_id in definitions and definitions[effect_id].get("enabled", True) is not True:
            continue
        confidence = _number(detection.get("icon_score"))
        previous = selected.get(effect_id)
        previous_confidence = _number(previous.get("icon_score")) if previous else None
        if previous is None or (confidence if confidence is not None else -1) > (
                previous_confidence if previous_confidence is not None else -1):
            selected[effect_id] = detection

    if unified_attack and 'support.attack.01' not in selected and 'support.attack.additional' in selected:
        selected['support.attack.01'] = dict(selected['support.attack.additional'],
            effect_id='support.attack.01', source_effect_id='support.attack.additional')

    # Only share the timer while the red attack buff is present.  Its identity,
    # icon, border, placement and emphasis remain those of the red detection.
    attack_timer = None
    if region_id == 'self_buff' and merged and 'support.attack.01' in selected:
        partners = ('support.attack.additional',)
        candidates = [_display_timer(selected[effect])
                      for effect in ('support.attack.01',) + partners if effect in selected]
        readable = [timer for timer in candidates if timer[1] is not None and timer[1] >= 0]
        if readable:
            attack_timer = max(readable, key=lambda timer: timer[1])

    if not merged:
        classes = [effect for effect in CLASS_ATTACK if effect in selected]
        if classes:
            winner = max(classes, key=lambda effect: (_number(selected[effect].get('icon_score')) or 0,
                                                      -CLASS_ATTACK.index(effect)))
            for effect in classes:
                if effect != winner:
                    del selected[effect]
            selected.pop('support.attack.additional', None)

    if unified_attack:
        for effect in (*CLASS_ATTACK, 'support.attack.additional'):
            selected.pop(effect, None)

    ordered = sorted(
        selected.items(),
        key=lambda pair: (
            priority.get(pair[0], len(priority)),
            -(_number(pair[1].get("icon_score")) or 0),
            pair[0],
        ),
    )[:limit]

    result = []
    for effect_id, detection in ordered:
        slot = display_slot(effect_id, profile)
        placement = layout.icons[slot if slot in layout.icons else effect_id]
        definition = definitions.get(effect_id, {})
        variants = definition.get("template_variants", [])
        icon_path = definition.get("overlay_icon_path")
        if not isinstance(icon_path, str) or not icon_path:
            icon_path = variants[0] if isinstance(variants, list) and variants and isinstance(variants[0], str) else None
        remaining_text, remaining_seconds, time_status = _recognized_text(detection.get("time"))
        stack_text, stack_value, stack_status = _recognized_text(detection.get("stack"))
        if "display_time" in detection:
            remaining_text = detection["display_time"].get("raw_text")
            remaining_seconds, time_status = None, "display"
        if effect_id == 'support.attack.01' and attack_timer is not None:
            remaining_text, remaining_seconds, time_status = attack_timer
        if "display_stack" in detection:
            stack_text = detection["display_stack"].get("raw_text")
            stack_value, stack_status = None, "display"
        if effect_id in GREEN_EFFECTS and remaining_text == '?':
            remaining_text = None
        if effect_id == 'support.attack.01':
            stack_text = None
            stack_value, stack_status = None, 'hidden'
        result.append(OverlayItem(
            effect_id=effect_id,
            region_id=region_id,
            x=placement.x,
            y=placement.y,
            size=placement.size,
            opacity=1.0 - placement.transparency_pct / 100.0,
            confidence=_number(detection.get("icon_score")),
            source_bbox=_source_bbox(detection.get("bbox")),
            icon_path=icon_path,
            remaining_text=remaining_text,
            remaining_seconds=remaining_seconds,
            time_status=time_status,
            stack_text=stack_text,
            stack_value=stack_value,
            stack_status=stack_status,
            source_effect_id=detection.get('source_effect_id', effect_id),
        ))
    return tuple(result)
