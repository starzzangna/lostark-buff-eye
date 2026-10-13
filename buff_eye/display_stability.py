"""Display continuity, kept separate from observed OCR and lifecycle events."""
from __future__ import annotations

import copy
import math
from .effect_groups import GREEN_EFFECTS
from dataclasses import dataclass


def seconds(detection):
    payload = detection.get('time', {})
    value = payload.get('value')
    if (payload.get('status') in ('read', 'recognized', 'ok')
            and isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0):
        return float(value)
    return None


@dataclass
class Instance:
    detection: dict
    seen: int
    deadline: int | None = None
    time_seen: int | None = None
    stack: dict | None = None
    stack_seen: int = 0
    last_value: float | None = None
    pending_jump: float | None = None
    pending_jump_seen: int = 0
    emphasis_seen: int | None = None


def format_seconds(value):
    """Use explicit leading zero and decimal point for sub-second values."""
    if value <= 0:
        return '0.0초'
    if value < 1:
        return f'{min(0.9, math.ceil(value*10)/10):.1f}초'
    return f'{math.ceil(value)}초'


def timer_text(track, now):
    if track.time_seen == now:
        return format_seconds(track.last_value), 'observed'
    if (track.last_value == 0 and track.time_seen is not None
            and now-track.time_seen <= 350_000_000 and track.seen == now):
        # Keep a witnessed terminal zero through a brief text-only miss.
        # Missing icons still expire normally; never fabricate zero from 1s.
        return '0.0초', 'held'
    if track.deadline is not None and now < track.deadline:
        # A recurring 1s buff must not oscillate 1 -> 0.9 -> 1 on OCR misses.
        # Only a real fractional observation starts its fractional countdown.
        if track.last_value == 1:
            return '1초', 'held'
        return format_seconds((track.deadline-now)/1e9), 'estimated'
    return '?', 'unreadable'


class DisplayStabilizer:
    """Hold an observed icon until its read deadline; never invent renewals.

    Common and class-specific attack buffs have different effect IDs.
    Duplicate candidates of one effect collapse by recognition confidence.
    Unknown text is held for at most 600ms; predicted timers retain estimated status.
    """

    grace_ns = 600_000_000
    green_stack_grace_ns = 1_500_000_000

    def __init__(self):
        self.groups = {}

    def update(self, now, observed, stabilized):
        grouped = {}
        for d in observed:
            grouped.setdefault((d['region_id'], d['effect_id']), []).append(d)
        for key in set(self.groups) | set(grouped):
            tracks = self.groups.setdefault(key, [])
            tracks[:] = [t for t in tracks if now-t.seen < self.grace_ns or
                         (t.deadline is not None and now < t.deadline)]
            group_icon = tracks[0].detection.get('_icon_frame') if tracks else None
            candidates = grouped.get(key, [])
            fresh = [max(candidates, key=lambda d: d.get('icon_score', -1))] if candidates else []
            available = list(tracks)
            # Read timers associate first so unreadable duplicate slots do not
            # steal the countdown instance from a clearly observed timer.
            for d in sorted(fresh, key=lambda d: seconds(d) is None):
                value = seconds(d)
                def cost(t):
                    predicted = max(0, (t.deadline-now)/1e9) if t.deadline is not None else None
                    time_cost = abs(value-predicted) if value is not None and predicted is not None else 100
                    return time_cost, abs(d['bbox'][0]-t.detection['bbox'][0])
                track = min(available, key=cost) if available else None
                if track is None:
                    track = Instance(copy.deepcopy(d), now)
                    tracks.append(track)
                else:
                    available = [t for t in available if t is not track]
                # Freeze the first valid bitmap throughout this appearance.
                icon = track.detection.get('_icon_frame')
                previous_border = track.detection.get('border_color')
                if group_icon is not None:
                    icon = group_icon
                track.detection = copy.deepcopy(d)
                # Freeze the first observed frame color with its bitmap. The
                # live border sampler can fluctuate under background/sheens.
                # Raw detection colors still reach OCR before this display step.
                if previous_border is not None:
                    track.detection['border_color'] = previous_border
                if icon is not None:
                    track.detection['_icon_frame'] = icon
                if group_icon is None:
                    group_icon = track.detection.get('_icon_frame')
                track.seen = now
                if d.get('merged_emphasis') is True:
                    track.emphasis_seen = now
                predicted = max(0, (track.deadline-now)/1e9) if track.deadline is not None else None
                suspicious = (value is not None and value > 0 and track.last_value is not None
                              and predicted is not None and
                              (abs(value-predicted) > 1.25 or
                               (value >= 2 and 0 < track.last_value < 1)))
                # Ordinary rounding/countdown changes pass through. Large
                # changes need a second consistent observation in either
                # direction; one bad 18 -> 8 must not reset the display clock.
                if suspicious:
                    confirmed = (track.pending_jump is not None
                                 and 0 <= track.pending_jump-value <= 1
                                 and 0 < now-track.pending_jump_seen <= 350_000_000)
                    if confirmed:
                        track.pending_jump = None
                    else:
                        track.pending_jump, track.pending_jump_seen = value, now
                        value = None
                else:
                    track.pending_jump = None
                if value is not None:
                    track.deadline = now + round(value*1e9)
                    track.time_seen = now
                    track.last_value = value
                stack = d.get('stack', {})
                if stack.get('status') in ('read', 'recognized', 'ok'):
                    track.stack, track.stack_seen = copy.deepcopy(stack), now
                elif stack.get('status') in ('absent', 'not_present'):
                    track.stack = None
            tracks[:] = [t for t in tracks if t.seen == now or
                         (t.deadline is not None and now < t.deadline) or
                         (t.deadline is None and now-t.seen < self.grace_ns)]
            if key[1] in GREEN_EFFECTS:
                # Stack completion can remove an icon before its last timer.
                # A predicted deadline must not keep that optional buff alive.
                tracks[:] = [t for t in tracks if now-t.seen < self.grace_ns]
            if not tracks:
                del self.groups[key]

        raw_by_key = {(d['region_id'], d['effect_id']): d for d in stabilized}
        result = []
        for key, tracks in self.groups.items():
            selected = tracks[0]
            base = copy.deepcopy(raw_by_key.get(key, tracks[0].detection))
            if not any(t.seen == now for t in tracks):
                base['time'] = {'status': 'unreadable', 'raw_text': None, 'reason': 'display_hold'}
                base['stack'] = {'status': 'unreadable', 'raw_text': None, 'reason': 'display_hold'}
                base['temporal'] = dict(base.get('temporal', {}), observed=False)
            base['_icon_frame'] = tracks[0].detection.get('_icon_frame')
            base['border_color'] = selected.detection.get('border_color')
            base['display_merged_emphasis'] = (
                selected.emphasis_seen is not None
                and now-selected.emphasis_seen < self.grace_ns)
            text, status = timer_text(selected, now)
            base['display_time'] = {'raw_text': text, 'status': 'display',
                                    'sources': [status], 'instance_count': len(tracks)}
            # Background can interrupt tiny stack OCR while the icon remains
            # verified. Hold only the last observed number, never infer a count.
            stack_grace = (self.green_stack_grace_ns
                           if key[1] in GREEN_EFFECTS and selected.seen == now
                           else self.grace_ns)
            stack = (selected.stack['raw_text'] if selected.stack and
                     now-selected.stack_seen < stack_grace else None)
            source = ('observed' if selected.stack_seen == now else 'held') if stack else 'unreadable'
            base['display_stack'] = {'raw_text': stack, 'status': 'display', 'source': source}
            result.append(base)
        return tuple(result)
