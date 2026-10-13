"""Bounded, opt-in diagnostic trace; detector output is not ground truth."""
from collections import Counter, deque
from copy import deepcopy

from buff_eye.timer_audit import timer_category


class TemporalAudit:
    def __init__(self, limit=2048):
        self.records = deque(maxlen=limit)
        self.limit = limit
        self.counts = Counter()
        self.previous = {}
        self.sequences = {}
        self.total = 0

    def observe(self, group, update):
        if self.sequences.get(group) == update.sequence:
            return
        self.sequences[group] = update.sequence
        current = {}
        for detection in update.detections:
            effect = detection.get('effect_id', 'unknown')
            current.setdefault(effect, []).append({
                'category': timer_category(detection),
                'time': deepcopy(detection.get('time')),
                'display_time': deepcopy(detection.get('display_time')),
            })
        previous = self.previous.get(group, {})
        for effect in sorted(current.keys() | previous.keys()):
            before, after = previous.get(effect, []), current.get(effect, [])
            # Confidence jitter must not fill the trace with identical values.
            def signature(items):
                return [(item['category'], (item['time'] or {}).get('raw_text'),
                         (item['display_time'] or {}).get('raw_text')) for item in items]
            if signature(before) == signature(after):
                continue
            event = 'appeared' if not before else 'removed' if not after else 'changed'
            self.counts[group + ':' + event] += 1
            self.total += 1
            self.records.append({'group': group, 'effect': effect, 'event': event,
                                 'sequence': update.sequence,
                                 'captured_at_ns': update.captured_at_ns,
                                 'before': before, 'after': after})
        self.previous[group] = current

    def report(self):
        return {'counts': dict(self.counts), 'records': list(self.records),
                'limit': self.limit, 'dropped': self.total - len(self.records),
                'meaning': 'Detector/display transitions, not independently verified game events.'}
