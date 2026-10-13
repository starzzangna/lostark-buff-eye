"""Bounded developer-only native crops for inspecting short timer transitions."""
from pathlib import Path
from collections import Counter
import json
import re
import cv2
from .timer_audit import timer_category


class ShortTimerEvidence:
    def __init__(self, folder, limit=400):
        self.folder = Path(folder) if folder else None
        self.limit = limit
        self.records = []
        self.counts = Counter()
        self.last = {}

    def observe(self, group, update):
        if self.folder is None or update.frame is None or len(self.records) >= self.limit:
            return
        for d in update.detections:
            if timer_category(d) != 'observed_read':
                continue
            raw = d.get('time', {}).get('raw_text') or ''
            if not re.fullmatch(r'(?:1|0(?:\.[0-9])?|00)초',raw):
                continue
            effect = d['effect_id']
            bucket = 'one' if raw == '1초' else 'fraction'
            key = (group,effect,bucket)
            if self.counts[key] >= 60:
                continue
            previous = self.last.get((group,effect))
            if previous and previous[0] == raw and update.captured_at_ns-previous[1] < 500_000_000:
                continue
            x,y,w,h = d['bbox']
            left,top = max(0,x-4),max(0,y-24)
            right,bottom = min(update.frame.shape[1],x+w+4),min(update.frame.shape[0],y+h+24)
            if right<=left or bottom<=top:
                continue
            name=f'short-{len(self.records):03d}.png'
            ok,encoded=cv2.imencode('.png',update.frame[top:bottom,left:right])
            if not ok:raise OSError('short timer crop encoding failed')
            (self.folder/name).write_bytes(encoded.tobytes())
            self.records.append(dict(image=name,group=group,effect=effect,sequence=update.sequence,
                captured_at_ns=update.captured_at_ns,bbox=[x-left,y-top,w,h],
                raw_text=raw,display_time=d.get('display_time')))
            self.counts[key]+=1
            self.last[(group,effect)]=(raw,update.captured_at_ns)

    def flush(self):
        if self.folder:
            (self.folder/'short-timers.json').write_text(json.dumps(self.records,ensure_ascii=False,indent=2),encoding='utf-8')
