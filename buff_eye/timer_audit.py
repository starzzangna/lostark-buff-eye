"""Opt-in developer diagnostics; never imported by the ordinary app loop."""
from collections import Counter
from pathlib import Path
import hashlib
import cv2


def timer_category(detection):
    timer=detection.get('time') or {}
    if timer.get('reason')=='display_hold' or detection.get('temporal',{}).get('observed') is False:
        return 'display_hold'
    if timer.get('status')=='read':
        return 'observed_read'
    if timer.get('status')=='unreadable':
        return 'observed_unreadable'
    return 'other'


class TimerAudit:
    def __init__(self, folder=None, limit=24, byte_limit=32*1024*1024):
        self.folder=Path(folder) if folder else None
        self.limit=limit
        self.byte_limit=byte_limit
        self.bytes_written=0
        self.counts=Counter()
        self.reasons=Counter()
        self.failures=[]
        self.saved=Counter()
        if self.folder:self.folder.mkdir(parents=True,exist_ok=True)

    def observe(self, group, update):
        for detection in update.detections:
            category=timer_category(detection)
            self.counts[group+':'+category]+=1
            timer=detection.get('time') or {}
            reason=str(timer.get('reason') or 'unspecified')
            effect=detection.get('effect_id','unknown')
            key=(group,effect,reason)
            if category!='observed_unreadable':continue
            self.reasons[':'.join(key)]+=1
            # Up to three examples per effect/reason, with a hard global cap.
            if (self.folder is None or update.frame is None or len(self.failures)>=self.limit
                    or self.saved[key]>=3 or self.bytes_written>=self.byte_limit):continue
            ok,encoded=cv2.imencode('.png',update.frame)
            if not ok:raise OSError('Could not encode diagnostic ROI')
            data=encoded.tobytes()
            if self.bytes_written+len(data)>self.byte_limit:continue
            name=f'failure-{len(self.failures):02d}.png'
            (self.folder/name).write_bytes(data)
            self.bytes_written+=len(data)
            self.saved[key]+=1
            self.failures.append({'image':name,'sha256':hashlib.sha256(data).hexdigest(),
                'group':group,'effect':effect,'sequence':update.sequence,
                'captured_at_ns':update.captured_at_ns,'frame_shape':list(update.frame.shape),
                'capture_status':getattr(update,'status',None),
                'bbox':detection.get('bbox'),'time':timer,'display_time':detection.get('display_time'),
                'temporal':detection.get('temporal')})

    def report(self):
        return {'counts':dict(self.counts),'reasons':dict(self.reasons),
                'failures':self.failures,'bytes_written':self.bytes_written,
                'limit':self.limit,'byte_limit':self.byte_limit}
