"""Bounded multi-frame evidence for an unclassified, whitening icon slot.

No predicted pixels and no prior identity: every vote compares visible pixels
to shipped assets. OCR always runs on the current original frame afterwards.
"""
from collections import deque
import cv2
import numpy as np
from .core import iou
from .slots import SlotProposer


def correlation(a, b):
    a=a.astype(np.float32); b=b.astype(np.float32)
    a-=a.mean(axis=0); b-=b.mean(axis=0)
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float((a*b).sum()/denominator) if denominator>1 else -1.


class TemporalIconEvidence:
    max_slots=17
    max_frames=5
    max_age=.5

    def __init__(self):
        self.reset()

    def reset(self):
        self.pending={}
        self.context=None
        self.last_scan=None

    def detect(self, frame, observed, now, search, sizes):
        context=(frame.shape, search.size)
        if context!=self.context:
            self.reset(); self.context=context
        if self.last_scan is not None and 0<=now-self.last_scan<.08:
            return []
        self.last_scan=now
        choices=(search.size,) if search.size is not None else tuple(sorted(set(sizes)))
        proposer=SlotProposer(dict(enumerate(choices)))
        h,w=frame.shape[:2]
        boxes=[p.bbox for p in proposer.propose(frame,[0,0,w,h],calibrated_row=False,
                                               max_candidates=self.max_slots)]
        return self.update(frame, boxes, observed, now, search.templates)

    def update(self, frame, boxes, observed, now, templates):
        fresh={}; recovered=[]
        for box in boxes[:self.max_slots]:
            if any(iou(box,d['bbox'])>.3 for d in observed):continue
            x,y,w,h=box
            if w!=h or w<16 or min(x,y)<0 or x+w>frame.shape[1] or y+h>frame.shape[0]:continue
            crop=frame[y+4:y+h-4,x+4:x+w-4].copy()
            hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
            keep=~((hsv[:,:,2]>220)&(hsv[:,:,1]<100))
            if not .2<=1-float(keep.mean())<=.7 or keep.sum()<48:continue
            ranks=sorted(((correlation(crop[keep],template[keep]),effect)
                          for (effect,size),template in templates.items()
                          if size==w and template.shape==crop.shape),reverse=True)
            if not ranks or ranks[0][0]<.90:continue
            if len(ranks)>1 and ranks[0][0]-ranks[1][0]<.06:continue
            score,effect=ranks[0]
            key=tuple(box)
            history=deque((entry for entry in self.pending.get(key,())
                           if 0<=now-entry[0]<=self.max_age and entry[1]==effect),maxlen=self.max_frames)
            # Replaying the same cached frame is not additional evidence.
            if history and np.array_equal(crop,history[-1][2]):
                fresh[key]=history
                continue
            history.append((now,effect,crop,keep))
            fresh[key]=history
            if len(history)<3:continue
            masks=np.stack([entry[3] for entry in history])
            coverage=masks.sum(axis=0)
            trusted=coverage>=2
            if float(trusted.mean())<.8:continue
            if float(np.any(masks!=masks[0],axis=0).mean())<.15:continue
            total=sum(entry[2].astype(np.float32)*entry[3][:,:,None] for entry in history)
            combined=total/np.maximum(coverage,1)[:,:,None]
            scores=sorted(((correlation(combined[trusted],template[trusted]),candidate)
                           for (candidate,size),template in templates.items()
                           if size==w and template.shape==crop.shape),reverse=True)
            if scores[0][1]!=effect or scores[0][0]<.94:continue
            if len(scores)>1 and scores[0][0]-scores[1][0]<.06:continue
            recovered.append(dict(effect_id=effect,region_id='self_buff',bbox=list(box),
                icon_score=scores[0][0],variant_id=f'{effect}-{w}',
                score_method='multi_frame_visible_asset',evidence_frames=len(history)))
        self.pending=fresh
        return recovered
