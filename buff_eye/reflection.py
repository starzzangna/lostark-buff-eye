"""Short, pixel-verified bridge over whitening animation; never a cold detector."""
import cv2
import numpy as np
from .core import iou


class ReflectionBridge:
    def __init__(self):
        self.reset()

    def reset(self):
        self.references=[]
        self.shape=None

    def update(self,frame,observed,now):
        if self.shape!=frame.shape:
            self.reset(); self.shape=frame.shape
        recovered=[]
        fresh=[]
        for detection in observed:
            x,y,w,h=detection['bbox']
            fresh.append((now,dict(detection),frame[y:y+h,x:x+w].copy()))
        for timestamp,detection,reference in self.references:
            if now-timestamp>.35:continue
            if any(iou(detection['bbox'],d['bbox'])>.3 for d in observed):continue
            x,y,w,h=detection['bbox']
            current=frame[y:y+h,x:x+w]
            # Ignore borders and text. Only whitening with reduced saturation
            # may hide pixels; ordinary scene changes cannot mask themselves.
            a=reference[3:-3,3:-3]; b=current[3:-3,3:-3]
            if a.shape!=b.shape or min(a.shape[:2])<8:continue
            ah=cv2.cvtColor(a,cv2.COLOR_BGR2HSV).astype(np.int16)
            bh=cv2.cvtColor(b,cv2.COLOR_BGR2HSV).astype(np.int16)
            glare=(bh[:,:,2]-ah[:,:,2]>=35)&(bh[:,:,1]<ah[:,:,1])
            fraction=float(glare.mean())
            if not .05<=fraction<=.65:continue
            keep=~glare
            if fraction>.4:
                # Stronger glare is allowed only with substantial original
                # texture remaining in multiple parts of the icon.
                if keep.sum()<80:continue
                hh,ww=keep.shape
                quadrants=(keep[:hh//2,:ww//2],keep[:hh//2,ww//2:],
                           keep[hh//2:,:ww//2],keep[hh//2:,ww//2:])
                if sum(int(part.sum())>=12 for part in quadrants)<2:continue
            av=a[keep].astype(float); bv=b[keep].astype(float)
            if np.abs(av-bv).mean()>15:continue
            av-=av.mean(axis=0);bv-=bv.mean(axis=0)
            denominator=float(np.linalg.norm(av)*np.linalg.norm(bv))
            score=float((av*bv).sum()/denominator) if denominator else 0
            if score<(.97 if fraction>.4 else .94):continue
            item={key:value for key,value in detection.items() if key not in ('time','stack')}
            item.update(icon_score=score,score_method='recent_icon_unoccluded_pixels')
            recovered.append(item)
            # Recovery never refreshes the clean reference or its expiry.
            fresh.append((timestamp,detection,reference))
        self.references=fresh[:17]
        return recovered
