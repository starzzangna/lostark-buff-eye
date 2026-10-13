"""Native-pixel asset search shared by support and target status rows.

UI percentages are not capture pixel sizes. Search the measured pixels without
resampling the ROI (especially its small decimal points). No minimum row count.
"""
import time
import cv2
import numpy as np
from .core import iou
from .target_matching import ColorSearchFrame


class IconSearch:
    def __init__(self, assets, region_id, *, sizes=range(18, 55), max_instances=1, size_tolerance=0):
        self.templates = {}
        self.outer_templates = {}
        self.sizes = tuple(sizes)
        self.region_id = region_id
        self.max_instances = max_instances
        self.size_tolerance = size_tolerance
        self.cancel_check=lambda:False
        for key, image in assets.items():
            for size in self.sizes:
                resized = cv2.resize(image, (size-4, size-4), interpolation=cv2.INTER_AREA)
                self.templates[key, size] = resized[2:-2, 2:-2].copy()
                if size<20:self.outer_templates[key,size]=resized
        self.reset()

    def _verify_small(self,frame,key,size,bx,by):
        # Do not shrink to tiny cores and accept correlations with scenery.
        # 18/19px icons retain >=10px artwork, a second wider artwork check,
        # and three sides of an actual colored frame.
        if size<18:return False
        crop=frame[by:by+size,bx:bx+size]
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        hue,sat,val=cv2.split(hsv)
        color=((hue<=12)|(hue>=170)) if self.region_id=='target_debuff' else ((hue>=16)&(hue<=95))
        ink=color&(sat>=85)&(val>=90)
        edges=(ink[:2,2:-2],ink[-2:,2:-2],ink[2:-2,:2],ink[2:-2,-2:])
        if sum(float(e.mean())>=.25 for e in edges)<3:return False
        inner=crop[2:-2,2:-2]
        score=float(cv2.matchTemplate(inner,self.outer_templates[key,size],cv2.TM_CCOEFF_NORMED)[0,0])
        return bool(np.isfinite(score) and score>=.80)

    def reset(self):
        self.size = None
        self.probe_at = 0
        self.shape = None

    def _search(self, frame, sizes, broad):
        search = ColorSearchFrame(frame) if broad else None
        candidates = []
        for (key, size), template in sorted(self.templates.items(), key=lambda item:item[0][1]):
            if self.cancel_check():return []
            if size not in sizes or min(frame.shape[:2]) < size:
                continue
            if search is not None:
                if not search.could_match(template) or float(search.scores(template).max()) < .859:
                    continue
            scores = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            if self.cancel_check():return []
            scores = np.nan_to_num(scores, nan=-1, posinf=-1, neginf=-1)
            for _ in range(self.max_instances):
                _, score, _, (x, y) = cv2.minMaxLoc(scores)
                if score < .86:
                    break
                bx, by = x-4, y-4
                if bx >= 0 and by >= 0 and bx+size <= frame.shape[1] and by+size <= frame.shape[0]:
                    if size>=20 or self._verify_small(frame,key,size,bx,by):candidates.append(dict(effect_id=key, region_id=self.region_id,
                        icon_score=score, bbox=[bx, by, size, size],
                        variant_id=f'{key}-{size}', score_method='asset_color_correlation'))
                scores[max(0,y-size//2):y+size//2+1, max(0,x-size//2):x+size//2+1] = -1
        accepted = []
        for d in sorted(candidates, key=lambda d:d['icon_score'], reverse=True):
            if any(iou(d['bbox'], other['bbox']) > .3 for other in accepted):
                continue
            if any(other['effect_id'] != d['effect_id'] and iou(d['bbox'], other['bbox']) > .3
                   and d['icon_score']-other['icon_score'] < .03 for other in candidates):
                continue
            if sum(other['effect_id'] == d['effect_id'] for other in accepted) >= self.max_instances:
                continue
            accepted.append(d)
        return accepted

    def detect(self, frame, now=None, size_hint=None):
        now = time.monotonic() if now is None else now
        if self.shape != frame.shape:
            self.reset()
            self.shape = frame.shape
        if size_hint in self.sizes:
            self.size = size_hint
        broad = size_hint is None and (self.size is None or now >= self.probe_at)
        local_sizes = tuple(s for s in self.sizes if self.size is not None
                            and abs(s-self.size) <= self.size_tolerance)
        found = self._search(frame, self.sizes if broad else local_sizes, broad)
        if not found and not broad and size_hint is None:
            found = self._search(frame, self.sizes, True)
            broad = True
        if found:
            self.size = max(found, key=lambda d:d['icon_score'])['bbox'][2]
            if broad:
                self.probe_at = now+10
        return found
