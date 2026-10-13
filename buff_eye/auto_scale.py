"""Infer supported native icon sizes from actual template matches in a ROI."""
import time


class AutoScale:
    def __init__(self):
        self.context = None
        self.scale = None
        self.next_probe = 0

    def resolve(self, detector, frame, context, now=None):
        now = time.monotonic() if now is None else now
        if context != self.context:
            self.context, self.scale, self.next_probe = context, None, 0
        if now < self.next_probe:
            return self.scale
        scores = []
        height, width = frame.shape[:2]
        for scale in (80, 90, 100, 110, 120):
            matches = detector.detect(frame, scale, search_mode='native', region_bbox=[0,0,width,height])
            values = sorted((d['icon_score'] for d in matches), reverse=True)[:3]
            if values:
                scores.append((sum(values)/len(values), len(values), scale))
        scores.sort(reverse=True)
        if scores:
            score, count, candidate = scores[0]
            margin = score-scores[1][0] if len(scores)>1 else 1
            if score >= .90 and margin >= .025 and (count >= 2 or score >= .98):
                self.scale = candidate
        self.next_probe = now + (10 if self.scale is not None else 2)
        return self.scale
