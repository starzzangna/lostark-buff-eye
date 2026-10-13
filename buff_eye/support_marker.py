"""Recognize the green merge emphasis above a common attack icon."""
import cv2
import numpy as np


class AttackPairResolver:
    """Resolve a pair after all search paths, with a short fixed-position hold."""
    def __init__(self):
        self.verified = None

    def resolve(self, detections, now):
        red = [d for d in detections if d['effect_id'] in
               ('support.attack.01', 'support.attack.additional')]
        if len(red) != 2:
            self.verified = None
            return detections
        marked = [d for d in red if d.get('merged_emphasis') is True]
        plain = [d for d in red if d.get('merged_emphasis') is False]
        if len(marked) == len(plain) == 1:
            marked[0]['effect_id'] = 'support.attack.01'
            plain[0]['effect_id'] = 'support.attack.additional'
            self.verified = (now, tuple(marked[0]['bbox']), tuple(plain[0]['bbox']))
        elif self.verified is not None:
            stamp, first, second = self.verified
            boxes = {tuple(d['bbox']): d for d in red}
            if (0 <= now-stamp <= .35 and not marked
                    and first in boxes and second in boxes and len(boxes)==2):
                boxes[first]['effect_id'] = 'support.attack.01'
                boxes[second]['effect_id'] = 'support.attack.additional'
            else:
                self.verified = None
        return detections


def green_mask(image):
    b, g, r = cv2.split(image.astype(np.float32))
    return ((g > 150) & (g > r * 1.15) & (g > b * 1.10)).astype(np.uint8)


def read_marker(frame, bbox, reference):
    x, y, w, h = bbox
    height = round(24 * w / 37)
    pad = 2
    if y-height-pad < 0 or x-pad < 0 or x+w+pad > frame.shape[1]:
        return None
    expected = cv2.resize(green_mask(reference), (w, height), interpolation=cv2.INTER_NEAREST)
    actual = green_mask(frame[y-height-pad:y+pad, x-pad:x+w+pad])
    if expected.sum() < 10 or actual.sum() < 10:
        return False
    score = cv2.matchTemplate(actual.astype(np.float32), expected.astype(np.float32), cv2.TM_CCORR_NORMED)
    return bool(np.nanmax(score) >= 0.85)
