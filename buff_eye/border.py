"""Sample colored frame pixels, excluding the icon artwork and outer scene."""
import cv2
import numpy as np


def border_color(frame, bbox):
    x, y, w, h = bbox
    if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0] or min(w,h) < 8:
        return None
    crop = frame[y:y+h,x:x+w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    band = max(2, round(min(w,h)*0.09))
    ring = np.ones((h,w), dtype=bool)
    ring[band:-band,band:-band] = False
    hue, sat, val = cv2.split(hsv)
    vivid = ring & (sat >= 85) & (val >= 90)
    colors = [vivid & ((hue <= 10) | (hue >= 170)),
              vivid & (hue >= 16) & (hue <= 38),
              vivid & (hue >= 39) & (hue <= 90)]
    selected = max(colors, key=np.count_nonzero)
    if np.count_nonzero(selected) < max(8, round((w+h)*0.15)):
        return None
    # Keep the source shade instead of imposing a single hard-coded yellow.
    b,g,r = np.median(crop[selected],axis=0)
    return [int(r),int(g),int(b)]
