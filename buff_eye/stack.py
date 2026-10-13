"""Conservative OCR for the green ``xN`` buff stack marker.

Stack state is intentionally independent from remaining-time OCR.  A missing,
unsupported, or low-confidence marker always returns ``unreadable``; it is
never converted to zero or one.
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np


STACK_LAYOUT = {
    80: (-8, -19, 16, 19),
    90: (-8, -20, 16, 20),
    100: (-10, -22, 20, 22),
    110: (-11, -23, 22, 23),
    120: (-12, -25, 24, 25),
}


def foreground(crop):
    if crop is None or not crop.size:
        return np.zeros((0, 0), np.uint8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    color = cv2.inRange(hsv, np.array([35, 35, 95]), np.array([100, 255, 255]))
    # Remove isolated scenery speckles without eroding the 80% glyph strokes.
    count = cv2.boxFilter((color > 0).astype(np.uint8), -1, (3, 3), normalize=False)
    return np.where((color > 0) & (count >= 2), 255, 0).astype(np.uint8)


def segments(mask):
    if not mask.size:
        return []
    columns = np.any(mask > 0, axis=0)
    edges = np.diff(np.pad(columns.astype(np.int8), (1, 1)))
    result = []
    for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        part = mask[:, left:right]
        ys = np.flatnonzero(np.any(part > 0, axis=1))
        if len(ys) < 4 or np.count_nonzero(part) < 7:
            continue
        top, bottom = int(ys[0]), int(ys[-1]) + 1
        result.append(([int(left), top, int(right-left), bottom-top], part[top:bottom].copy()))
    return result


def normalize(glyph):
    height, width = glyph.shape
    target_width = max(1, min(28, round(width * 20 / height)))
    resized = cv2.resize(glyph, (target_width, 20), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((24, 32), np.uint8)
    left = (32 - target_width) // 2
    canvas[2:22, left:left+target_width] = resized
    return canvas


def stack_bbox(icon_bbox, scale):
    if scale not in STACK_LAYOUT:
        raise ValueError(f"unsupported buff scale: {scale}")
    x, y, width, _ = icon_bbox
    dx, dy, extra_width, height = STACK_LAYOUT[scale]
    return [x + dx, y + dy, width + extra_width, height]


class StackReader:
    def __init__(self, bank_path, exclude_asset=None):
        bank = json.loads(Path(bank_path).read_text(encoding="utf-8-sig"))
        if bank.get("schema_version") != 1:
            raise ValueError("Unsupported stack glyph bank schema")
        self.threshold = float(bank.get("threshold", 0.76))
        self.margin = float(bank.get("margin", 0.035))
        self.glyphs = [(item["character"], np.array(item["bitmap"], np.uint8))
                       for item in bank.get("glyphs", [])
                       if item.get("source_asset_id") != exclude_asset]
        self.coverage = {character for character, _ in self.glyphs}

    def _rank(self, glyph, allowed):
        candidate = normalize(glyph) > 0
        scores = {}
        for character, template in self.glyphs:
            if character not in allowed:
                continue
            best = 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    shifted = cv2.warpAffine(template, np.float32([[1, 0, dx], [0, 1, dy]]),
                                             (32, 24), flags=cv2.INTER_NEAREST) > 0
                    denominator = np.count_nonzero(candidate) + np.count_nonzero(shifted)
                    score = 2 * np.count_nonzero(candidate & shifted) / denominator if denominator else 0
                    best = max(best, score)
            scores[character] = max(scores.get(character, 0), best)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def read(self, crop):
        result = {"status": "unreadable", "raw_text": None, "value": None,
                  "unit": "stack", "score": None, "source": "observed",
                  "method": "calibrated_stack_glyphs", "reason": "segmentation",
                  "characters": []}
        pieces = segments(foreground(crop))
        if crop is not None and crop.size:
            height, width = crop.shape[:2]
            # Runtime crops are wider than the marker so x10 is not clipped.
            # Green expiry arrows or adjacent borders entering at either crop
            # edge are context, not stack glyphs.  Tall top-touching scenery is
            # rejected for the same reason.
            pieces = [(box, glyph) for box, glyph in pieces
                      if box[0] >= 4 and box[0] + box[2] <= width - 4
                      and not (box[1] <= 1 and box[3] >= height * 0.75)]
        if len(pieces) not in (2, 3):
            return result
        text = []
        for index, (box, glyph) in enumerate(pieces):
            allowed = {"x"} if index == 0 else set("0123456789")
            available = allowed & self.coverage
            if not available:
                result["reason"] = "insufficient_glyph_coverage"
                return result
            ranked = self._rank(glyph, available)
            character, score = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
            margin = score - runner_up
            result["characters"].append({"bbox": box, "candidate": character,
                                         "score": score, "margin": margin})
            if score < self.threshold or margin < self.margin:
                result["reason"] = "low_confidence"
                return result
            text.append(character)
        raw = "".join(text)
        if not re.fullmatch(r"x[0-9]{1,2}", raw):
            result["reason"] = "unsupported_format"
            return result
        value = int(raw[1:])
        result.update(status="read", raw_text=raw, value=value,
                      score=min(item["score"] for item in result["characters"]), reason=None)
        return result

    def read_region(self, frame, bbox):
        x, y, width, height = bbox
        if (min(x, y) < 0 or min(width, height) <= 0 or
                x + width > frame.shape[1] or y + height > frame.shape[0]):
            result = self.read(None)
            result["reason"] = "crop_outside_frame"
            return result
        return self.read(frame[y:y+height, x:x+width])

    def read_for_icon(self, frame, icon_bbox, scale):
        return self.read_region(frame, stack_bbox(icon_bbox, scale))
