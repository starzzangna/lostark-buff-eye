"""Fast, identity-agnostic buff icon slot proposals.

The proposer only locates square icon frames.  It deliberately does not inspect
effect labels, time text, or stack text, so its output can be fed to a separate
classifier without leaking ground-truth identities into candidate generation.
"""
from dataclasses import dataclass

import cv2
import numpy as np


BUFF_ICON_SIZES = {80: 30, 90: 33, 100: 37, 110: 41, 120: 45}
# Distance from the bottom of the standard self-buff ROI to the bottom of the
# icon row.  This is UI layout geometry, not learned effect data.
ROW_BOTTOM_MARGIN = {80: 21, 90: 23, 100: 25, 110: 26, 120: 27}
SLOT_STRIDES = {80: 31, 90: 35, 100: 39, 110: 43, 120: 46}


@dataclass(frozen=True)
class SlotProposal:
    """An absolute image-space square proposed for icon classification."""

    bbox: list[int]
    score: float
    border_score: float
    square_score: float
    interior_score: float

    def as_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "score": self.score,
            "border_score": self.border_score,
            "square_score": self.square_score,
            "interior_score": self.interior_score,
        }


def _rect_sums(integral: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return every height-by-width sum, indexed by its top-left pixel."""
    return (integral[height:, width:] - integral[:-height, width:]
            - integral[height:, :-width] + integral[:-height, :-width])


def _intersection_over_union(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    return overlap / (a[2] * a[3] + b[2] * b[3] - overlap)


class SlotProposer:
    """Locate colored square buff frames inside a configured screen region."""

    def __init__(self, icon_sizes: dict[int, int] | None = None):
        self.icon_sizes = dict(icon_sizes or BUFF_ICON_SIZES)

    @staticmethod
    def _border_mask(roi: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        # Game buff frames are green through yellow.  Saturation rejects the
        # thin black outer frame, white expiry sheen, and gray time shadows.
        colored = (((hue >= 15) & (hue <= 100) & (saturation >= 70) & (value >= 60)) |
                   ((hue >= 10) & (hue <= 45) & (saturation >= 105) & (value >= 48)))
        return colored.astype(np.uint8)

    @staticmethod
    def _score_size(roi: np.ndarray, mask: np.ndarray, size: int, expected_y: int | None):
        height, width = mask.shape
        if size > height or size > width:
            return []
        band = max(2, round(size / 16))
        integral = cv2.integral(mask, sdepth=cv2.CV_32S)
        top = _rect_sums(integral, size, band)[:height-size+1]
        bottom = _rect_sums(integral, size, band)[size-band:height-band+1]
        left = _rect_sums(integral, band, size)[:, :width-size+1]
        right = _rect_sums(integral, band, size)[:, size-band:width-band+1]
        sides = np.stack((top, bottom, left, right), axis=0).astype(np.float32)
        sides /= float(size * band)

        # One side may be partly covered by the stack marker or green expiry
        # arrow.  Reward the three strongest sides, while retaining a smaller
        # all-sides term to reject isolated horizontal text strokes.
        ordered = np.sort(sides, axis=0)
        strong_three = ordered[1:].mean(axis=0)
        all_four = sides.mean(axis=0)
        weakest = ordered[0]
        border = 0.58 * strong_three + 0.30 * all_four + 0.12 * weakest

        # A real slot has comparable horizontal and vertical evidence.  This
        # suppresses time glyphs, stack text, and long scenery edges.
        horizontal = (sides[0] + sides[1]) * 0.5
        vertical = (sides[2] + sides[3]) * 0.5
        square = 1.0 - np.abs(horizontal - vertical) / np.maximum(
            np.maximum(horizontal, vertical), 0.08)
        square = np.clip(square, 0.0, 1.0)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gradient_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        edges = ((np.abs(gradient_x) + np.abs(gradient_y)) >= 100).astype(np.uint8)
        edge_integral = cv2.integral(edges, sdepth=cv2.CV_32S)
        inset = max(3, round(size * 0.16))
        inner_size = size - 2 * inset
        inner = _rect_sums(edge_integral, inner_size, inner_size)[
            inset:inset+height-size+1, inset:inset+width-size+1]
        interior = np.clip(inner.astype(np.float32) / (inner_size * inner_size * 0.60), 0, 1)

        score = 0.72 * border + 0.16 * square + 0.12 * interior
        # Thresholds are intentionally candidate-oriented: the downstream
        # classifier is responsible for identity and final rejection.
        # The 120% UI sample is deliberately dim and expiring 0.x icons can
        # be washed toward white.  Keep this permissive enough to preserve
        # recall; row geometry and downstream classification bound the noise.
        valid = ((score >= 0.34) & (border >= 0.19) &
                 (strong_three >= 0.21) & (square >= 0.24))
        if expected_y is None:
            peaks = cv2.dilate(score, np.ones((max(3, size // 2), max(3, size // 2)), np.uint8))
            ys, xs = np.nonzero(valid & (score == peaks))
            return [(int(x), int(y), float(score[y, x]), float(border[y, x]),
                     float(square[y, x]), float(interior[y, x])) for y, x in zip(ys, xs)]
        # Collapse the calibrated row band to a 1-D horizontal signal first.
        # Bright stack arrows can pull a 2-D maximum upward; choosing the best
        # y inside the band keeps the icon bbox anchored while tolerating JPEG
        # ringing and the one-pixel layout rounding seen between UI scales.
        y0 = max(0, expected_y - 5)
        y1 = min(score.shape[0], expected_y + 6)
        row_scores = score[y0:y1]
        best_offsets = np.argmax(row_scores, axis=0)
        xs_all = np.arange(score.shape[1])
        ys_all = y0 + best_offsets
        best_scores = score[ys_all, xs_all]
        best_valid = valid[ys_all, xs_all]
        # Wide horizontal suppression yields one proposal per icon-sized cell,
        # instead of several shifted boxes over the same colorful icon.
        dilated = cv2.dilate(best_scores[None, :],
                             np.ones((1, max(3, size // 2)), np.uint8))[0]
        xs = np.nonzero(best_valid & (best_scores == dilated))[0]
        ys = ys_all[xs]
        return [(int(x), int(y), float(score[y, x]), float(border[y, x]),
                 float(square[y, x]), float(interior[y, x])) for y, x in zip(ys, xs)]

    def propose(self, frame: np.ndarray, region_bbox: list[int], scale: int | None = None,
                max_candidates: int = 17, *, calibrated_row: bool = True,
                row_y: int | None = None) -> list[SlotProposal]:
        """Return non-overlapping absolute bboxes ordered by proposal score.

        ``scale`` limits the scan to its native icon size.  Passing ``None``
        scans all five supported sizes, which is useful during calibration.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image")
        if len(region_bbox) != 4:
            raise ValueError("region_bbox must be [x, y, width, height]")
        x0, y0, width, height = region_bbox
        if (min(x0, y0) < 0 or min(width, height) <= 0 or
                x0 + width > frame.shape[1] or y0 + height > frame.shape[0]):
            raise ValueError("region_bbox lies outside frame")
        if scale is not None and scale not in self.icon_sizes:
            raise ValueError(f"unsupported buff scale: {scale}")

        roi = frame[y0:y0+height, x0:x0+width]
        mask = self._border_mask(roi)
        scale_size_pairs = ([(scale, self.icon_sizes[scale])] if scale is not None else
                            sorted(self.icon_sizes.items()))
        raw = []
        for candidate_scale, size in scale_size_pairs:
            expected_y = height - ROW_BOTTOM_MARGIN[candidate_scale] - size if calibrated_row else None
            if row_y is not None:
                expected_y = row_y
            for x, y, score, border, square, interior in self._score_size(
                    roi, mask, size, expected_y):
                raw.append(SlotProposal([x0+x, y0+y, size, size], score, border, square, interior))
        raw.sort(key=lambda item: item.score, reverse=True)

        accepted = []
        for candidate in raw:
            if any((_intersection_over_union(candidate.bbox, other.bbox) > 0.25 or
                    ((calibrated_row or abs(candidate.bbox[1] - other.bbox[1]) < candidate.bbox[3] * .5) and
                     abs(candidate.bbox[0] + candidate.bbox[2] / 2 -
                         (other.bbox[0] + other.bbox[2] / 2)) <
                     min(candidate.bbox[2], other.bbox[2]) * 0.70))
                   for other in accepted):
                continue
            accepted.append(candidate)

        # A white expiry sheen can erase most of one colored frame while the
        # two neighbours remain clear.  Fill only a single missing cell between
        # two proposals separated by approximately two native strides.  This
        # uses fixed UI geometry and never effect labels or evaluation truth.
        if scale is not None and len(accepted) >= 2:
            stride = SLOT_STRIDES[scale]
            by_x = sorted(accepted, key=lambda item: item.bbox[0])
            additions = []
            for left, right in zip(by_x, by_x[1:]):
                if not calibrated_row and abs(left.bbox[1] - right.bbox[1]) > 5:
                    continue
                gap = right.bbox[0] - left.bbox[0]
                if abs(gap - 2 * stride) > max(4, round(stride * 0.16)):
                    continue
                size = self.icon_sizes[scale]
                x = round((left.bbox[0] + right.bbox[0]) / 2)
                y = (y0 + height - ROW_BOTTOM_MARGIN[scale] - size if calibrated_row
                     else round((left.bbox[1] + right.bbox[1]) / 2))
                if row_y is not None:
                    y = y0 + row_y
                box = [x, y, size, size]
                if any(_intersection_over_union(box, item.bbox) > 0.25 for item in accepted):
                    continue
                confidence = min(left.score, right.score) * 0.88
                additions.append(SlotProposal(
                    box, confidence, min(left.border_score, right.border_score),
                    min(left.square_score, right.square_score),
                    min(left.interior_score, right.interior_score)))
            accepted.extend(additions)
        accepted.sort(key=lambda item: item.score, reverse=True)
        return accepted[:max_candidates]
