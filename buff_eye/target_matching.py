"""Shared color statistics for full-size target searches, without frame skipping."""
import cv2
import numpy as np


class ColorSearchFrame:
    def __init__(self, frame):
        self.frame = np.ascontiguousarray(frame)
        self.interleaved = self.frame.reshape(frame.shape[0], -1)
        self.sums, self.squares = cv2.integral2(self.frame, sdepth=cv2.CV_64F)
        self.square_sum = self.squares.sum(axis=2)
        self.gray_sum_integral = self.sums.sum(axis=2)/3
        self.gray = self.frame.astype(np.float32).mean(axis=2)
        self.gray_squares = cv2.integral(self.gray.astype(np.float64)**2)
        self.windows = {}

    def statistics(self, h, w):
        key = (h, w)
        if key not in self.windows:
            self.windows.clear()
            def area(integral):
                return integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
            sums = area(self.sums)
            variance = np.maximum(0, area(self.square_sum) - np.einsum('ijk,ijk->ij',sums,sums)/(h*w))
            gray_sum = area(self.gray_sum_integral)
            gray_variance = np.maximum(0, area(self.gray_squares) - gray_sum**2/(h*w))
            self.windows[key] = sums, np.sqrt(variance), gray_sum, np.sqrt(np.maximum(0, variance-3*gray_variance))
        return self.windows[key]

    def could_match(self, template, threshold=.859):
        return bool(np.any(self.possible_positions(template,threshold)))

    def possible_positions(self, template, threshold=.859):
        """Cauchy-Schwarz upper bound on full color correlation, not gray OCR."""
        h,w=template.shape[:2]
        sums,norm,gray_sum,residual_norm=self.statistics(h,w)
        values=template.astype(np.float64)
        centered=values-values.mean(axis=(0,1))
        template_norm=np.sqrt((centered**2).sum())
        # OpenCV has special behavior for constant templates. Let the original
        # matcher decide instead of rejecting them with a zero denominator.
        if template_norm<=1e-9:return np.ones(norm.shape,dtype=bool)
        gray=template.astype(np.float32).mean(axis=2)
        gray_energy=((gray.astype(np.float64)-gray.mean(dtype=np.float64))**2).sum()*3
        residual=np.sqrt(max(0,template_norm**2-gray_energy))
        cross=cv2.matchTemplate(self.gray,gray,cv2.TM_CCORR)
        numerator=3*(cross-gray_sum*gray.mean(dtype=np.float64))+residual_norm*residual
        # Compare before division. Constant windows carry no color evidence.
        denominator=norm*template_norm
        return (denominator>1e-9) & (numerator>=threshold*denominator)

    def scores(self, template):
        h, w = template.shape[:2]
        sums, norm, _, _ = self.statistics(h,w)
        values = template.astype(np.float64)
        mean = values.mean(axis=(0, 1))
        template_norm = np.sqrt(((values-mean)**2).sum())
        # Every third column aligns B/G/R with B/G/R, giving the same color
        # dot product as three-channel correlation with one transform.
        cross = cv2.matchTemplate(self.interleaved, np.ascontiguousarray(template).reshape(h, -1), cv2.TM_CCORR)[:, ::3]
        numerator = cross.astype(np.float64) - np.einsum('ijk,k->ij', sums, mean)
        denominator = norm*template_norm
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator>1e-9)
