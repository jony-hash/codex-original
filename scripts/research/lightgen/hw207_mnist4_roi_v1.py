"""Four-region intensity readout with explicitly supplied diagnostic geometry."""

import math

import numpy as np


def readout(intensity, rectangles, target=None):
    """Return region energies and decision; ties and zero energy are undecided.

    Rectangles are (row_start, row_end, col_start, col_end), exclusive ends,
    ordered top-left, top-right, bottom-left, bottom-right.
    """
    image = np.asarray(intensity)
    if image.ndim != 2 or image.dtype.kind not in "buif" or not np.isfinite(image).all() or (image < 0).any():
        raise ValueError("intensity must be a finite nonnegative 2-D image")
    if len(rectangles) != 4:
        raise ValueError("exactly four rectangles required")
    mask = np.zeros(image.shape, dtype=bool)
    scores, areas, centers = [], [], []
    for rect in rectangles:
        if len(rect) != 4 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in rect):
            raise ValueError("rectangle coordinates must be integers")
        y0, y1, x0, x1 = rect
        if not (0 <= y0 < y1 <= image.shape[0] and 0 <= x0 < x1 <= image.shape[1]):
            raise ValueError("rectangle outside image or empty")
        if mask[y0:y1, x0:x1].any():
            raise ValueError("rectangles overlap")
        mask[y0:y1, x0:x1] = True
        scores.append(float(image[y0:y1, x0:x1].sum(dtype=np.float64)))
        areas.append((y1-y0)*(x1-x0))
        centers.append(((y0+y1)/2, (x0+x1)/2))
    if len(set(areas)) != 1 or not (
        centers[0][0] == centers[1][0] < centers[2][0] == centers[3][0]
        and centers[0][1] == centers[2][1] < centers[1][1] == centers[3][1]
    ):
        raise ValueError("rectangles must form an equal-area 2x2 grid")
    if target is not None and (isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer)) or target not in range(4)):
        raise ValueError("invalid target")
    ordered = sorted(scores, reverse=True)
    prediction = int(np.argmax(scores)) if ordered[0] > 0 and ordered[0] > ordered[1] else None
    total = float(image.sum(dtype=np.float64))
    if not math.isfinite(total):
        raise ValueError("integrated energy exceeds float64 range")
    return {
        "scores": scores, "prediction": prediction, "roi_total": sum(scores),
        "outside_energy": float(image[~mask].sum(dtype=np.float64)),
        "correct_fraction": scores[target]/total if target is not None and total > 0 else None,
        "strongest_wrong": max(s for i, s in enumerate(scores) if i != target) if target is not None else None,
        "top_margin": ordered[0]-ordered[1],
    }
