from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from .utils import delta_e

# ── Thresholds ───────────────────────────────────────────────────────────────
BORDER_DELTA_E_LOW  = 20.0   # ΔE below this → we are in the border
DESIGN_DELTA_E_HIGH = 30.0   # ΔE above this → design has begun
SCAN_SMOOTH_K       = 7      # window for smoothing the ΔE profile
SCAN_COL_FRAC       = (0.25, 0.75)  # fraction of card width used for top/bottom scan columns
SCAN_ROW_FRAC       = (0.25, 0.75)  # fraction of card height used for left/right scan rows
MIN_BORDER_PX       = 15     # minimum credible border width returned

# PSA centering grade ceilings: (max dominant-side %, grade)
CENTERING_THRESHOLDS_FRONT: List[Tuple[float, int]] = [
    (55.0, 10),
    (60.0,  9),
    (65.0,  8),
    (70.0,  7),
    (80.0,  6),
    (85.0,  5),
    (90.0,  4),
]

CENTERING_THRESHOLDS_BACK: List[Tuple[float, int]] = [
    (75.0, 10),
    (90.0,  9),  # PSA back centering is flat 90/10 for grades 9 down through 1-4
                 # (grading-schema.md) -- centering isn't the limiting pillar below
                 # 90/10 either way, so the ceiling here just needs to not be lower
                 # than the best non-10 grade.
]


def measure_centering(result: Dict) -> Dict:
    """
    Measure border widths and compute L/R and T/B ratios with a PSA grade ceiling.

    Full-art fronts are unmeasurable and return measurable=False with no grade penalty.
    """
    face = result['face']
    clf  = result.get('classification') or {}

    if face == 'front' and clf.get('type') == 'full_art':
        return {
            'left_px': None, 'right_px': None,
            'top_px':  None, 'bottom_px': None,
            'lr_ratio': None, 'tb_ratio': None,
            'centering_grade': 10,
            'measurable': False,
        }

    warped = result['warped']
    ref    = result['rois']['edge_reference_colors']
    h, w   = warped.shape[:2]

    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)

    col_lo = int(w * SCAN_COL_FRAC[0])
    col_hi = int(w * SCAN_COL_FRAC[1])
    row_lo = int(h * SCAN_ROW_FRAC[0])
    row_hi = int(h * SCAN_ROW_FRAC[1])

    top_px    = _scan_border(lab[:h // 2,    col_lo:col_hi],          ref['top'],    axis='row')
    bottom_px = _scan_border(lab[h // 2:,    col_lo:col_hi][::-1],    ref['bottom'], axis='row')
    left_px   = _scan_border(lab[row_lo:row_hi, :w // 2],             ref['left'],   axis='col')
    right_px  = _scan_border(lab[row_lo:row_hi, w // 2:][:, ::-1],    ref['right'],  axis='col')

    lr_ratio = _ratio(left_px, right_px)
    tb_ratio = _ratio(top_px, bottom_px)

    worst = max(max(lr_ratio), max(tb_ratio))
    thresholds = CENTERING_THRESHOLDS_BACK if face == 'back' else CENTERING_THRESHOLDS_FRONT

    return {
        'left_px':        left_px,
        'right_px':       right_px,
        'top_px':         top_px,
        'bottom_px':      bottom_px,
        'lr_ratio':       lr_ratio,
        'tb_ratio':       tb_ratio,
        'centering_grade': _centering_grade(worst, thresholds),
        'measurable':     True,
    }


# ── Internal helpers ─────────────────────────────────────────────────────────

def _scan_border(region: np.ndarray, ref_lab: np.ndarray, axis: str) -> int:
    """
    Scan a LAB region inward from the card edge to find where the border ends.
    region must be oriented so that axis-0 runs from card edge toward center.
    Returns border width in pixels from the card edge.
    """
    if axis == 'col':
        region = region.transpose(1, 0, 2)  # (cols, rows, 3) → scan cols as rows

    n = region.shape[0]
    # Per-position median across the perpendicular dimension
    profile = np.median(region, axis=1)  # (n, 3)

    de = delta_e(profile, ref_lab)  # (n,)

    k = SCAN_SMOOTH_K
    smoothed = np.convolve(de, np.ones(k) / k, mode='same')

    in_border = False
    for i, val in enumerate(smoothed):
        if not in_border:
            if val < BORDER_DELTA_E_LOW:
                in_border = True
        else:
            if val > DESIGN_DELTA_E_HIGH:
                return max(i, MIN_BORDER_PX)

    return max(n // 4, MIN_BORDER_PX)


def _ratio(a: int, b: int) -> Tuple[float, float]:
    total = a + b
    if total == 0:
        return (50.0, 50.0)
    return (round(a / total * 100, 1), round(b / total * 100, 1))


def _centering_grade(dominant_pct: float, thresholds: List[Tuple[float, int]]) -> int:
    for max_pct, grade in thresholds:
        if dominant_pct <= max_pct:
            return grade
    return 1
