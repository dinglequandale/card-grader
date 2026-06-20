"""
Perimeter pillar: edges + corners, ONE reference-free detector
(phase3.edge_inspect) on the temporal median, plus a border-band pass of
the detect-then-aggregate saliency for edge scratch/chipping/print (which
edge_inspect's whitening-only column statistics don't catch -- see
handoff_pipeline_revamp.md section 4).

Corners are NOT a separate detector: they're the adjacent strip-ends (outer
CORNER_END_PCT of each edge) of the same 4 edge strips -- a defect run that
touches a strip's start/end contributes to the corner at that end.

The median (not a single frame) is used because it's glare-free and the
edge inspector's column statistics are sensitive to specular noise.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import List

from phase3 import edge_inspect
from video.temporal import TemporalBank

CORNER_END_PCT = 12.0   # outer % of an edge's length treated as "corner-adjacent"

# corner each edge's (start, end) is adjacent to, per edge_inspect.rectify_edge_strip
# (top: TL->TR, bottom: BL->BR, left: TL->BL, right: TR->BR)
_EDGE_CORNERS = {
    "top":    ("TL", "TR"),
    "bottom": ("BL", "BR"),
    "left":   ("TL", "BL"),
    "right":  ("TR", "BR"),
}

# EdgeDefect.peak_sigma (MAD-multiples) -> severity 1-5
_SEVERITY_BUCKETS = [(4.0, 1), (6.0, 2), (9.0, 3), (13.0, 4)]  # else 5


def _severity_from_sigma(sigma: float) -> int:
    for cap, sev in _SEVERITY_BUCKETS:
        if sigma <= cap:
            return sev
    return 5


def _canonical_corners(size: tuple) -> np.ndarray:
    W, H = size
    return np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)


def _bbox_for_strip_run(side: str, start_pct: float, extent_pct: float,
                         size: tuple, band_frac: float = edge_inspect.STRIP_WIDTH_FRAC) -> list:
    """Approximate canonical-frame bbox for an edge_inspect defect run, for
    eval overlap matching against annotations (which are boxed in the full
    candidate frame, not the rectified strip)."""
    W, H = size
    band_w = int(W * band_frac)
    band_h = int(H * band_frac)
    if side == "top":
        x0, x1 = int(start_pct / 100 * W), int((start_pct + extent_pct) / 100 * W)
        return [x0, 0, max(x1 - x0, 1), band_h]
    if side == "bottom":
        x0, x1 = int(start_pct / 100 * W), int((start_pct + extent_pct) / 100 * W)
        return [x0, H - band_h, max(x1 - x0, 1), band_h]
    if side == "left":
        y0, y1 = int(start_pct / 100 * H), int((start_pct + extent_pct) / 100 * H)
        return [0, y0, band_w, max(y1 - y0, 1)]
    if side == "right":
        y0, y1 = int(start_pct / 100 * H), int((start_pct + extent_pct) / 100 * H)
        return [W - band_w, y0, band_w, max(y1 - y0, 1)]
    raise ValueError(side)


def detect_edge_defects(bank: TemporalBank, size: tuple) -> List[dict]:
    """Whitening/darkening/desaturation per edge, via edge_inspect (VALIDATED
    reference-free detector). Returns verdicts shaped for
    grade_synthesis.edges_grade_from_verdicts: {side, type, severity, ...}.
    """
    corners = _canonical_corners(size)
    results = edge_inspect.inspect_all_edges(bank.median, corners)
    verdicts = []
    for side, out in results.items():
        for d in out["defects"]:
            verdicts.append({
                "side": side,
                "type": d.type_hint,
                "severity": _severity_from_sigma(d.peak_sigma),
                "start_pct": d.start_pct,
                "extent_pct": d.extent_pct,
                "peak_sigma": d.peak_sigma,
                "bbox": _bbox_for_strip_run(side, d.start_pct, d.extent_pct, size),
            })
    return verdicts


def detect_corner_defects(edge_verdicts: List[dict]) -> List[dict]:
    """Promote edge-strip-end defects (outer CORNER_END_PCT) to corner verdicts.
    Shaped for grade_synthesis.corners_grade_from_verdicts: {side: 'corner_XX', ...}."""
    corner_verdicts = []
    for v in edge_verdicts:
        side = v["side"]
        if side not in _EDGE_CORNERS or "start_pct" not in v:
            continue
        start_corner, end_corner = _EDGE_CORNERS[side]
        s, e = v["start_pct"], v["start_pct"] + v["extent_pct"]
        if s < CORNER_END_PCT:
            corner_verdicts.append({**v, "side": f"corner_{start_corner}"})
        if e > 100 - CORNER_END_PCT:
            corner_verdicts.append({**v, "side": f"corner_{end_corner}"})
    return corner_verdicts


def detect_edge_surface_defects(bank: TemporalBank, percentile: float = 97.0,
                                 band_frac: float = 0.06) -> List[dict]:
    """Border-band pass of defect_saliency for edge scratch/chipping/print --
    defects edge_inspect's column-statistics whitening detector doesn't see.
    Thresholded against the BAND's own statistics (not the whole-card
    percentile from video/surface.py), since the border naturally has
    different saliency baseline than the interior.
    """
    sal = bank.defect_saliency
    H, W = sal.shape
    m = int(min(H, W) * band_frac)
    band_mask = np.zeros((H, W), dtype=bool)
    band_mask[:m, :] = band_mask[-m:, :] = True
    band_mask[:, :m] = band_mask[:, -m:] = True

    band_vals = sal[band_mask]
    if band_vals.size == 0:
        return []
    thr = float(np.percentile(band_vals, percentile))
    binm = ((sal > thr) & band_mask).astype(np.uint8) * 255
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(binm, 8)

    verdicts = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 6:
            continue
        side = _side_for_bbox(x, y, w, h, W, H, m)
        peak = float(sal[y:y + h, x:x + w].max())
        verdicts.append({
            "side": side,
            "type": "scratch",
            "severity": _severity_from_magnitude(peak),
            "bbox": [int(x), int(y), int(w), int(h)],
        })
    return verdicts


def _severity_from_magnitude(peak: float) -> int:
    from video.surface import _severity_from_magnitude as _f
    return _f(peak)


def _side_for_bbox(x, y, w, h, W, H, margin) -> str:
    cx, cy = x + w / 2, y + h / 2
    d = {"top": cy, "bottom": H - cy, "left": cx, "right": W - cx}
    return min(d, key=d.get)
