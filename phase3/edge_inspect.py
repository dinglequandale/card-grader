"""
Reference-free edge inspection (Phase 3, perimeter pillar).

The TCG-API reference images are tight-cropped inside the yellow border —
they contain no card-edge signal at all.  So edge defects (whitening,
fraying, chipping, rough cuts) cannot be detected by reference diff.  This
module solves the same problem the way human graders do: it looks at the
candidate's own edge for relative wear against the edge's own interior.

Algorithm per edge:
  1. Rectify the edge into a thin axis-aligned strip via perspective warp
     using the candidate's 4 detected corners.  Strip width = full edge
     length, strip height = a few % of card width inward from the physical
     edge.  No background pixels survive this rectification.
  2. Within the strip, sample two colour-statistics bands:
       - OUTER band: the first OUTER_BAND_FRAC of the strip's depth (the
         actual paper edge).
       - INTERIOR band: a deeper interior slab known to be plain coloured
         border (not artwork / not text).
  3. Whitening signature: per-column, compare outer-band L (luminance) and
     chroma to interior-band's robust median + MAD.  A column where the
     outer band is significantly lighter and/or less saturated than the
     interior is whitened.
  4. Connected-components along the strip length → discrete defect regions
     with bbox-in-strip and severity = number of MAD-standard-deviations.

Each function is pure; the runner does I/O and visualisation.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

import cv2
import numpy as np


# ── Tunables ──────────────────────────────────────────────────────────────────

STRIP_WIDTH_FRAC      = 0.05   # depth inward as % of card width.  Generous so
                               # we always include the full border + a small
                               # buffer past it; we'll find the border
                               # boundary adaptively inside the strip.
OUTER_BAND_DEPTH_PX   = 3      # outermost rows treated as the "edge" zone
BORDER_DETECT_SIGMA   = 8.0    # row-mean ΔE change that flags the border→interior
                               # transition (LAB units of luma+chroma)
COLUMN_SMOOTH_PX      = 5      # 1-D smoothing along the edge before thresholding
SEVERITY_THRESHOLD    = 3.0    # MAD-multiples — column is "defect" when above
MIN_RUN_LENGTH_PCT    = 0.5    # min run of contiguous defect columns as % of edge


@dataclass
class EdgeDefect:
    """One edge defect, located along a single edge of the card."""
    side:         str           # 'top' | 'bottom' | 'left' | 'right'
    start_pct:    float         # 0-100 along the edge (start of left/top)
    extent_pct:   float         # 0-100 length along the edge
    peak_sigma:   float         # max severity in MAD-multiples within the run
    luma_drop:    float         # mean L drop (candidate's outer vs interior)
    chroma_drop:  float         # mean chroma drop
    type_hint:    str           # 'whitening' | 'darkening' | 'desaturation'

    def asdict(self) -> dict:
        return asdict(self)


# ── Strip rectification ───────────────────────────────────────────────────────

def rectify_edge_strip(img: np.ndarray, corners: np.ndarray, side: str,
                       width_frac: float = STRIP_WIDTH_FRAC,
                       ) -> np.ndarray:
    """
    Perspective-warp the card's `side` edge into a strip whose TOP row is the
    card's physical outer edge and whose y axis runs inward.

    corners: (4,2) array, TL/TR/BR/BL in original-image pixel coordinates.
    """
    TL, TR, BR, BL = corners.astype(np.float32)
    card_w = (np.linalg.norm(TR - TL) + np.linalg.norm(BR - BL)) / 2
    card_h = (np.linalg.norm(BL - TL) + np.linalg.norm(BR - TR)) / 2

    if   side == 'top':    p0, p1, u_in_pair, long_len, in_len = TL, TR, (BL - TL), card_w, card_h * width_frac
    elif side == 'bottom': p0, p1, u_in_pair, long_len, in_len = BL, BR, (TL - BL), card_w, card_h * width_frac
    elif side == 'left':   p0, p1, u_in_pair, long_len, in_len = TL, BL, (TR - TL), card_h, card_w * width_frac
    elif side == 'right':  p0, p1, u_in_pair, long_len, in_len = TR, BR, (TL - TR), card_h, card_w * width_frac
    else: raise ValueError(side)

    u_in = u_in_pair / np.linalg.norm(u_in_pair)
    src = np.array([
        p0,
        p1,
        p1 + u_in * in_len,
        p0 + u_in * in_len,
    ], dtype=np.float32)
    out_w = int(round(long_len))
    out_h = int(round(in_len))
    dst = np.array([
        [0,      0],
        [out_w,  0],
        [out_w,  out_h],
        [0,      out_h],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR)


# ── Per-edge inspection ───────────────────────────────────────────────────────

def _column_stats_lab(strip: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """
    Per-column LAB-space stats for the band y in [y0, y1).
    Returns array of shape (W, 4): [L_med, a_med, b_med, chroma_med]
    where chroma = sqrt((a-128)^2 + (b-128)^2).
    """
    band = cv2.cvtColor(strip[y0:y1], cv2.COLOR_BGR2LAB).astype(np.float32)
    L = np.median(band[..., 0], axis=0)
    a = np.median(band[..., 1], axis=0)
    b = np.median(band[..., 2], axis=0)
    chroma = np.sqrt((a - 128) ** 2 + (b - 128) ** 2)
    return np.stack([L, a, b, chroma], axis=1)


def _robust_median_mad(x: np.ndarray) -> tuple[float, float]:
    """Return (median, MAD_as_stdev_estimate). MAD scaled by 1.4826."""
    m = float(np.median(x))
    mad = float(np.median(np.abs(x - m))) * 1.4826
    return m, max(mad, 1.0)


def find_border_interior_band(strip: np.ndarray,
                              outer_skip: int = OUTER_BAND_DEPTH_PX,
                              change_sigma: float = BORDER_DETECT_SIGMA,
                              ) -> tuple[int, int]:
    """
    Adaptively find the row range inside the rectified strip that lies
    SQUARELY in the card's coloured border (just past the worn edge zone,
    before any artwork/text transition).

    Algorithm: starting a few rows in from the outer edge (skipping past
    any wear), find the longest stable stretch where the per-row colour
    barely changes.  Stop when the row-to-row colour delta exceeds
    `change_sigma`.  Return (y_start, y_end) of that band.
    """
    H = strip.shape[0]
    if H < outer_skip + 4:
        return outer_skip, max(outer_skip + 1, H - 1)

    lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Per-row median LAB triple (column-robust)
    row_med = np.median(lab, axis=1)            # shape (H, 3)
    # Anchor: row just past the outer-wear zone is assumed to be border
    anchor = row_med[outer_skip]
    # Walk inward: stop when per-row distance from anchor exceeds threshold
    y_end = H - 1
    for r in range(outer_skip + 1, H):
        d = float(np.linalg.norm(row_med[r] - anchor))
        if d > change_sigma:
            y_end = r - 1
            break
    # Must be at least 3 rows thick to be useful as a calibration zone.
    y_start = outer_skip
    if y_end - y_start < 3:
        y_end = min(H - 1, y_start + 3)
    return y_start, y_end


def inspect_edge(strip: np.ndarray,
                 outer_band_depth_px: int   = OUTER_BAND_DEPTH_PX,
                 smooth_px:           int   = COLUMN_SMOOTH_PX,
                 severity_threshold:  float = SEVERITY_THRESHOLD,
                 min_run_length_pct:  float = MIN_RUN_LENGTH_PCT,
                 side:                str   = '?',
                 ) -> tuple[list[EdgeDefect], dict]:
    """
    Inspect a rectified edge strip for outer-edge whitening / darkening
    / desaturation relative to its own interior.  Returns (defects, diag).
    """
    H, W = strip.shape[:2]
    if H < 6 or W < 20:
        return [], {'reason': 'strip too small'}

    y_outer_end = max(1, min(outer_band_depth_px, H - 4))
    y_int_start, y_int_end = find_border_interior_band(strip, outer_skip=y_outer_end + 1)

    outer    = _column_stats_lab(strip, 0, y_outer_end)         # (W, 4)
    interior = _column_stats_lab(strip, y_int_start, y_int_end) # (W, 4)

    # Reference distribution = the INTERIOR band's per-column stats, pooled.
    # We use ROBUST stats (median + MAD) so an occasional artwork pixel or
    # text glyph in the interior band doesn't poison the calibration.
    L_med, L_mad         = _robust_median_mad(interior[:, 0])
    chroma_med, chroma_mad = _robust_median_mad(interior[:, 3])

    luma_diff   = outer[:, 0] - L_med           # positive = outer is brighter
    chroma_diff = chroma_med - outer[:, 3]      # positive = outer is less saturated

    # Smooth column signals 1-D so single-column noise doesn't create runs.
    k = max(1, smooth_px) | 1
    luma_smooth   = cv2.GaussianBlur(luma_diff.reshape(1, -1).astype(np.float32),
                                     (k, 1), 0).ravel()
    chroma_smooth = cv2.GaussianBlur(chroma_diff.reshape(1, -1).astype(np.float32),
                                     (k, 1), 0).ravel()

    # Severity in MAD-standard-deviations.  Take the MAX of luma-brightening
    # and chroma-loss, since either signal indicates whitening.
    sigma_luma   = luma_smooth   / L_mad
    sigma_chroma = chroma_smooth / chroma_mad
    severity = np.maximum(sigma_luma, sigma_chroma)

    above = severity > severity_threshold

    # Find runs of contiguous defect columns; reject runs shorter than min length.
    min_run = max(3, int(W * min_run_length_pct / 100.0))
    defects: list[EdgeDefect] = []
    i = 0
    while i < W:
        if above[i]:
            j = i
            while j < W and above[j]:
                j += 1
            if (j - i) >= min_run:
                peak = float(severity[i:j].max())
                luma_drop_mean   = float(luma_smooth[i:j].mean())
                chroma_drop_mean = float(chroma_smooth[i:j].mean())
                # Classify: which channel drove it
                if luma_drop_mean > 0 and chroma_drop_mean > 0:
                    type_hint = 'whitening'
                elif luma_drop_mean < 0:
                    type_hint = 'darkening'
                else:
                    type_hint = 'desaturation'
                defects.append(EdgeDefect(
                    side       = side,
                    start_pct  = 100.0 * i / W,
                    extent_pct = 100.0 * (j - i) / W,
                    peak_sigma = peak,
                    luma_drop  = luma_drop_mean,
                    chroma_drop= chroma_drop_mean,
                    type_hint  = type_hint,
                ))
            i = j
        else:
            i += 1

    diag = {
        'W': W, 'H': H,
        'outer_band': (0, y_outer_end),
        'interior_band': (int(y_int_start), int(y_int_end)),
        'L_interior_median': L_med, 'L_interior_mad': L_mad,
        'chroma_interior_median': chroma_med, 'chroma_interior_mad': chroma_mad,
        'severity_signal': severity,
        'luma_signal':     luma_smooth,
        'chroma_signal':   chroma_smooth,
    }
    return defects, diag


# ── Top-level ─────────────────────────────────────────────────────────────────

SIDES = ('top', 'bottom', 'left', 'right')


def inspect_all_edges(candidate: np.ndarray,
                      candidate_corners: np.ndarray,
                      width_frac: float = STRIP_WIDTH_FRAC,
                      ) -> dict:
    """
    Run the reference-free edge inspector on all 4 edges of the candidate.
    Returns a dict per side with the rectified strip, defects, and diagnostics.
    """
    out: dict = {}
    for side in SIDES:
        strip = rectify_edge_strip(candidate, candidate_corners, side, width_frac)
        defects, diag = inspect_edge(strip, side=side)
        out[side] = {'strip': strip, 'defects': defects, 'diagnostics': diag}
    return out


# ── Visualisation ─────────────────────────────────────────────────────────────

def annotate_edge_strip(strip: np.ndarray, defects: list[EdgeDefect],
                        diag: dict | None = None) -> np.ndarray:
    """Overlay defect bars + severity signal on a rectified strip."""
    vis = strip.copy()
    H, W = vis.shape[:2]
    # Draw a thin reference line at the outer-band boundary
    if diag is not None and 'outer_band' in diag:
        y = diag['outer_band'][1]
        cv2.line(vis, (0, y), (W - 1, y), (200, 200, 200), 1, cv2.LINE_AA)
    for d in defects:
        x0 = int(d.start_pct / 100 * W)
        x1 = int((d.start_pct + d.extent_pct) / 100 * W)
        color = (0, 0, 255) if d.peak_sigma > 6 else (0, 140, 255)
        cv2.rectangle(vis, (x0, 0), (x1, H - 1), color, 2)
        label = f"{d.type_hint} σ={d.peak_sigma:.1f}"
        cv2.putText(vis, label, (x0 + 2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (x0 + 2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return vis


def severity_plot(strip_shape: tuple, diag: dict, threshold: float = SEVERITY_THRESHOLD,
                  plot_height: int = 100) -> np.ndarray:
    """1-D severity-vs-column plot, height matching strip width."""
    W = strip_shape[1] if isinstance(strip_shape, tuple) else strip_shape
    sev = diag.get('severity_signal')
    if sev is None or len(sev) == 0:
        return np.zeros((plot_height, W, 3), dtype=np.uint8)
    plot = np.zeros((plot_height, W, 3), dtype=np.uint8)
    vmax = max(threshold * 2, float(np.percentile(sev, 99)))
    # threshold line
    y_thresh = int(plot_height - threshold / vmax * plot_height)
    cv2.line(plot, (0, y_thresh), (W - 1, y_thresh), (80, 80, 80), 1, cv2.LINE_AA)
    # severity curve
    ys = (plot_height - sev / vmax * plot_height).clip(0, plot_height - 1).astype(int)
    pts = np.stack([np.arange(W), ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(plot, [pts], False, (0, 255, 255), 1, cv2.LINE_AA)
    return plot
