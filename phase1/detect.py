from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple

# Pokemon card physical ratio: 63mm wide x 88mm tall
ASPECT_RATIO = 63.0 / 88.0
ASPECT_TOLERANCE = 0.15

def detect_card(img: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """
    Find the card in a raw photo. Returns (corners, method).

    All methods funnel through the same downstream pipeline:
      mask | edges
        → boundary (Canny on filled mask, or the edges directly)
        → 4 dominant Hough lines (split horiz/vert, dedup, median per group)
        → 4 line intersections (corners — extrapolates past rounded physical corners)
        → sub-pixel line refinement (perpendicular gradient peak + TLS refit)
        → edge_evidence_score

    The method with the highest min-edge-score wins.

    An ML fallback via `phase1.detect_rembg` was prototyped (see
    investigate_detect_vs_ml.py). It improves outer-edge localisation on
    cards with whitening but is not in the production path: Phase 3's edge
    inspector is expected to sample perpendicular profiles in the original
    image, which makes the exact quad position less critical and removes
    the need for ML segmentation here. Revisit if that design changes.

    Method order (cost-ordered; final winner is score-based, not order-based):
      colored_border  — HSV mask on solid card border; survives slabbed cards.
      bg_subtraction  — chromaticity-only LAB distance from corner-sampled bg.
      blurred_canny   — heavy Gaussian blur Canny.
      canny           — fine-detail Canny.
      adaptive        — local adaptive threshold.
      grabcut         — last-resort foreground segmentation.
    """
    methods = [
        ('colored_border', _mask_colored_border),
        ('bg_subtraction', _mask_bg_subtraction),
        ('blurred_canny',  _edges_blurred_canny),
        ('canny',          _edges_canny),
        ('adaptive',       _edges_adaptive),
        ('grabcut',        _mask_grabcut),
    ]

    candidates = []  # list of (corners, score, name)
    for name, fn in methods:
        try:
            result = fn(img)
        except Exception:
            continue
        if result is None:
            continue
        kind, data = result
        boundary = data if kind == 'edges' else _mask_to_boundary(data, img.shape)
        if boundary is None:
            continue
        corners = _corners_from_hough(boundary, img.shape)
        if corners is None or not _aspect_ok(corners):
            continue
        refined = _refine_lines_subpixel(img, corners)
        if refined is None or not _aspect_ok(refined):
            refined = corners
        score = edge_evidence_score(img, refined)['min_edge_score']
        candidates.append((refined, score, name))

    if not candidates:
        return None, 'failed'
    best = max(candidates, key=lambda c: c[1])
    return _ensure_portrait(best[0]), best[2]


# ── public helpers for debug visualisation ────────────────────────────────────

def get_bg_mask(img: np.ndarray) -> np.ndarray:
    """
    Binary foreground mask via corner background sampling, chromaticity-only.

    Samples the outer ~4% of each image corner in LAB space, then thresholds
    Euclidean distance in the (a*, b*) plane (luminance dropped). This handles
    dark-card-on-dark-bg cases where L is similar but chroma differs strongly
    (e.g. dark-blue card on dark-brown wood).
    """
    h, w = img.shape[:2]
    m = max(int(min(h, w) * 0.04), 10)

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_pixels = np.vstack([
        img_lab[:m,  :m ].reshape(-1, 3),
        img_lab[:m,  w-m:].reshape(-1, 3),
        img_lab[h-m:, :m ].reshape(-1, 3),
        img_lab[h-m:, w-m:].reshape(-1, 3),
    ])
    bg_mean_ab = np.median(bg_pixels[:, 1:], axis=0)
    bg_std_ab  = np.maximum(bg_pixels[:, 1:].std(axis=0), 5.0)

    dist = np.linalg.norm((img_lab[..., 1:] - bg_mean_ab) / bg_std_ab, axis=2)
    mask = (dist > 2.5).astype(np.uint8) * 255

    k9 = np.ones((9, 9), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k9, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5, iterations=2)
    return mask


def get_canny_edges(img: np.ndarray) -> np.ndarray:
    """Fine-detail Canny edge map for debug visualisation."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    blurred = cv2.GaussianBlur(lab[:, :, 0], (5, 5), 0)
    edges = cv2.bitwise_or(cv2.Canny(blurred, 30, 100), cv2.Canny(blurred, 50, 150))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def get_blurred_canny_edges(img: np.ndarray) -> np.ndarray:
    """Heavy-blur Canny edge map for debug visualisation."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    blurred = cv2.GaussianBlur(lab[:, :, 0], (21, 21), 0)
    edges = cv2.Canny(blurred, 15, 50)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)


# ── corner ordering ───────────────────────────────────────────────────────────

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points into TL, TR, BR, BL via the sum/difference trick."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # TL: smallest x+y
        pts[np.argmin(d)],   # TR: smallest x-y
        pts[np.argmax(s)],   # BR: largest x+y
        pts[np.argmax(d)],   # BL: largest x-y
    ], dtype=float)


def _ensure_portrait(corners: np.ndarray) -> np.ndarray:
    """Rotate corner ordering 90 deg CW if the detected quad is wider than tall."""
    tl, tr, _, bl = corners
    if np.linalg.norm(tr - tl) > np.linalg.norm(bl - tl):
        corners = np.array([corners[3], corners[0], corners[1], corners[2]], dtype=float)
    return corners


def _aspect_ok(corners: np.ndarray) -> bool:
    w_len = np.linalg.norm(corners[1] - corners[0])
    h_len = np.linalg.norm(corners[3] - corners[0])
    if h_len == 0 or w_len == 0:
        return False
    ratio = min(w_len, h_len) / max(w_len, h_len)
    return abs(ratio - ASPECT_RATIO) / ASPECT_RATIO <= ASPECT_TOLERANCE


# ── boundary extraction (shared by mask-producing methods) ────────────────────

def _mask_to_boundary(mask: np.ndarray, img_shape: tuple,
                      min_frac: float = 0.08) -> Optional[np.ndarray]:
    """
    Aggressive close → largest CC → fill → fill internal holes → Canny.

    The aggressive close (kernel ~3% of image dimension) bridges:
      - Thin-ring masks (colored_border) → solid card blob.
      - Fragmented chromaticity masks (holo / dark-on-dark) → unified card.
      - Missing-border-corner gaps that the convex-hull approach would have
        turned into diagonal phantom edges.

    Hole-filling (flood-fill from outside) eliminates interior dark patches
    in the card region so the Canny boundary is just the 4 outer sides.
    """
    h, w = img_shape[:2]
    k_size = max(min(h, w) // 30, 7)
    if k_size % 2 == 0:
        k_size += 1
    k = np.ones((k_size, k_size), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_frac * h * w:
        return None

    clean = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(clean, [largest], -1, 255, -1)
    # Fill any interior holes in the card blob so Canny captures only the
    # outer perimeter (4 card edges, not interior contour rings).
    flood = clean.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(clean, holes)
    return cv2.Canny(filled, 50, 150)


def _edges_to_mask(edges: np.ndarray, kernel_size: int = 7,
                   iters: int = 2) -> np.ndarray:
    """
    Convert a Canny-style edge image into a filled foreground mask.

    Closes gaps, then picks the contour whose CONVEX HULL has the largest
    area, and fills the hull. Using hull-area (not contour-area) is robust
    to:
      - The card outline being broken into fragments by Canny gaps (its hull
        still spans the full card).
      - Interior-text blobs winning on raw contour area (their hulls are
        smaller than the card's hull bounding region).
    The hull is only used to identify which region IS the card; the downstream
    pipeline still extracts the precise 4 edges from this filled-hull mask
    via `_mask_to_boundary` → Canny → Hough.
    """
    k = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=iters)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(edges)
    best = max(contours, key=lambda c: cv2.contourArea(cv2.convexHull(c)))
    hull = cv2.convexHull(best)
    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    return mask


# ── Hough line helpers ────────────────────────────────────────────────────────

def _line_intersect(l1: np.ndarray, l2: np.ndarray) -> Optional[np.ndarray]:
    """Intersect two Hough lines given as (rho, theta). Returns (x, y) or None."""
    c1, s1 = np.cos(l1[1]), np.sin(l1[1])
    c2, s2 = np.cos(l2[1]), np.sin(l2[1])
    det = c1 * s2 - s1 * c2
    if abs(det) < 1e-6:
        return None
    x = (l1[0] * s2 - l2[0] * s1) / det
    y = (l2[0] * c1 - l1[0] * c2) / det
    return np.array([x, y], dtype=float)


def _split_two_lines(lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a set of co-directional Hough lines into two groups (the two
    opposing card edges) by finding the largest gap in rho. Returns the
    median (rho, theta) of each group.

    Canonicalisation: near-vertical lines can be encoded by Hough as either
    (rho, theta≈0) or (-rho, theta≈π). Naively flipping rho's sign and adding
    π puts theta near 360°, then the median over a cluster mixing 0° and 360°
    snaps to 180° — which represents a line through x=-rho, completely wrong.
    Instead we map any theta > π/2 to theta-π (with rho negated correspondingly),
    so a near-vertical cluster sits in [-π/2, π/2] with no wrap. Same line,
    sensible median.
    """
    rho   = lines[:, 0].copy().astype(float)
    theta = lines[:, 1].copy().astype(float)

    # Detect orientation by the group's median theta (caller already split
    # by direction, so the group is internally consistent).
    median_theta = float(np.median(theta))
    if median_theta < np.pi / 4 or median_theta > 3 * np.pi / 4:
        # Near-vertical: canonicalise theta to be near 0 (within [-π/2, π/2]).
        mask = theta > np.pi / 2
        theta[mask] = theta[mask] - np.pi
        rho[mask] = -rho[mask]
        # All near-vertical lines now share orientation; rho carries the signed
        # x-intercept directly. Sort+split on rho cleanly separates left vs right.
    # Near-horizontal lines: theta stays in (π/4, 3π/4); no wrap issue,
    # rho is signed y-intercept-ish, sort by rho works as-is.

    order = np.argsort(rho)
    rho_s = rho[order]
    theta_s = theta[order]
    split = int(np.argmax(np.diff(rho_s))) + 1

    g1 = np.stack([rho_s[:split], theta_s[:split]], axis=1)
    g2 = np.stack([rho_s[split:], theta_s[split:]], axis=1)
    return np.median(g1, axis=0), np.median(g2, axis=0)


def _corners_from_hough(edges: np.ndarray, img_shape: tuple) -> Optional[np.ndarray]:
    """
    Find the 4 dominant straight lines on a boundary edge image and return the
    4 line-intersection corners. Extrapolates cleanly past rounded physical
    card corners — the corners are the virtual intersections of the extended
    straight edges, not points on the contour.
    """
    h, w = img_shape[:2]
    # Threshold sized so a clean card edge always exceeds it, but a partially
    # jagged edge (broken into ~500-px segments on high-res phone photos)
    # still contributes at least one line per direction. _split_two_lines's
    # median-per-group deduplicates the resulting cluster of near-duplicate
    # lines per edge.
    threshold = max(min(h, w) // 15, 50)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    if lines is None or len(lines) < 4:
        return None

    rho_theta = lines[:, 0, :]
    theta = rho_theta[:, 1]

    horiz_mask = (theta >= np.pi / 4) & (theta < 3 * np.pi / 4)
    horiz = rho_theta[horiz_mask]
    vert  = rho_theta[~horiz_mask]
    if len(horiz) < 2 or len(vert) < 2:
        return None

    h1, h2 = _split_two_lines(horiz)
    v1, v2 = _split_two_lines(vert)

    pts = []
    for hh, vv in [(h1, v1), (h1, v2), (h2, v2), (h2, v1)]:
        pt = _line_intersect(hh, vv)
        if pt is None:
            return None
        pts.append(pt)

    return order_corners(np.array(pts, dtype=float))


# ── sub-pixel line refinement ─────────────────────────────────────────────────

def _refine_lines_subpixel(img: np.ndarray, corners: np.ndarray,
                           n_samples: int = 80,
                           perp_window: int = 8) -> Optional[np.ndarray]:
    """
    Refine the 4 corners by re-fitting each edge LINE (not corner) to
    sub-pixel-located gradient peaks, then re-intersecting.

    For each of the 4 edges:
      1. Sample n_samples points along the current edge (5%–95% of length).
      2. At each sample, look perpendicular ±perp_window and find the max
         gradient magnitude.
      3. Parabolic sub-pixel interpolation on the 3-point neighbourhood
         around that peak.
      4. Total-least-squares line fit to the collected sub-pixel points.
    Then re-intersect adjacent line pairs for sub-pixel corners.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    h, w = grad_mag.shape

    edges = [(corners[0], corners[1]),  # top:    TL → TR
             (corners[1], corners[2]),  # right:  TR → BR
             (corners[2], corners[3]),  # bottom: BR → BL
             (corners[3], corners[0])]  # left:   BL → TL

    refined_lines = []  # each line as (nx, ny, c) with nx*x + ny*y = c
    for p0, p1 in edges:
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        edge_vec = p1 - p0
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len < 1:
            return None
        edge_dir = edge_vec / edge_len
        perp = np.array([-edge_dir[1], edge_dir[0]])

        pts = []
        for t in np.linspace(0.05, 0.95, n_samples):
            base = p0 + t * edge_vec
            mags = np.zeros(2 * perp_window + 1, dtype=float)
            for j, k in enumerate(range(-perp_window, perp_window + 1)):
                p = base + k * perp
                x, y = int(round(p[0])), int(round(p[1]))
                if 0 <= x < w and 0 <= y < h:
                    mags[j] = float(grad_mag[y, x])
            idx = int(np.argmax(mags))
            if mags[idx] <= 0:
                continue
            # Parabolic sub-pixel offset on 3-point neighbourhood
            if 0 < idx < len(mags) - 1 and mags[idx - 1] > 0 and mags[idx + 1] > 0:
                denom = mags[idx - 1] - 2 * mags[idx] + mags[idx + 1]
                if abs(denom) > 1e-6:
                    delta = 0.5 * (mags[idx - 1] - mags[idx + 1]) / denom
                    delta = float(np.clip(delta, -1.0, 1.0))
                else:
                    delta = 0.0
            else:
                delta = 0.0
            k_sub = (idx - perp_window) + delta
            pts.append(base + k_sub * perp)

        if len(pts) < 10:
            return None
        pts_arr = np.asarray(pts, dtype=float)
        centroid = pts_arr.mean(axis=0)
        centered = pts_arr - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]                                # principal axis
        normal = np.array([-direction[1], direction[0]])  # unit normal
        c = float(normal @ centroid)
        refined_lines.append((normal[0], normal[1], c))

    # Re-intersect refined lines.
    # edge 0 = top, 1 = right, 2 = bottom, 3 = left
    # TL = top ∩ left, TR = top ∩ right, BR = bottom ∩ right, BL = bottom ∩ left
    pairs = [(0, 3), (0, 1), (2, 1), (2, 3)]
    new_corners = []
    for i, j in pairs:
        pt = _intersect_abc(refined_lines[i], refined_lines[j])
        if pt is None:
            return None
        new_corners.append(pt)
    return np.asarray(new_corners, dtype=float)


def _intersect_abc(l1: tuple, l2: tuple) -> Optional[np.ndarray]:
    """Intersect two lines parameterised as (a, b, c) with a*x + b*y = c."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    return np.array([x, y], dtype=float)


# ── edge-evidence quality score (public) ──────────────────────────────────────

def edge_evidence_score(img: np.ndarray, corners: np.ndarray,
                        n_samples: int = 80, perp_window: int = 6) -> dict:
    """
    Score each of the 4 quad edges for boundary evidence.

    Per sample point: find max-gradient pixel within ±perp_window of the line,
    require magnitude > image p60 AND gradient direction within ~45° of the
    line normal (|cos| > 0.7). Per-edge score = hits / valid_samples. Quad
    score = MIN across edges (worst edge determines warp quality).

    Returns {'edge_0..3', 'min_edge_score', 'mean_edge_score'}. A `min` of
    ≥0.85 means "trust this quad"; lower means we should try the ML fallback
    or prompt for a reshoot.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    h, w = grad_mag.shape
    threshold = float(np.percentile(grad_mag, 60))

    edges = [(corners[0], corners[1]),
             (corners[1], corners[2]),
             (corners[2], corners[3]),
             (corners[3], corners[0])]

    out: dict = {}
    for i, (p0, p1) in enumerate(edges):
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        edge_vec = p1 - p0
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len < 1:
            out[f'edge_{i}'] = 0.0
            continue
        edge_dir = edge_vec / edge_len
        perp = np.array([-edge_dir[1], edge_dir[0]])

        hits = 0
        valid = 0
        for t in np.linspace(0.05, 0.95, n_samples):
            base = p0 + t * edge_vec
            best_mag = 0.0
            best_gx = 0.0
            best_gy = 0.0
            for k in range(-perp_window, perp_window + 1):
                p = base + k * perp
                x, y = int(round(p[0])), int(round(p[1]))
                if 0 <= x < w and 0 <= y < h:
                    m = float(grad_mag[y, x])
                    if m > best_mag:
                        best_mag = m
                        best_gx = float(gx[y, x])
                        best_gy = float(gy[y, x])
            if best_mag <= 0:
                continue
            valid += 1
            if best_mag < threshold:
                continue
            cos_align = (best_gx * perp[0] + best_gy * perp[1]) / best_mag
            if abs(cos_align) > 0.7:
                hits += 1
        out[f'edge_{i}'] = (hits / valid) if valid > 0 else 0.0

    scores = [out[f'edge_{i}'] for i in range(4)]
    out['min_edge_score'] = float(min(scores))
    out['mean_edge_score'] = float(np.mean(scores))
    return out


# ── per-method producers (return ('mask'|'edges', ndarray) or None) ───────────

# Coloured-border HSV ranges, in priority order. Tried sequentially within
# _mask_colored_border; the first range whose largest CC covers > min_frac of
# the image is returned as the mask. Tunable — add ranges as needed.
BORDER_HSV_RANGES = [
    ("yellow",  ( 18,  80, 100), ( 35, 255, 255)),
    ("black",   (  0,   0,   0), (180,  60,  55)),
    ("silver",  (  0,   0, 150), (180,  20, 220)),
    ("white",   (  0,   0, 220), (180,  25, 255)),
]


def _mask_colored_border(img: np.ndarray):
    """HSV-range mask of the dominant card-border colour. Slabbed-card friendly."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_area = img.shape[0] * img.shape[1]
    k = np.ones((5, 5), np.uint8)
    for _name, lo, hi in BORDER_HSV_RANGES:
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
        # Quick area check — skip ranges that hit barely anything
        if int(np.count_nonzero(mask)) < 0.08 * img_area:
            continue
        return ('mask', mask)
    return None


def _mask_bg_subtraction(img: np.ndarray):
    return ('mask', get_bg_mask(img))


def _edges_canny(img: np.ndarray):
    return ('mask', _edges_to_mask(get_canny_edges(img)))


def _edges_blurred_canny(img: np.ndarray):
    return ('mask', _edges_to_mask(get_blurred_canny_edges(img)))


def _edges_adaptive(img: np.ndarray):
    gray   = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    edges  = cv2.Canny(thresh, 50, 150)
    return ('mask', _edges_to_mask(edges))


def _mask_grabcut(img: np.ndarray):
    h, w  = img.shape[:2]
    mask  = np.zeros((h, w), np.uint8)
    rect  = (w // 10, h // 10, w * 8 // 10, h * 8 // 10)
    bgd   = np.zeros((1, 65), np.float64)
    fgd   = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    return ('mask', fg)


