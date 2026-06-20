"""
Region characterizer: deterministic pre-filter between PatchCore and VLM.

PatchCore over-recalls. Many of its regions are not defects but imaging
artefacts: alignment residual at content edges, holographic colour-cast, glare
hotspots, sharpness mismatch. Sending all to a VLM is expensive AND
risky (VLMs hallucinate on borderline noisy inputs).

The core empirical insight (from sample_2 front diff heatmap analysis): the
dominant PatchCore false positive is **alignment residual**, which shows up as
ΔE concentrated at content edges (Lugia silhouette, text edges, lightning
bolts) — anywhere the candidate has high local gradient. A real defect
(stain, scratch) creates ΔE in regions where the candidate is FLAT (low
gradient). The two signals are separable by looking at where the diff lives.

Five rules, OR-rejection (any rule rejects):

  edge_distance         : centroid is in the outer ~5% margin. Edge_inspect
                          owns the boundary in the original image. Reject.
  diff_edge_corr        : Pearson correlation between local ΔE and candidate
                          gradient magnitude. > 0.5 ⇒ diff lives on content
                          edges ⇒ alignment residual. **The key new metric.**
  mean_delta_e          : mean ΔE across the region. Below ~3 ⇒ PatchCore
                          flagged something feature-space-only (sharpness,
                          tonemap) with no visible colour signal ⇒ noise.
  chroma_lum_ratio      : mean Δ(a*,b*) / max(ΔL, 1). Large ⇒ colour cast,
                          not defect.
  specular_fraction     : candidate saturated >30% while reference <10% ⇒ glare.

Thresholds tunable; validate against ground truth:
  - sample_1/front: must KEEP the two confirmed brown stains.
  - sample_2/front: should reject ~all of PatchCore's 32 noise regions.
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── thresholds (tunable; validated against ground-truth test set) ────────────
EDGE_FRAC                  = 0.05  # < this fraction of min(h,w) from border → reject
DIFF_EDGE_CORR_THRESHOLD   = 0.50  # > this Pearson corr (ΔE vs candidate-gradient) → reject
MAX_BLOB_AREA_MIN          = 50    # < this many pixels in largest hi-ΔE blob → no localised signal
BLOB_THRESHOLD_K           = 2.0   # ΔE > k * median(ΔE) defines a "high-ΔE" pixel
BLOB_THRESHOLD_FLOOR       = 25.0  # but never below this absolute ΔE (avoid noisy thresholds in flat regions)
CHROMA_LUM_RATIO_THRESH    = 2.0   # > this ratio → reject (colour cast dominates)
SPECULAR_CAND_FRAC         = 0.30  # > this fraction saturated in cand
SPECULAR_REF_FRAC          = 0.10  # AND < this fraction in ref → glare

L_SATURATION_LEVEL         = 240


@dataclass
class Characterization:
    region: dict
    keep: bool
    reasons: List[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


# ── metric helpers ───────────────────────────────────────────────────────────

def _edge_distance(bbox: Tuple[int, int, int, int],
                   card_shape: Tuple[int, int]) -> float:
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    H, W = card_shape
    return min(cx, cy, W - cx, H - cy) / min(H, W)


def _delta_e_map(cand_crop: np.ndarray, ref_crop: np.ndarray) -> np.ndarray:
    """Per-pixel CIE76 ΔE between candidate and reference crops."""
    c = cv2.cvtColor(cand_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref_crop,  cv2.COLOR_BGR2LAB).astype(np.float32)
    d = c - r
    return np.sqrt(np.sum(d * d, axis=2))


def _gradient_mag(crop: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _candidate_gradient_mag(cand_crop: np.ndarray) -> np.ndarray:
    return _gradient_mag(cand_crop)


def _diff_edge_correlation(delta_e: np.ndarray, grad_mag: np.ndarray) -> float:
    """
    Pearson correlation between ΔE and candidate gradient magnitude.

    Alignment residual concentrates ΔE on content edges (where grad_mag is
    high). A real defect creates ΔE in flat areas (where grad_mag is low).
    """
    if delta_e.size == 0:
        return 0.0
    a = delta_e.ravel()
    b = grad_mag.ravel()
    if a.std() < 1e-3 or b.std() < 1e-3:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _chroma_lum_ratio(cand_crop: np.ndarray,
                      ref_crop: np.ndarray) -> Tuple[float, float, float]:
    c_lab = cv2.cvtColor(cand_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    r_lab = cv2.cvtColor(ref_crop,  cv2.COLOR_BGR2LAB).astype(np.float32)
    dL = float(np.mean(np.abs(c_lab[..., 0] - r_lab[..., 0])))
    da = c_lab[..., 1] - r_lab[..., 1]
    db = c_lab[..., 2] - r_lab[..., 2]
    dC = float(np.mean(np.sqrt(da * da + db * db)))
    return dC, dL, dC / max(dL, 1.0)


def _specular_fraction(cand_crop: np.ndarray,
                       ref_crop: np.ndarray) -> Tuple[float, float]:
    c_l = cv2.cvtColor(cand_crop, cv2.COLOR_BGR2LAB)[..., 0]
    r_l = cv2.cvtColor(ref_crop,  cv2.COLOR_BGR2LAB)[..., 0]
    return float((c_l > L_SATURATION_LEVEL).mean()), float((r_l > L_SATURATION_LEVEL).mean())


def _max_blob_area(delta_e: np.ndarray) -> int:
    """
    Size (pixels) of the largest connected component of high-ΔE pixels.

    Thresholds the per-pixel ΔE map at k × median(ΔE) (with an absolute floor)
    and counts pixels in the biggest blob. Discriminates:
      stain  → one large coherent blob, area ≫ 50
      scratch → elongated line blob, area moderate (50–300)
      sparkle → many tiny scattered points, largest CC ~5–20 px
      uniform noise → no coherent blob at all, largest CC ≤ a few px
    """
    if delta_e.size == 0:
        return 0
    median = float(np.median(delta_e))
    threshold = max(median * BLOB_THRESHOLD_K, BLOB_THRESHOLD_FLOOR)
    mask = (delta_e > threshold).astype(np.uint8) * 255
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


def _crop(img: np.ndarray, bbox: Tuple[int, int, int, int],
          pad: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad); y1 = min(h, y + bh + pad)
    return img[y0:y1, x0:x1]


# ── top-level ────────────────────────────────────────────────────────────────

def characterize_regions(candidate: np.ndarray,
                         reference: np.ndarray,
                         regions: List[dict],
                         card_shape: Optional[Tuple[int, int]] = None
                         ) -> List[Characterization]:
    if card_shape is None:
        card_shape = candidate.shape[:2]

    results: List[Characterization] = []
    for region in regions:
        bbox = tuple(region['bbox'])  # type: ignore[arg-type]
        cand_crop = _crop(candidate, bbox)
        ref_crop  = _crop(reference, bbox)

        delta_e  = _delta_e_map(cand_crop, ref_crop)
        cand_grad = _gradient_mag(cand_crop)
        ref_grad  = _gradient_mag(ref_crop)
        mean_dE  = float(delta_e.mean()) if delta_e.size else 0.0
        max_dE   = float(delta_e.max())  if delta_e.size else 0.0
        ref_detail = float(ref_grad.mean()) if ref_grad.size else 0.0
        cand_detail = float(cand_grad.mean()) if cand_grad.size else 0.0
        max_blob = _max_blob_area(delta_e)

        edge_d   = _edge_distance(bbox, card_shape)
        diff_corr = _diff_edge_correlation(delta_e, cand_grad)
        dC, dL, ratio = _chroma_lum_ratio(cand_crop, ref_crop)
        cand_sat, ref_sat = _specular_fraction(cand_crop, ref_crop)

        reasons: List[str] = []
        if edge_d < EDGE_FRAC:
            reasons.append('edge_zone')
        if diff_corr > DIFF_EDGE_CORR_THRESHOLD:
            reasons.append('alignment_residual')
        if max_blob < MAX_BLOB_AREA_MIN:
            reasons.append('no_localised_signal')
        if ratio > CHROMA_LUM_RATIO_THRESH:
            reasons.append('chroma_dominant')
        if cand_sat > SPECULAR_CAND_FRAC and ref_sat < SPECULAR_REF_FRAC:
            reasons.append('specular')

        results.append(Characterization(
            region=region,
            keep=(len(reasons) == 0),
            reasons=reasons,
            metrics={
                'edge_distance':       edge_d,
                'diff_edge_corr':      diff_corr,
                'mean_delta_e':        mean_dE,
                'max_delta_e':         max_dE,
                'max_blob_area':       max_blob,
                'ref_detail':          ref_detail,
                'cand_detail':         cand_detail,
                'chroma_diff':         dC,
                'luminance_diff':      dL,
                'chroma_lum_ratio':    ratio,
                'cand_sat_frac':       cand_sat,
                'ref_sat_frac':        ref_sat,
            },
        ))
    return results
