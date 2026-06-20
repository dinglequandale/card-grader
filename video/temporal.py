"""
Temporal bank: turn a registered frame stack (video/ingest.py) into the
three products the rest of the pipeline depends on.

  median           clean, glare-free reference. Centering/perimeter run on
                    this; it is also a defect SUPPRESSOR (central tendency
                    averages away the one tilt that reveals a subtle
                    defect) so do NOT detect defects on it directly.
  foil_mask         per-pixel float map, high = foil/shimmer (angle-
                    dependent local variance), low = matte. Used to route
                    surface detection (foil regions can't use the DB
                    template) and to suppress "is this foil?" false
                    positives.
  defect_saliency   the detection substrate. DETECT-THEN-AGGREGATE: for each
                    frame, compute a per-frame defect response against the
                    median (so static artwork/foil cancels), then take the
                    per-pixel MAX across frames so each defect's
                    best-reveal tilt wins. Collapsing frames first
                    (median/avg of a response) was the bug this replaces.

Validated recipe -- see memory/v2_temporal_validation.md. Two response
channels feed the max-projection:
  (a) Frangi ridge delta: frangi(frame) - frangi(median).  Scratches/
      creases read as ridges; subtracting the median's own ridge response
      cancels static line-like artwork and foil texture.
  (b) patch-tolerant |frame - median| deviation, gradient-attenuated so
      residual sub-pixel misalignment at content edges doesn't fake a
      defect (the recurring "alignment residual" villain from Phase 3).
A third (c) chroma-deviation channel is REQUIRED for small colour marks,
which have near-zero luminance signal (see handoff section 4) -- it is
computed the same patch-tolerant way, on a*/b* instead of L.

Glare is broad and low-frequency, so the ridge/high-pass operators in (a)
and (b) reject it for free; no explicit glare-handling needed here.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from skimage.filters import frangi

FRANGI_SIGMAS = [1, 2, 3]
PATCH_TOLERANCE_RADIUS = 2
RIDGE_DILATE_RADIUS = 1         # static-ridge cancel tolerance (px). Smaller than
                                 # PATCH_TOLERANCE_RADIUS on purpose: larger radii
                                 # over-cancel real line-defects on busy artwork.
GRADIENT_ATTENUATION = 0.85
GRADIENT_PERCENTILE = 95
RESPONSE_NORM_PERCENTILE = 99.5
FOIL_VARIANCE_PERCENTILE = 75   # std above this percentile (post photometric-norm) = foil


@dataclass
class TemporalBank:
    median: np.ndarray          # uint8 BGR, canonical size
    foil_mask: np.ndarray        # float32 [0,1], canonical size
    defect_saliency: np.ndarray  # float32 >=0, canonical size (unclipped above; ~1.0 is "typical" peak)
    foil_threshold: float        # foil_mask cutoff (same [0,1] scale): foil_mask > foil_threshold == foil region


def _photometric_normalize(grays: list) -> list:
    """Scale each frame to a common mean so cross-frame variance reflects
    LOCAL angle-dependent shimmer (foil), not whole-frame exposure drift."""
    gmean = float(np.mean([g.mean() for g in grays]))
    return [g * (gmean / (g.mean() + 1e-6)) for g in grays]


def _patch_tolerant_delta(channel: np.ndarray, ref: np.ndarray,
                           r: int = PATCH_TOLERANCE_RADIUS) -> np.ndarray:
    """min-over-a-small-window |channel - ref|: tolerant of the sub-pixel
    misalignment that deviation maps amplify (the median forgives it;
    deviation maps don't)."""
    h, w = channel.shape
    ref_pad = cv2.copyMakeBorder(ref, r, r, r, r, cv2.BORDER_REFLECT)
    best = np.full((h, w), np.inf, np.float32)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            best = np.minimum(best, np.abs(channel - ref_pad[dy:dy + h, dx:dx + w]))
    return best


def _gradient_attenuation_map(median_gray: np.ndarray) -> np.ndarray:
    """1 in flat regions, down to (1 - GRADIENT_ATTENUATION) at the
    sharpest content edges -- suppresses alignment-residual false flares."""
    gm = cv2.GaussianBlur(median_gray.astype(np.float32), (0, 0), 2)
    gx = cv2.Sobel(gm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gm, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    return 1.0 - GRADIENT_ATTENUATION * np.clip(
        gmag / (np.percentile(gmag, GRADIENT_PERCENTILE) + 1e-6), 0, 1)


def build_bank(frames: list) -> TemporalBank:
    """frames: list of registered BGR canonical-size frames (video.ingest.IngestResult.frames)."""
    if len(frames) < 3:
        raise ValueError(f"need >=3 registered frames, got {len(frames)}")

    stack_bgr = np.stack(frames).astype(np.float32)
    median_bgr = np.median(stack_bgr, axis=0).astype(np.uint8)

    grays = _photometric_normalize(
        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
    grays_stack = np.stack(grays)
    median_gray = np.median(grays_stack, axis=0)
    std_gray = grays_stack.std(axis=0)

    foil_mask = np.clip(std_gray / (std_gray.max() + 1e-6), 0, 1).astype(np.float32)
    # Normalized to the SAME [0,1] scale as foil_mask, so callers can route
    # with `foil_mask > foil_threshold` directly.
    foil_threshold = float(np.percentile(foil_mask, FOIL_VARIANCE_PERCENTILE))

    # ── defect_saliency: detect-then-aggregate, max-projected across frames ──
    atten = _gradient_attenuation_map(median_gray)
    f_median = frangi(median_gray / 255.0, sigmas=FRANGI_SIGMAS, black_ridges=True).astype(np.float32)
    # Patch-tolerant static-ridge reference: dilating the median's ridge map by
    # RIDGE_DILATE_RADIUS lets a static artwork/foil ridge that is displaced by
    # sub-pixel registration residual still cancel against it. Without this the
    # ridge channel flagged every artwork edge whose frangi response shifts
    # frame-to-frame -- the dominant clean-area false-positive source. This is
    # the ridge analogue of _patch_tolerant_delta's tolerance, but with a SMALLER
    # radius (1, not PATCH_TOLERANCE_RADIUS=2): per-defect controlled testing on
    # sample_3 back showed radius 2 over-cancels real scratches/creases that sit
    # on busy artwork (lost a scratch + a crease), while radius 1 keeps full
    # recall AND still cuts surface FPs ~59% (37->15). Provisional, n=1.
    _k = 2 * RIDGE_DILATE_RADIUS + 1
    f_median_dil = cv2.dilate(f_median, np.ones((_k, _k), np.float32))

    labs = [cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32) for f in frames]
    lab_stack = np.stack(labs)
    median_ab = np.median(lab_stack[..., 1:], axis=0)   # (H, W, 2)

    H, W = median_gray.shape
    r_ridge = np.zeros((H, W), np.float32)
    r_dev = np.zeros((H, W), np.float32)
    r_chroma = np.zeros((H, W), np.float32)
    for g, lab in zip(grays, labs):
        ridge = np.clip(
            frangi(g / 255.0, sigmas=FRANGI_SIGMAS, black_ridges=True).astype(np.float32) - f_median_dil,
            0, None)
        r_ridge = np.maximum(r_ridge, ridge)
        r_dev = np.maximum(r_dev, _patch_tolerant_delta(g, median_gray))
        d_a = _patch_tolerant_delta(lab[..., 1], median_ab[..., 0])
        d_b = _patch_tolerant_delta(lab[..., 2], median_ab[..., 1])
        r_chroma = np.maximum(r_chroma, np.sqrt(d_a * d_a + d_b * d_b))

    # NOTE: r_ridge is deliberately NOT multiplied by `atten`. The dev/chroma
    # channels measure |frame - median|, whose edge response is mostly alignment
    # residual, so attenuating them at content edges is correct. But a real
    # scratch/crease IS a ridge, often crossing or near artwork edges -- gradient
    # attenuation suppresses the actual defect signal there and craters recall
    # (sweep: ridge-atten drops front surface recall 8/8 -> 5/8). The patch-
    # tolerant static-ridge cancel above (f_median_dil) is what suppresses the
    # ridge channel's residual FPs, with no recall cost.
    r_dev *= atten
    r_chroma *= atten

    def _norm(x):
        return x / (np.percentile(x, RESPONSE_NORM_PERCENTILE) + 1e-6)

    # Do NOT clip each channel to [0,1] before combining: a clip caps each
    # channel's top ~0.5% at the same value 1.0, so the union of three
    # independently-saturated channels can exceed 1% of pixels at the cap --
    # which then makes percentile-99 thresholding find nothing (the cap IS
    # the 99th percentile). Keep the combined map unclipped; callers/display
    # clip only at the very end.
    saliency = np.maximum(np.maximum(_norm(r_ridge), _norm(r_dev)), _norm(r_chroma))
    saliency = np.clip(saliency, 0, None)

    return TemporalBank(
        median=median_bgr,
        foil_mask=foil_mask,
        defect_saliency=saliency.astype(np.float32),
        foil_threshold=foil_threshold,
    )


def save_debug(bank: TemporalBank, out_dir: str) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(f"{out_dir}/median.png", bank.median)
    cv2.imwrite(f"{out_dir}/foil_mask.png",
                cv2.applyColorMap((bank.foil_mask * 255).astype(np.uint8), cv2.COLORMAP_JET))
    sal_u8 = np.clip(bank.defect_saliency * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{out_dir}/defect_saliency.png",
                cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET))
    heat = cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET)
    cv2.imwrite(f"{out_dir}/defect_saliency_on_median.png",
                cv2.addWeighted(bank.median, 0.45, heat, 0.55, 0))
