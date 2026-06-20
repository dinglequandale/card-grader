"""
Surface pillar: connected-components on `defect_saliency` (video/temporal.py),
foil-routed.

  non-foil regions: keep CC blobs as-is. If a reference is available
                     (DB template / canonical back), cross-check via
                     phase3.region_characterizer to drop intended print
                     (the characterizer's edge_distance + diff_edge_corr
                     rules also kill frame-border alignment-residual FPs).
                     No reference -> keep all (degrade gracefully).
  foil regions:      template can't apply (no foil in the TCG render); a
                     defect here is a local disruption of the smooth
                     shimmer, which the saliency substrate already captures.
                     Never blanket-mask foil -- never drop a blob just
                     because it's in a foil region.

Severity bucketing from saliency magnitude is a coarse first cut (the
substrate's raw response, not a calibrated physical unit); refine against
the eval, not by eye.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional

from video.temporal import TemporalBank

SALIENCY_PERCENTILE = 97.0     # threshold for "defect" pixels. Tuned against
                                # sample_3/front: 99.0 -> 2/8 surface recall,
                                # 97.0 -> 8/8 (the fine scratches/marks need
                                # the lower floor). Reference cross-check
                                # (region_characterizer) trims ~25% of the
                                # resulting blobs without losing recall --
                                # always prefer passing `reference` in.
MIN_BLOB_AREA = 8               # px, drop noise specks
SMALL_BLOB_MIN_AREA = 4         # floor for the peak-aware exception below; under
                                 # this is single-speck territory, never kept
SMALL_BLOB_PEAK_KEEP = 1.1      # keep a 4-7px blob only if its raw-saliency peak
                                 # is this strong. The back's sev4 scratch is a
                                 # 5px blob, peak 1.16 -- recovered here; the 1px
                                 # sev2 mark (area 1) and weak specks stay dropped.
BORDER_MARGIN_FRAC = 0.02       # drop blobs whose centroid sits in the outer
                                 # margin -- that's the perimeter pillar's job,
                                 # not surface's (avoids double-counting)
LOCAL_BG_KERNEL = 151           # white top-hat kernel (px, canonical frame).
                                 # The global percentile alone is dominated by
                                 # busy artwork (the back's swirl): its high
                                 # local saliency floor buries small defects
                                 # elsewhere. A white top-hat subtracts the
                                 # slowly-varying LOCAL background so the
                                 # percentile becomes effectively local-adaptive
                                 # (same lesson the perimeter band-percentile
                                 # already embodies). Kernel > the biggest
                                 # defect we keep (creases ~140px) so those
                                 # survive; card-scale structure is removed.
                                 # Static artwork is already cancelled upstream
                                 # in the detect-then-aggregate saliency, so
                                 # what top-hat strips here is residual, not signal.

# saliency-magnitude -> severity 1-5 (coarse; tune against eval)
_SEVERITY_BUCKETS = [(1.1, 1), (1.3, 2), (1.6, 3), (2.0, 4)]  # else 5


def _severity_from_magnitude(peak: float) -> int:
    for cap, sev in _SEVERITY_BUCKETS:
        if peak <= cap:
            return sev
    return 5


def _blob_type(saliency_peak_ratio_to_area: float) -> str:
    """Coarse shape-based type hint: thin/elongated -> scratch-ish, blobby ->
    mark/stain. Refined typing (crease vs scratch) needs the Frangi channel
    split, deferred until the eval shows it matters."""
    return "scratch" if saliency_peak_ratio_to_area > 0.15 else "mark"


def detect_surface_defects(bank: TemporalBank,
                            reference: Optional[np.ndarray] = None,
                            percentile: float = SALIENCY_PERCENTILE) -> List[dict]:
    """Returns a list of verdict dicts compatible with
    phase3.grade_synthesis.surface_grade_from_verdicts:
      {region_idx, bbox, verdict: {defect, type, severity, reasoning}}
    """
    sal = bank.defect_saliency
    H, W = sal.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LOCAL_BG_KERNEL, LOCAL_BG_KERNEL))
    sal_local = cv2.morphologyEx(sal, cv2.MORPH_TOPHAT, k)  # local-adaptive floor
    thr = float(np.percentile(sal_local, percentile))
    binm = cv2.morphologyEx((sal_local > thr).astype(np.uint8) * 255,
                             cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binm, 8)

    margin = int(min(H, W) * BORDER_MARGIN_FRAC)
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        peak = float(sal[y:y + h, x:x + w].max())
        # Keep a blob if it's large enough, OR if it's a small-but-intense peak
        # (a sharp strong peak is a real scratch/mark; a weak small blob is
        # noise). Recovers e.g. the sev4 back scratch -- a 5px blob with a strong
        # peak -- without blanket-lowering MIN_BLOB_AREA (which floods specks).
        if area < MIN_BLOB_AREA and not (area >= SMALL_BLOB_MIN_AREA and peak >= SMALL_BLOB_PEAK_KEEP):
            continue
        cx, cy = x + w / 2, y + h / 2
        if cx < margin or cy < margin or cx > W - margin or cy > H - margin:
            continue  # perimeter pillar's territory
        regions.append({"bbox": [int(x), int(y), int(w), int(h)], "_area": int(area), "_peak": peak})

    if not regions:
        return []

    keep_mask = np.ones(len(regions), dtype=bool)
    if reference is not None:
        from phase3.region_characterizer import characterize_regions
        foil_frac = np.array([
            bank.foil_mask[y:y + h, x:x + w].mean()
            for (x, y, w, h) in (r["bbox"] for r in regions)
        ])
        non_foil_idx = [i for i, f in enumerate(foil_frac) if f <= bank.foil_threshold]
        if non_foil_idx:
            non_foil_regions = [regions[i] for i in non_foil_idx]
            results = characterize_regions(bank.median, reference, non_foil_regions, (H, W))
            for i, res in zip(non_foil_idx, results):
                keep_mask[i] = res.keep

    verdicts = []
    for idx, region in enumerate(regions):
        if not keep_mask[idx]:
            continue
        x, y, w, h = region["bbox"]
        peak = region["_peak"]
        verdicts.append({
            "region_idx": idx,
            "bbox": region["bbox"],
            "verdict": {
                "defect": True,
                "type": _blob_type(peak / max(region["_area"], 1)),
                "severity": _severity_from_magnitude(peak),
                "reasoning": f"detect-then-aggregate saliency peak={peak:.2f}, area={region['_area']}px",
            },
        })
    return verdicts
