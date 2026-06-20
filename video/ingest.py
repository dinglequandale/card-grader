"""
Video ingest: short tilt video -> a stack of canonical-frame, mutually
registered card images, ready for the temporal bank (video/temporal.py).

Recipe (VALIDATED on real Fearow footage by probe_temporal.py /
probe_detect_aggregate.py -- see memory/v2_temporal_validation.md):
  1. Sample N frames evenly across the clip, keep the K sharpest
     (Laplacian variance) to drop motion blur.
  2. Per kept frame: phase1.detect.detect_card -> perspective warp to the
     canonical frame size.
  3. Register every warped frame onto the single SHARPEST warped frame via
     ORB feature matching + RANSAC homography. ORB sits on stable structure
     (text/artwork/border), so foil shimmer doesn't drag it the way
     intensity-based ECC does.

The global ORB homography is sufficient when capture follows CAPTURE.md
(flat, grippy, stable) -- do not add local/grid refinement (proven
unnecessary; see v2_temporal_validation.md "Edge falloff").
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

from phase1.detect import detect_card

CANONICAL_SIZE = (600, 825)   # (W, H) -- the pipeline's canonical card frame
N_SAMPLE = 32
N_KEEP = 24
MIN_REGISTERED = 3
ORB_FEATURES = 3000
ORB_MATCH_RATIO = 0.75
MIN_GOOD_MATCHES = 15
RANSAC_REPROJ_THRESH = 3.0


@dataclass
class IngestResult:
    frames: list           # list of BGR np.ndarray, canonical size, registered
    size: tuple             # (W, H)
    n_sampled: int
    n_sharp_kept: int
    n_detected: int
    n_registered: int
    detect_methods: dict     # method name -> count
    anchor_index: int        # index into `frames` of the registration anchor


def _laplacian_var(img: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _read_sampled_frames(path: str, n_sample: int) -> list:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"could not read frame count from {path}")
    frames = []
    for i in np.linspace(0, total - 1, min(n_sample, total)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames


def _detect_and_warp(frames: list, size: tuple) -> tuple:
    """Returns (warped_frames, n_detected, method_counts)."""
    W, H = size
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    warped, methods = [], {}
    for f in frames:
        corners, method = detect_card(f)
        if corners is None:
            continue
        methods[method] = methods.get(method, 0) + 1
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        warped.append(cv2.warpPerspective(f, M, (W, H)))
    return warped, len(warped), methods


def _register_orb(anchor_bgr: np.ndarray, img: np.ndarray, size: tuple,
                   orb: cv2.ORB) -> tuple:
    """Feature-based alignment of `img` onto `anchor_bgr`. Returns (registered, moved)."""
    a = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None:
        return img, False
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pair in bf.knnMatch(db, da, k=2):
        if len(pair) == 2 and pair[0].distance < ORB_MATCH_RATIO * pair[1].distance:
            good.append(pair[0])
    if len(good) < MIN_GOOD_MATCHES:
        return img, False
    src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    Mh, _ = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    if Mh is None:
        return img, False
    return cv2.warpPerspective(img, Mh, size), True


def ingest_video(path: str, size: tuple = CANONICAL_SIZE,
                  n_sample: int = N_SAMPLE, n_keep: int = N_KEEP) -> IngestResult:
    """Extract, detect+warp, and ORB-register a short tilt video into a
    stack of canonical-frame card images ready for temporal aggregation.

    Raises ValueError if fewer than MIN_REGISTERED frames survive detection
    (capture/aspect problem -- report, don't tune around it; see handoff
    section 6).
    """
    frames = _read_sampled_frames(path, n_sample)
    if len(frames) < n_keep:
        n_keep = len(frames)
    sharp_sorted = sorted(frames, key=_laplacian_var, reverse=True)
    kept = sharp_sorted[:n_keep]

    warped, n_detected, methods = _detect_and_warp(kept, size)
    if len(warped) < MIN_REGISTERED:
        raise ValueError(
            f"only {len(warped)}/{len(kept)} frames detected the card -- "
            f"capture or aspect-gate problem, not a registration problem. "
            f"detect methods so far: {methods}"
        )

    anchor_idx = int(np.argmax([_laplacian_var(w) for w in warped]))
    anchor = warped[anchor_idx]
    orb = cv2.ORB_create(ORB_FEATURES)
    registered, n_moved = [], 0
    for w in warped:
        reg, moved = _register_orb(anchor, w, size, orb)
        registered.append(reg)
        n_moved += int(moved)

    return IngestResult(
        frames=registered,
        size=size,
        n_sampled=len(frames),
        n_sharp_kept=len(kept),
        n_detected=n_detected,
        n_registered=n_moved,
        detect_methods=methods,
        anchor_index=anchor_idx,
    )
