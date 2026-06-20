from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Optional

# Width of the outer strip sampled for border classification (~2.4% of card height)
BORDER_STRIP_PX = 15

# Mean per-channel LAB std above which a side is considered artwork (not solid border).
# Solid borders — even metallic silver/gold — stay below ~20.
# Artwork seeping to the edge typically exceeds 30+.
VARIANCE_THRESHOLD = 25.0


def detect_holo(warped: np.ndarray) -> Dict:
    """
    Detect whether the card's surface finish is holographic.

    PLACEHOLDER — not yet calibrated. Returns is_holo=False with confidence='none'
    until a holo sample cohort is available to fit a threshold against. The
    function shape is fixed so downstream code can already consume the field.

    Holo detection signal candidates (to evaluate once samples arrive):
      - HSV V-channel high-pass energy across the inner card (shimmer ⇒ high)
      - Per-pixel saturation gradient std in expected-flat regions
      - Fraction of near-saturated bright pixels in a non-glare-masked inner crop

    Returns:
        {'is_holo': bool, 'confidence': 'high'|'medium'|'low'|'none', 'signal': float}
    """
    return {
        'is_holo':    False,
        'confidence': 'none',
        'signal':     0.0,
    }


def classify_card_type(warped: np.ndarray,
                       holo_override: Optional[bool] = None) -> Dict:
    """
    Determine whether the front face has a solid border or artwork-to-edge,
    and whether the card's surface is holographic.

    `holo_override`: if not None, sets `is_holo` directly and skips auto-detect.
    Used when the user (or a runner CLI flag) supplies the holo status manually
    — necessary while the auto-detector is uncalibrated.

    Returns:
        {
          'type':           'bordered' | 'full_art',
          'side_variances': {'top': float, 'bottom': float, 'left': float, 'right': float},
          'threshold':      float,
          'is_holo':        bool,
          'holo_confidence':'high'|'medium'|'low'|'none'|'manual',
        }

    Only meaningful for the front face — the back is always 'bordered' and
    non-holo (Pokémon card backs are always matte).
    """
    h, w = warped.shape[:2]
    s = BORDER_STRIP_PX

    strips = {
        'top':    warped[:s, :],
        'bottom': warped[h - s:, :],
        'left':   warped[:, :s],
        'right':  warped[:, w - s:],
    }

    variances = {}
    for side, strip in strips.items():
        lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Mean standard deviation across L, a, b channels independently
        variances[side] = float(np.mean([np.std(lab[:, :, c]) for c in range(3)]))

    card_type = 'full_art' if max(variances.values()) > VARIANCE_THRESHOLD else 'bordered'

    if holo_override is not None:
        is_holo = bool(holo_override)
        holo_conf = 'manual'
    else:
        h_result = detect_holo(warped)
        is_holo = h_result['is_holo']
        holo_conf = h_result['confidence']

    return {
        'type':            card_type,
        'side_variances':  variances,
        'threshold':       VARIANCE_THRESHOLD,
        'is_holo':         is_holo,
        'holo_confidence': holo_conf,
    }
