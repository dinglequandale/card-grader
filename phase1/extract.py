from __future__ import annotations
import cv2
import numpy as np
from typing import Dict

CORNER_SIZE = 70   # px square at each corner
EDGE_WIDTH = 20    # px wide strip along each side (excluding corner overlap)


def extract_rois(warped: np.ndarray) -> Dict:
    """
    Slice the warped card into fixed analysis regions.

    Returned dict:
        corners:              {'TL', 'TR', 'BR', 'BL'} — 70x70 px patches
        edges:                {'top', 'bottom', 'left', 'right'} — 20px strips
        surface:              inner rectangle (all 4 borders removed)
        edge_reference_colors: per-edge LAB median sampled from the middle 60%
                               of each strip (ground truth for that edge's color)
    """
    h, w = warped.shape[:2]
    c = CORNER_SIZE
    e = EDGE_WIDTH

    corners = {
        'TL': warped[:c, :c].copy(),
        'TR': warped[:c, w - c:].copy(),
        'BR': warped[h - c:, w - c:].copy(),
        'BL': warped[h - c:, :c].copy(),
    }

    # Edge strips run between the corner patches so corners don't bias them
    edges = {
        'top':    warped[:e,      c:w - c].copy(),
        'bottom': warped[h - e:,  c:w - c].copy(),
        'left':   warped[c:h - c, :e].copy(),
        'right':  warped[c:h - c, w - e:].copy(),
    }

    surface = warped[e:h - e, e:w - e].copy()

    ref_colors = _sample_edge_reference_colors(edges)

    return {
        'corners': corners,
        'edges': edges,
        'surface': surface,
        'edge_reference_colors': ref_colors,
    }


def _sample_edge_reference_colors(edges: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    For each edge strip, sample the middle 30% of its length and return the
    LAB median color as a float array [L, a, b] (OpenCV uint8 LAB scale).
    Sampling the middle avoids corner damage contaminating the reference.
    """
    ref = {}
    for side, strip in edges.items():
        sh, sw = strip.shape[:2]
        if side in ('top', 'bottom'):
            lo, hi = sw // 5, 4 * sw // 5
            sample = strip[:, lo:hi]
        else:
            lo, hi = sh // 5, 4 * sh // 5
            sample = strip[lo:hi, :]
        lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)
        ref[side] = np.median(lab.reshape(-1, 3), axis=0)  # shape (3,)
    return ref
