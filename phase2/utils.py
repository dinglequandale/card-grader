from __future__ import annotations
import cv2
import numpy as np

BG_SAMPLE_SIZE = 10  # px square sampled from each warped-image corner for background colour


def lab_to_float(lab: np.ndarray) -> np.ndarray:
    """Convert OpenCV uint8 LAB to float L*a*b* (L 0-100, a/b ±127)."""
    out = lab.astype(np.float32)
    out[..., 0] = out[..., 0] * (100.0 / 255.0)
    out[..., 1] = out[..., 1] - 128.0
    out[..., 2] = out[..., 2] - 128.0
    return out


def delta_e(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """
    ΔE76 between two arrays of OpenCV uint8 LAB pixels.
    Broadcasts: one arg can be (3,) against the other (..., 3).
    """
    fa = lab_to_float(np.asarray(lab_a, dtype=np.float32))
    fb = lab_to_float(np.asarray(lab_b, dtype=np.float32))
    return np.sqrt(np.sum((fa - fb) ** 2, axis=-1))


def sample_background_lab(warped: np.ndarray) -> np.ndarray:
    """
    Estimate background colour from the 4 corner tips of the warped image.
    Card corners are rounded, so the warped-image corners are always background.
    Returns (3,) float array in OpenCV uint8 LAB encoding.
    """
    h, w = warped.shape[:2]
    n = BG_SAMPLE_SIZE
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    samples = [
        lab[:n, :n].reshape(-1, 3),
        lab[:n, w - n:].reshape(-1, 3),
        lab[h - n:, w - n:].reshape(-1, 3),
        lab[h - n:, :n].reshape(-1, 3),
    ]
    return np.median(np.vstack(samples), axis=0)


def background_mask(lab_img: np.ndarray, bg_lab: np.ndarray, threshold: float) -> np.ndarray:
    """
    Boolean mask — True where pixel is background.
    lab_img: (..., 3) OpenCV uint8 LAB.  bg_lab: (3,) same encoding.
    """
    return delta_e(lab_img, bg_lab) < threshold
