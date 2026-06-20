"""
Fetch a pristine reference image from the Pokémon TCG API.
Card identity is a `set_id-number` string, e.g. 'ex15-46'.

CRITICAL: the API's `large` (hires) URL is RGB only — the rounded-corner
regions were rendered with fabricated yellow border content, so the actual
card-edge geometry is lost.  The `small` URL is RGBA — the alpha channel
exactly defines the card boundary (rounded corners are alpha=0).

We fetch BOTH, upscale the small's alpha to large's resolution, and
composite the large RGB onto a white background using that alpha.  Result:
a high-resolution reference with the small's accurate card boundary.
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import requests

TCG_API = "https://api.pokemontcg.io/v2/cards/{card_id}"

# Card backs are identical across the whole Pokemon TCG -- one canonical image,
# no per-card TCG fetch needed (unlike fronts).
BACK_REFERENCE_PATH = Path(__file__).resolve().parent.parent / "references" / "pokemon_back.jpg"


def get_back_reference() -> Path:
    """Path to the canonical card-back reference image. Caller is responsible
    for checking existence -- handoff_pipeline_revamp.md Step 6 says to mark
    back grading blocked, not hack a reference, if this is missing."""
    return BACK_REFERENCE_PATH


def _composite_with_alpha(large_rgb: np.ndarray, small_rgba: np.ndarray
                          ) -> np.ndarray:
    """
    Build a high-res reference by compositing the large RGB onto white using
    the small image's alpha (upscaled to match large's dimensions).

    Where small's alpha = 0 (rounded corner exterior, anything outside the
    card), the output is white.  Where alpha = 255, output uses large's
    pixel.  Anti-aliased edges blend.
    """
    if small_rgba.ndim != 3 or small_rgba.shape[2] != 4:
        # No alpha in small — return large unchanged.
        return large_rgb
    h_l, w_l = large_rgb.shape[:2]
    alpha_small = small_rgba[..., 3]
    alpha_large = cv2.resize(alpha_small, (w_l, h_l), interpolation=cv2.INTER_LINEAR)
    alpha_f = (alpha_large.astype(np.float32) / 255.0)[..., None]
    white = np.full_like(large_rgb, 255)
    composited = (large_rgb.astype(np.float32) * alpha_f
                  + white.astype(np.float32) * (1.0 - alpha_f))
    return composited.clip(0, 255).astype(np.uint8)


def fetch_reference(card_id: str, dst: Path) -> dict:
    """
    Downloads the reference image for `card_id` to `dst` (if not cached).

    Always composites onto white using the small image's alpha mask so the
    saved reference has accurate card geometry — including rounded corners
    and the actual paper edge — that the large-only URL is missing.

    Returns the card metadata dict.
    """
    if dst.exists():
        return {'id': card_id, 'cached': True}

    r = requests.get(TCG_API.format(card_id=card_id), timeout=30)
    r.raise_for_status()
    data = r.json()['data']

    large_bytes = requests.get(data['images']['large'], timeout=60).content
    small_bytes = requests.get(data['images']['small'], timeout=60).content

    large = cv2.imdecode(np.frombuffer(large_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    small = cv2.imdecode(np.frombuffer(small_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if large is None or small is None:
        raise RuntimeError(f"failed to decode reference images for {card_id}")
    # Strip large's alpha if it happens to have one — we want plain BGR.
    if large.ndim == 3 and large.shape[2] == 4:
        large = cv2.cvtColor(large, cv2.COLOR_BGRA2BGR)

    composited = _composite_with_alpha(large, small)

    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), composited)
    return data
