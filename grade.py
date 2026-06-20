"""
End-to-end v2 card grader: short tilt video in, PSA-style grade out.

No VLM, no DB-template dependency (reference is optional, only helps the
non-foil surface cross-check -- see handoff_pipeline_revamp.md).

Usage:
    python grade.py samples/sample_3 --face front
    python grade.py samples/sample_3 --face front --reference samples/sample_3/reference_hires.png
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2

from video.temporal import save_debug
from video.ingest import CANONICAL_SIZE
from video_pipeline import grade_video


def _find_video(sample_dir: Path, face: str) -> Path | None:
    for name in (f"{face}_no_palm.MOV", f"{face}.MOV", f"{face}.mp4", f"{face}_no_palm.mp4"):
        p = sample_dir / name
        if p.exists():
            return p
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sample_dir", type=Path)
    p.add_argument("--face", choices=("front", "back"), default="front")
    p.add_argument("--reference", type=Path,
                   help="optional reference image (TCG render or canonical back) "
                        "for the non-foil surface cross-check")
    args = p.parse_args()

    video_path = _find_video(args.sample_dir, args.face)
    if video_path is None:
        sys.exit(f"no {args.face}.MOV / {args.face}_no_palm.MOV found in {args.sample_dir}")

    reference = None
    ref_path = args.reference or next(
        (p for p in (args.sample_dir / "reference_hires.png", args.sample_dir / "reference.png") if p.exists()),
        None)
    if ref_path is None and args.face == "back":
        from phase3.reference import get_back_reference
        back_ref = get_back_reference()
        if back_ref.exists():
            ref_path = back_ref
        else:
            sys.exit(f"back reference missing: {back_ref} (Step 6 input #2) -- not hacking around it")
    if ref_path is not None:
        ref_img = cv2.imread(str(ref_path))
        if ref_img is not None:
            reference = cv2.resize(ref_img, CANONICAL_SIZE)

    print(f"Ingesting {video_path} ...")
    result = grade_video(str(video_path), face=args.face, reference=reference)

    debug_dir = args.sample_dir / "debug" / f"{args.face}_temporal"
    save_debug(result["bank"], str(debug_dir))

    final = result["final"]
    print(f"\n{'='*50}")
    print(f"  PSA grade:    {final['grade_psa']}   (limiting: {final['limiting_pillar']})")
    print(f"  Score /1000:  {final['score_1000']}  (-> grade {final['grade_score']})")
    print(f"{'='*50}")
    for pillar, grade in final["pillar_grades"].items():
        print(f"  {pillar:10}: {grade}")
    print(f"\nDebug: {debug_dir}")


if __name__ == "__main__":
    main()
