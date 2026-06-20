"""
v2 pipeline orchestrator: video in, PSA-style grade out. No VLM, no DB-
template dependency by default (reference is optional and only helps the
non-foil surface cross-check / back grading -- see handoff_pipeline_revamp.md).

    CAPTURE -> INGEST -> TEMPORAL -> {CENTERING, PERIMETER, SURFACE} -> GRADE

Centering and perimeter are reference-free (phase1/phase2, phase3.edge_inspect).
Surface is detect-then-aggregate on the temporal saliency (video/surface.py),
optionally cross-checked against a reference image for the non-foil region.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from phase1.classify import classify_card_type
from phase1.extract import extract_rois
from phase2.centering import measure_centering
from phase3.grade_synthesis import (
    aggregate,
    corners_grade_from_verdicts,
    edges_grade_from_verdicts,
    surface_grade_from_verdicts,
)
from video.ingest import ingest_video, CANONICAL_SIZE
from video.temporal import build_bank, TemporalBank
from video.perimeter import (detect_edge_defects, detect_corner_defects,
                             detect_edge_surface_defects, merge_edge_runs)
from video.surface import detect_surface_defects


def grade_video(video_path: str, face: str = "front",
                 reference: Optional[np.ndarray] = None) -> dict:
    """Run the full v2 pipeline on a single face's tilt video.

    Returns a dict with the temporal bank, per-pillar detections (with
    canonical-frame bboxes, for eval overlap matching), pillar grades, and
    the final aggregate grade.
    """
    ingest = ingest_video(video_path)
    bank = build_bank(ingest.frames)

    classification = classify_card_type(bank.median) if face == "front" else {"type": "bordered"}
    rois = extract_rois(bank.median)
    centering = measure_centering({
        "face": face, "warped": bank.median, "rois": rois, "classification": classification,
    })

    edge_verdicts_raw = detect_edge_defects(bank, bank.median.shape[1::-1])
    edge_verdicts_raw += detect_edge_surface_defects(bank)
    corner_verdicts = detect_corner_defects(edge_verdicts_raw)   # needs start_pct
    edge_verdicts = merge_edge_runs(edge_verdicts_raw, bank.median.shape[1::-1])
    surface_verdicts = detect_surface_defects(bank, reference=reference)

    corners = corners_grade_from_verdicts(corner_verdicts)
    edges = edges_grade_from_verdicts(edge_verdicts)
    surface = surface_grade_from_verdicts(surface_verdicts)

    final = aggregate(centering, corners, edges, surface)

    return {
        "face": face,
        "bank": bank,
        "ingest": ingest,
        "centering": centering,
        "corners": corners,
        "edges": edges,
        "surface": surface,
        "edge_verdicts": edge_verdicts,
        "corner_verdicts": corner_verdicts,
        "surface_verdicts": surface_verdicts,
        "final": final,
    }


def _detections_for_eval(result: dict) -> list:
    """Flatten all pillar detections into dicts, canonical 600x825 frame, for
    eval/run_eval.py overlap matching against annotations (recall + precision).
    Each: {"bbox": (x,y,w,h), "pillar": str, "severity": int|None}."""
    dets = []
    for v in result["surface_verdicts"]:
        dets.append({"bbox": tuple(v["bbox"]), "pillar": "surface",
                     "severity": v["verdict"]["severity"]})
    for v in result["edge_verdicts"]:
        if "bbox" in v:
            dets.append({"bbox": tuple(v["bbox"]), "pillar": "edges",
                         "severity": v.get("severity")})
    for v in result["corner_verdicts"]:
        if "bbox" in v:
            dets.append({"bbox": tuple(v["bbox"]), "pillar": "corners",
                         "severity": v.get("severity")})
    return dets
