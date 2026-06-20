"""
Grade synthesis — combine per-pillar VLM/algorithmic outputs into a single
PSA-scale grade plus a 0-1000 weighted score.

Two views of the same data:
  - PSA-style grade (min of pillars, with half-grade bumps when the limiting
    pillar is the sole limiter) — matches how PSA actually grades.
  - 0-1000 weighted score — finer granularity, accumulates wear across
    pillars instead of being dominated by the worst one.

Per-pillar grade ceilings are derived from defect counts + severities + types,
following the grading-schema.md cheat-sheets.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

# ── Pillar weights for the 0-1000 score ──────────────────────────────────────
# Corners weighted heaviest because PSA penalises corner wear most aggressively.
WEIGHTS = {
    "corners":   300,
    "edges":     250,
    "surface":   250,
    "centering": 200,
}
TOTAL_WEIGHT = sum(WEIGHTS.values())   # = 1000


# ── Corners pillar ────────────────────────────────────────────────────────────
# Inputs: VLM edge verdicts where side starts with 'corner_'
# Each verdict has type ('corner_wear', 'whitening', etc.) and severity 1-5

# (max_severity, n_affected_corners) → grade ceiling
# n_affected ∈ {1, 2, 3+} buckets
_CORNERS_TABLE: List[Tuple[int, int, int]] = [
    # (max_sev, max_n_affected, grade)
    (0, 4, 10),
    (1, 1,  9),
    (1, 4,  8),
    (2, 2,  8),
    (2, 4,  7),
    (3, 2,  7),
    (3, 4,  6),
    (4, 1,  6),
    (4, 4,  5),
    (5, 1,  4),
    (5, 2,  3),
    (5, 4,  2),
]


def _lookup(table, max_sev, n_affected):
    for sev_cap, n_cap, grade in table:
        if max_sev <= sev_cap and n_affected <= n_cap:
            return grade
    return 1


def corners_grade_from_verdicts(edge_verdicts: List[Dict]) -> Dict:
    corner = [v for v in edge_verdicts if v.get("side", "").startswith("corner_")]
    if not corner:
        return {"grade": 10, "n_affected": 0, "max_severity": 0, "defects": []}
    sides = {v["side"] for v in corner}
    max_sev = max(v.get("severity", 0) for v in corner)
    grade = _lookup(_CORNERS_TABLE, max_sev, len(sides))
    return {
        "grade":         grade,
        "n_affected":    len(sides),
        "max_severity":  max_sev,
        "defects":       corner,
    }


# ── Edges pillar ──────────────────────────────────────────────────────────────
# Same shape as corners but slightly more forgiving (edges accumulate wear
# more universally than corners).
_EDGES_TABLE: List[Tuple[int, int, int]] = [
    (0, 4, 10),
    (1, 2,  9),
    (1, 4,  8),
    (2, 2,  8),
    (2, 4,  7),
    (3, 2,  7),
    (3, 4,  6),
    (4, 2,  6),
    (4, 4,  5),
    (5, 1,  4),
    (5, 2,  3),
    (5, 4,  2),
]


def edges_grade_from_verdicts(edge_verdicts: List[Dict]) -> Dict:
    side_verdicts = [v for v in edge_verdicts
                     if v.get("side", "") in ("top", "bottom", "left", "right")]
    if not side_verdicts:
        return {"grade": 10, "n_affected": 0, "max_severity": 0, "defects": []}
    sides = {v["side"] for v in side_verdicts}
    max_sev = max(v.get("severity", 0) for v in side_verdicts)
    grade = _lookup(_EDGES_TABLE, max_sev, len(sides))
    return {
        "grade":         grade,
        "n_affected":    len(sides),
        "max_severity":  max_sev,
        "defects":       side_verdicts,
    }


# ── Surface pillar ────────────────────────────────────────────────────────────
# Inputs: VLM filter verdicts on PatchCore regions
# Each verdict: {region_idx, bbox, verdict: {defect, type, severity, reasoning}}
# Approach: only confirmed defects (defect=true) contribute. Per-type weight
# reflects PSA's severity table (crease > stain > scratch > whitening).

_TYPE_WEIGHT = {
    "crease":        3.0,   # creases permanently alter cardstock — heaviest
    "chipping":      2.5,
    "stain":         2.0,
    "mark":          2.0,
    "scratch":       1.5,
    "whitening":     1.5,
    "print_defect":  1.0,
}

# (max-defect-penalty, max-total-penalty) → grade
# Per-defect penalty = severity × type_weight (so max single-defect = 5 × 3 = 15)
# Per-card total penalty accumulates all confirmed defects.
_SURFACE_TABLE: List[Tuple[float, float, int]] = [
    # (max_defect_penalty, max_total_penalty, grade)
    (0.0,    0.0, 10),
    (1.5,    3.0,  9),    # one tiny defect → PSA 9
    (3.0,    6.0,  8),    # small visible scratch / faint mark
    (6.0,   12.0,  7),    # clearly visible defect or 2-3 minor ones
    (9.0,   18.0,  6),    # significant defect or multiple small
    (12.0,  25.0,  5),
    (15.0,  40.0,  4),    # crease at sev 4
    (15.0,  60.0,  3),    # major crease or many defects
    (15.0,  90.0,  2),
]


def surface_grade_from_verdicts(surface_verdicts: List[Dict]) -> Dict:
    confirmed = [v for v in surface_verdicts
                 if (v.get("verdict") or {}).get("defect")]
    if not confirmed:
        return {"grade": 10, "n_defects": 0, "max_penalty": 0.0,
                "total_penalty": 0.0, "defects": []}

    per_defect = []
    for v in confirmed:
        vd = v["verdict"]
        w  = _TYPE_WEIGHT.get(vd.get("type"), 1.0)
        p  = vd.get("severity", 0) * w
        per_defect.append({**v, "_penalty": p})

    max_p   = max(d["_penalty"] for d in per_defect)
    total_p = sum(d["_penalty"] for d in per_defect)

    grade = 1
    for cap_def, cap_total, g in _SURFACE_TABLE:
        if max_p <= cap_def and total_p <= cap_total:
            grade = g
            break
    return {
        "grade":         grade,
        "n_defects":     len(per_defect),
        "max_penalty":   round(max_p, 2),
        "total_penalty": round(total_p, 2),
        "defects":       per_defect,
    }


# ── Aggregation: 0-1000 score + PSA grade ─────────────────────────────────────

def _pillar_penalty(grade: int, weight: int) -> float:
    """Linear: grade 10 → 0 deduction, grade 1 → full weight deducted."""
    return (10 - grade) / 9.0 * weight


# 0-1000 score → PSA 1-10 grade. PSA does NOT issue 9.5.
_SCORE_TO_GRADE: List[Tuple[float, float]] = [
    (950.0, 10.0),
    (900.0,  9.0),    # no 9.5
    (850.0,  8.5),
    (800.0,  8.0),
    (750.0,  7.5),
    (700.0,  7.0),
    (650.0,  6.5),
    (600.0,  6.0),
    (550.0,  5.5),
    (500.0,  5.0),
    (450.0,  4.5),
    (400.0,  4.0),
    (350.0,  3.5),
    (300.0,  3.0),
    (250.0,  2.5),
    (200.0,  2.0),
    (150.0,  1.5),
    (0.0,    1.0),
]


def score_to_grade(score: float) -> float:
    for threshold, grade in _SCORE_TO_GRADE:
        if score >= threshold:
            return grade
    return 1.0


def aggregate(centering: Dict,
              corners:   Dict,
              edges:     Dict,
              surface:   Dict,
              ) -> Dict:
    """
    Take per-pillar dicts (each with at least 'grade' key 1-10) and produce
    final grade outputs.

    Returns:
      {
        'score_1000': float,        # weighted total penalty subtracted from 1000
        'grade_score': float,       # PSA grade derived from score_1000
        'grade_psa':   float,       # PSA-style min-of-pillars + half-grade bump
        'limiting_pillar': str,
        'pillar_grades': dict,
        'pillar_penalties': dict,
      }
    """
    pillar_grades = {
        "centering": centering.get("centering_grade", centering.get("grade", 10)),
        "corners":   corners.get("grade", 10),
        "edges":     edges.get("grade", 10),
        "surface":   surface.get("grade", 10),
    }
    pillar_penalties = {
        k: round(_pillar_penalty(g, WEIGHTS[k]), 1)
        for k, g in pillar_grades.items()
    }
    score_1000  = round(TOTAL_WEIGHT - sum(pillar_penalties.values()), 1)
    grade_score = score_to_grade(score_1000)

    # PSA-style limiting pillar
    limiting = min(pillar_grades, key=pillar_grades.get)
    base     = pillar_grades[limiting]
    grade_psa = float(base)
    if base < 10 and base != 9:
        others_clear_next = all(v >= base + 1
                                for k, v in pillar_grades.items() if k != limiting)
        if others_clear_next:
            grade_psa = base + 0.5

    return {
        "score_1000":       score_1000,
        "grade_score":      grade_score,
        "grade_psa":        grade_psa,
        "limiting_pillar":  limiting,
        "pillar_grades":    pillar_grades,
        "pillar_penalties": pillar_penalties,
    }
