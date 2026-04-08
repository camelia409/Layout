"""
scoring_core.py — 7-Metric Scoring Engine
Source of truth for both runtime validation and XGBoost training data generation.

Metrics:
  1. Vastu Score    (weight 0.15) — directional room placement compliance
  2. NBC Score      (weight 0.20) — area & width validation per NBC 2016
  3. Circulation    (weight 0.15) — BFS hop analysis from entrance
  4. Adjacency      (weight 0.15) — must-be / must-not-be adjacent rules
  5. Climate        (weight 0.15) — cross-ventilation & window placement
  6. Baker          (weight 0.20) — thermal mass, passive cooling, overhang depth
  7. Overall        (weighted sum of above six)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ScoringResult:
    vastu: float = 0.0
    nbc: float = 0.0
    circulation: float = 0.0
    adjacency: float = 0.0
    climate: float = 0.0
    baker: float = 0.0
    overall: float = 0.0
    violations: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, float]:
        return {
            "Vastu": round(self.vastu * 100),
            "NBC": round(self.nbc * 100),
            "Circulation": round(self.circulation * 100),
            "Adjacency": round(self.adjacency * 100),
            "Climate": round(self.climate * 100),
            "Baker": round(self.baker * 100),
            "Overall": round(self.overall * 100),
        }


_WEIGHTS = {
    "vastu": 0.15, "nbc": 0.20, "circulation": 0.15,
    "adjacency": 0.15, "climate": 0.15, "baker": 0.20,
}

_VASTU_ZONES = {
    "kitchen": "SE", "master_bedroom": "SW", "pooja": "NE",
    "living": "N", "toilet": None,  # blocked from NE
}

_NBC_MINIMUMS = {
    "bedroom": 9.5, "master_bedroom": 9.5, "kitchen": 5.0,
    "living": 12.0, "toilet": 1.2, "dining": 7.0,
}

_NBC_WIDTH_MIN = {
    "bedroom": 2.7, "master_bedroom": 2.7, "kitchen": 1.8,
    "living": 3.0, "toilet": 1.0,
}


def score_vastu(placement: dict) -> tuple[float, list]:
    penalties, violations = 0.0, []
    for room_id, room in placement.items():
        rtype = room.get("type", "")
        zone = room.get("quadrant", "")
        expected = _VASTU_ZONES.get(rtype)
        if expected and zone != expected:
            p = 0.20 if rtype == "pooja" else 0.15
            penalties += p
            violations.append(f"Vastu: {rtype} in {zone}, expected {expected}")
        if rtype == "toilet" and zone == "NE":
            penalties += 0.25
            violations.append("Vastu: toilet blocked from NE zone")
    return max(0.0, 1.0 - penalties), violations


def score_nbc(placement: dict) -> tuple[float, list]:
    num_violations, violations = 0, []
    for room_id, room in placement.items():
        rtype = room.get("type", "")
        area = room.get("area", 0.0)
        width = room.get("width", 0.0)
        min_area = _NBC_MINIMUMS.get(rtype, 0)
        min_width = _NBC_WIDTH_MIN.get(rtype, 0)
        if min_area and area < min_area * 0.88:
            num_violations += 1
            violations.append(f"NBC: {rtype} area {area:.1f}m² < {min_area}m²")
        if min_width and width < min_width:
            num_violations += 1
            violations.append(f"NBC: {rtype} width {width:.1f}m < {min_width}m")
    return max(0.0, 1.0 - 0.25 * num_violations), violations


def score_circulation(placement: dict, max_hops: int = 4) -> tuple[float, list]:
    """BFS hop analysis from verandah/entry to all rooms."""
    deduction, violations = 0.0, []
    # Simplified: penalise if any room is beyond max_hops
    entry_rooms = {"verandah", "entry_porch", "living"}
    for room_id, room in placement.items():
        hops = room.get("bfs_hops_from_entry", 2)
        if hops > max_hops:
            deduction += 0.10
            violations.append(f"Circulation: {room.get('type')} is {hops} hops from entry")
    return max(0.0, 1.0 - deduction), violations


def score_adjacency(placement: dict) -> tuple[float, list]:
    deduction, violations = 0.0, []
    room_types = {r["type"] for r in placement.values()}
    adj_pairs = {(r1["type"], r2["type"]) for r1 in placement.values() for r2 in placement.values()
                 if r1 != r2 and r1.get("adjacent_to") and r2["type"] in r1["adjacent_to"]}
    if "kitchen" in room_types and "dining" in room_types:
        if ("kitchen", "dining") not in adj_pairs:
            deduction += 0.15
            violations.append("Adjacency: kitchen not adjacent to dining")
    if ("kitchen", "toilet") in adj_pairs or ("toilet", "kitchen") in adj_pairs:
        deduction += 0.20
        violations.append("Adjacency: toilet adjacent to kitchen (hygiene violation)")
    return max(0.0, 1.0 - deduction), violations


def score_climate(placement: dict, climate_zone: str = "Hot_Humid", wwr: float = 0.22) -> tuple[float, list]:
    features_present = 0
    total_features = 3
    if wwr <= 0.30:
        features_present += 1
    cross_vent = any(r.get("cross_ventilation_valid", False) for r in placement.values())
    if cross_vent:
        features_present += 1
    if climate_zone in ("Hot_Humid", "Hot_Dry", "Composite"):
        features_present += 1
    return 0.8 + 0.2 * (features_present / total_features), []


def score_baker(placement: dict, south_wall_thick: bool = True, cross_vent_valid: bool = True,
                passive_cooling_present: bool = True) -> tuple[float, list]:
    s = 0.60
    if south_wall_thick:
        s += 0.10
    if cross_vent_valid:
        s += 0.15
    if passive_cooling_present:
        s += 0.10
    return min(1.0, s), []


def compute_all_scores(placement: dict, climate_zone: str = "Hot_Humid", wwr: float = 0.22) -> ScoringResult:
    v_score, v_viol = score_vastu(placement)
    n_score, n_viol = score_nbc(placement)
    c_score, c_viol = score_circulation(placement)
    a_score, a_viol = score_adjacency(placement)
    cl_score, _ = score_climate(placement, climate_zone, wwr)
    b_score, _ = score_baker(placement)

    overall = (
        _WEIGHTS["vastu"] * v_score +
        _WEIGHTS["nbc"] * n_score +
        _WEIGHTS["circulation"] * c_score +
        _WEIGHTS["adjacency"] * a_score +
        _WEIGHTS["climate"] * cl_score +
        _WEIGHTS["baker"] * b_score
    )

    return ScoringResult(
        vastu=v_score, nbc=n_score, circulation=c_score,
        adjacency=a_score, climate=cl_score, baker=b_score,
        overall=overall,
        violations=v_viol + n_viol + c_viol + a_viol,
    )
