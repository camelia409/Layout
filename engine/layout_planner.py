"""
layout_planner.py — Zoning, corridor, and room-order decisions.

Separates WHAT to build (decisions) from WHERE to place it (geometry).
Called ONCE per generate() invocation, before the placement retry loop.
_place_rooms() reads the returned plan dict and performs only coordinate
placement and geometry — no rule-engine calls, no ordering logic.

Public API
----------
  create_layout_plan(inputs, rules) -> dict

Inputs dict keys
----------------
  rooms      : list[str]   room set after EWS/small_plot filtering (from engine.get_room_set)
  net_w      : float       net buildable width  (m)
  net_d      : float       net buildable depth  (m)
  bhk        : int         effective BHK (post-EWS adjustment: always 1 on EWS plots)
  facing     : str         'N' | 'S' | 'E' | 'W'
  floors     : int         total floors (1 or 2)
  ews_plot   : bool        net_area < 50 sqm
  small_plot : bool        net_area < 90 sqm
  net_area   : float       net_w × net_d

Rules parameter
---------------
  RuleSet — from rule_engine.resolve_conflicts(load_all_rules(bhk, strategy_mode)).
  layout_planner imports ONLY from engine.rule_engine (no engine.engine import —
  avoids circular dependency).

Returns dict keys
-----------------
  rooms          list[str]         unchanged from inputs
  bhk            int
  facing         str
  floors         int
  ews_plot       bool
  small_plot     bool
  net_area       float
  strategy_mode  str               resolved mode ('compact'|'standard'|'luxury')
  dim_scale      float             geometry scale factor from PlacementConstraints
  corr_d_target  float             corridor depth target (m) from PlacementConstraints
  zones          dict[str,str]     room_type → zone name (from DB rules.zoning)
  corridor       dict              {'type': str, 'depth_target': float}
  room_order     dict              {'private': list[str]}  priority-sorted
  pooja_band     str|None          'b2' | 'b3' | None  (facing-dependent Vastu NE)
  decision_log   list[str]         human-readable record of every decision made
"""

from __future__ import annotations

from typing import Dict, List, Optional

from engine.rule_engine import (
    RuleSet,
    provide_constraints as _provide_constraints,
)


def create_layout_plan(inputs: dict, rules: RuleSet) -> dict:
    """Derive all layout decisions from inputs and DB rules.

    The result is stable across retry-loop seeds: the same rooms, the same
    corridor type, the same private-zone order on every attempt.  Only the
    random number generator seed (passed separately to _place_rooms) varies.
    """
    rooms      = list(inputs["rooms"])
    net_w      = float(inputs["net_w"])
    net_d      = float(inputs["net_d"])
    bhk        = int(inputs["bhk"])
    facing     = str(inputs["facing"])
    floors     = int(inputs["floors"])
    ews_plot   = bool(inputs["ews_plot"])
    small_plot = bool(inputs["small_plot"])
    net_area   = float(inputs["net_area"])

    mode = rules.strategy_mode
    pc   = _provide_constraints(rules, net_w, net_d, ews_plot)

    log: List[str] = []

    # ── Zone assignments from DB rules ────────────────────────────────────────
    # rules.zoning maps room_type → zone name as loaded from strategy_rules table.
    zones: Dict[str, str] = dict(rules.zoning)
    for rt, zone in zones.items():
        log.append(f"Applied zoning: {rt} → {zone} [strategy_rules]")

    # ── Corridor ──────────────────────────────────────────────────────────────
    corr_type = rules.corridor.get("type", "linear")
    log.append(f"Corridor type: {corr_type} [strategy_rules]")
    corridor = {
        "type":         corr_type,
        "depth_target": pc.corr_d_target,
    }

    # ── Private zone room order ───────────────────────────────────────────────
    # master_bedroom + toilet_attached anchor the west cluster (Vastu SW,
    # NBC required adjacency).  Secondary bedrooms sorted by DB priority
    # (ascending = more prominent = placed further west).
    pri_map        = {rt: p for rt, p in rules.priority}
    secondary_beds = [rt for rt in ("bedroom_2", "bedroom_3", "bedroom_4")
                      if rt in rooms]
    secondary_beds.sort(key=lambda rt: pri_map.get(rt, 100))

    private_order: List[str] = ["master_bedroom"]
    if "toilet_attached" in rooms:
        private_order.append("toilet_attached")
    private_order.extend(secondary_beds)
    log.append(
        f"Private zone order: {' → '.join(private_order)} [strategy_rules priority]"
    )

    # ── Pooja band assignment (Vastu NE compliance, facing-dependent) ─────────
    # N/E-facing: pooja goes in Band 2 (public zone) → high cy_pct → NE after rotation.
    # S/W-facing: pooja goes in Band 3 south edge → low y_int → NE after S/W rotation.
    # small_plot: pooja is excluded from the room set entirely (never in `rooms`).
    if "pooja" in rooms and not small_plot:
        pooja_band: Optional[str] = "b2" if facing in ("N", "E") else "b3"
    else:
        pooja_band = None

    return {
        "rooms":         rooms,
        "bhk":           bhk,
        "facing":        facing,
        "floors":        floors,
        "ews_plot":      ews_plot,
        "small_plot":    small_plot,
        "net_area":      net_area,
        "strategy_mode": mode,
        "dim_scale":     pc.dim_scale,
        "corr_d_target": pc.corr_d_target,
        "zones":         zones,
        "corridor":      corridor,
        "room_order":    {"private": private_order},
        "pooja_band":    pooja_band,
        "decision_log":  log,
    }
