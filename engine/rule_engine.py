"""
rule_engine.py — Centralised rule loading, conflict resolution, and placement constraint
derivation for the Tamil Nadu Floor Plan Generator.

This module is the SINGLE entry point for DB rule queries used during room placement.
engine.py imports from here; rule_engine.py imports only stdlib + sqlite3 (no circular deps).

Public API:
  load_all_rules(bhk, strategy_mode) -> RuleSet
  resolve_conflicts(ruleset)         -> RuleSet
  provide_constraints(ruleset, net_w, net_d, ews_plot) -> PlacementConstraints
"""

import copy
import functools
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from config import DB_PATH

VALID_STRATEGY_MODES = frozenset({'compact', 'standard', 'luxury'})
DEFAULT_STRATEGY_MODE = 'standard'

# ── Fallback constants (used ONLY when DB query fails) ────────────────────────
# Primary values are always loaded from strategy_rules table.
_DIM_SCALE_FALLBACK: Dict[str, float] = {
    'compact':  0.92,
    'standard': 1.00,
    'luxury':   1.08,
}
_CORR_D_FALLBACK: Dict[Tuple[str, bool], float] = {
    ('compact',  False): 0.90,
    ('standard', False): 1.20,
    ('luxury',   False): 1.50,
    ('compact',  True):  0.75,
    ('standard', True):  0.90,
    ('luxury',   True):  1.20,
}


@dataclass
class RuleSet:
    """Resolved rule set for a given BHK + strategy_mode combination.

    Fields
    ------
    zoning          : room_type → zone_name  (front/public/service/private/passage)
    placement       : room_type → compass direction  (N/NE/E/SE/S/SW/W/NW)
    corridor        : DB corridor sub-rules  (type, position)
    dim_scale       : dimension scale factor for the active strategy_mode
    corridor_depths : {'standard': float, 'ews': float} corridor depths per plot class
    priority        : [(room_type, priority_int)] sorted ascending
    conflicts       : list of conflict description strings (populated by resolve_conflicts)
    """
    bhk:             int
    strategy_mode:   str
    zoning:          Dict[str, str]         = field(default_factory=dict)
    placement:       Dict[str, str]         = field(default_factory=dict)
    corridor:        Dict[str, str]         = field(default_factory=dict)
    dim_scale:       float                  = 1.0
    corridor_depths: Dict[str, float]       = field(default_factory=lambda: {'standard': 1.20, 'ews': 0.90})
    priority:        List[Tuple[str, int]]  = field(default_factory=list)
    conflicts:       List[str]              = field(default_factory=list)


@dataclass
class PlacementConstraints:
    """Geometry-ready constraints derived from a resolved RuleSet.

    Consumed directly by _place_rooms() in engine.py.  All values are
    in the same units as the engine (metres, dimensionless scale factors).
    """
    dim_scale:          float       = 1.0
    corr_d_target:      float       = 1.20
    private_room_order: List[str]   = field(default_factory=list)
    placement_hints:    Dict[str, str] = field(default_factory=dict)


# ── Internal DB helpers ───────────────────────────────────────────────────────

def _query_strategy_rows(bhk: int) -> list:
    """Single DB round-trip for ALL strategy_rules rows applicable to bhk."""
    bhk_tag = f'{bhk}BHK'
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur  = conn.cursor()
            cur.execute(
                """
                SELECT rule_type, rule_key, rule_value, bhk_applicability, priority
                FROM strategy_rules
                WHERE bhk_applicability = 'ALL'
                   OR bhk_applicability LIKE ?
                ORDER BY priority ASC
                """,
                (f'%{bhk_tag}%',),
            )
            rows = cur.fetchall()
            return rows
    except Exception as exc:
        print(f"[RULE ENGINE WARNING] strategy_rules query failed ({exc}); "
              f"fallback placement values will be used.")
        return []


# ── Public API ────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=12)   # 4 BHK values × 3 strategy modes
def load_all_rules(bhk: int, strategy_mode: str) -> RuleSet:
    """Load and assemble all strategy rules for a given BHK + strategy_mode.

    Queries strategy_rules once per (bhk, mode) pair (lru_cached).
    Falls back gracefully to _DIM_SCALE_FALLBACK / _CORR_D_FALLBACK on DB error.

    The DB stores dimension_scale and corridor_depth as strategy_rules rows:
      rule_type='dimension_scale', rule_key=<mode>, rule_value=<float>
      rule_type='corridor_depth',  rule_key=<mode>, rule_value=<float>
      rule_type='corridor_depth_ews', rule_key=<mode>, rule_value=<float>

    Returns a RuleSet with all fields populated.
    """
    mode = strategy_mode if strategy_mode in VALID_STRATEGY_MODES else DEFAULT_STRATEGY_MODE
    rs   = RuleSet(bhk=bhk, strategy_mode=mode)
    rows = _query_strategy_rows(bhk)

    _dim_scale_found      = False
    _corr_d_found         = False
    _corr_d_ews_found     = False

    for rule_type, rule_key, rule_value, _bhk_app, priority in rows:
        rt   = str(rule_key).strip().lower()
        rval = str(rule_value).strip()
        pri  = int(priority)

        if rule_type == 'zoning':
            rs.zoning[rt] = rval
            rs.priority.append((rt, pri))

        elif rule_type == 'placement':
            rs.placement[rt] = rval.upper()
            rs.priority.append((rt, pri))

        elif rule_type == 'corridor':
            rs.corridor[rt] = rval

        elif rule_type == 'dimension_scale':
            if rt == mode:
                try:
                    rs.dim_scale = float(rval)
                    _dim_scale_found = True
                except ValueError:
                    pass

        elif rule_type == 'corridor_depth':
            if rt == mode:
                try:
                    rs.corridor_depths['standard'] = float(rval)
                    _corr_d_found = True
                except ValueError:
                    pass

        elif rule_type == 'corridor_depth_ews':
            if rt == mode:
                try:
                    rs.corridor_depths['ews'] = float(rval)
                    _corr_d_ews_found = True
                except ValueError:
                    pass

    # Apply fallbacks for any values not found in DB
    if not _dim_scale_found:
        rs.dim_scale = _DIM_SCALE_FALLBACK.get(mode, 1.0)
    if not _corr_d_found:
        rs.corridor_depths['standard'] = _CORR_D_FALLBACK.get((mode, False), 1.20)
    if not _corr_d_ews_found:
        rs.corridor_depths['ews'] = _CORR_D_FALLBACK.get((mode, True), 0.90)

    # Sort priority list ascending
    rs.priority.sort(key=lambda pair: pair[1])
    return rs


def resolve_conflicts(ruleset: RuleSet) -> RuleSet:
    """Detect and log conflicting rules in a RuleSet.

    Conflict types checked:
      1. Room has both a zoning rule AND a placement hint whose compass direction
         is geometrically inconsistent with the zone (e.g. kitchen=public + SE hint).
      2. Duplicate zoning entries are already deduplicated by load_all_rules()
         (first row wins due to dict assignment order).

    Conflicts are logged to ruleset.conflicts; the zoning rule takes precedence
    (geometry wins over preference hints).  Returns a deep copy with conflicts noted.
    """
    rs = copy.deepcopy(ruleset)

    # Approximate zone→compatible-compass mapping
    _zone_compass: Dict[str, frozenset] = {
        'front':   frozenset(['N', 'NE', 'NW']),
        'public':  frozenset(['N', 'NE', 'NW', 'E', 'W']),
        'service': frozenset(['E', 'SE', 'W', 'SW', 'S']),
        'private': frozenset(['S', 'SW', 'SE', 'W']),
        'passage': frozenset(['N', 'NE', 'NW', 'E', 'W', 'S', 'SW', 'SE']),
    }

    for rt, zone in rs.zoning.items():
        compass = rs.placement.get(rt)
        if compass is None:
            continue
        allowed = _zone_compass.get(zone, frozenset())
        if compass not in allowed:
            msg = (f"{rt}: zone={zone} vs placement={compass} — "
                   f"zone constraint takes precedence in geometry")
            rs.conflicts.append(msg)
            print(f"[RULE ENGINE CONFLICT] {msg}")

    return rs


def provide_constraints(
    ruleset:  RuleSet,
    net_w:    float,
    net_d:    float,
    ews_plot: bool,
) -> PlacementConstraints:
    """Translate a resolved RuleSet into geometry-ready placement constraints.

    Parameters
    ----------
    ruleset   : resolved RuleSet (from resolve_conflicts())
    net_w     : net buildable width (m)
    net_d     : net buildable depth (m)
    ews_plot  : True when net area < 50 sqm (EWS plot class)

    Returns PlacementConstraints consumed directly by _place_rooms().
    """
    # Corridor depth: EWS or standard
    plot_class = 'ews' if ews_plot else 'standard'
    corr_d     = ruleset.corridor_depths.get(plot_class,
                     0.90 if ews_plot else 1.20)

    # Private zone room order: master_bedroom + toilet_attached anchor west cluster;
    # secondary bedrooms sorted by DB priority (ascending = more prominent = west side).
    _pri_map   = {rt: p for rt, p in ruleset.priority}
    _beds      = ['bedroom_2', 'bedroom_3', 'bedroom_4']
    _beds.sort(key=lambda rt: _pri_map.get(rt, 100))
    priv_order = ['master_bedroom', 'toilet_attached'] + _beds

    return PlacementConstraints(
        dim_scale          = ruleset.dim_scale,
        corr_d_target      = corr_d,
        private_room_order = priv_order,
        placement_hints    = dict(ruleset.placement),
    )
