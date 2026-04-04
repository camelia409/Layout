import os, sqlite3, time, warnings, math, functools
import numpy as np
from engine.rule_engine import load_all_rules as _re_load_all_rules
from engine.rule_engine import resolve_conflicts as _re_resolve_conflicts
from engine.rule_engine import provide_constraints as _re_provide_constraints
from engine.rule_engine import VALID_STRATEGY_MODES as _re_valid_modes
from engine.layout_planner import create_layout_plan
import pandas as pd
import joblib
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
warnings.filterwarnings('ignore')

# SECTION 1 — IMPORTS AND CONSTANTS
from config import DB_PATH, MODELS_DIR

# build_wall_polygons() geometry constants (DO NOT change without updating CLAUDE.md §6)
GAP_OVERCUT  = 0.06   # extra margin so gap box fully punches through wall solid
MIN_WALL_LEN = 0.02   # minimum wall length to produce a box (0.02 m = 20 mm)

WALL_EXT = 0.230      # exterior wall thickness (m)
WALL_INT = 0.115      # interior wall thickness (m)
HALF_EXT = 0.115
HALF_INT = 0.0575

FACING_MAP  = {'N': 0, 'S': 1, 'E': 2, 'W': 3}
CLIMATE_MAP = {'Hot_Humid': 0, 'Hot_Dry': 1, 'Composite': 2, 'Warm_Humid': 3}
ROAD_SIDE = {'N': 'north', 'S': 'south', 'E': 'east', 'W': 'west'}

# ── Universal Layout Engine — data structures (Part 8) ───────────────────────
# These replace the hardcoded ROOM_LISTS dict and scattered band parameters.
# DO NOT hardcode per-BHK room lists anywhere else. Use get_room_set() instead.

# Explicit adjacency graph: room -> [(neighbour, door_type, clear_width_m)]
# Engine iterates this to place doors. Absent rooms: edge skipped silently.

# Explicit zone assignments: zone_id -> ordered room list (west->east within band)
# Zone order south->north (pre-rotation): private / passage / service / public / front
ZONE_ASSIGNMENTS = {
    'front':   ['verandah'],
    'public':  ['living', 'dining', 'pooja'],
    'service': ['toilet_common', 'staircase', 'utility', 'kitchen', 'store'],
    'passage': ['corridor'],
    'private': ['master_bedroom', 'toilet_attached',
                'bedroom_2', 'bedroom_3', 'bedroom_4'],
}
ZONE_ORDER = ['private', 'passage', 'service', 'public', 'front']  # S->N pre-rotation

ROOM_LISTS = {
    1: ['master_bedroom', 'toilet_attached', 'living', 'kitchen', 'verandah'],
    2: ['master_bedroom', 'toilet_attached', 'bedroom_2', 'living', 'dining', 'kitchen', 'toilet_common', 'utility', 'verandah'],
    3: ['master_bedroom', 'toilet_attached', 'bedroom_2', 'bedroom_3', 'living', 'dining', 'kitchen', 'toilet_common', 'utility', 'verandah', 'pooja'],
    4: ['master_bedroom', 'toilet_attached', 'bedroom_2', 'bedroom_3', 'bedroom_4', 'living', 'dining', 'kitchen', 'toilet_common', 'utility', 'verandah', 'pooja', 'store'],
}  # kept for legacy callers; new code uses get_room_set()

# ── NBC fallback constants ────────────────────────────────────────────────────
# Activated only when DB is unavailable or a specific value fails validation.
# Derived from _NBC_STANDARDS (below) plus rooms not in NBC Table 3.
# NBC 2016 Table 3 reference standards — authoritative minimums used for normalization.
# Engine uses normalized values only; this dict is never read at placement/scoring time.
_NBC_STANDARDS = {
    'master_bedroom':  {'min_area': 9.5,  'min_width': 2.4},
    'bedroom_2':       {'min_area': 7.5,  'min_width': 2.1},
    'bedroom_3':       {'min_area': 7.5,  'min_width': 2.1},
    'bedroom_4':       {'min_area': 7.5,  'min_width': 2.1},
    'living':          {'min_area': 9.5,  'min_width': 2.4},
    'dining':          {'min_area': 5.0,  'min_width': 1.8},
    'kitchen':         {'min_area': 4.5,  'min_width': 1.8},
    'toilet_attached': {'min_area': 2.5,  'min_width': 1.2},
    'toilet_common':   {'min_area': 2.5,  'min_width': 1.2},
    'utility':         {'min_area': 2.0,  'min_width': 1.0},
    'verandah':        {'min_area': 4.0,  'min_width': 1.5},
    'corridor':        {'min_area': None, 'min_width': 1.0},
    'staircase':       {'min_area': None, 'min_width': 1.0},
}

# Derived programmatically from _NBC_STANDARDS; extra rooms not in Table 3 appended.
_NBC_AREA_FALLBACK = {rt: v['min_area'] for rt, v in _NBC_STANDARDS.items()
                      if v['min_area'] is not None}
_NBC_AREA_FALLBACK.update({'pooja': 1.2, 'store': 2.0})

_NBC_WIDTH_FALLBACK = {rt: v['min_width'] for rt, v in _NBC_STANDARDS.items()
                       if v['min_width'] is not None}

# Full mapping: nbc_codes.room_type (DB uppercase) -> engine room_type(s).
# Covers all ground-floor room types. First-floor types (balcony, dry_kitchen,
# staircase_head) are excluded — they are outside ROOM_UNIVERSE and not scored.
_DB_TO_ENGINE_ROOM = {
    'MASTER_BEDROOM':    ['master_bedroom'],
    'BEDROOM':           ['bedroom_2', 'bedroom_3', 'bedroom_4'],
    'LIVING_ROOM':       ['living'],
    'KITCHEN':           ['kitchen'],
    'BATHROOM_COMMON':   ['toilet_common'],
    'BATHROOM_ATTACHED': ['toilet_attached'],
    'DINING_ROOM':       ['dining'],
    'UTILITY_AREA':      ['utility'],
    'VERANDAH':          ['verandah'],
    'POOJA_ROOM':        ['pooja'],
    'STORE_ROOM':        ['store'],
    'INTERNAL_CORRIDOR': ['corridor'],
    'STAIRCASE':         ['staircase'],
}


def normalize_nbc_data(raw_db_rows):
    """Clean raw NBC DB rows before engine consumption.

    All normalization happens here, once, at module load time.
    Engine logic never compares against fallback or applies corrections at runtime.

    Normalization rules applied in order:
      Rule 1 — BATHROOM_ATTACHED / toilet_attached: area must be >= 2.5 sqm.
               DB stores 1.8 (standalone WC); combined bath+WC requires 2.5.
      Rule 2 — NBC conflict resolution: if DB value < NBC 2016 Table 3 standard,
               raise to standard.  DB values above standard are kept (stricter = valid).
      Rule 3 — WC vs bathroom separation: handled by _DB_TO_ENGINE_ROOM mapping
               (BATHROOM_ATTACHED -> toilet_attached, BATHROOM_COMMON -> toilet_common).
      Rule 4 — Every correction emits a [NORMALIZED] log line.

    Parameters
    ----------
    raw_db_rows : list of (db_room_type, parameter_name, min_value) tuples
        As returned by SELECT room_type, parameter_name, min_value FROM nbc_codes.

    Returns
    -------
    dict: {eng_room_type: {'min_area': float, 'min_width': float}}
        Only rooms present in raw_db_rows are included.
        Rooms absent from the DB must be filled from fallback by the caller.
    """
    # Pass 1: parse rows into {eng_rt: {'area': float, 'width': float}}
    raw = {}
    for db_rt, param, raw_val in raw_db_rows:
        engine_rts = _DB_TO_ENGINE_ROOM.get(str(db_rt).upper())
        if engine_rts is None:
            continue
        try:
            v = float(raw_val)
        except (TypeError, ValueError):
            continue
        if not (v > 0):
            continue
        is_area  = 'floor area' in str(param).lower()
        is_width = 'width'      in str(param).lower() and not is_area
        if not (is_area or is_width):
            continue
        for eng_rt in engine_rts:
            if eng_rt not in raw:
                raw[eng_rt] = {}
            if is_area:
                raw[eng_rt]['area']  = v
            else:
                raw[eng_rt]['width'] = v

    # Pass 2: apply normalization rules, log every correction
    clean = {}
    for eng_rt, vals in raw.items():
        std = _NBC_STANDARDS.get(eng_rt, {})
        clean[eng_rt] = {}

        if 'area' in vals:
            db_val  = vals['area']
            nbc_min = std.get('min_area')

            # Rule 1 — toilet_attached: combined bath+WC minimum 2.5 sqm
            if eng_rt == 'toilet_attached' and db_val < 2.5:
                print(f"[NORMALIZED] {eng_rt}: area {db_val} -> 2.5 "
                      f"(Rule 1: combined bath+WC standard; DB value is WC-only)")
                clean[eng_rt]['min_area'] = 2.5
            # Rule 2 — NBC conflict: DB below Table 3 minimum
            elif nbc_min is not None and db_val < nbc_min:
                print(f"[NORMALIZED] {eng_rt}: area {db_val} -> {nbc_min} "
                      f"(Rule 2: NBC 2016 Table 3 standard)")
                clean[eng_rt]['min_area'] = nbc_min
            else:
                clean[eng_rt]['min_area'] = db_val

        if 'width' in vals:
            db_val  = vals['width']
            nbc_min = std.get('min_width')

            # Rule 2 — width conflict: DB below Table 3 minimum
            if nbc_min is not None and db_val < nbc_min:
                print(f"[NORMALIZED] {eng_rt}: width {db_val} -> {nbc_min} "
                      f"(Rule 2: NBC 2016 Table 3 standard)")
                clean[eng_rt]['min_width'] = nbc_min
            else:
                clean[eng_rt]['min_width'] = db_val

    return clean


def _load_nbc_rules():
    """Load NBC minimum area and width from the nbc_codes DB table.

    DB rows are normalized via normalize_nbc_data() before engine use.
    No runtime comparisons against fallback — normalization is a one-time
    pre-processing step at module load.

    Fallbacks cover only rooms entirely absent from the DB.

    Returns
    -------
    tuple(dict, dict)
        (nbc_min_area, nbc_min_width) — both keyed by engine room_type string.
    """
    area  = dict(_NBC_AREA_FALLBACK)
    width = dict(_NBC_WIDTH_FALLBACK)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur  = conn.cursor()
            cur.execute("""
                SELECT room_type, parameter_name, min_value
                FROM   nbc_codes
                WHERE  parameter_name IN ('Minimum floor area', 'Minimum width')
                  AND  min_value IS NOT NULL
            """)
            rows = cur.fetchall()
    except Exception as exc:
        print(f"[NBC WARNING] DB query failed ({exc}); all fallback values in use.")
        return area, width

    # Normalize DB data before engine consumption (Rule 1–4)
    clean = normalize_nbc_data(rows)

    for eng_rt, vals in clean.items():
        if 'min_area'  in vals:
            area[eng_rt]  = vals['min_area']
        if 'min_width' in vals:
            width[eng_rt] = vals['min_width']

    return area, width


# Module-level NBC dicts — DB-driven at import time, fallback-safe.
NBC_MIN_AREA, NBC_MIN_WIDTH = _load_nbc_rules()

ROOM_UNIVERSE = [
    "master_bedroom", "toilet_attached", "living", "kitchen", "verandah",
    "bedroom_2", "bedroom_3", "dining", "toilet_common", "utility", "pooja", "bedroom_4", "store",
    "staircase",
]

HARDCODED_DEFAULTS = {
    "master_bedroom": (3.2, 3.0),
    "bedroom_2": (2.9, 2.9),
    "bedroom_3": (2.7, 2.7),
    "bedroom_4": (2.7, 2.7),
    "living": (4.2, 3.6),
    "dining": (2.6, 2.4),
    "kitchen": (2.6, 2.4),
    "toilet_attached": (1.5, 1.5),
    "toilet_common": (1.4, 1.4),
    "utility": (1.6, 1.4),
    "verandah": (0.0, 1.5),
    "pooja": (1.2, 1.2),
    "store": (1.5, 1.2),
}

ROOM_ZONE = {
    "verandah": "public_zone", "living": "public_zone", "dining": "public_zone", "pooja": "public_zone",
    "master_bedroom": "private_zone", "bedroom_2": "private_zone", "bedroom_3": "private_zone", "bedroom_4": "private_zone",
    "kitchen": "wet_zone", "utility": "wet_zone",
    "toilet_attached": "service_zone", "toilet_common": "service_zone", "store": "service_zone",
}

def _interval_union(ivs, tol=1e-6):
    if not ivs:
        return []
    ivs = sorted((min(a, b), max(a, b)) for a, b in ivs)
    out = [ivs[0]]
    for a, b in ivs[1:]:
        la, lb = out[-1]
        if a <= lb + tol:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out

def _adj(a, b, tol=0.05):
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["d"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["d"]
    if abs(ax1 - bx0) <= tol or abs(bx1 - ax0) <= tol:
        if min(ay1, by1) - max(ay0, by0) > tol:
            return True
    if abs(ay1 - by0) <= tol or abs(by1 - ay0) <= tol:
        if min(ax1, bx1) - max(ax0, bx0) > tol:
            return True
    return False

def _wall_stats(pl, net_w, net_d, tol=0.05):
    rooms = list(pl.keys())
    north, south, east, west = [], [], [], []
    for rt in rooms:
        a = pl[rt]
        x0, y0 = a["x"], a["y"]
        x1, y1 = x0 + a["w"], y0 + a["d"]
        if abs(y1 - net_d) <= tol:
            north.append((x0, x1))
        if abs(y0 - 0.0) <= tol:
            south.append((x0, x1))
        if abs(x1 - net_w) <= tol:
            east.append((y0, y1))
        if abs(x0 - 0.0) <= tol:
            west.append((y0, y1))
    north_u = _interval_union(north, tol=tol)
    south_u = _interval_union(south, tol=tol)
    east_u = _interval_union(east, tol=tol)
    west_u = _interval_union(west, tol=tol)
    ext_c = len(north_u) + len(south_u) + len(east_u) + len(west_u)
    ext_l = sum(b - a for a, b in north_u + south_u + east_u + west_u)

    seen, int_c, int_l = set(), 0, 0.0
    for i, ra in enumerate(rooms):
        a = pl[ra]
        ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["d"]
        for rb in rooms[i + 1:]:
            b = pl[rb]
            bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["d"]
            if abs(ax1 - bx0) <= tol or abs(bx1 - ax0) <= tol:
                y0 = max(ay0, by0)
                y1 = min(ay1, by1)
                if y1 - y0 > tol:
                    x = ax1 if abs(ax1 - bx0) <= tol else bx1
                    key = ("V", round(x, 3), round(y0, 3), round(y1, 3))
                    if key not in seen:
                        seen.add(key); int_c += 1; int_l += (y1 - y0)
            if abs(ay1 - by0) <= tol or abs(by1 - ay0) <= tol:
                x0 = max(ax0, bx0)
                x1 = min(ax1, bx1)
                if x1 - x0 > tol:
                    y = ay1 if abs(ay1 - by0) <= tol else by1
                    key = ("H", round(y, 3), round(x0, 3), round(x1, 3))
                    if key not in seen:
                        seen.add(key); int_c += 1; int_l += (x1 - x0)
    return int(ext_c), int(int_c), float(round(ext_l, 3)), float(round(int_l, 3))

def _rotate(rooms_n, net_w, net_d, facing):
    if facing == "N":
        return rooms_n
    out = []
    if facing == "S":
        for rt, x, y, w, d in rooms_n:
            out.append((rt, x, round(net_d - y - d, 2), w, d))
        return out
    if facing == "E":
        for rt, x, y, w, d in rooms_n:
            out.append((rt, round(y, 2), round(net_w - x - w, 2), d, w))
        return out
    for rt, x, y, w, d in rooms_n:
        out.append((rt, round(net_d - y - d, 2), round(x, 2), d, w))
    return out

# ── Universal Layout Engine — helper functions (Part 8) ──────────────────────

def get_room_set(bhk: int, net_area: float, floors: int = 1,
                 ews_plot: bool = False, small_plot: bool = False) -> list:
    """
    Build room list dynamically from BHK count, net area, and floor count.
    Replaces ROOM_LISTS[bhk] static dict. All selection rules are explicit.
    Called by _place_rooms(); do NOT call ROOM_LISTS[bhk] in new code.
    """
    rooms = ['verandah', 'living', 'master_bedroom', 'toilet_attached', 'kitchen']

    # Bedrooms scale linearly with BHK (2BHK adds bedroom_2, etc.)
    for i in range(2, bhk + 1):
        rooms.append(f'bedroom_{i}')

    # Dining: requires space (net_area > 60 sqm) and not EWS / small plot
    if net_area > 60.0 and not ews_plot and not small_plot:
        rooms.append('dining')

    # Common toilet: required from 2BHK upward
    if bhk >= 2 and not ews_plot:
        rooms.append('toilet_common')

    # Utility: REQUIRED separator between toilet_common and kitchen.
    # Without utility, tc and kitchen are directly adjacent -> forbidden adjacency.
    # Area threshold removed: always include when bhk >= 2 (toilet_common present).
    if bhk >= 2 and not ews_plot:
        rooms.append('utility')

    # Optional rooms added when area budget allows
    if net_area > 100.0 and not ews_plot and not small_plot:
        rooms.append('pooja')
    if net_area > 150.0 and not ews_plot and not small_plot:
        rooms.append('store')

    # Staircase: only for multi-floor projects (ADR-6)
    if floors >= 2:
        rooms.append('staircase')

    return rooms


def compute_bands(rooms: list, net_d: float, net_w: float,
                  bhk: int, ews_plot: bool = False) -> dict:
    """
    Derive zone band heights from actual room space demands.
    Eliminates dead space caused by random band-height sampling (rng.uniform).
    Returns dict: zone_id -> (y_start, height).

    Band convention matches _place_rooms() internal variables:
      b4_h = 'private' height = bedroom space + corridor space combined.
             The corridor is carved from the TOP of b4 by _place_rooms() itself.
      b3_h = 'service' height
      b2_h = 'public' height
      b1_h = 'front'  height  (verandah, fixed 1.5 m)

    Band order south->north (pre-rotation): private(b4) / service(b3) / public(b2) / front(b1)

    ADR-8: deterministic — no rng calls. Variation comes from retry seeds, not band heights.
    """
    b1_h = 1.50   # verandah: fixed NBC 1.5 m minimum

    # Private zone (b4): deepest bedroom drives minimum; corridor space added on top
    bed_rooms = [r for r in rooms if r in ZONE_ASSIGNMENTS['private']]
    bed_min   = max(
        (NBC_MIN_AREA.get(r, 7.5) / max(NBC_MIN_WIDTH.get(r, 2.1), 0.1)
         for r in bed_rooms),
        default=3.0,
    )
    bed_min  = round(max(bed_min, 2.8), 2)
    corr_h   = 0.9 if ews_plot else 1.2           # corridor carved from top of b4
    b4_h     = round(bed_min + corr_h, 2)         # total private band incl. corridor
    # Guard: if b4 would consume > 55% of net_d, omit corridor
    if b4_h > net_d * 0.55:
        corr_h = 0.0
        b4_h   = bed_min

    # Service zone (b3): deepest service room drives minimum
    svc_rooms = [r for r in rooms if r in ZONE_ASSIGNMENTS['service']]
    svc_min   = max(
        (NBC_MIN_AREA.get(r, 2.0) / max(NBC_MIN_WIDTH.get(r, 1.0), 0.1)
         for r in svc_rooms),
        default=2.0,
    )
    b3_h = round(max(svc_min, 2.0), 2)

    # Public zone (b2): whatever remains after private + service + front
    b2_h = round(net_d - b1_h - b3_h - b4_h, 2)
    b2_h = max(b2_h, 2.4)   # living ≥ 2.4 m deep (NBC min)

    # Rebalance: if zones exceed net_d, shrink public first (most flexible)
    total = b1_h + b2_h + b3_h + b4_h
    if total > net_d + 0.02:
        b2_h = round(max(net_d - b1_h - b3_h - b4_h, 2.0), 2)

    # Compute y_start (south->north)
    y_b4 = 0.0
    y_b3 = round(b4_h,             3)
    y_b2 = round(b4_h + b3_h,      3)
    y_b1 = round(net_d - b1_h,     3)

    return {
        'private': (y_b4, b4_h),   # maps to _place_rooms() b4_h, y_b4
        'service': (y_b3, b3_h),   # maps to _place_rooms() b3_h, y_b3
        'public':  (y_b2, b2_h),   # maps to _place_rooms() b2_h, y_b2
        'front':   (y_b1, b1_h),   # maps to _place_rooms() b1_h, y_b1
        'passage': (0.0,  0.0),    # corridor carved dynamically from b4 top
    }


def eliminate_dead_space(placed: list, net_w: float, tol: float = 0.10) -> list:
    """
    Post-placement gap filler: expand the easternmost room in each band
    to fill any uncovered x-range between its right edge and net_w.
    Eliminates sliver gaps caused by width scaling and rounding.
    Operates on placed list of (room_type, x, y, w, d) tuples.
    """
    from collections import defaultdict
    by_band = defaultdict(list)
    for entry in placed:
        rt, x, y, w, d = entry
        by_band[round(y, 2)].append(list(entry))

    result = []
    for y_key in sorted(by_band):
        band = sorted(by_band[y_key], key=lambda e: e[1])   # sort by x
        if not band:
            continue
        last = band[-1]
        gap  = round(net_w - (last[1] + last[3]), 3)
        if gap > tol:
            last[3] = round(last[3] + gap, 3)   # expand width of easternmost room
        result.extend(tuple(e) for e in band)
    return result


# ── Part 9 — Layout Optimisation Layer ───────────────────────────────────────
# Constants governing rule thresholds (do NOT make these per-call parameters;
# they are system invariants calibrated against the canonical test suite).

ALIGN_TOL      = 0.03   # R2: snap adjacent wall gap < 3 cm (float artefact closure)
SLIVER_MIN     = 0.05   # R4: dead-space strip < 5 cm is absorbed into neighbour
RATIO_MAX      = 3.0    # R3: aspect-ratio threshold above which redistribution fires
MIN_CORR_DEPTH = 1.00   # R1: corridor must be ≥ 1.0 m deep (CLAUDE.md §6 corridor)
VARY_FACTOR    = 0.08   # R5: ±8% bedroom-width nudge when all widths are identical


def optimize_layout(
    placed: list,
    net_w: float,
    y_b3: float,
    ews_plot: bool = False,
    corr_d_min: float = MIN_CORR_DEPTH,
) -> list:
    """
    Rule-based layout refinement pass.

    Precondition : called inside _place_rooms() AFTER all try_add() calls,
                   BEFORE _rotate().  Input is the mutable `placed` list of
                   (room_type, x, y, w, d) pre-rotation tuples.
    Postcondition: returns a refined placed list; zone memberships and band
                   depths (y, d) are NEVER changed — only x-axis (width)
                   values and x-positions are adjusted.

    Rules applied in order:
      R1 — Corridor depth ≥ MIN_CORR_DEPTH (1.0 m)
      R2 — Adjacent wall snap: close sub-ALIGN_TOL gaps in each band row
      R3 — Proportion correction: redistribute secondary bedroom widths when
           extreme aspect ratio (> RATIO_MAX) detected in the private zone
      R4 — Dead space absorption (delegates to eliminate_dead_space)
      R5 — Bedroom variation: apply ±VARY_FACTOR nudge when all secondary
           bedroom widths are nearly identical (creates visual diversity)
    """
    import math as _math
    from collections import defaultdict as _dd

    if not placed:
        return placed

    # Work on mutable list-of-lists; y and d fields are READ-ONLY in this function.
    rooms = [list(p) for p in placed]   # [rt, x, y, w, d]

    # ── R1: Corridor depth ≥ MIN_CORR_DEPTH ──────────────────────────────────
    # Corridor sits at the top of Band 4 (y = y_b3 - corr_d).  If its depth
    # fell below 1.0 m (e.g. on plots where b4_h was barely above 2.8 m),
    # enforce the minimum by shifting its y down.  Depth (d) change is the
    # ONLY exception to the "no depth change" rule — the corridor is not a
    # band-filling room; it is carved from Band 4 and has no NBC area minimum.
    for r in rooms:
        if r[0] == 'corridor' and r[4] < corr_d_min:
            deficit = round(corr_d_min - r[4], 3)
            r[4]    = corr_d_min
            r[2]    = round(max(r[2] - deficit, 0.0), 3)   # shift y down, floor at 0

    # ── R2: Adjacent wall snap ────────────────────────────────────────────────
    # Group rooms by their y-row (same y = same band-row).  Within each row,
    # sort by x and close any gap < ALIGN_TOL between consecutive rooms.
    by_band: dict = _dd(list)
    for idx, r in enumerate(rooms):
        by_band[round(r[2], 2)].append(idx)

    for y_key, indices in by_band.items():
        row = sorted(indices, key=lambda i: rooms[i][1])
        for k in range(len(row) - 1):
            curr = rooms[row[k]]
            nxt  = rooms[row[k + 1]]
            gap  = round(nxt[1] - (curr[1] + curr[3]), 4)
            if 0.0 < gap < ALIGN_TOL:
                curr[3] = round(curr[3] + gap, 3)  # extend current to close gap

    # ── R3: Proportion correction — private zone secondary bedrooms ───────────
    # Redistribute widths among bedroom_2/3/4 when any one room has aspect
    # ratio > RATIO_MAX.  Width redistribution only (depth is band-fixed).
    # master_bedroom and toilet_attached are excluded — they have independent
    # sizing logic.  kitchen is always excluded (ADR-4).
    _secondary = ['bedroom_2', 'bedroom_3', 'bedroom_4']
    bed_indices = [i for i, r in enumerate(rooms)
                   if r[0] in _secondary and r[2] < y_b3]  # in private zone only
    if len(bed_indices) >= 2:
        extreme = [
            i for i in bed_indices
            if rooms[i][3] > 0 and rooms[i][4] > 0
            and max(rooms[i][3] / rooms[i][4],
                    rooms[i][4] / max(rooms[i][3], 0.01)) > RATIO_MAX
        ]
        if extreme:
            total_w = sum(rooms[i][3] for i in bed_indices)
            avg_w   = round(total_w / len(bed_indices), 3)
            # 50% blend toward average — preserves variation while reducing extremes
            new_ws  = []
            for i in bed_indices:
                nbc_w = NBC_MIN_WIDTH.get(rooms[i][0], 2.1)
                nw    = round(rooms[i][3] * 0.5 + avg_w * 0.5, 2)
                new_ws.append(max(nw, nbc_w))
            # Normalise so total width is preserved exactly
            scale = total_w / max(sum(new_ws), 0.01)
            new_ws = [round(w * scale, 2) for w in new_ws]
            # Re-seat x positions (sorted west->east by current x)
            row_sorted = sorted(bed_indices, key=lambda i: rooms[i][1])
            x_cur = rooms[row_sorted[0]][1]   # anchor at first bedroom's x
            for k, i in enumerate(row_sorted):
                rooms[i][3] = new_ws[k]
                rooms[i][1] = round(x_cur, 2)
                x_cur = round(x_cur + rooms[i][3], 2)

    # ── R4: Dead space absorption (micro-gaps only) ───────────────────────────
    # NOTE: we do NOT call eliminate_dead_space() here — that function expands
    # the rightmost room per y-row to net_w, but Band 4 bedrooms are staggered
    # (y_off=0.30/0.15) so they sit in different y-rows; calling it would
    # erroneously expand toilet_attached into the bedroom column.
    # Instead we close only sub-ALIGN_TOL slivers (handled by R2 above).
    # Full band-level dead-space absorption is the responsibility of each band's
    # placement logic in _place_rooms().

    # ── R5: Bedroom width variation ────────────────────────────────────────────
    # When secondary bedroom widths are nearly identical (Δ < 5 cm), apply a
    # ±VARY_FACTOR alternating nudge so each bedroom has a distinct proportion.
    # Total width is preserved; NBC minimums are enforced post-nudge.
    rooms2 = [list(r) for r in [tuple(r) for r in rooms]]
    bed_idx2 = [i for i, r in enumerate(rooms2)
                if r[0] in _secondary and r[2] < y_b3]
    if len(bed_idx2) >= 2:
        widths = [rooms2[i][3] for i in bed_idx2]
        if max(widths) - min(widths) < 0.05:   # all nearly identical -> add variation
            n      = len(bed_idx2)
            total  = sum(widths)
            # Alternating ±VARY_FACTOR pattern (sums to ≈ total when even count)
            raw_ws = [
                round(widths[k] * (1.0 + VARY_FACTOR * (1 if k % 2 == 0 else -1)), 2)
                for k in range(n)
            ]
            # Scale to preserve total exactly
            scale2 = total / max(sum(raw_ws), 0.01)
            raw_ws = [round(w * scale2, 2) for w in raw_ws]
            # Apply, enforcing NBC minimums, re-seat x
            row2   = sorted(bed_idx2, key=lambda i: rooms2[i][1])
            x_cur2 = rooms2[row2[0]][1]
            for k, i in enumerate(row2):
                nbc_w = NBC_MIN_WIDTH.get(rooms2[i][0], 2.1)
                rooms2[i][3] = max(raw_ws[k], nbc_w)
                rooms2[i][1] = round(x_cur2, 2)
                x_cur2 = round(x_cur2 + rooms2[i][3], 2)

    return [tuple(r) for r in rooms2]


def _place_rooms(net_w, net_d, t, rng, layout_plan, err_p=0.05, strategy_log=None):
    """Pure coordinate placement and geometry.

    All decisions (which rooms, corridor type, zone assignments, room ordering)
    were resolved by create_layout_plan() before the retry loop and are read
    from `layout_plan`.  This function contains no rule-engine calls.
    """
    reference_net_area = 108.0
    tol  = 0.05
    step = 0.1

    # ── Read all decisions from layout_plan ───────────────────────────────────
    # Decision logic (zoning, ordering, strategy params) was resolved once by
    # create_layout_plan() before the retry loop.  _place_rooms() is now pure
    # coordinate placement and geometry — it does not call the rule engine.
    rooms      = list(layout_plan["rooms"])
    bhk        = int(layout_plan["bhk"])
    facing     = str(layout_plan["facing"])
    floors     = int(layout_plan["floors"])
    ews_plot   = bool(layout_plan["ews_plot"])
    small_plot = bool(layout_plan["small_plot"])
    net_area   = float(layout_plan["net_area"])
    _mode           = str(layout_plan["strategy_mode"])
    _dim_scale      = float(layout_plan["dim_scale"])
    _corr_d_target  = float(layout_plan["corr_d_target"])

    area_scale_factor = max(1.0, net_area / reference_net_area)

    # strategy_log accumulates placement-phase entries (corridor depth).
    # Decision-phase entries (zoning, corridor type, private zone order) live
    # in layout_plan["decision_log"] and are combined by generate() after the
    # retry loop.  Clearing here means the last attempt's entries are kept.
    if strategy_log is None:
        strategy_log = []
    strategy_log.clear()
    # ─────────────────────────────────────────────────────────────────────────

    # Placement minimum rules derived from NBC_MIN_AREA / NBC_MIN_WIDTH.
    # These are the DB-driven values loaded by _load_nbc_rules() at module init.
    # Coupling placement to scoring ensures:  placed_area = NBC_min * scale_factor
    # and since scale_factor = max(1.0, ...) >= 1.0 > 0.88, the NBC scoring
    # check (area >= NBC_min * 0.88) is always satisfied — mathematical guarantee.
    # Rooms absent from NBC dicts fall back to HARDCODED_DEFAULTS dimensions.
    # Verandah: min_area is always net_w * 1.5 (NBC §3.4.5 full-width verandah).
    _PLACEMENT_ROOMS = [
        'master_bedroom', 'bedroom_2', 'bedroom_3', 'bedroom_4',
        'living', 'dining', 'kitchen',
        'toilet_attached', 'toilet_common', 'utility',
        'pooja', 'store', 'staircase', 'corridor',
    ]
    room_rules = {}
    for _rt in _PLACEMENT_ROOMS:
        _def_w, _def_d = HARDCODED_DEFAULTS.get(_rt, (2.4, 2.4))
        room_rules[_rt] = {
            'min_area':  NBC_MIN_AREA.get(_rt,  _def_w * _def_d),
            'min_width': NBC_MIN_WIDTH.get(_rt, _def_w),
        }
    room_rules['verandah'] = {'min_area': net_w * 1.5, 'min_width': net_w}

    # ADR-5: EWS plots use 1.1 m toilet minimum, not the NBC standard 1.2 m
    if ews_plot:
        room_rules['toilet_attached']['min_width'] = 1.1
        room_rules['toilet_common']['min_width']   = 1.1

    def target_area(rt):
        if rt == "verandah":
            return float(net_w * 1.5)
        # _dim_scale from strategy mode: compact=0.92 | standard=1.00 | luxury=1.08
        return round(float(room_rules[rt]["min_area"] * area_scale_factor * _dim_scale), 3)

    def scaled_dims(rt):
        base_w, base_d = t.get(rt, HARDCODED_DEFAULTS.get(rt, (2.4, 2.4)))
        base_w = max(float(base_w), 0.1)
        base_d = max(float(base_d), 0.1)
        ratio = _clamp(base_w / base_d, 0.55, 2.40)
        if small_plot:
            # Avoid extreme wide/shallow proportions on small plots; improves scorer alignment.
            ratio = _clamp(ratio, 0.75, 1.60)
        area = target_area(rt)
        min_w = float(room_rules[rt]["min_width"])
        w = max(min_w, math.sqrt(area * ratio))
        d = area / w
        if w * d < area:
            d = area / w
        return round(float(w), 2), round(float(d), 2)

    def overlaps(a, b, pad=tol):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return (
            ax0 < bx1 - pad and
            ax1 > bx0 + pad and
            ay0 < by1 - pad and
            ay1 > by0 + pad
        )

    placed = []
    skipped_overlap = False

    def can_place(x, y, w, d, x_min, x_max, y_min, y_max):
        if x < x_min - 1e-6 or y < y_min - 1e-6:
            return False
        if x + w > x_max + 1e-6 or y + d > y_max + 1e-6:
            return False
        rect = (x, y, x + w, y + d)
        for _, px, py, pw, pd in placed:
            if overlaps(rect, (px, py, px + pw, py + pd)):
                return False
        return True

    def try_add(rt, x, y, w, d, x_min, x_max, y_min, y_max):
        nonlocal skipped_overlap
        attempts = [(0.0, 0.0)]
        for i in range(1, 6):
            delta = round(i * step, 2)
            attempts.extend([
                (delta, 0.0), (-delta, 0.0),
                (0.0, delta), (0.0, -delta),
                (delta, delta), (-delta, delta),
            ])
        for dx, dy in attempts:
            xx = round(x + dx, 2)
            yy = round(y + dy, 2)
            if can_place(xx, yy, w, d, x_min, x_max, y_min, y_max):
                placed.append((rt, xx, yy, round(w, 2), round(d, 2)))
                return True
        skipped_overlap = True
        return False

    # ── Universal band heights — compute_bands() replaces rng.uniform sampling ─
    # y=0 south, y=net_d north. Bands are ordered south->north (private->front).
    _bands = compute_bands(rooms, net_d, net_w, bhk, ews_plot)
    b1_h = _bands['front'][1]
    b2_h = _bands['public'][1]
    b3_h = _bands['service'][1]
    b4_h = _bands['private'][1]

    y_b4 = 0.0
    y_b3 = round(b4_h, 2)
    y_b2 = round(b4_h + b3_h, 2)
    y_b1 = round(net_d - b1_h, 2)

    if small_plot:
        y_b3 = round(y_b3 + 0.115, 2)
        b3_h = round(max(b3_h - 0.115, 0.5), 2)
        y_b2 = round(y_b3 + b3_h, 2)

    # ── Pre-compute pooja size (placement is facing-dependent for Vastu NE) ──────
    # Band assignment (b2 vs b3) was decided by create_layout_plan() from facing.
    # Dimension math stays here because it depends on band heights (b2_h, b3_h).
    _pooja_in_b2 = layout_plan["pooja_band"] == "b2"
    _pooja_in_b3 = layout_plan["pooja_band"] == "b3"
    _pooja_w_pre = 0.0
    _pooja_d_pre = 0.0
    if _pooja_in_b2 or _pooja_in_b3:
        _pooja_area = target_area("pooja")
        _, _pooja_d_nat = scaled_dims("pooja")
        _pooja_d_pre = min(max(_pooja_d_nat, room_rules["pooja"]["min_width"]), b2_h)
        _pooja_w_pre = round(max(room_rules["pooja"]["min_width"],
                                  _pooja_area / max(_pooja_d_pre, 0.1)), 2)
        _pooja_w_pre = round(min(_pooja_w_pre, net_w * 0.20), 2)

    # ── Outer-scope corridor variables (updated via nonlocal in _do_circulation) ──
    _corr_d    = 0.0
    _corr_y    = y_b3
    _b4_h_eff  = b4_h
    _y_b3_eff  = y_b3

    # ── Nested band-placement functions ──────────────────────────────────────────

    def _apply_zoning():
        """Section A — Band 1 (front zone): place verandah at full plot width."""
        if "verandah" in rooms:
            try_add("verandah", 0.0, y_b1, net_w, b1_h, 0.0, net_w, y_b1, net_d)

    def _apply_strategy():
        """Section B — Band 2 (public zone): place living, dining, and pooja rooms."""
        # Reserve pooja space: N facing -> east of dining; E facing -> west of living.
        _b2_pooja_w = _pooja_w_pre if _pooja_in_b2 else 0.0
        _b2_avail_w = net_w - _b2_pooja_w
        _b2_x_off   = _b2_pooja_w if facing == "E" else 0.0

        if small_plot or "dining" not in rooms:
            try_add("living", _b2_x_off, y_b2, round(_b2_avail_w, 2), round(b2_h, 2),
                    0.0, net_w, y_b2, y_b1)
        else:
            living_area = target_area("living")
            dining_area = target_area("dining")
            living_share = float(rng.uniform(0.55, 0.65))
            living_w = round(_b2_avail_w * living_share, 2)
            dining_w = round(_b2_avail_w - living_w, 2)
            living_req_w = max(room_rules["living"]["min_width"],
                               living_area / max(b2_h, 0.1))
            dining_req_w = max(room_rules["dining"]["min_width"],
                               dining_area / max(b2_h, 0.1))
            living_w = max(living_w, living_req_w)
            dining_w = max(dining_w, dining_req_w)
            total_public_w = living_w + dining_w
            if total_public_w > _b2_avail_w:
                scale = _b2_avail_w / total_public_w
                living_w = round(living_w * scale, 2)
                dining_w = round(_b2_avail_w - living_w, 2)
            min_liv = room_rules["living"]["min_width"]
            min_din = room_rules["dining"]["min_width"]
            if _b2_avail_w >= (min_liv + min_din):
                living_w = max(living_w, min_liv)
                dining_w = round(_b2_avail_w - living_w, 2)
                if dining_w < min_din:
                    dining_w = min_din
                    living_w = round(_b2_avail_w - dining_w, 2)
                living_w = max(living_w, min_liv)
                dining_w = round(_b2_avail_w - living_w, 2)
            else:
                living_w = round(_b2_avail_w, 2)
                dining_w = 0.0

            if dining_w > 0:
                try_add("living", _b2_x_off, y_b2, living_w, b2_h,
                        0.0, net_w, y_b2, y_b1)
                try_add("dining", round(_b2_x_off + living_w, 2), y_b2, dining_w, b2_h,
                        0.0, net_w, y_b2, y_b1)
            else:
                try_add("living", _b2_x_off, y_b2, round(_b2_avail_w, 2), round(b2_h, 2),
                        0.0, net_w, y_b2, y_b1)

        # Place pooja in Band 2 for N/E-facing — Vastu NE after rotation.
        # Anchor at TOP of Band 2 (y_b1 - d) so cy_pct > 0.5 even on large plots
        # where Band 2 is tall (e.g. 20x25 b2_h≈10.84m). Placing at y_b2 (bottom)
        # would give cy_pct ≈ 0.46, failing the NE quadrant check.
        if _pooja_in_b2 and _b2_pooja_w > 0.0:
            _b2_px = 0.0 if facing == "E" else round(net_w - _b2_pooja_w, 2)
            _b2_pd = min(_pooja_d_pre, b2_h)
            _b2_py = round(y_b1 - _b2_pd, 2)   # top-anchored: cy always near y_b1
            try_add("pooja", _b2_px, _b2_py, _b2_pooja_w, _b2_pd,
                    _b2_px, _b2_px + _b2_pooja_w, _b2_py, y_b1)

    def _place_core_rooms():
        """Section C — Band 3 (service zone): place toilet_common, staircase, kitchen, utility, store, pooja."""
        # ── Band 3: service zone ──────────────────────────────────────────────────────
        if any(r in rooms for r in ("toilet_common", "kitchen", "utility", "pooja", "store")):
            tc_w = 0.0
            if "toilet_common" in rooms:
                tc_w_calc, _ = scaled_dims("toilet_common")
                tc_w = round(max(tc_w_calc, room_rules["toilet_common"]["min_width"]), 2)
                tc_w = round(min(tc_w, net_w * 0.35), 2)
                try_add("toilet_common", 0.0, y_b3, tc_w, b3_h, 0.0, net_w, y_b3, y_b2)

            # G+1 plans need a staircase in the service band.
            # Place it immediately east of toilet_common (x = tc_w).
            # Fixed footprint: 1.0m wide × 2.5m deep (NBC stair clearance).
            # try_add returns False if it can't fit — cluster_x stays at tc_w
            # so kitchen placement is unaffected.
            stair_placed_w = 0.0
            if floors >= 2:
                _stair_w = 1.0
                _stair_d = 2.5
                _stair_x = tc_w
                if try_add("staircase", _stair_x, y_b3, _stair_w, _stair_d,
                            0.0, net_w, y_b3, y_b2):
                    stair_placed_w = _stair_w

            cluster_x = round(tc_w + stair_placed_w, 2)
            cluster_w = round(max(net_w - cluster_x, 0.0), 2)

            if cluster_w > 0.6:
                # Reserve east edge of cluster for pooja (S/W-facing; N/E pooja is in Band 2)
                _b3_pooja_w = _pooja_w_pre if _pooja_in_b3 else 0.0
                _b3_avail_w = round(cluster_w - _b3_pooja_w, 2)

                # Horizontal layout (west->east): utility | kitchen | store
                # Utility acts as a buffer between toilet_common and kitchen,
                # satisfying the FORBIDDEN toilet_common ↔ kitchen rule while
                # keeping REQUIRED kitchen ↔ utility adjacency (shared vertical wall).
                util_w = 0.0
                if "utility" in rooms:
                    util_area = target_area("utility")
                    util_w = round(max(room_rules["utility"]["min_width"],
                                       util_area / max(b3_h, 0.1)), 2)
                    util_w = round(min(util_w, _b3_avail_w * 0.35), 2)

                store_w = 0.0
                if "store" in rooms and not small_plot and util_w > 0:
                    store_area = target_area("store")
                    store_w = round(max(room_rules["store"]["min_width"],
                                        store_area / max(b3_h, 0.1)), 2)
                    store_w = round(min(store_w, _b3_avail_w * 0.25), 2)

                kitchen_x = round(cluster_x + util_w, 2)
                kitchen_w = round(_b3_avail_w - util_w - store_w, 2)
                kitchen_w = max(kitchen_w, 0.0)
                _kitch_natural_w = kitchen_w                # save uncapped width
                kitchen_d = b3_h                             # must reach y_b2 (ADR #4)
                # NBC minimum width: kitchen area must meet 4.5 sqm * 0.88 threshold.
                # On EWS plots (no utility/store) the natural width is all of Band 3 but
                # the old 35%-cap made it too narrow -> drop the hard cap in favour of:
                #   max(35% of net_w,  NBC_min_w) so large plots stay reasonable.
                _kitch_nbc_min_w = round(NBC_MIN_AREA.get('kitchen', 4.5) / max(kitchen_d, 0.1), 2)
                kitchen_w = min(kitchen_w, max(net_w * 0.35, _kitch_nbc_min_w))
                # Guarantee kitchen↔dining REQUIRED adjacency: kitchen right edge must
                # reach at least 0.20 m into dining's x-range (WALL_TOL = 0.14 m).
                _dining_pl = next((p for p in placed if p[0] == 'dining'), None)
                if _dining_pl is not None:
                    _dining_x0 = float(_dining_pl[1])
                    _kitch_need = round(_dining_x0 - kitchen_x + 0.25, 2)
                    kitchen_w = max(kitchen_w, min(_kitch_need, _kitch_natural_w))

                # Kitchen full Band 3 height, anchored at top — always touches Band 2
                if "kitchen" in rooms and kitchen_w >= room_rules["kitchen"]["min_width"]:
                    try_add("kitchen", kitchen_x, y_b3, kitchen_w, kitchen_d,
                            kitchen_x, round(kitchen_x + kitchen_w, 2), y_b3, y_b2)

                # Utility west of kitchen — shares vertical wall (REQUIRED adjacency)
                if util_w >= room_rules["utility"]["min_width"] and "utility" in rooms:
                    try_add("utility", cluster_x, y_b3, util_w, b3_h,
                            cluster_x, round(cluster_x + util_w, 2), y_b3, y_b2)

                # Store east of kitchen — keeps utility ↔ kitchen adjacency intact
                if store_w >= room_rules["store"]["min_width"] and "store" in rooms and not small_plot:
                    _store_x = round(kitchen_x + kitchen_w, 2)
                    try_add("store", _store_x, y_b3, store_w, b3_h,
                            _store_x, round(_store_x + store_w, 2), y_b3, y_b2)

                # Pooja at Band 3 south-east for S/W-facing (Vastu NE after S/W rotation)
                if _b3_pooja_w > 0.0 and _pooja_in_b3:
                    _b3_px = round(net_w - _b3_pooja_w, 2)
                    _b3_pd = min(_pooja_d_pre, b3_h)
                    try_add("pooja", _b3_px, y_b3, _b3_pooja_w, _b3_pd,
                            _b3_px, net_w, y_b3, y_b2)

    def _do_circulation():
        """Section D — Corridor: hallway between service zone and private zone."""
        nonlocal _corr_d, _corr_y, _b4_h_eff, _y_b3_eff
        # ── Corridor: hallway between service zone and private zone ─────────────────
        # Carves space from the TOP of Band 4 so every bedroom door opens into
        # a named passage rather than directly into the service zone.
        # Minimum b4_min_eff is preserved so bedrooms stay wide enough.
        # Corridor depth: driven by strategy_mode (DB corridor.type already confirmed
        # as 'linear'; depth magnitude is a mode-level parameter).
        strategy_log.append(
            f"Corridor depth target: {_corr_d_target}m "
            f"({'ews' if ews_plot else _mode} mode) [strategy_rules]"
        )
        _b4_min_eff    = 2.4   # minimum bedroom depth after corridor carved out
        _corr_d = round(min(_corr_d_target, max(0.0, b4_h - _b4_min_eff)), 2)
        if _corr_d >= 0.6:
            _corr_y    = round(y_b3 - _corr_d, 2)   # corridor bottom y
            _b4_h_eff  = round(b4_h - _corr_d, 2)   # effective bedroom band height
            _y_b3_eff  = _corr_y                     # bedrooms' north boundary
            try_add("corridor", 0.0, _corr_y, round(net_w, 2), _corr_d,
                    0.0, net_w, 0.0, y_b3)
        else:
            _corr_y   = y_b3
            _b4_h_eff = b4_h
            _y_b3_eff = y_b3

    def _place_private_rooms():
        """Section E — Band 4 (private zone): place master_bedroom, toilet_attached, and secondary bedrooms."""
        # Private zone room order: pre-determined by create_layout_plan() from DB
        # priority rules.  Filtered here to rooms actually present on this floor.
        # Decision log entry lives in layout_plan["decision_log"].
        private_rooms = [rt for rt in layout_plan["room_order"]["private"]
                         if rt in rooms]

        desired_widths = {}
        for rt in private_rooms:
            w, _ = scaled_dims(rt)
            desired_widths[rt] = round(w, 2)

        total_private_w = sum(desired_widths.values())
        if total_private_w > net_w and total_private_w > 0:
            scale = net_w / total_private_w
            for rt in private_rooms:
                desired_widths[rt] = round(max(room_rules[rt]["min_width"], desired_widths[rt] * scale), 2)
            for rt in private_rooms:
                desired_widths[rt] = max(desired_widths[rt], room_rules[rt]["min_width"])
            total_after = sum(desired_widths.values())
            if total_after > net_w:
                # Allow tiny overshoot caused by per-room rounding; shrink the last room to fit.
                overshoot = float(total_after - net_w)
                if overshoot <= 0.05 and private_rooms:
                    last = private_rooms[-1]
                    desired_widths[last] = round(max(room_rules[last]["min_width"], desired_widths[last] - overshoot), 2)
                    total_after = sum(desired_widths.values())
                if total_after > net_w + 0.01:
                    return False  # signals plot_too_small to orchestrator

        total_private_w = sum(desired_widths.values())
        if total_private_w < net_w and total_private_w > 0:
            growables = [rt for rt in private_rooms if rt != "toilet_attached"]
            extra_w = round(net_w - total_private_w, 2)
            per_room_add = round(extra_w / max(len(growables), 1), 2)
            for rt in growables:
                desired_widths[rt] = round(desired_widths[rt] + per_room_add, 2)
            remainder = round(net_w - sum(desired_widths.values()), 2)
            if growables and abs(remainder) > 0.001:
                desired_widths[growables[-1]] = round(desired_widths[growables[-1]] + remainder, 2)

        x_cursor = 0.0
        private_y_offsets = {
            "master_bedroom": 0.0,
            "toilet_attached": 0.0,
            # Stagger bedroom y positions slightly to avoid "bedroom row" patterns.
            # Offsets reduce depth accordingly so room tops still align to band 3.
            "bedroom_2": 0.3,
            "bedroom_3": 0.15,
            "bedroom_4": 0.0,
        }
        for rt in private_rooms:
            w = round(min(desired_widths[rt], max(net_w - x_cursor, 0.6)), 2)
            _, d0 = scaled_dims(rt)
            y_off = private_y_offsets.get(rt, 0.0)
            if rt == "toilet_attached":
                # Toilet occupies the south portion of its column; keep top below corridor.
                # FIX 3: ensure area meets NBC minimum (NBC_MIN_AREA * 0.88 threshold in tests)
                _ta_nbc = NBC_MIN_AREA.get('toilet_attached', 2.5)
                _min_d_nbc = round(_ta_nbc * 0.90 / max(w, 0.1), 2)
                d = round(max(_b4_h_eff * 0.55, _min_d_nbc, 0.8), 2)
            elif rt in ("bedroom_2", "bedroom_3", "bedroom_4"):
                # Bedrooms extend to the top of the effective private band (minus stagger).
                d = round(max(_b4_h_eff - y_off, 0.8), 2)
            else:
                # Default: cap by scaled dimension prediction.
                d = round(min(d0, max(_b4_h_eff - y_off, 0.8)), 2)
            try_add(rt, round(x_cursor, 2), round(y_b4 + y_off, 2), w, d, 0.0, net_w, y_b4, _y_b3_eff)
            x_cursor = round(x_cursor + w, 2)

        return True  # success

    def _finalize_layout():
        """Section F — Part 9: layout optimisation, rotation, error injection, and dict-building."""
        nonlocal placed
        # ── Part 9: layout optimisation pass (R1–R5) ─────────────────────────────
        # Runs AFTER all try_add() calls, BEFORE _rotate().
        # Only x-axis (width/x-position) values are adjusted; y and d are read-only.
        # Pass _corr_d as the corridor floor so R1 respects the active strategy mode.
        # compact=0.9 m, standard=1.2 m, luxury=1.5 m — never enforce standard over compact.
        placed = optimize_layout(placed, net_w, y_b3, ews_plot,
                                 corr_d_min=max(0.6, _corr_d))

        rn = _rotate(placed, net_w, net_d, facing)
        err = "overlap_skip" if skipped_overlap else None
        if float(rng.random()) < float(err_p):
            candidates = [rt for rt in rooms if rt in NBC_MIN_AREA]
            if candidates:
                rt = str(rng.choice(candidates))
                for i, tup in enumerate(rn):
                    if tup[0] == rt:
                        rtt, xx, yy, ww, dd = tup
                        rn[i] = (rtt, xx, yy, ww, round(max(0.6, dd * 0.75), 2))
                        err = "nbc_area_violation"
                        break

        pl = {}
        for rt, x, y, w, d in rn:
            pl[rt] = {
                "x": float(round(x, 3)),
                "y": float(round(y, 3)),
                "w": float(round(w, 3)),
                "d": float(round(d, 3)),
            }
            pl[rt]["cx"] = float(round(pl[rt]["x"] + pl[rt]["w"] / 2.0, 3))
            pl[rt]["cy"] = float(round(pl[rt]["y"] + pl[rt]["d"] / 2.0, 3))
        return pl, err

    # ── Orchestrate band-placement phases ────────────────────────────────────
    _apply_zoning()
    _apply_strategy()
    _place_core_rooms()
    _do_circulation()
    if not _place_private_rooms():
        return {}, "plot_too_small", b4_h, y_b3, y_b2
    pl, err = _finalize_layout()
    return pl, err, b4_h, y_b3, y_b2

# ── Adjacency DB support ─────────────────────────────────────────────────────
#
# DB SCHEMA CONTRACT (immutable — changes require DB migration + re-seed)
# ────────────────────────────────────────────────────────────────────────────
# Table: adjacency_rules
#   room_a            TEXT    — first room of the pair (engine room_type or DB alias)
#   room_b            TEXT    — second room of the pair
#   relationship      TEXT    — MUST_SHARE_WALL | SHOULD_BE_ADJACENT |
#                               PREFERRED_ADJACENT | MUST_NOT_BE_ADJACENT |
#                               FORBIDDEN_ADJACENT
#   door_required     INTEGER — 1 if a physical door must be placed, 0 otherwise
#   door_width_m      REAL    — clear door opening width in metres (nullable)
#   bhk_applicability TEXT    — comma-separated BHK tags: '1BHK,2BHK,3BHK,4BHK'
#   priority          INTEGER — lower number = higher priority (used for ORDER BY)
#
# Table: movement_paths
#   from_space          TEXT    — origin room (engine room_type or DB alias)
#   to_space            TEXT    — destination room
#   path_type           TEXT    — DIRECT_ENTRY | PASSAGE_REQUIRED | PRIVATE_ONLY |
#                                 SERVICE_ONLY | FORBIDDEN | EMERGENCY_EXIT
#   min_passage_width_m REAL    — minimum clear width of the connection in metres
#   bhk_applicability   TEXT    — comma-separated BHK tags
#
# Engine never reads CSV files. This block is the single authoritative contract.
# ────────────────────────────────────────────────────────────────────────────

# DB space names that differ from engine room_type names.
# All other names are passed through as-is after lower-casing.
_DB_SPACE_TO_ENGINE = {
    'passage':   'corridor',
    'puja_room': 'pooja',
}

# All valid engine room_type values — used to filter out unknown DB space names.
_ENGINE_ROOM_TYPES = frozenset({
    'master_bedroom', 'bedroom_2', 'bedroom_3', 'bedroom_4',
    'living', 'dining', 'kitchen', 'toilet_attached', 'toilet_common',
    'utility', 'verandah', 'pooja', 'store', 'staircase', 'corridor',
})

# Critical pairs that MUST be covered by the DB for every BHK.
# If absent after querying, a [DB WARNING] is emitted per missing pair.
# format: (room_a, room_b) — order-insensitive (checked as frozenset)
_CRITICAL_ADJ_PAIRS = [
    ('living',          'verandah'),
    ('master_bedroom',  'toilet_attached'),
    ('living',          'kitchen'),
    ('kitchen',         'dining'),
]

# Fallback — used ONLY when the DB query itself fails (OperationalError etc.).
# Never used for partially-populated DB. Matches former hardcoded list.
_REQUIRED_CONNECTIONS_FALLBACK = [
    ('verandah',         'living',            'archway', 1.20),
    ('living',           'master_bedroom',    'swing',   0.90),
    ('master_bedroom',   'toilet_attached',   'swing',   0.75),
    ('living',           'kitchen',           'swing',   0.90),
    ('kitchen',          'utility',           'swing',   0.75),
    ('living',           'toilet_common',     'swing',   0.75),
    ('living',           'bedroom_2',         'swing',   0.90),
    ('living',           'bedroom_3',         'swing',   0.90),
    ('living',           'bedroom_4',         'swing',   0.90),
    ('living',           'pooja',             'archway', 0.75),
    ('dining',           'living',            'archway', 1.20),
    ('corridor',         'master_bedroom',    'swing',   0.90),
    ('corridor',         'bedroom_2',         'swing',   0.90),
    ('corridor',         'bedroom_3',         'swing',   0.90),
    ('corridor',         'bedroom_4',         'swing',   0.90),
    ('kitchen',          'corridor',          'swing',   0.75),
]


def _normalize_space(name):
    """Map a DB space name to an engine room_type.

    Applies _DB_SPACE_TO_ENGINE aliases first, then checks membership in
    _ENGINE_ROOM_TYPES. Returns None for any name that cannot be mapped to
    a known engine room type (e.g. 'outside', 'courtyard', 'car_porch').
    Never reads from CSV or any file — pure in-memory lookup.
    """
    n = _DB_SPACE_TO_ENGINE.get(str(name).strip().lower(), str(name).strip().lower())
    return n if n in _ENGINE_ROOM_TYPES else None


def _validate_adjacency_coverage(bhk, adj_door_rows, path_rows):
    """Emit [DB WARNING] for each critical pair absent from the combined DB result.

    Parameters
    ----------
    bhk           : int  — BHK count for the current plan
    adj_door_rows : list — (room_a, room_b, door_width_m) from adjacency_rules
    path_rows     : list — (from_space, to_space, width_m) from movement_paths

    A pair is considered 'covered' if either source contains both rooms after
    space-name normalisation.  Warnings are informational; they do not abort
    generation.  The lru_cache means each BHK emits warnings at most once per
    process lifetime.
    """
    covered = set()
    for room_a, room_b, _ in adj_door_rows:
        ra, rb = _normalize_space(room_a), _normalize_space(room_b)
        if ra and rb:
            covered.add(frozenset([ra, rb]))
    for from_s, to_s, _ in path_rows:
        ra, rb = _normalize_space(from_s), _normalize_space(to_s)
        if ra and rb:
            covered.add(frozenset([ra, rb]))

    for ra, rb in _CRITICAL_ADJ_PAIRS:
        if frozenset([ra, rb]) not in covered:
            print(f"[DB WARNING] Missing adjacency rule for: {ra} <-> {rb}  "
                  f"(BHK={bhk}; fallback connection will be used by place_doors)")


@functools.lru_cache(maxsize=4)
def build_adjacency_graph_from_db(bhk):
    """Build the adjacency graph and required door connections from the DB for a given BHK.

    Source tables (DB-only — no CSV files):
      • adjacency_rules   — rows where door_required=1 and relationship is positive;
                            ordered by priority ASC so mandatory rules take precedence.
      • movement_paths    — DIRECT_ENTRY / PASSAGE_REQUIRED / PRIVATE_ONLY / SERVICE_ONLY
                            rows with a non-null min_passage_width_m.

    Both tables are filtered to the given BHK via bhk_applicability LIKE '%{N}BHK%'.
    DB space names are normalised via _normalize_space(); unknown spaces are skipped.
    Duplicate pairs are deduplicated by frozenset — the first occurrence wins.

    A coverage check via _validate_adjacency_coverage() emits [DB WARNING] for any
    critical pair absent from the DB result.  Warnings do not abort generation.

    Returns
    -------
    adjacency_graph : dict[str, list[str]]
        Bidirectional map of room_type -> [adjacent room_types].
        Derived from all positive-relationship adjacency_rules rows.
    required_connections : list[(from_r, to_r, door_type, width_m)]
        Consumed by place_doors() — one entry per unique unordered pair.
        door_type is 'archway' when width_m >= 1.0, else 'swing'.

    Fallback
    --------
    On any DB exception the function returns ({}, _REQUIRED_CONNECTIONS_FALLBACK)
    and emits [ADJ WARNING].  Missing rows (partial DB) do NOT trigger the fallback;
    they trigger [DB WARNING] only.
    """
    bhk_tag = f'{bhk}BHK'
    adjacency  = {}
    req_conns  = []
    seen_pairs = set()   # frozenset — deduplicates door connection entries

    def _adj(ra, rb):
        adjacency.setdefault(ra, [])
        adjacency.setdefault(rb, [])
        if rb not in adjacency[ra]: adjacency[ra].append(rb)
        if ra not in adjacency[rb]: adjacency[rb].append(ra)

    def _conn(ra, rb, w):
        key = frozenset([ra, rb])
        if key in seen_pairs:
            return
        req_conns.append((ra, rb, 'archway' if w >= 1.0 else 'swing', round(w, 2)))
        seen_pairs.add(key)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur  = conn.cursor()

            # ── adjacency_rules: door-requiring rows, ordered by priority ──────────
            # Columns used: room_a, room_b, door_width_m
            # Filter:       door_required=1, positive relationship, BHK match
            cur.execute("""
                SELECT room_a, room_b, door_width_m
                FROM   adjacency_rules
                WHERE  door_required = 1
                  AND  relationship IN ('MUST_SHARE_WALL','SHOULD_BE_ADJACENT','PREFERRED_ADJACENT')
                  AND  bhk_applicability LIKE ?
                ORDER  BY priority ASC
            """, (f'%{bhk_tag}%',))
            adj_door_rows = cur.fetchall()

            # ── adjacency_rules: all positive rows for graph (includes non-door) ───
            # Columns used: room_a, room_b
            cur.execute("""
                SELECT room_a, room_b
                FROM   adjacency_rules
                WHERE  relationship IN ('MUST_SHARE_WALL','SHOULD_BE_ADJACENT','PREFERRED_ADJACENT')
                  AND  bhk_applicability LIKE ?
                ORDER  BY priority ASC
            """, (f'%{bhk_tag}%',))
            adj_all_rows = cur.fetchall()

            # ── movement_paths: direct and private/service connections ─────────────
            # Columns used: from_space, to_space, min_passage_width_m
            # Filter:       traversable path types, non-null width, BHK match
            cur.execute("""
                SELECT from_space, to_space, min_passage_width_m
                FROM   movement_paths
                WHERE  path_type IN ('DIRECT_ENTRY','PASSAGE_REQUIRED','PRIVATE_ONLY','SERVICE_ONLY')
                  AND  min_passage_width_m IS NOT NULL
                  AND  bhk_applicability LIKE ?
            """, (f'%{bhk_tag}%',))
            path_rows = cur.fetchall()

    except Exception as exc:
        print(f"[ADJ WARNING] DB query failed ({exc}); using hardcoded fallback connections.")
        return {}, list(_REQUIRED_CONNECTIONS_FALLBACK)

    # ── No rows at all → BHK value has no DB coverage (e.g. first-floor bhk=5) ─
    if not adj_door_rows and not path_rows:
        print(f"[ADJ WARNING] No adjacency rows in DB for BHK={bhk}; "
              f"using fallback connections.")
        return {}, list(_REQUIRED_CONNECTIONS_FALLBACK)

    # ── Validation: warn on missing critical pairs ─────────────────────────────
    _validate_adjacency_coverage(bhk, adj_door_rows, path_rows)

    # ── Build bidirectional graph from all positive adjacency rules ────────────
    for room_a, room_b in adj_all_rows:
        ra, rb = _normalize_space(room_a), _normalize_space(room_b)
        if ra and rb:
            _adj(ra, rb)

    # ── Required door connections from adjacency_rules (priority-ordered) ─────
    for room_a, room_b, door_w in adj_door_rows:
        ra, rb = _normalize_space(room_a), _normalize_space(room_b)
        if ra and rb:
            _adj(ra, rb)
            _conn(ra, rb, float(door_w) if door_w else 0.9)

    # ── Supplement with movement path connections (dedup by pair) ─────────────
    for from_s, to_s, width_m in path_rows:
        ra, rb = _normalize_space(from_s), _normalize_space(to_s)
        if ra and rb:
            _adj(ra, rb)
            _conn(ra, rb, float(width_m))

    return adjacency, req_conns


# ── Active Strategy Engine ────────────────────────────────────────────────────
#
# strategy_rules DB table drives three placement decisions:
#   1. zoning        — which zone (band) each room_type belongs to
#   2. corridor      — corridor layout type + position
#   3. placement     — preferred compass direction per room_type
#
# strategy_mode (standard | compact | luxury) tunes two geometry parameters
# that are NOT stored in the DB:
#   corridor depth   — compact: 0.90 m  |  standard: 1.20 m  |  luxury: 1.50 m
#   dim scale factor — compact: 0.92    |  standard: 1.00    |  luxury: 1.08
#
# These are defined below as module-level constants so they are visible to
# both _place_rooms() and any future callers.
# ─────────────────────────────────────────────────────────────────────────────

# Fallback corridor depth (m) per strategy mode: (standard_target, ews_target).
# Used ONLY when the rule_engine DB query fails.  Primary values come from
# strategy_rules table via engine/rule_engine.py → provide_constraints().
_STRATEGY_CORR_D_FALLBACK = {
    'compact':  (0.90, 0.75),
    'standard': (1.20, 0.90),
    'luxury':   (1.50, 1.20),
}

# Fallback dimension scale factors.  Primary values from strategy_rules DB table.
_STRATEGY_DIM_SCALE_FALLBACK = {
    'compact':  0.92,
    'standard': 1.00,
    'luxury':   1.08,
}


PASSAGE_TYPE_MAP = {
    ('verandah', 'living'): 'ARCHWAY_LIVING_VERANDAH',
    ('living', 'master_bedroom'): 'BEDROOM_DOOR',
    ('master_bedroom', 'toilet_attached'): 'TOILET_DOOR',
    ('living', 'kitchen'): 'KITCHEN_DOOR',
    ('kitchen', 'utility'): 'UTILITY_DOOR',
    ('living', 'toilet_common'): 'TOILET_DOOR',
    ('living', 'bedroom_2'): 'BEDROOM_DOOR',
    ('living', 'bedroom_3'): 'BEDROOM_DOOR',
    ('living', 'bedroom_4'): 'BEDROOM_DOOR',
    ('living', 'pooja'): 'ARCHWAY_LIVING_VERANDAH',
    ('dining', 'living'): 'ARCHWAY_LIVING_DINING',
    ('outside', 'verandah'): 'MAIN_ENTRANCE_DOOR',
    ('corridor', 'master_bedroom'): 'BEDROOM_DOOR',
    ('corridor', 'bedroom_2'):      'BEDROOM_DOOR',
    ('corridor', 'bedroom_3'):      'BEDROOM_DOOR',
    ('corridor', 'bedroom_4'):      'BEDROOM_DOOR',
    ('kitchen', 'corridor'):        'KITCHEN_DOOR',
}

CURRENT_FACING = 'N'

# SECTION 2 — DATA CLASSES
@dataclass
class WallSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    thickness: float
    wall_type: str
    room_left: str
    room_right: str
    has_opening: bool = False

    @property
    def length(self) -> float:
        return math.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def direction(self) -> str:
        return 'H' if abs(self.y2 - self.y1) < 0.01 else 'V'


@dataclass
class DoorOpening:
    label: str
    wall: WallSegment
    position: float
    width: float
    door_type: str
    hinge_side: str
    swing_into: str
    room_from: str
    room_to: str


@dataclass
class WindowOpening:
    label: str
    wall: WallSegment
    position: float
    width: float
    height: float
    sill_height: float
    room_type: str
    is_ventilator: bool = False


@dataclass
class WallGeometry:
    """
    Final Shapely wall solids after gap subtraction.
    Produced by build_wall_polygons(); consumed by renderer only.
    Three invariants (CLAUDE.md §6):
      1. door gaps subtract from interior solid only
      2. window gaps subtract from exterior solid only
      3. floor_index filtering applied before all other operations
    """
    exterior: object   # Shapely Polygon/MultiPolygon or None — exterior wall solid
    interior: object   # Shapely Polygon/MultiPolygon or None — interior wall solid


@dataclass
class Room:
    room_type: str
    x: float
    y: float
    width: float
    depth: float
    area: float
    cx_pct: float
    cy_pct: float
    compass: str


@dataclass
class FloorPlan:
    plot_w: float
    plot_d: float
    bhk: int
    facing: str
    district: str
    net_w: float
    net_d: float
    setback_front: float
    setback_rear: float
    setback_side: float
    rooms: List = field(default_factory=list)
    walls: List = field(default_factory=list)
    doors: List = field(default_factory=list)
    windows: List = field(default_factory=list)
    score_valid: float = 0.0
    score_vastu: float = 0.0
    score_nbc: float = 0.0
    score_circulation: float = 0.0
    score_adjacency: float = 0.0
    score_overall: float = 0.0
    climate_zone: str = ''
    facing_code: int = 0
    climate_code: int = 0
    explanations: dict = field(default_factory=dict)
    shap_values: dict = field(default_factory=dict)
    materials: List = field(default_factory=list)
    baker_principles: List = field(default_factory=list)
    wall_geometry: object = None  # Shapely geometry, set after generate()
    placement: dict = field(default_factory=dict)
    band_b4_h: float = 0.0
    band_y_b3: float = 0.0
    band_y_b2: float = 0.0
    ml_debug: dict = field(default_factory=dict)
    generation_time_s: float = 0.0
    seed: int = 42

    @property
    def rooms_list(self) -> list:
        """Return rooms as a flat List[Room] regardless of internal representation.

        engine_api.generate_plan() stores rooms as Dict[str, Room] keyed by room_type
        for O(1) lookup during scoring.  Renderer and app always need a flat list.
        Use this property at every engine→renderer boundary instead of isinstance checks.
        """
        if isinstance(self.rooms, dict):
            return list(self.rooms.values())
        return list(self.rooms)


# SECTION 3 — MODEL LOADER (singleton)
class ModelLoader:
    _clf = None
    _dim_model = None
    _explainer = None
    _loaded = False

    @classmethod
    def get(cls):
        if not cls._loaded:
            print('Loading models...', end=' ', flush=True)
            paths = {
                'scorer': os.path.join(MODELS_DIR, 'constraint_scorer.pkl'),
                'dims': os.path.join(MODELS_DIR, 'room_dimensions.h5'),
                'shap': os.path.join(MODELS_DIR, 'shap_explainer.pkl'),
            }
            for _, path in paths.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f'Missing model: {path}\nDownload from Colab and place in models\\')
            cls._clf = joblib.load(paths['scorer'])
            # Inference-only load to avoid Keras H5 deserialization issues across versions.
            # Inference-only load; be tolerant to Keras version diffs across environments.
            try:
                cls._dim_model = tf.keras.models.load_model(paths['dims'], compile=False)
            except Exception:
                # Some saved models include Dense(..., quantization_config=None); ignore that arg.
                from tensorflow.keras.layers import Dense as _Dense
                class DenseCompat(_Dense):
                    def __init__(self, *args, **kwargs):
                        kwargs.pop('quantization_config', None)
                        super().__init__(*args, **kwargs)
                cls._dim_model = tf.keras.models.load_model(
                    paths['dims'], compile=False, custom_objects={'Dense': DenseCompat}
                )
            cls._explainer = joblib.load(paths['shap'])
            cls._loaded = True
            print('OK')
        return cls._clf, cls._dim_model, cls._explainer


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _pair(a: str, b: str) -> frozenset:
    return frozenset([a, b])


def _compass(cx_pct: float, cy_pct: float) -> str:
    if cx_pct > 0.6 and cy_pct > 0.6:
        return 'NE'
    if cx_pct < 0.4 and cy_pct > 0.6:
        return 'NW'
    if cx_pct > 0.6 and cy_pct < 0.4:
        return 'SE'
    if cx_pct < 0.4 and cy_pct < 0.4:
        return 'SW'
    if cy_pct > 0.6:
        return 'N'
    if cy_pct < 0.4:
        return 'S'
    if cx_pct > 0.6:
        return 'E'
    if cx_pct < 0.4:
        return 'W'
    return 'C'


def _cardinal_for_wall(wall: WallSegment, net_w: float, net_d: float) -> str:
    tol = 0.05
    if wall.direction == 'H':
        return 'N' if abs(wall.y1 - net_d) < tol else 'S'
    return 'E' if abs(wall.x1 - net_w) < tol else 'W'


# SECTION 4 — DATABASE HELPERS
def get_setbacks(plot_area: float, district: str) -> Tuple[float, float, float]:
    q = '''
        SELECT front_setback_m, rear_setback_m,
               side_setback_left_m, side_setback_right_m
        FROM tn_setbacks
        WHERE plot_area_min_sqm <= ?
          AND (plot_area_max_sqm >= ? OR plot_area_max_sqm IS NULL)
        ORDER BY plot_area_min_sqm DESC LIMIT 1
    '''
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(q, (plot_area, plot_area)).fetchone()
    if not row:
        return (2.0, 1.5, 1.0)
    return (float(row[0]), float(row[1]), round((float(row[2]) + float(row[3])) / 2.0, 2))


def get_climate_zone(district: str) -> str:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute('SELECT climate_zone FROM climate_data WHERE district = ? LIMIT 1', (district,)).fetchone()
    return row[0] if row and row[0] else 'Composite'


def get_window_scores(district: str) -> dict:
    q = '''
        SELECT window_north_score, window_south_score,
               window_east_score, window_west_score
        FROM climate_data WHERE district = ? LIMIT 1
    '''
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(q, (district,)).fetchone()
    if not row:
        return {'N': 1.0, 'S': 0.6, 'E': 0.8, 'W': 0.3}
    return {'N': float(row[0]), 'S': float(row[1]), 'E': float(row[2]), 'W': float(row[3])}


def get_door_width_from_db(passage_type: str) -> float:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute('SELECT min_clear_width_m FROM passage_dimensions WHERE passage_type = ? LIMIT 1', (passage_type,)).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.9


def get_materials(district: str, climate_zone: str) -> List[dict]:
    q = '''
        SELECT material_name, material_category,
               cost_per_unit_inr_avg, unit,
               thermal_performance, sustainability_rating,
               baker_recommended, climate_zone_suitability,
               wall_drawing_color_hex, hatch_pattern
        FROM materials_db
        WHERE (districts_available LIKE ? OR districts_available = 'ALL')
          AND nbc_approved = 1
        ORDER BY sustainability_rating DESC, cost_per_unit_inr_avg ASC
        LIMIT 12
    '''
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(q, conn, params=(f'%{district}%',))
    return df.to_dict(orient='records')


def get_baker_principles(plot_area: float, climate_zone: str) -> List[dict]:
    q = '''
        SELECT principle_name, category, description,
               cost_saving_pct, drawing_impact, wall_thickness_mm
        FROM baker_principles
        WHERE (plot_area_min_sqm <= ? OR plot_area_min_sqm IS NULL)
          AND (plot_area_max_sqm >= ? OR plot_area_max_sqm IS NULL)
          AND (climate_zones LIKE ? OR climate_zones LIKE '%ALL%')
        ORDER BY cost_saving_pct DESC LIMIT 6
    '''
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(q, conn, params=(plot_area, plot_area, f'%{climate_zone}%'))
    return df.to_dict(orient='records')


# SECTION 5 — DIMENSION MODEL PREDICTION
def predict_room_dims(plot_w, plot_d, bhk, facing_code, climate_code, net_w, net_d, dim_model) -> dict:
    """Predict room dimension templates (width, depth) using the Keras model.

    IMPORTANT: Output index mapping must match the Colab DIM_TARGET_COLS order:
      0..25 = 13 rooms * (w,d)
      26..38 = areas (ignored here)
    """
    arr = np.array([[plot_w, plot_d, plot_w * plot_d, net_w, net_d, net_w * net_d, bhk,
                     facing_code, climate_code]], dtype=float)
    pred = np.asarray(dim_model.predict(arr, verbose=0)[0], dtype=float)

    raw = {
        'master_bedroom': (pred[0], pred[1]),
        'toilet_attached': (pred[2], pred[3]),
        'living': (pred[4], pred[5]),
        'kitchen': (pred[6], pred[7]),
        'verandah': (pred[8], pred[9]),
        'bedroom_2': (pred[10], pred[11]),
        'bedroom_3': (pred[12], pred[13]),
        'dining': (pred[14], pred[15]),
        'toilet_common': (pred[16], pred[17]),
        'utility': (pred[18], pred[19]),
        'pooja': (pred[20], pred[21]),
        'bedroom_4': (pred[22], pred[23]),
        'store': (pred[24], pred[25]),
    }

    out = {}
    for room, (w, d) in raw.items():
        w = max(float(w), NBC_MIN_WIDTH.get(room, 1.0))
        d = max(float(d), 0.8)
        if room in NBC_MIN_AREA and w * d < NBC_MIN_AREA[room]:
            d = NBC_MIN_AREA[room] / w + 0.1
        out[room] = (round(w, 2), round(d, 2))
    return out


# SECTION 6 — 4-BAND HORIZONTAL ROOM PLACEMENT
def apply_wall_offsets(rooms_n, net_w, net_d):
    """
    Adjusts room (x, y, w, d) tuples to account for wall thickness.
    Rooms are shrunk inward so gaps between them = wall thickness.

    Rule:
      Each room edge touching net boundary gets HALF_EXT offset.
      Each room edge touching another room gets HALF_INT offset.

    Since rooms in our layout share edges exactly (gap=0),
    we classify each edge:
      x == 0        -> touching west exterior wall
      x+w == net_w  -> touching east exterior wall
      y == 0        -> touching south exterior wall
      y+d == net_d  -> touching north exterior wall
      otherwise     -> touching interior wall
    """
    tol = 0.05  # tolerance for edge detection
    adjusted = []
    for room_type, x, y, w, d in rooms_n:
        # Determine offsets for each edge
        left_off   = HALF_EXT if x < tol else HALF_INT
        right_off  = HALF_EXT if abs(x + w - net_w) < tol else HALF_INT
        bottom_off = HALF_EXT if y < tol else HALF_INT
        top_off    = HALF_EXT if abs(y + d - net_d) < tol else HALF_INT

        new_x = round(x + left_off,   3)
        new_y = round(y + bottom_off, 3)
        new_w = round(w - left_off - right_off, 3)
        new_d = round(d - bottom_off - top_off, 3)

        # Safety: ensure minimum room size
        new_w = max(new_w, 0.5)
        new_d = max(new_d, 0.5)

        adjusted.append((room_type, new_x, new_y, new_w, new_d))
    return adjusted

def place_rooms_in_bands(net_w, net_d, bhk, predicted_dims, facing) -> List[Room]:
    rooms_n = []
    # Band 1: verandah
    b1_h = _clamp(predicted_dims.get('verandah', (net_w, 1.8))[1], 1.5, net_d * 0.22)
    b1_h = round(b1_h, 2)

    # Band 3: service zone height from kitchen depth
    b3_h = _clamp(predicted_dims.get('kitchen', (2.5, 2.2))[1], 2.0, net_d * 0.28)
    b3_h = round(b3_h, 2)

    # Band 4: bedroom zone height from predicted dims (not remaining height)
    _mb_d = round(predicted_dims.get('master_bedroom', (3.0, 2.8))[1], 2)
    _ta_d = round(predicted_dims.get('toilet_attached', (1.5, 1.5))[1], 2) if 'toilet_attached' in ROOM_LISTS[bhk] else 0.0
    _extra_beds = [r for r in ('bedroom_2', 'bedroom_3', 'bedroom_4') if r in ROOM_LISTS[bhk]]
    _max_bed_d = max((predicted_dims.get(r, (2.5, 2.5))[1] for r in _extra_beds), default=_mb_d)
    _private_col_h = round(_mb_d + _ta_d, 2)
    b4_h = round(max(_private_col_h, _max_bed_d, 2.5), 2)
    b4_h = round(min(b4_h, net_d * 0.55), 2)

    # Band 2: ALL remaining space (no dead zone)
    b2_h = round(net_d - b1_h - b3_h - b4_h, 2)
    b2_h = max(b2_h, 2.4)

    # Scale if total exceeds net_d
    total = b1_h + b2_h + b3_h + b4_h
    if total > net_d:
        s = net_d / total
        b1_h = round(b1_h * s, 2)
        b2_h = round(b2_h * s, 2)
        b3_h = round(b3_h * s, 2)
        b4_h = round(net_d - b1_h - b2_h - b3_h, 2)

    y_b4 = 0.0
    y_b3 = round(b4_h, 2)
    y_b2 = round(b4_h + b3_h, 2)
    y_b1 = round(b4_h + b3_h + b2_h, 2)

    svc_basis = [predicted_dims.get(r, (0.0, 0.0))[0] for r in ('kitchen', 'utility', 'toilet_common') if r in ROOM_LISTS[bhk]]
    svc_w = round(_clamp((max(svc_basis) + 0.1) if svc_basis else net_w * 0.3, net_w * 0.28, net_w * 0.42), 2)
    lv_w = round(net_w - svc_w, 2)

    rooms_n.append(('verandah', 0.0, y_b1, net_w, b1_h))

    pooja_w = 0.0
    if 'pooja' in ROOM_LISTS[bhk]:
        pooja_w = round(min(predicted_dims.get('pooja', (1.2, 1.2))[0], svc_w), 2)
    band2_left = round((net_w - pooja_w) if pooja_w else lv_w, 2)

    public_y = y_b3 if bhk == 2 else y_b2
    public_d = round(y_b1 - y_b3, 2) if bhk == 2 else b2_h
    living_w = band2_left if 'dining' not in ROOM_LISTS[bhk] else round(max(band2_left * 0.58, NBC_MIN_WIDTH['living']), 2)
    living_w = min(living_w, band2_left)
    rooms_n.append(('living', 0.0, public_y, living_w, public_d))

    if 'dining' in ROOM_LISTS[bhk]:
        dining_w = round(max(band2_left - living_w, 0.6), 2)
        rooms_n.append(('dining', living_w, public_y, dining_w, public_d))
    if 'pooja' in ROOM_LISTS[bhk]:
        rooms_n.append(('pooja', round(net_w - pooja_w, 2), y_b2, pooja_w, b2_h))

    tcom_w = 0.0
    if 'toilet_common' in ROOM_LISTS[bhk]:
        tcom_w = round(max(predicted_dims.get('toilet_common', (1.4, 1.4))[0], NBC_MIN_WIDTH['toilet_common']), 2)

    kitchen_w = round(max(predicted_dims.get('kitchen', (2.5, 2.2))[0], NBC_MIN_WIDTH['kitchen']), 2) if 'kitchen' in ROOM_LISTS[bhk] else 0.0
    utility_w = round(max(predicted_dims.get('utility', (1.5, 1.2))[0], NBC_MIN_WIDTH['utility']), 2) if 'utility' in ROOM_LISTS[bhk] else 0.0
    store_w = round(max(predicted_dims.get('store', (1.5, 1.2))[0], NBC_MIN_WIDTH.get('store', 0.9)), 2) if 'store' in ROOM_LISTS[bhk] else 0.0

    # Valid training layouts consistently keep the service zone on the east side.
    # For 2BHK, match that cluster more closely with a stacked east-side service column.
    if bhk == 2:
        svc_col_w = round(max(kitchen_w, utility_w, tcom_w, net_w * 0.26), 2)
        svc_col_w = round(_clamp(svc_col_w, net_w * 0.26, net_w * 0.34), 2)
        left_block_w = round(net_w - svc_col_w, 2)

        living_w = round(max(left_block_w * 0.62, NBC_MIN_WIDTH['living']), 2)
        living_w = round(min(living_w, left_block_w), 2)
        public_d = round(y_b1 - y_b3, 2)
        for idx, item in enumerate(rooms_n):
            if item[0] == 'living':
                rooms_n[idx] = ('living', 0.0, y_b3, living_w, public_d)
            elif item[0] == 'dining':
                dining_w = round(max(left_block_w - living_w, 0.6), 2)
                rooms_n[idx] = ('dining', living_w, y_b3, dining_w, public_d)

        svc_x = round(net_w - svc_col_w, 2)

        if 'kitchen' in ROOM_LISTS[bhk]:
            kitchen_d = round(min(predicted_dims.get('kitchen', (2.5, 2.2))[1], b2_h), 2)
            kitchen_d = round(max(kitchen_d, 2.0), 2)
            kitchen_y = round(y_b1 - kitchen_d, 2)
            rooms_n.append(('kitchen', svc_x, kitchen_y, svc_col_w, kitchen_d))

        y_cursor = y_b3
        if 'toilet_common' in ROOM_LISTS[bhk]:
            tcom_d = round(min(predicted_dims.get('toilet_common', (1.4, 1.4))[1], b3_h * 0.55), 2)
            tcom_d = round(max(tcom_d, 1.4), 2)
            rooms_n.append(('toilet_common', svc_x, y_cursor, svc_col_w, tcom_d))
            y_cursor = round(y_cursor + tcom_d, 2)

        if 'utility' in ROOM_LISTS[bhk]:
            util_d = round(max(predicted_dims.get('utility', (1.5, 1.2))[1], 1.2), 2)
            util_d = round(min(util_d, max(net_d - y_cursor, 0.6)), 2)
            rooms_n.append(('utility', svc_x, y_cursor, svc_col_w, util_d))

    else:
        gap = 0.20 if 'toilet_common' in ROOM_LISTS[bhk] else 0.0
        if 'toilet_common' in ROOM_LISTS[bhk]:
            tcom_w = round(min(predicted_dims.get('toilet_common', (1.4, 1.4))[0], max(net_w * 0.22, 1.2)), 2)
            rooms_n.append(('toilet_common', 0.0, y_b3, tcom_w, b3_h))

        service_rooms = [r for r in ('kitchen', 'utility', 'store') if r in ROOM_LISTS[bhk]]
        widths = {r: round(max(predicted_dims.get(r, (1.5, 1.5))[0], NBC_MIN_WIDTH.get(r, 0.9)), 2) for r in service_rooms}
        total_svc = sum(widths.values())
        svc_start = round(max(net_w - total_svc, tcom_w + gap), 2)
        x = svc_start
        for r in service_rooms:
            w = round(min(widths[r], max(net_w - x, 0.6)), 2)
            rooms_n.append((r, x, y_b3, w, b3_h))
            x = round(x + w, 2)


    # master_bedroom: SW corner, depth from h5 model prediction
    mb_w = round(max(
        min(predicted_dims.get('master_bedroom', (3.2, 3.0))[0],
            net_w * 0.45),
        NBC_MIN_WIDTH['master_bedroom']), 2)
    mb_d_pred = round(predicted_dims.get(
        'master_bedroom', (3.2, 2.8))[1], 2)
    # toilet_attached: depth from h5 model prediction
    ta_w = 0.0
    ta_d_pred = 0.0
    if 'toilet_attached' in ROOM_LISTS[bhk]:
        ta_w = round(max(
            min(predicted_dims.get('toilet_attached', (1.5, 1.5))[0],
                mb_w),
            NBC_MIN_WIDTH['toilet_attached']), 2)
        ta_d_pred = round(predicted_dims.get(
            'toilet_attached', (1.5, 1.5))[1], 2)
    # Scale both to fit b4_h using predicted proportions (no hardcoding)
    required_d = mb_d_pred + ta_d_pred
    if required_d > b4_h and required_d > 0:
        scale = b4_h / required_d
        mb_d_actual = round(max(mb_d_pred * scale,
            NBC_MIN_AREA['master_bedroom'] / mb_w), 2)
        ta_d_actual = round(max(ta_d_pred * scale, 1.2), 2)
    else:
        mb_d_actual = round(max(mb_d_pred,
            NBC_MIN_AREA['master_bedroom'] / mb_w), 2)
        mb_d_actual = round(min(mb_d_actual, b4_h - 1.2), 2)
        ta_d_actual = round(max(ta_d_pred, 1.2), 2)
        ta_d_actual = round(min(ta_d_actual, b4_h - mb_d_actual), 2)
    # Place master_bedroom at SW (x=0, y=0)
    rooms_n.append(('master_bedroom', 0.0, 0.0, mb_w, mb_d_actual))
    # Place toilet_attached ABOVE master_bedroom (shared horizontal wall)
    # This satisfies adjacency_rules MUST_SHARE_WALL constraint from DB
    if 'toilet_attached' in ROOM_LISTS[bhk]:
        if bhk == 2:
            ta_w = mb_w
        rooms_n.append((
            'toilet_attached', 0.0, mb_d_actual,
            ta_w, ta_d_actual))

    extra = [r for r in ('bedroom_2', 'bedroom_3', 'bedroom_4')
             if r in ROOM_LISTS[bhk]]
    pred_w = {r: round(max(
                  predicted_dims.get(r, (2.5, 2.5))[0],
                  NBC_MIN_WIDTH.get(r, 2.1)), 2)
              for r in extra}
    pred_d = {r: round(max(
                  predicted_dims.get(r, (2.5, 2.5))[1],
                  NBC_MIN_AREA.get(r, 7.5) / pred_w[r]), 2)
              for r in extra}
    available_w = round(net_w - mb_w, 2)
    total_pred = sum(pred_w.values())
    if total_pred > available_w and total_pred > 0:
        s = available_w / total_pred
        pred_w = {r: round(max(w * s,
                               NBC_MIN_WIDTH.get(r, 2.1)), 2)
                  for r, w in pred_w.items()}
    x = round(mb_w, 2)
    for r in extra:
        w = round(min(pred_w[r], max(net_w - x, 0.6)), 2)
        d = round(min(pred_d[r], b4_h), 2)
        rooms_n.append((r, x, y_b4, w, d))
        x = round(x + w, 2)

    out = []
    rooms_n = apply_wall_offsets(rooms_n, net_w, net_d)
    for rt, x, y, w, d in rooms_n:
        if facing == 'S':
            nx, ny, nw, nd = x, net_d - y - d, w, d
        elif facing == 'E':
            x0, y0, w0, d0 = x / net_w, y / net_d, w / net_w, d / net_d
            nx, ny, nw, nd = net_w * (1 - (y0 + d0)), net_d * x0, net_w * d0, net_d * w0
        elif facing == 'W':
            x0, y0, w0, d0 = x / net_w, y / net_d, w / net_w, d / net_d
            nx, ny, nw, nd = net_w * y0, net_d * (1 - (x0 + w0)), net_w * d0, net_d * w0
        else:
            nx, ny, nw, nd = x, y, w, d
        nx, ny = round(_clamp(nx, 0.0, max(net_w - nw, 0.0)), 2), round(_clamp(ny, 0.0, max(net_d - nd, 0.0)), 2)
        nw, nd = round(max(nw, 0.6), 2), round(max(nd, 0.6), 2)
        area = round(nw * nd, 2)
        cx_pct = round((nx + nw / 2) / net_w, 3)
        cy_pct = round((ny + nd / 2) / net_d, 3)
        out.append(Room(rt, nx, ny, nw, nd, area, cx_pct, cy_pct, _compass(cx_pct, cy_pct)))
    return out

def build_wall_geometry(fp):
    """
    Builds Shapely wall mass from room voids.
    After wall offset fix, rooms no longer fill net area.
    Gaps between rooms = actual wall material.
    Returns Shapely Polygon or MultiPolygon.
    """
    from shapely.geometry import box as shapely_box
    from shapely.ops import unary_union
    net_poly = shapely_box(0, 0, fp.net_w, fp.net_d)
    room_voids = unary_union([
        shapely_box(r.x, r.y, r.x + r.width, r.y + r.depth)
        for r in fp.rooms
    ])
    wall_mass = net_poly.difference(room_voids)
    return wall_mass


def build_wall_polygons(walls, doors, windows, floor_index=None):
    """
    Convert WallSegment list + openings into final Shapely wall geometry.

    Three mandatory fixes applied (see CLAUDE.md §6):

    FIX 1 — Split gap subtraction by wall type:
        door gaps   -> subtracted from interior solid only
        window gaps -> subtracted from exterior solid only
        (Subtracting all gaps from both solids creates unnecessary boolean
         operations and micro-geometry errors at wall junctions.)

    FIX 2 — MIN_WALL_LEN = 0.02 m (not 0.05):
        0.05 m threshold discards valid toilet and utility walls (~0.03 m overlap
        segments). 0.02 m retains all valid segments while still rejecting
        degenerate zero-length artefacts from float rounding.

    FIX 3 — floor_index filtering as first gate:
        When walls/doors/windows from multiple floors are passed together (e.g.
        combined G+1 context), filter to the target floor before any geometry
        operation. Uses getattr fallback so function works with objects that
        predate the floor_index field.

    Returns WallGeometry(exterior, interior) — no matplotlib, no ax parameter.
    """
    from shapely.geometry import box as _box
    from shapely.ops import unary_union as _uu

    # ── FIX 3: floor isolation ─────────────────────────────────────────────────
    if floor_index is not None:
        walls   = [w for w in walls   if getattr(w, 'floor_index', floor_index) == floor_index]
        doors   = [d for d in doors   if getattr(d, 'floor_index', floor_index) == floor_index]
        windows = [w for w in windows if getattr(w, 'floor_index', floor_index) == floor_index]

    # ── Helpers (use current WallSegment field names: x1,y1,x2,y2,direction) ──
    def _wbx(w):
        """WallSegment -> Shapely box using half-thickness expansion."""
        if w.length < MIN_WALL_LEN:          # FIX 2: 0.02 m threshold
            return None
        t2 = w.thickness / 2.0
        if w.direction == 'H':
            x0, x1 = sorted((w.x1, w.x2))
            return _box(x0, w.y1 - t2, x1, w.y1 + t2)
        y0, y1 = sorted((w.y1, w.y2))
        return _box(w.x1 - t2, y0, w.x1 + t2, y1)

    def _gbx(wall, width, position):
        """Opening -> overcut gap box. position is fractional [0, 1] along wall."""
        t2 = (wall.thickness + GAP_OVERCUT) / 2.0
        if wall.direction == 'H':
            cx = wall.x1 + position * (wall.x2 - wall.x1)
            return _box(cx - width / 2.0, wall.y1 - t2, cx + width / 2.0, wall.y1 + t2)
        cy = wall.y1 + position * (wall.y2 - wall.y1)
        return _box(wall.x1 - t2, cy - width / 2.0, wall.x1 + t2, cy + width / 2.0)

    # ── Phase 1: wall boxes ────────────────────────────────────────────────────
    ext_boxes = [b for w in walls if w.wall_type == 'exterior'
                   for b in [_wbx(w)] if b is not None]
    int_boxes  = [b for w in walls if w.wall_type != 'exterior'
                   for b in [_wbx(w)] if b is not None]

    # ── Phase 2: merge into two solids ────────────────────────────────────────
    ext_solid = _uu(ext_boxes) if ext_boxes else None
    int_solid = _uu(int_boxes)  if int_boxes  else None

    # ── Phase 3: build gap sets split by opening type (FIX 1) ─────────────────
    door_gaps   = [_gbx(d.wall, d.width, d.position)
                   for d in doors
                   if getattr(d, 'wall', None) is not None]

    window_gaps = [_gbx(w.wall, w.width, w.position)
                   for w in windows
                   if getattr(w, 'wall', None) is not None
                   and w.wall.wall_type == 'exterior']   # windows never on interior walls

    # ── Phase 4: subtract — doors from interior, windows from exterior (FIX 1) ─
    if door_gaps and int_solid is not None:
        int_solid = int_solid.difference(_uu(door_gaps))

    if window_gaps and ext_solid is not None:
        ext_solid = ext_solid.difference(_uu(window_gaps))

    return WallGeometry(exterior=ext_solid, interior=int_solid)


# SECTION 7 — WALL NETWORK BUILDER
def build_wall_network(rooms: List[Room], net_w: float, net_d: float) -> List[WallSegment]:
    # Wall thickness tolerance: rooms now have WALL_INT (0.115m) gaps
    # between them and HALF_EXT (0.115m) offsets to net boundary.
    tol = 0.14      # gap detection (covers ~0.115m)
    min_len = 0.10  # minimum wall segment length

    walls, seen = [], set()

    # Interior walls: find near-parallel room edges separated by wall gap and
    # place the wall segment at the midpoint of that gap.
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            # Vertical wall between rooms (side-by-side)
            if abs((a.x + a.width) - b.x) < tol or abs((b.x + b.width) - a.x) < tol:
                y1 = max(a.y, b.y)
                y2 = min(a.y + a.depth, b.y + b.depth)
                if y2 - y1 > min_len:
                    if abs((a.x + a.width) - b.x) < tol:
                        x = round((a.x + a.width + b.x) / 2, 3)
                        west, east = a.room_type, b.room_type
                    else:
                        x = round((b.x + b.width + a.x) / 2, 3)
                        west, east = b.room_type, a.room_type
                    key = ('V', x, round(y1, 3), round(y2, 3))
                    if key not in seen:
                        walls.append(WallSegment(x, round(y1, 2), x, round(y2, 2),
                                                 WALL_INT, 'interior', west, east, False))
                        seen.add(key)

            # Horizontal wall between rooms (stacked)
            if abs((a.y + a.depth) - b.y) < tol or abs((b.y + b.depth) - a.y) < tol:
                x1 = max(a.x, b.x)
                x2 = min(a.x + a.width, b.x + b.width)
                if x2 - x1 > min_len:
                    if abs((a.y + a.depth) - b.y) < tol:
                        y = round((a.y + a.depth + b.y) / 2, 3)
                        south, north = a.room_type, b.room_type
                    else:
                        y = round((b.y + b.depth + a.y) / 2, 3)
                        south, north = b.room_type, a.room_type
                    key = ('H', y, round(x1, 3), round(x2, 3))
                    if key not in seen:
                        walls.append(WallSegment(round(x1, 2), y, round(x2, 2), y,
                                                 WALL_INT, 'interior', south, north, False))
                        seen.add(key)

    # Exterior walls: create segments ON the net boundary, attributed to the
    # nearest room edge within the exterior offset tolerance.
    ext_tol = HALF_EXT + 0.03  # ~0.145m
    for r in rooms:
        # West boundary
        if abs(r.x - HALF_EXT) < ext_tol or r.x < ext_tol:
            key = ('VW', 0.0, round(r.y, 3), round(r.y + r.depth, 3), r.room_type)
            if key not in seen and r.depth > min_len:
                walls.append(WallSegment(0.0, round(r.y, 2), 0.0, round(r.y + r.depth, 2),
                                         WALL_EXT, 'exterior', r.room_type, 'outside', False))
                seen.add(key)
        # East boundary
        if abs((net_w - (r.x + r.width)) - HALF_EXT) < ext_tol or abs(r.x + r.width - (net_w - HALF_EXT)) < ext_tol:
            key = ('VE', net_w, round(r.y, 3), round(r.y + r.depth, 3), r.room_type)
            if key not in seen and r.depth > min_len:
                walls.append(WallSegment(round(net_w, 2), round(r.y, 2), round(net_w, 2), round(r.y + r.depth, 2),
                                         WALL_EXT, 'exterior', r.room_type, 'outside', False))
                seen.add(key)
        # South boundary
        if abs(r.y - HALF_EXT) < ext_tol or r.y < ext_tol:
            key = ('HS', 0.0, round(r.x, 3), round(r.x + r.width, 3), r.room_type)
            if key not in seen and r.width > min_len:
                walls.append(WallSegment(round(r.x, 2), 0.0, round(r.x + r.width, 2), 0.0,
                                         WALL_EXT, 'exterior', r.room_type, 'outside', False))
                seen.add(key)
        # North boundary
        if abs((net_d - (r.y + r.depth)) - HALF_EXT) < ext_tol or abs(r.y + r.depth - (net_d - HALF_EXT)) < ext_tol:
            key = ('HN', net_d, round(r.x, 3), round(r.x + r.width, 3), r.room_type)
            if key not in seen and r.width > min_len:
                walls.append(WallSegment(round(r.x, 2), round(net_d, 2), round(r.x + r.width, 2), round(net_d, 2),
                                         WALL_EXT, 'exterior', r.room_type, 'outside', False))
                seen.add(key)

    return walls


# SECTION 8 — DOOR PLACEMENT
DOOR_CORNER_CLEAR = 0.60   # 600 mm minimum from wall corner to door edge


def _safe_door_pos(wall_length: float, door_width: float,
                   corner_clear: float = DOOR_CORNER_CLEAR) -> float:
    """Return fractional position (0–1) for door centre that keeps
    ≥600 mm clearance between the door edge and each wall corner.
    Falls back to 0.5 if the wall is too short to satisfy the constraint."""
    min_centre = door_width / 2.0 + corner_clear
    max_centre = wall_length - door_width / 2.0 - corner_clear
    if max_centre < min_centre:
        return 0.5                        # wall too short; centre is best we can do
    centre = max(min_centre, min(wall_length / 2.0, max_centre))
    return round(centre / wall_length, 4)


def _hinge_for_pos(pos: float) -> str:
    """Hinge toward the nearer wall end so the door leaf lies clear of the corner."""
    return 'left' if pos <= 0.5 else 'right'


def place_doors(rooms: List[Room], walls: List[WallSegment], bhk: int) -> List[DoorOpening]:
    room_map = {r.room_type: r for r in rooms}
    doors = []
    vwalls = [w for w in walls if w.wall_type == 'exterior' and w.room_left == 'verandah']
    if vwalls:
        if CURRENT_FACING == 'N':
            wall = max(vwalls, key=lambda w: w.midpoint[1])
        elif CURRENT_FACING == 'S':
            wall = min(vwalls, key=lambda w: w.midpoint[1])
        elif CURRENT_FACING == 'E':
            wall = max(vwalls, key=lambda w: w.midpoint[0])
        else:
            wall = min(vwalls, key=lambda w: w.midpoint[0])
        wall.has_opening = True
        _me_w   = round(get_door_width_from_db('MAIN_ENTRANCE_DOOR'), 2)
        _me_pos = _safe_door_pos(wall.length, _me_w)
        doors.append(DoorOpening('MAIN ENTRANCE', wall, _me_pos, _me_w,
                                 'swing', _hinge_for_pos(_me_pos),
                                 'verandah', 'outside', 'verandah'))
    counter = 1
    _, _req_conns = build_adjacency_graph_from_db(bhk)
    for from_r, to_r, door_type, fallback in _req_conns:
        if from_r not in room_map or to_r not in room_map:
            continue
        matches = [w for w in walls if _pair(w.room_left, w.room_right) == _pair(from_r, to_r)]
        if not matches:
            # Fallback: living->bedroom has no shared wall (Band 3 separates them).
            # Find an intermediate Band 3 room that shares a wall with the bedroom
            # and create a door there. Access path: living->kitchen(existing)->bedroom.
            if from_r == 'living' and (to_r.startswith('bedroom') or to_r == 'master_bedroom'):
                _intermediates = ['kitchen', 'utility', 'store', 'toilet_common']
                _placed = False
                for mid_r in _intermediates:
                    if mid_r not in room_map:
                        continue
                    wall_mid_bed = [w for w in walls
                                    if _pair(w.room_left, w.room_right) == _pair(mid_r, to_r)]
                    if wall_mid_bed:
                        wall = max(wall_mid_bed, key=lambda w: w.length)
                        ptype = PASSAGE_TYPE_MAP.get((from_r, to_r),
                                    PASSAGE_TYPE_MAP.get((to_r, from_r), 'BEDROOM_DOOR'))
                        width = get_door_width_from_db(ptype)
                        if wall.length < width + 0.4:
                            width = max(round(wall.length * 0.55, 2), 0.6)
                        wall.has_opening = True
                        _fb_pos = _safe_door_pos(wall.length, width)
                        doors.append(DoorOpening(
                            f'D{counter}', wall, _fb_pos, round(width, 2),
                            door_type, _hinge_for_pos(_fb_pos), to_r, mid_r, to_r))
                        counter += 1
                        _placed = True
                        break
                if _placed:
                    continue
            continue
        wall = max(matches, key=lambda w: w.length)
        ptype = PASSAGE_TYPE_MAP.get((from_r, to_r), PASSAGE_TYPE_MAP.get((to_r, from_r), 'BEDROOM_DOOR'))
        width = get_door_width_from_db(ptype)
        if width == 0.9 and door_type == 'archway':
            width = 1.2
        if wall.length < width + 0.4:
            width = max(round(wall.length * 0.55, 2), 0.6)
        wall.has_opening = True
        _dw  = round(width if width else fallback, 2)
        _pos = _safe_door_pos(wall.length, _dw)
        doors.append(DoorOpening(f'D{counter}', wall, _pos, _dw,
                                 door_type, _hinge_for_pos(_pos), to_r, from_r, to_r))
        counter += 1
    return doors


# SECTION 9 — WINDOW PLACEMENT
def place_windows(rooms: List[Room], walls: List[WallSegment], window_scores: dict, facing: str) -> List[WindowOpening]:
    net_w = max(max(w.x1, w.x2) for w in walls) if walls else 0.0
    net_d = max(max(w.y1, w.y2) for w in walls) if walls else 0.0
    by_room = {}
    for w in walls:
        if w.wall_type == 'exterior':
            by_room.setdefault(w.room_left, []).append(w)
    habitable = {'master_bedroom', 'bedroom_2', 'bedroom_3', 'bedroom_4', 'living', 'dining', 'kitchen'}
    wet = {'toilet_attached', 'toilet_common', 'utility'}
    wins, counter = [], 1
    for r in rooms:
        rws = by_room.get(r.room_type, [])
        if not rws:
            continue
        if r.room_type in habitable:
            scored = sorted([(window_scores.get(_cardinal_for_wall(w, net_w, net_d), 0.5), w) for w in rws], key=lambda t: t[0], reverse=True)
            placed = 0
            for _, w in scored:
                if w.has_opening:
                    continue
                if placed == 0:
                    ww = max(min(w.length * 0.50, 1.50), 0.60)
                    wins.append(WindowOpening(f'W{counter}', w, 0.50, round(ww, 2), 1.20, 0.90, r.room_type, False))
                    counter += 1
                    placed += 1
                elif r.room_type.startswith('bedroom') or r.room_type == 'master_bedroom':
                    if w.length >= 0.8:
                        ww = max(min(w.length * 0.40, 1.05), 0.45)
                        wins.append(WindowOpening(f'W{counter}', w, 0.50, round(ww, 2), 1.20, 0.90, r.room_type, False))
                        counter += 1
                    break
        elif r.room_type in wet:
            for w in rws:
                if not w.has_opening:
                    wins.append(WindowOpening(f'W{counter}', w, 0.50, 0.45, 0.45, 1.50, r.room_type, True))
                    counter += 1
                    break
    return wins


# SECTION 10 — FEATURE VECTOR BUILDER
def _pfx(rt: str) -> str:
    return {
        "master_bedroom": "masterbedr",
        "toilet_attached": "toiletatta",
        "living": "living",
        "kitchen": "kitchen",
        "verandah": "verandah",
        "bedroom_2": "bedroom2",
        "bedroom_3": "bedroom3",
        "dining": "dining",
        "toilet_common": "toiletcomm",
        "utility": "utility",
        "pooja": "pooja",
        "bedroom_4": "bedroom4",
        "store": "store",
    }.get(rt, rt.replace("_", ""))


def build_feature_vector(fp: FloorPlan, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Build the exact model feature vector.

    Order must match the trained FEATURE_COLS / clf.feature_names_in_.
    Values are computed from fp.placement (output of _place_rooms).
    """
    if feature_cols is None:
        try:
            clf, _, _ = ModelLoader.get()
            feature_cols = list(getattr(clf, "feature_names_in_", []))
        except Exception:
            feature_cols = []

    pl = fp.placement or {}
    net_w = float(fp.net_w)
    net_d = float(fp.net_d)

    row = {c: 0.0 for c in (feature_cols or [])}

    # Base inputs
    base = {
        "plot_w": float(fp.plot_w),
        "plot_d": float(fp.plot_d),
        "plot_area": float(round(fp.plot_w * fp.plot_d, 2)),
        "net_w": float(fp.net_w),
        "net_d": float(fp.net_d),
        "net_area": float(round(fp.net_w * fp.net_d, 2)),
        "bhk": float(fp.bhk),
        "facing_code": float(fp.facing_code),
        "climate_code": float(fp.climate_code),
    }
    for k, v in base.items():
        if k in row:
            row[k] = v

    # Per-room (w, d, area, cx_pct, cy_pct)
    for rt in ROOM_UNIVERSE:
        p = _pfx(rt)
        if rt in pl:
            a = pl[rt]
            w = float(a.get("w", 0.0))
            d = float(a.get("d", 0.0))
            cx = float(a.get("cx", float(a.get("x", 0.0)) + w / 2.0))
            cy = float(a.get("cy", float(a.get("y", 0.0)) + d / 2.0))
            row[f"{p}_w"] = round(w, 2)
            row[f"{p}_d"] = round(d, 2)
            row[f"{p}_area"] = round(w * d, 2)
            row[f"{p}_cx_pct"] = round(cx / max(net_w, 0.01), 3)
            row[f"{p}_cy_pct"] = round((net_d - cy) / max(net_d, 0.01), 3)
        else:
            row[f"{p}_w"] = row[f"{p}_d"] = row[f"{p}_area"] = 0.0
            row[f"{p}_cx_pct"] = row[f"{p}_cy_pct"] = 0.0

    # Absolute positions + zones
    zone_area = {"public_zone": 0.0, "private_zone": 0.0, "wet_zone": 0.0, "service_zone": 0.0}
    for rt in ROOM_UNIVERSE:
        p = _pfx(rt)
        if rt in pl:
            row[f"{p}_x_abs"] = float(pl[rt].get("x", 0.0))
            y0 = float(pl[rt].get("y", 0.0))
            d0 = float(pl[rt].get("d", 0.0))
            row[f"{p}_y_abs"] = float(round(net_d - (y0 + d0), 3))
            z = ROOM_ZONE.get(rt, "public_zone")
            zone_area[z] += float(pl[rt].get("w", 0.0) * pl[rt].get("d", 0.0))
        else:
            row[f"{p}_x_abs"] = row[f"{p}_y_abs"] = 0.0

    denom = max(float(net_w * net_d), 0.01)
    row["zone_public_area_pct"] = round(zone_area["public_zone"] / denom, 6)
    row["zone_private_area_pct"] = round(zone_area["private_zone"] / denom, 6)
    row["zone_wet_area_pct"] = round(zone_area["wet_zone"] / denom, 6)
    row["zone_service_area_pct"] = round(zone_area["service_zone"] / denom, 6)

    # Wall geometry (match training helper)
    ext_c, int_c, ext_l, int_l = _wall_stats(pl, net_w, net_d, tol=0.05)
    gross = float(sum(float(a.get("w", 0.0)) * float(a.get("d", 0.0)) for a in pl.values()))
    wall_area = float(ext_l * 0.23 + int_l * 0.115)
    row["wall_count_ext"] = float(ext_c)
    row["wall_count_int"] = float(int_c)
    row["wall_total_length_ext"] = float(ext_l)
    row["wall_total_length_int"] = float(int_l)
    row["gross_built_area"] = round(gross, 3)
    row["net_carpet_area"] = round(max(gross - wall_area, 0.0), 3)

    # Adjacency flags (tolerance 0.5m as requested)
    def adj(rt1, rt2):
        return int(rt1 in pl and rt2 in pl and _adj(pl[rt1], pl[rt2], tol=0.5))

    row["adj_living_verandah"] = adj("living", "verandah")
    row["adj_kitchen_utility"] = adj("kitchen", "utility")
    row["adj_master_toilet"] = adj("master_bedroom", "toilet_attached")
    row["adj_kitchen_dining"] = adj("kitchen", "dining")
    row["adj_living_dining"] = adj("living", "dining")
    row["adj_toilet_common_bedroom"] = int(any(adj("toilet_common", b) for b in ("master_bedroom", "bedroom_2", "bedroom_3", "bedroom_4")))

    # Plumbing cluster valid (same as training band-membership check)
    y_b3 = float(getattr(fp, "band_y_b3", 0.0))
    y_b2 = float(getattr(fp, "band_y_b2", 0.0))
    # Training/model convention uses y=0 at NORTH. Flip band bounds to match.
    y_b3_f = float(net_d - y_b2)
    y_b2_f = float(net_d - y_b3)

    def in_service_band(room_key):
        a = pl.get(room_key, {})
        y0 = float(a.get("y", -999))
        d0 = float(a.get("d", 0.0))
        y_f = float(net_d - (y0 + d0))
        return (y_b3_f - 0.3) <= y_f <= (y_b2_f + 0.3)

    def centroid_dist(r1, r2):
        if r1 not in pl or r2 not in pl:
            return 0.0
        return math.sqrt((pl[r1]["cx"] - pl[r2]["cx"]) ** 2 + (pl[r1]["cy"] - pl[r2]["cy"]) ** 2)

    plumbing_ok = 1
    required = {"kitchen", "utility", "toilet_common"}
    present = required.intersection(set(pl.keys()))
    if len(present) >= 2:
        if all(r in pl for r in ("kitchen", "utility", "toilet_common")):
            kit_util_close = centroid_dist("kitchen", "utility") < 4.5
            plumbing_ok = int(in_service_band("kitchen") and in_service_band("utility") and in_service_band("toilet_common") and kit_util_close)
        elif "kitchen" in pl and "utility" in pl:
            plumbing_ok = int(centroid_dist("kitchen", "utility") < 4.5)

    row["plumbing_cluster_valid"] = float(plumbing_ok)

    # Corridor (match training: corridor exists if living + any bedroom exist)
    bedrooms_present = any(r in pl for r in ("master_bedroom", "bedroom_2", "bedroom_3", "bedroom_4"))
    has_corridor = int("living" in pl and bedrooms_present)
    row["has_corridor"] = float(has_corridor)
    row["corridor_width"] = float(0.6 if has_corridor else 0.0)

    df = pd.DataFrame([row])
    if feature_cols:
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0.0
        df = df[feature_cols]
    return df


def score_and_explain(fp: FloorPlan, clf, explainer) -> FloorPlan:
    X = build_feature_vector(fp, feature_cols=list(getattr(clf, 'feature_names_in_', [])))
    fp.score_valid = round(float(clf.predict_proba(X)[0][1]), 3)
    room_map = {r.room_type: r for r in fp.rooms}

    sv = 1.0
    if 'kitchen' in room_map:
        k = room_map['kitchen']
        if k.cx_pct < 0.35 and k.cy_pct < 0.35:
            sv -= 0.40
    if 'master_bedroom' in room_map:
        mb = room_map['master_bedroom']
        if mb.cx_pct > 0.65 and mb.cy_pct > 0.65:
            sv -= 0.40
    if 'pooja' in room_map:
        p = room_map['pooja']
        if not (p.cx_pct > 0.55 and p.cy_pct > 0.55):
            sv -= 0.20

    sn = 1.0
    n_rooms = len([r for r in fp.rooms if r.room_type in NBC_MIN_AREA])
    if n_rooms:
        _net_area = fp.net_w * fp.net_d
        for r in fp.rooms:
            if r.room_type in NBC_MIN_AREA:
                nbc_min = NBC_MIN_AREA[r.room_type]
                # FIX 3: EWS plots use relaxed toilet minimum (NBC EWS exemption)
                if r.room_type in ('toilet_attached', 'toilet_common') and _net_area < 50.0:
                    nbc_min = 1.2
                if r.area < nbc_min * 0.88:
                    sn -= (1.0 / n_rooms)

    sc = 1.0
    pairs = [_pair(w.room_left, w.room_right) for w in fp.walls]
    if _pair('living', 'verandah') not in pairs:
        sc -= 0.35
    if _pair('master_bedroom', 'toilet_attached') not in pairs:
        sc -= 0.35
    if len(fp.doors) == 0:
        sc -= 0.30

    sa = 1.0
    penalties = {
        frozenset(['kitchen', 'master_bedroom']): 0.20,
        frozenset(['kitchen', 'bedroom_2']): 0.20,
        frozenset(['kitchen', 'bedroom_3']): 0.20,
        frozenset(['toilet_attached', 'kitchen']): 0.20,
        frozenset(['toilet_common', 'kitchen']): 0.20,
        frozenset(['toilet_attached', 'dining']): 0.15,
        frozenset(['toilet_common', 'dining']): 0.15,
    }
    seen = set()
    for w in fp.walls:
        if w.has_opening:
            continue
        pair = frozenset([w.room_left, w.room_right])
        if pair in penalties and pair not in seen:
            sa -= penalties[pair]
            seen.add(pair)

    clamp = lambda x: round(max(0.0, min(1.0, x)), 3)
    fp.score_vastu = clamp(sv)
    fp.score_nbc = clamp(sn)
    fp.score_circulation = clamp(sc)
    fp.score_adjacency = clamp(sa)
    fp.score_overall = clamp(0.30 * fp.score_vastu + 0.25 * fp.score_nbc + 0.25 * fp.score_circulation + 0.20 * fp.score_adjacency)

    try:
        shap_vals = explainer.shap_values(X)
        arr = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        idxs = np.argsort(np.abs(arr))[::-1][:5]
        fp.shap_values = {X.columns[i]: round(float(arr[i]), 4) for i in idxs}
    except Exception:
        fp.shap_values = {}

    VASTU_EXPLAIN = {
        'NE': 'NE corner - vastu Ishanya zone, auspicious.',
        'SW': 'SW corner - vastu Nairuthi zone, stable.',
        'NW': 'NW corner - vastu Vayavya zone, movement.',
        'SE': 'SE corner - vastu Agni zone, fire element.',
        'N': 'North side - faces road, public zone.',
        'S': 'South side - private, away from road.',
        'E': 'East side - morning light, positive.',
        'W': 'West side - afternoon, service zone.',
        'C': 'Central position - core circulation.',
    }
    for r in fp.rooms:
        ok = r.area >= NBC_MIN_AREA.get(r.room_type, 0) * 0.88
        fp.explanations[r.room_type] = f"{r.room_type.replace('_', ' ').title()} ({r.width:.1f}m x {r.depth:.1f}m, {r.area:.1f}m2) - {VASTU_EXPLAIN.get(r.compass, '')} Area {'meets NBC' if ok else 'below NBC min'}."
    fp.explanations['overall'] = (
        f"Plan scores {fp.score_overall:.0%} overall. Vastu {fp.score_vastu:.0%}, NBC {fp.score_nbc:.0%}, Circulation {fp.score_circulation:.0%}. "
        f"{'All critical circulation paths connected.' if fp.score_circulation > 0.7 else 'Some circulation paths incomplete.'}"
    )
    return fp


# SECTION 12 — MAIN GENERATE FUNCTION + SELF TEST
def generate(params: dict) -> FloorPlan:
    t0 = time.time()
    clf, dim_model, explainer = ModelLoader.get()
    plot_w, plot_d = float(params['plot_w']), float(params['plot_d'])
    bhk, facing = int(params['bhk']), str(params['facing']).upper()
    district = str(params['district'])
    floors        = int(params.get('floors', 1))
    strategy_mode = str(params.get('strategy_mode', 'standard')).lower()
    strategy_log: List[str] = []   # mutable; populated by _place_rooms() on each attempt
    base_seed = params.get('seed', None)
    if base_seed is None:
        # Deterministic default so the retry loop converges quickly without user-provided seeds.
        base_seed = 82
    else:
        base_seed = int(base_seed)
    seed = int(base_seed)
    plot_area = plot_w * plot_d
    front, rear, side = get_setbacks(plot_area, district)
    climate_zone = get_climate_zone(district)
    window_scores = get_window_scores(district)
    materials = get_materials(district, climate_zone)
    baker_principles = get_baker_principles(plot_area, climate_zone)
    net_w = round(max(plot_w - 2 * side, 3.0), 2)
    net_d = round(max(plot_d - front - rear, 3.0), 2)
    fp = FloorPlan(plot_w, plot_d, bhk, facing, district, net_w, net_d, front, rear, side, climate_zone=climate_zone, facing_code=FACING_MAP.get(facing, 0), climate_code=CLIMATE_MAP.get(climate_zone, 2), materials=materials, baker_principles=baker_principles, seed=seed)
    dims = predict_room_dims(plot_w, plot_d, bhk, fp.facing_code, fp.climate_code, net_w, net_d, dim_model)
    global CURRENT_FACING
    CURRENT_FACING = facing

    # ── Layout plan: computed ONCE, shared across all retry-loop attempts ──────
    # Decisions (room set, zone assignments, corridor type, private-zone order,
    # pooja band) are stable across retries — only the rng seed varies.
    _net_area_gp  = net_w * net_d
    _ews_gp       = _net_area_gp < 50.0
    _small_gp     = _net_area_gp < 90.0
    _bhk_eff      = bhk   # effective BHK for rule loading; may become 1 on EWS plots

    _rooms_gp = get_room_set(bhk, _net_area_gp, floors, _ews_gp, _small_gp)
    if _ews_gp:
        _rooms_gp = [r for r in _rooms_gp
                     if r not in ("dining", "pooja", "store",
                                  "bedroom_2", "bedroom_3", "bedroom_4", "utility")]
        _bhk_eff = 1
    if _small_gp:
        _rooms_gp = [r for r in _rooms_gp if r not in ("dining", "pooja", "store")]

    _mode_gp = strategy_mode if strategy_mode in _re_valid_modes else 'standard'
    _rs_gp   = _re_resolve_conflicts(_re_load_all_rules(_bhk_eff, _mode_gp))

    layout_plan = create_layout_plan(
        {
            "rooms":      _rooms_gp,
            "net_w":      net_w,
            "net_d":      net_d,
            "bhk":        _bhk_eff,
            "facing":     facing,
            "floors":     floors,
            "ews_plot":   _ews_gp,
            "small_plot": _small_gp,
            "net_area":   _net_area_gp,
        },
        _rs_gp,
    )
    # ─────────────────────────────────────────────────────────────────────────

    MAX_ATTEMPTS = 15
    best_pl = None
    best_score = -1.0
    best_seed = seed
    best_attempt = 0
    best_err = None
    best_b4_h = 0.0
    best_y_b3 = 0.0
    best_y_b2 = 0.0
    for attempt in range(MAX_ATTEMPTS):
        seed_try = int(seed) + int(attempt)
        rng = np.random.default_rng(seed_try)
        pl_try, err_type, b4_h_try, y_b3_try, y_b2_try = _place_rooms(
            net_w, net_d, dims, rng, layout_plan, err_p=0.0,
            strategy_log=strategy_log,
        )
        if not pl_try:
            best_err = err_type
            continue
        # Score directly from the constraint_scorer model
        fp.placement = pl_try
        fp.band_b4_h = float(b4_h_try)
        fp.band_y_b3 = float(y_b3_try)
        fp.band_y_b2 = float(y_b2_try)
        X_try = build_feature_vector(fp, feature_cols=list(getattr(clf, "feature_names_in_", [])))
        score_try = float(clf.predict_proba(X_try)[0][1])
        if score_try > best_score:
            best_score = score_try
            best_pl = pl_try
            best_seed = seed_try
            best_attempt = attempt + 1
            best_b4_h = b4_h_try
            best_y_b3 = y_b3_try
            best_y_b2 = y_b2_try
        if score_try > 0.4:
            break
    if best_pl is None:
        raise ValueError(f"Room placement failed after {MAX_ATTEMPTS} attempts: {best_err}")
    # Use the best attempt found
    seed = int(best_seed)
    pl = best_pl
    err_type = None
    b4_h = best_b4_h
    y_b3 = best_y_b3
    y_b2 = best_y_b2
    fp.seed = seed
    fp.seed_attempt = int(best_attempt)
    # Store strategy context: decision log (from layout_plan, computed once) +
    # placement log (from strategy_log, last retry attempt's corridor depth entry).
    fp.strategy_mode = strategy_mode
    fp.strategy_log  = list(layout_plan["decision_log"]) + list(strategy_log)
    # Store placement + band coordinates for feature engineering
    fp.placement = pl
    fp.band_b4_h = float(b4_h)
    fp.band_y_b3 = float(y_b3)
    fp.band_y_b2 = float(y_b2)

    # Convert placement dict into Room objects
    fp.rooms = []
    room_order = list(ROOM_LISTS.get(bhk, []))
    for rt in room_order:
        if rt not in pl:
            continue
        a = pl[rt]
        x, y = float(a['x']), float(a['y'])
        w, d = float(a['w']), float(a['d'])
        cx = float(a.get('cx', x + w / 2.0))
        cy = float(a.get('cy', y + d / 2.0))
        fp.rooms.append(Room(
            rt,
            round(x, 3),
            round(y, 3),
            round(w, 3),
            round(d, 3),
            round(w * d, 3),
            round(cx / max(net_w, 0.01), 3),
            round(cy / max(net_d, 0.01), 3),
            _compass(cx / max(net_w, 0.01), cy / max(net_d, 0.01)),
        ))

    # "corridor" is not in ROOM_LISTS — append it as a Room if placed.
    if "corridor" in pl:
        a = pl["corridor"]
        x, y = float(a['x']), float(a['y'])
        w, d = float(a['w']), float(a['d'])
        cx = float(a.get('cx', x + w / 2.0))
        cy = float(a.get('cy', y + d / 2.0))
        fp.rooms.append(Room(
            "corridor",
            round(x, 3), round(y, 3), round(w, 3), round(d, 3),
            round(w * d, 3),
            round(cx / max(net_w, 0.01), 3),
            round(cy / max(net_d, 0.01), 3),
            _compass(cx / max(net_w, 0.01), cy / max(net_d, 0.01)),
        ))

    # "staircase" is not in ROOM_LISTS (it only appears for G+1 plans)
    # but try_add may have placed it — append it as a Room if so.
    if "staircase" in pl:
        a = pl["staircase"]
        x, y = float(a['x']), float(a['y'])
        w, d = float(a['w']), float(a['d'])
        cx = float(a.get('cx', x + w / 2.0))
        cy = float(a.get('cy', y + d / 2.0))
        fp.rooms.append(Room(
            "staircase",
            round(x, 3), round(y, 3), round(w, 3), round(d, 3),
            round(w * d, 3),
            round(cx / max(net_w, 0.01), 3),
            round(cy / max(net_d, 0.01), 3),
            _compass(cx / max(net_w, 0.01), cy / max(net_d, 0.01)),
        ))

    fp.walls = build_wall_network(fp.rooms, net_w, net_d)
    fp.doors = place_doors(fp.rooms, fp.walls, bhk)
    fp.windows = place_windows(fp.rooms, fp.walls, window_scores, facing)
    fp = score_and_explain(fp, clf, explainer)
    fp.wall_geometry = build_wall_geometry(fp)
    fp.generation_time_s = round(time.time() - t0, 3)
    return fp


if __name__ == '__main__':
    test_cases = [
        {'plot_w': 12, 'plot_d': 15, 'bhk': 2, 'facing': 'N', 'district': 'Coimbatore', 'seed': 42},
        {'plot_w': 9, 'plot_d': 12, 'bhk': 2, 'facing': 'S', 'district': 'Chennai', 'seed': 42},
        {'plot_w': 15, 'plot_d': 20, 'bhk': 3, 'facing': 'E', 'district': 'Madurai', 'seed': 42},
        {'plot_w': 20, 'plot_d': 25, 'bhk': 4, 'facing': 'W', 'district': 'Salem', 'seed': 42},
    ]
    for i, params in enumerate(test_cases):
        print(f"\n{'=' * 60}")
        print(f"Test {i+1}: {params['plot_w']}x{params['plot_d']}m  {params['bhk']}BHK  {params['facing']}-facing  {params['district']}")
        fp = generate(params)
        print(f"  Net area:  {fp.net_w}x{fp.net_d}m")
        print(f"  Setbacks:  front={fp.setback_front}m rear={fp.setback_rear}m side={fp.setback_side}m")
        print(f"  Climate:   {fp.climate_zone}")
        print(f"  Rooms:     {len(fp.rooms)}")
        print(f"  Walls:     {len(fp.walls)} segments ({sum(1 for w in fp.walls if w.wall_type=='exterior')} ext, {sum(1 for w in fp.walls if w.wall_type=='interior')} int)")
        print(f"  Doors:     {len(fp.doors)}")
        print(f"  Windows:   {len(fp.windows)}")
        print(f"  Scores:    valid={fp.score_valid:.2f}  vastu={fp.score_vastu:.2f}  nbc={fp.score_nbc:.2f}  circulation={fp.score_circulation:.2f}  overall={fp.score_overall:.2f}")
        print(f"  Time:      {fp.generation_time_s}s")
        print(f"\n  ROOMS:")
        for r in fp.rooms:
            nbc = NBC_MIN_AREA.get(r.room_type, 0)
            ok = 'OK' if r.area >= nbc * 0.88 else 'X'
            print(f"    {r.room_type:<22} x={r.x:.1f} y={r.y:.1f} w={r.width:.1f} d={r.depth:.1f} area={r.area:.1f}m2 {ok} compass={r.compass}")
        print(f"\n  WALLS:")
        for w in fp.walls:
            print(f"    [{w.wall_type[:3].upper()}] ({w.x1:.1f},{w.y1:.1f})-({w.x2:.1f},{w.y2:.1f}) len={w.length:.2f}m t={w.thickness*1000:.0f}mm {'[OPENING]' if w.has_opening else ''}")
        print(f"\n  DOORS:")
        for d in fp.doors:
            print(f"    {d.label:<14} {d.room_from} -> {d.room_to}  type={d.door_type}  width={d.width}m")
        print(f"\n  WINDOWS:")
        for w in fp.windows:
            kind = 'ventilator' if w.is_ventilator else 'window'
            print(f"    {w.label:<6} {w.room_type:<22} width={w.width:.2f}m  sill={w.sill_height}m  [{kind}]")
        print(f"\n  OVERALL: {fp.explanations.get('overall', '')}")
        print(f"  Materials: {len(fp.materials)} recommended for {params['district']}")
        print(f"  Baker principles: {len(fp.baker_principles)} applicable")









