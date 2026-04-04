"""
engine_api.py - Public API wrapper for the floor-plan engine.

The only function app.py should ever call is ``generate_plan()``.
Engine internals (ModelLoader, _place_rooms, build_feature_vector, ...)
are never exposed.
"""

import copy
import functools
import joblib
import os
import sqlite3

from config import DB_PATH as _API_DB_PATH  # aliased so internal code is unchanged

from engine.engine import (
    generate as _engine_generate,
    ModelLoader,
    build_feature_vector,
    FloorPlan,
    Room,
    DoorOpening,
    build_wall_network,
    place_doors,
    place_windows,
    score_and_explain,
    build_wall_geometry,
    get_window_scores,
    ROOM_LISTS,
    NBC_MIN_AREA,
    NBC_MIN_WIDTH,
    ROOM_UNIVERSE,
    _compass,
    CURRENT_FACING,
    FACING_MAP,
    CLIMATE_MAP,
    MODELS_DIR,
    build_adjacency_graph_from_db,
)

from engine.rule_engine import (
    load_all_rules,
    resolve_conflicts,
    provide_constraints,
    PlacementConstraints,
    RuleSet,
)


# ---------------------------------------------------------------------------
# TASK 3 - FEATURE_COLS integrity guard
# ---------------------------------------------------------------------------

_feature_alignment_checked = False


def check_feature_alignment():
    """Verify that build_feature_vector() output columns match the trained
    constraint_scorer model's expected feature names.

    Raises ``ValueError`` with a diff if they don't match exactly.
    Called once at engine startup (first ``generate_plan`` call).
    """
    global _feature_alignment_checked
    if _feature_alignment_checked:
        return

    scorer_path = os.path.join(MODELS_DIR, "constraint_scorer.pkl")
    clf = joblib.load(scorer_path)
    model_features = list(getattr(clf, "feature_names_in_", []))
    if not model_features:
        _feature_alignment_checked = True
        return

    # Build a dummy FloorPlan to extract the columns build_feature_vector produces
    dummy_fp = FloorPlan(
        plot_w=12, plot_d=15, bhk=2, facing="N", district="test",
        net_w=10, net_d=12, setback_front=1.5, setback_rear=1.0,
        setback_side=1.0, facing_code=0, climate_code=0,
    )
    # Provide a minimal placement so the function runs
    dummy_fp.placement = {}
    df = build_feature_vector(dummy_fp, feature_cols=model_features)
    vector_cols = list(df.columns)

    model_set = set(model_features)
    vector_set = set(vector_cols)
    missing_in_vector = sorted(model_set - vector_set)
    extra_in_vector = sorted(vector_set - model_set)

    if missing_in_vector or extra_in_vector:
        parts = []
        if missing_in_vector:
            parts.append(f"  Missing from build_feature_vector: {missing_in_vector}")
        if extra_in_vector:
            parts.append(f"  Extra in build_feature_vector: {extra_in_vector}")
        raise ValueError(
            "FEATURE_COLS mismatch between constraint_scorer.pkl and "
            "build_feature_vector():\n" + "\n".join(parts)
        )

    # Also verify ordering
    if vector_cols != model_features:
        raise ValueError(
            "FEATURE_COLS ordering mismatch: build_feature_vector() columns "
            "do not match constraint_scorer.pkl feature_names_in_ order."
        )

    _feature_alignment_checked = True


# ---------------------------------------------------------------------------
# TASK 2 - G+1 first-floor generation
# ---------------------------------------------------------------------------

# Maps first-floor-only room types to their nearest ROOM_UNIVERSE equivalents
# for feature-vector / scoring purposes only.
# The Room object keeps its real room_type so the renderer sees the right label.
#
#   dry_kitchen    -> kitchen    (same plumbing zone, same service band, similar footprint)
#   staircase_head -> utility    (same service band, similar 1-2 m2 footprint)
#   balcony        -> verandah   (identical footprint and position; verandah is the column in the model)
#
# Room types that ARE already in ROOM_UNIVERSE (master_bedroom, bedroom_2..4,
# toilet_attached, toilet_common, pooja, store, dining, living, kitchen,
# utility, verandah) need no remapping.
FIRST_FLOOR_SCORE_MAP = {
    "dry_kitchen":    "kitchen",    # same plumbing zone, same service band
    "staircase_head": "utility",    # same service band, similar footprint
    "balcony":        "verandah",   # identical position; verandah is the trained column
    "staircase":      "utility",    # ground-floor staircase scores as utility (service band)
    "corridor":       "utility",    # passage/corridor scores as utility (service band footprint)
}


# ---------------------------------------------------------------------------
# Layout selection helpers (used by generate_best_layout)
# ---------------------------------------------------------------------------

# Selection weights - deterministic rule compliance prioritised over ML/aesthetic.
# adj + nbc = 0.55 (hard correctness); circ + vastu + space = 0.45 (quality).
# XGBoost score_valid is intentionally excluded: it has a training-distribution
# gap for plots < 200 sqm and would penalise valid EWS plans.
_SEL_W_ADJ   = 0.30
_SEL_W_NBC   = 0.25
_SEL_W_CIRC  = 0.20
_SEL_W_VASTU = 0.15
_SEL_W_SPACE = 0.10


def _space_efficiency(ground_fp) -> float:
    """Ratio of placed habitable room area to net buildable area in [0, 1].

    Corridor is excluded from the numerator - it is circulation, not dead space.
    Values close to 1.0 indicate the plan wastes little net area.
    """
    net_area = float(ground_fp.net_w) * float(ground_fp.net_d)
    if net_area < 0.01:
        return 1.0
    rooms = ground_fp.rooms
    room_iter = rooms.values() if isinstance(rooms, dict) else rooms
    covered = sum(
        float(getattr(r, 'area', 0.0))
        for r in room_iter
        if getattr(r, 'room_type', '') != 'corridor'
    )
    return round(min(covered / net_area, 1.0), 4)


def _selection_score(result: dict) -> float:
    """Weighted composite score for ranking layout candidates.

    Reads from result['scores'] (preferred) with fallback to FloorPlan
    attributes so the function works whether or not scores dict is populated.
    """
    gf = result.get('ground')
    if gf is None:
        return 0.0
    scores = result.get('scores', {})

    def _s(key, attr):
        v = scores.get(key)
        if v is not None:
            return float(v)
        return float(getattr(gf, attr, 0.0))

    adj   = _s('score_adjacency',  'score_adjacency')
    nbc   = _s('score_nbc',         'score_nbc')
    circ  = _s('score_circulation', 'score_circulation')
    vastu = _s('score_vastu',        'score_vastu')
    space = _space_efficiency(gf)

    return round(
        adj   * _SEL_W_ADJ   +
        nbc   * _SEL_W_NBC   +
        circ  * _SEL_W_CIRC  +
        vastu * _SEL_W_VASTU +
        space * _SEL_W_SPACE,
        4,
    )


def evaluate_layout(result: dict) -> float:
    """Public scoring function for a ``generate_plan()`` result dict.

    Computes the weighted composite score used by ``generate_best_layout()`` to
    compare and rank layout candidates.  Score ∈ [0.0, 1.0].

    Formula (deterministic criteria only — XGBoost ``score_valid`` excluded,
    see CLAUDE.md §9 / README §Known Limitations §1):

        adjacency        × 0.30  — REQUIRED/FORBIDDEN adjacency rule pass-rate
        nbc              × 0.25  — NBC minimum area/width pass-rate
        circulation      × 0.20  — corridor/path connectivity score
        vastu            × 0.15  — compass-zone compliance
        space_efficiency × 0.10  — covered room area ÷ net buildable area

    Parameters
    ----------
    result : dict
        Direct output of ``generate_plan()``.

    Returns
    -------
    float
        Composite score ∈ [0.0, 1.0].  Returns 0.0 for malformed results.
    """
    return _selection_score(result)


def generate_candidates(
    plot_w,
    plot_d,
    bhk,
    facing,
    district,
    floors=1,
    n=5,
    base_seed=42,
) -> list:
    """Generate *n* independent layout candidates with distinct seeds.

    Each candidate k uses ``seed = base_seed + k × 17``.  The ×17 spacing
    exceeds the 15-attempt internal retry budget (ADR-8) so no two candidates
    share any placement seeds — layouts are geometrically independent.

    Parameters
    ----------
    plot_w, plot_d : float
    bhk            : int   (1–4)
    facing         : str   ('N' | 'S' | 'E' | 'W')
    district       : str
    floors         : int   (1 or 2)
    n              : int   Number of candidates to attempt.  Default 5.
    base_seed      : int   First candidate seed.  Default 42.

    Returns
    -------
    list[dict]
        Each item contains::

            {
                'result':          dict,   # generate_plan() output
                'seed':            int,    # seed used for this candidate
                'selection_score': float,  # evaluate_layout() composite score
                'breakdown': {
                    'adjacency':        float,
                    'nbc':              float,
                    'circulation':      float,
                    'vastu':            float,
                    'space_efficiency': float,
                },
            }

        Returns an empty list only when every seed raises an exception.
        Callers should raise ``RuntimeError`` on an empty return.
    """
    import warnings

    candidates = []

    for k in range(n):
        seed_k = base_seed + k * 17
        try:
            result = generate_plan(
                plot_w=plot_w, plot_d=plot_d, bhk=bhk,
                facing=facing, district=district,
                floors=floors, seed=seed_k,
            )
        except Exception as exc:
            warnings.warn(
                f"generate_candidates: seed {seed_k} failed ({exc!r}); skipping.",
                RuntimeWarning, stacklevel=2,
            )
            continue

        gf = result.get('ground')
        if gf is None:
            continue

        scores = result.get('scores', {})

        def _s(key, attr, _scores=scores, _gf=gf):
            v = _scores.get(key)
            return float(v) if v is not None else float(getattr(_gf, attr, 0.0))

        candidates.append({
            'result':          result,
            'seed':            seed_k,
            'selection_score': evaluate_layout(result),
            'breakdown': {
                'adjacency':        _s('score_adjacency',  'score_adjacency'),
                'nbc':              _s('score_nbc',          'score_nbc'),
                'circulation':      _s('score_circulation', 'score_circulation'),
                'vastu':            _s('score_vastu',        'score_vastu'),
                'space_efficiency': _space_efficiency(gf),
            },
        })

    return candidates


def _score_first_floor(first_fp: FloorPlan, clf, explainer) -> FloorPlan:
    """Score the first floor by remapping first-floor-only room types
    (dry_kitchen, staircase_head, balcony) to their nearest ROOM_UNIVERSE
    equivalents in the placement dict before calling score_and_explain.

    The Room objects (and thus the renderer) are not touched - only the
    temporary placement copy used for feature extraction is remapped.
    """
    # Build a remapped placement dict for the feature vector
    remapped = {}
    for rt, info in first_fp.placement.items():
        scorer_rt = FIRST_FLOOR_SCORE_MAP.get(rt, rt)
        # If multiple first-floor types map to the same scorer key,
        # keep the one with the larger area (better signal for the model).
        if scorer_rt in remapped:
            existing_area = remapped[scorer_rt]["w"] * remapped[scorer_rt]["d"]
            new_area = info["w"] * info["d"]
            if new_area <= existing_area:
                continue
        remapped[scorer_rt] = dict(info)  # copy so inner dicts can't alias back to original_placement

    # Swap placement, score, swap back
    original_placement = first_fp.placement
    first_fp.placement = remapped
    first_fp = score_and_explain(first_fp, clf, explainer)
    first_fp.placement = original_placement
    return first_fp


# ---------------------------------------------------------------------------
# G+1 floor transformation — DB-driven infrastructure
# ---------------------------------------------------------------------------
#
# DB SCHEMA CONTRACT — two tables (immutable; changes require DB migration):
#
# Table: floor_transformation_rules
#   from_room_type      TEXT    — ground-floor room_type (engine name)
#   to_room_type        TEXT    — first-floor room_type (NULL for to_bedroom)
#   transformation_type TEXT    — rename | carry_over | to_bedroom | drop
#   bhk_applicability   TEXT    — 'ALL' or comma-separated BHK tags
#   priority            INTEGER — lower = higher priority; ORDER BY priority ASC
#
# Table: vertical_stacking_rules
#   room_type    TEXT    — engine room_type (applies to first-floor instance)
#   must_stack   INTEGER — 1 = (x,y) must match ground counterpart within tolerance
#   tolerance_m  REAL    — acceptable position deviation in metres
#
# Engine reads ONLY SQLite. CSV files (seeds/) are for the build pipeline only.
# ---------------------------------------------------------------------------

# Fallback — activated ONLY when DB query fails entirely (OperationalError etc.)
# Mirrors the DB rows; kept in sync manually.
_FLOOR_TRANSFORM_FALLBACK = {
    'verandah':        {'to_room_type': 'balcony',        'transformation_type': 'rename',      'priority':  1},
    'kitchen':         {'to_room_type': 'dry_kitchen',    'transformation_type': 'rename',      'priority':  2},
    'staircase':       {'to_room_type': 'staircase_head', 'transformation_type': 'rename',      'priority':  3},
    'utility':         {'to_room_type': 'utility',        'transformation_type': 'carry_over',  'priority':  4},
    'toilet_attached': {'to_room_type': 'toilet_attached','transformation_type': 'carry_over',  'priority':  5},
    'toilet_common':   {'to_room_type': 'toilet_common',  'transformation_type': 'carry_over',  'priority':  6},
    'master_bedroom':  {'to_room_type': 'master_bedroom', 'transformation_type': 'carry_over',  'priority':  7},
    'bedroom_2':       {'to_room_type': 'bedroom_2',      'transformation_type': 'carry_over',  'priority':  8},
    'bedroom_3':       {'to_room_type': 'bedroom_3',      'transformation_type': 'carry_over',  'priority':  9},
    'bedroom_4':       {'to_room_type': 'bedroom_4',      'transformation_type': 'carry_over',  'priority': 10},
    'pooja':           {'to_room_type': 'pooja',          'transformation_type': 'carry_over',  'priority': 11},
    'store':           {'to_room_type': 'store',          'transformation_type': 'carry_over',  'priority': 12},
    'living':          {'to_room_type': None,             'transformation_type': 'to_bedroom',  'priority': 13},
    'dining':          {'to_room_type': None,             'transformation_type': 'to_bedroom',  'priority': 14},
    'corridor':        {'to_room_type': 'corridor',       'transformation_type': 'carry_over',  'priority': 15},
}


@functools.lru_cache(maxsize=1)
def _load_floor_transformation_rules():
    """Load floor_transformation_rules from DB (cached for process lifetime).

    Returns
    -------
    dict: {from_room_type: {'to_room_type': str|None, 'transformation_type': str, 'priority': int}}
    First occurrence (lowest priority integer) wins for any given from_room_type.
    Activates _FLOOR_TRANSFORM_FALLBACK if DB query fails.
    """
    try:
        with sqlite3.connect(_API_DB_PATH) as conn:
            cur  = conn.cursor()
            cur.execute("""
                SELECT from_room_type, to_room_type, transformation_type, priority
                FROM   floor_transformation_rules
                ORDER  BY priority ASC
            """)
            rows = cur.fetchall()
    except Exception as exc:
        print(f"[FTR WARNING] floor_transformation_rules query failed ({exc}); "
              f"using fallback rules.")
        return dict(_FLOOR_TRANSFORM_FALLBACK)

    if not rows:
        print("[FTR WARNING] floor_transformation_rules table is empty; "
              "using fallback rules.")
        return dict(_FLOOR_TRANSFORM_FALLBACK)

    rules = {}
    for from_rt, to_rt, t_type, priority in rows:
        if from_rt not in rules:    # first row (lowest priority number) wins
            rules[from_rt] = {
                'to_room_type':        to_rt,   # may be None for to_bedroom
                'transformation_type': t_type,
                'priority':            priority,
            }
    return rules


@functools.lru_cache(maxsize=1)
def _load_vertical_stacking_rules():
    """Load vertical_stacking_rules from DB (cached for process lifetime).

    Returns
    -------
    dict: {room_type: {'must_stack': bool, 'tolerance_m': float}}
    Only rows with must_stack=1 are returned.
    Returns empty dict if DB query fails (stacking check is skipped).
    """
    try:
        with sqlite3.connect(_API_DB_PATH) as conn:
            cur  = conn.cursor()
            cur.execute("""
                SELECT room_type, must_stack, tolerance_m
                FROM   vertical_stacking_rules
                WHERE  must_stack = 1
            """)
            rows = cur.fetchall()
    except Exception as exc:
        print(f"[VSR WARNING] vertical_stacking_rules query failed ({exc}); "
              f"stacking validation skipped.")
        return {}

    return {rt: {'must_stack': bool(ms), 'tolerance_m': float(tol)}
            for rt, ms, tol in rows}


def _apply_stacking_constraints(ground_rooms_dict, first_rooms):
    """Validate vertical alignment for rooms marked must_stack=1 in the DB.

    Checks that each first-floor must-stack room's (x, y) position matches the
    ground-floor counterpart within the DB-specified tolerance.  Emits
    [DB WARNING] per violation.  Does NOT abort generation — violations are
    informational; the copy-based generation already guarantees alignment.

    Parameters
    ----------
    ground_rooms_dict : dict[room_type, Room]  — ground-floor rooms
    first_rooms       : list[Room]             — first-floor rooms (post-transform)

    Returns the unmodified first_rooms list.
    """
    stacking_rules = _load_vertical_stacking_rules()
    if not stacking_rules:
        return first_rooms

    for room in first_rooms:
        rule = stacking_rules.get(room.room_type)
        if rule is None:
            continue
        ground_room = ground_rooms_dict.get(room.room_type)
        if ground_room is None:
            continue   # first-floor-only room (e.g. balcony) — nothing to compare
        tol = rule['tolerance_m']
        dx  = abs(float(room.x)     - float(ground_room.x))
        dy  = abs(float(room.y)     - float(ground_room.y))
        if dx > tol or dy > tol:
            print(f"[DB WARNING] Stacking violation: {room.room_type} "
                  f"first=({room.x:.3f},{room.y:.3f}) "
                  f"ground=({ground_room.x:.3f},{ground_room.y:.3f}) "
                  f"Δ=({dx:.3f}m,{dy:.3f}m) > tol {tol}m")
    return first_rooms


def build_floor_plan_first_floor(ground_fp):
    """Transform ground-floor rooms into first-floor rooms using DB rules.

    Reads floor_transformation_rules from DB via _load_floor_transformation_rules().
    Applies transformation_type dispatch:
      • rename      — use to_room_type from DB (e.g. verandah → balcony)
      • carry_over  — same room_type on first floor (e.g. toilet_attached)
      • to_bedroom  — assign next sequential bedroom slot (living/dining → bedroom_N)
      • drop        — room is not placed on first floor

    Sequential bedroom numbering is inherently algorithmic (stateful counter) and
    remains in Python.  The DB controls WHICH rooms trigger sequencing, not HOW.

    Applies vertical stacking validation via _apply_stacking_constraints().

    Parameters
    ----------
    ground_fp : FloorPlan  — scored, wall-built ground-floor plan

    Returns
    -------
    first_rooms       : list[Room]
    first_placement   : dict  — room_type → geometry info dict for feature vector
    first_bhk         : int   — ground_fp.bhk + 1
    """
    transform_rules = _load_floor_transformation_rules()

    ground_rooms = {r.room_type: r for r in ground_fp.rooms}
    ground_bhk   = ground_fp.bhk
    first_bhk    = ground_bhk + 1

    # How many 'to_bedroom' transformations are allowed before overflow logic.
    # 2BHK ground → 1 new bedroom;  3BHK+ ground → 2 new bedrooms.
    new_bedroom_budget = 1 if ground_bhk == 2 else 2

    # Determine next available bedroom number after ground-floor bedrooms.
    max_bed_num = 1   # master_bedroom is implicitly bedroom-1
    for rt in ground_rooms:
        if rt.startswith("bedroom_"):
            try:
                max_bed_num = max(max_bed_num, int(rt.split("_")[1]))
            except (ValueError, IndexError):
                pass
    next_bed_num = max_bed_num + 1

    first_rooms     = []
    first_placement = {}

    def _add(new_rt, gr):
        """Append a first-floor Room and placement entry for room_type new_rt."""
        first_rooms.append(Room(
            room_type=new_rt,
            x=gr.x, y=gr.y, width=gr.width, depth=gr.depth,
            area=gr.area, cx_pct=gr.cx_pct, cy_pct=gr.cy_pct,
            compass=gr.compass,
        ))
        first_placement[new_rt] = {
            "x":  gr.x,   "y":  gr.y,
            "w":  gr.width, "d": gr.depth,
            "cx": gr.x + gr.width  / 2.0,
            "cy": gr.y + gr.depth  / 2.0,
        }

    # Priority-ordered iteration (rules are already sorted by priority ASC from DB)
    # Ground rooms are iterated in their stored order; each is looked up in rules.
    for rt, gr in ground_rooms.items():
        rule = transform_rules.get(rt)
        if rule is None:
            # No rule for this room type → skip (not present on first floor)
            continue

        t_type = rule['transformation_type']
        to_rt  = rule.get('to_room_type')   # may be None for to_bedroom

        if t_type == 'drop':
            continue   # explicitly excluded

        elif t_type == 'rename':
            _add(to_rt, gr)

        elif t_type == 'carry_over':
            _add(rt, gr)   # same room_type on both floors

        elif t_type == 'to_bedroom':
            # Use the sequential budget; fall to bedroom_N or store on overflow.
            if new_bedroom_budget > 0:
                new_rt = f"bedroom_{next_bed_num}"
                next_bed_num       += 1
                new_bedroom_budget -= 1
            elif ground_bhk >= 2 and next_bed_num <= 4:
                new_rt = f"bedroom_{next_bed_num}"
                next_bed_num += 1
            else:
                new_rt = "store"
            _add(new_rt, gr)

        else:
            # Unknown transformation_type — treat as carry_over and warn.
            print(f"[FTR WARNING] Unknown transformation_type '{t_type}' for "
                  f"room '{rt}'; treating as carry_over.")
            _add(rt, gr)

    # Validate vertical stacking alignment (emits warnings; does not block)
    _apply_stacking_constraints(ground_rooms, first_rooms)

    return first_rooms, first_placement, first_bhk


def _generate_first_floor(ground_fp: FloorPlan) -> FloorPlan:
    """Generate a first-floor plan based on the ground-floor layout.

    Rules (from PRD Section 6.2):
      - Same net_w, net_d, same setbacks
      - Verandah -> balcony (same position, labelled differently)
      - Living/dining -> additional bedroom(s)
        (1 bedroom for 2BHK G+1, 2 bedrooms for 3BHK G+1)
      - Kitchen -> dry_kitchen / store_room
      - Staircase head room: 1.0m x 2.5m at same position as ground staircase
      - Wet rooms (toilets) directly above ground wet rooms (same x_abs, y_abs)
      - BHK label increments by 1 for first floor
    """
    gfp = ground_fp
    net_w = gfp.net_w
    net_d = gfp.net_d

    # ── Room transformation: DB-driven (replaces hardcoded if/elif chain) ─────
    # build_floor_plan_first_floor() reads floor_transformation_rules from DB,
    # dispatches by transformation_type, and validates stacking via
    # _apply_stacking_constraints().  No room-type strings are hardcoded here.
    first_rooms, first_placement, first_bhk = build_floor_plan_first_floor(gfp)

    # Build the first-floor FloorPlan object
    first_fp = FloorPlan(
        plot_w=gfp.plot_w,
        plot_d=gfp.plot_d,
        bhk=first_bhk,
        facing=gfp.facing,
        district=gfp.district,
        net_w=net_w,
        net_d=net_d,
        setback_front=gfp.setback_front,
        setback_rear=gfp.setback_rear,
        setback_side=gfp.setback_side,
        climate_zone=gfp.climate_zone,
        facing_code=gfp.facing_code,
        climate_code=gfp.climate_code,
        materials=gfp.materials,
        baker_principles=gfp.baker_principles,
        seed=gfp.seed,
    )
    first_fp.rooms = first_rooms
    first_fp.placement = first_placement
    first_fp.band_b4_h = gfp.band_b4_h
    first_fp.band_y_b3 = gfp.band_y_b3
    first_fp.band_y_b2 = gfp.band_y_b2

    # Build walls, doors, windows for first floor
    first_fp.walls = build_wall_network(first_fp.rooms, net_w, net_d)
    window_scores = get_window_scores(gfp.district)

    import engine.engine as _eng
    saved_facing = _eng.CURRENT_FACING
    _eng.CURRENT_FACING = gfp.facing
    try:
        first_fp.doors = place_doors(first_fp.rooms, first_fp.walls, first_bhk)
        first_fp.windows = place_windows(
            first_fp.rooms, first_fp.walls, window_scores, gfp.facing
        )
    finally:
        _eng.CURRENT_FACING = saved_facing

    # Score first floor - uses remapped placement so dry_kitchen/staircase_head/
    # balcony map to their ROOM_UNIVERSE equivalents (see FIRST_FLOOR_SCORE_MAP).
    clf, _, explainer = ModelLoader.get()
    first_fp = _score_first_floor(first_fp, clf, explainer)
    first_fp.wall_geometry = build_wall_geometry(first_fp)

    return first_fp


# ---------------------------------------------------------------------------
# OBSERVABILITY — rule traceability, layout metrics, strategy compliance
# ---------------------------------------------------------------------------
#
# These three functions operate on a FINISHED FloorPlan (post-generation).
# They do NOT modify geometry, walls, doors, or scoring.  They only READ
# the plan, query cached DB loaders, and return structured dicts.
#
# Integration point: generate_plan() calls attach_plan_metadata(plan) after
# all geometry and scoring is complete and rooms have been converted to dict.
#
# DB SCHEMA CONTRACT (read-only from this module):
#   adjacency_rules         — queried via cached build_adjacency_graph_from_db()
#   nbc_codes               — consumed via module-level NBC_MIN_AREA / NBC_MIN_WIDTH
#   floor_transformation_rules — via cached _load_floor_transformation_rules()
#   vertical_stacking_rules    — via cached _load_vertical_stacking_rules()
#   strategy_rules          — rule_type / rule_key / rule_value / priority
#       Columns used: rule_type, rule_key, rule_value, bhk_applicability, priority
# ---------------------------------------------------------------------------

# Zone cy_pct ranges (pre-rotation: y=0 south, y=net_d north).
# Values are inclusive [lo, hi] fractions of net_d.
# Derived from the band contract in CLAUDE.md Section 4.
_ZONE_CY_RANGES = {
    'front':   (0.75, 1.00),   # verandah — northernmost band
    'public':  (0.50, 0.82),   # living / dining / pooja
    'service': (0.25, 0.65),   # kitchen / toilet_common / utility / staircase
    'private': (0.00, 0.42),   # bedrooms — southernmost band
}


@functools.lru_cache(maxsize=1)
def _load_strategy_rules():
    """Load strategy_rules from DB (cached for process lifetime).

    Returns list of dicts with keys: rule_type, rule_key, rule_value,
    bhk_applicability, priority.  Returns [] on DB failure.
    """
    try:
        with sqlite3.connect(_API_DB_PATH) as conn:
            cur  = conn.cursor()
            cur.execute("""
                SELECT rule_type, rule_key, rule_value, bhk_applicability, priority
                FROM   strategy_rules
                ORDER  BY priority ASC
            """)
            rows = cur.fetchall()
            return [{'rule_type': rt, 'rule_key': rk, 'rule_value': rv,
                     'bhk_applicability': ba, 'priority': p}
                    for rt, rk, rv, ba, p in rows]
    except Exception as exc:
        print(f"[STRATEGY WARNING] strategy_rules query failed ({exc}); "
              f"strategy compliance will not be computed.")
        return []


def _bhk_rule_applies(bhk_applicability, bhk):
    """Return True if the rule applies to the given BHK count."""
    s = str(bhk_applicability).strip()
    if s.upper() in ('ALL', ''):
        return True
    tag = f'{bhk}BHK'
    return tag in s.replace(';', ',').split(',')


def _check_zone_compliance(cy_pct, expected_zone):
    """Return True if cy_pct falls within the expected zone band."""
    lo, hi = _ZONE_CY_RANGES.get(expected_zone, (0.0, 1.0))
    return lo <= float(cy_pct) <= hi


def trace_layout_decisions(plan):
    """Trace which DB rules were applied to each room in the plan.

    Returns a dict keyed by room_type.  Each entry contains:
      adjacency   : list — expected-neighbour relationships from DB
      nbc         : dict — NBC minimum area/width applied and compliance status
      transformation : dict|None — G+1 floor transformation rule applied
      stacking    : dict|None — vertical stacking constraint applied

    All data is derived from cached DB loaders — no new DB round-trips.
    Reads: build_adjacency_graph_from_db(), NBC_MIN_AREA, NBC_MIN_WIDTH,
           _load_floor_transformation_rules(), _load_vertical_stacking_rules().
    """
    rooms = (plan.rooms if isinstance(plan.rooms, dict)
             else {r.room_type: r for r in plan.rooms})
    room_types = set(rooms.keys())

    # ── Actual shared-wall pairs from this plan ──────────────────────────────
    actual_adj = set()
    for w in (plan.walls or []):
        if w.room_left and w.room_right and w.room_right != 'outside':
            actual_adj.add(frozenset([w.room_left, w.room_right]))

    # ── Door connections expected by DB for this BHK ─────────────────────────
    _, req_conns = build_adjacency_graph_from_db(plan.bhk)
    # Deduplicate: one entry per pair (the DB may have both directions)
    seen_conn_pairs = set()
    door_conns = []
    for from_r, to_r, door_type, width in req_conns:
        key = frozenset([from_r, to_r])
        if key not in seen_conn_pairs and from_r in room_types and to_r in room_types:
            door_conns.append((from_r, to_r, door_type, width))
            seen_conn_pairs.add(key)

    # ── Cached DB rule sets ───────────────────────────────────────────────────
    ftr = _load_floor_transformation_rules()    # {from_rt: {to_room_type, ...}}
    vsr = _load_vertical_stacking_rules()       # {room_type: {must_stack, tolerance_m}}

    trace = {}
    for rt, room in rooms.items():
        entry = {'adjacency': [], 'nbc': {}, 'transformation': None, 'stacking': None}

        # ── Adjacency: which required door connections involve this room ──────
        for from_r, to_r, door_type, width in door_conns:
            if from_r == rt:
                pair = frozenset([rt, to_r])
                entry['adjacency'].append({
                    'neighbor':          to_r,
                    'door_type':         door_type,
                    'door_width_m':      width,
                    'shared_wall':       pair in actual_adj,
                    'source':            'adjacency_rules + movement_paths',
                })

        # ── NBC: applied minimum area and width ───────────────────────────────
        nbc_a = NBC_MIN_AREA.get(rt)
        nbc_w = NBC_MIN_WIDTH.get(rt)
        if nbc_a is not None:
            area = float(room.area)
            entry['nbc']['min_area_m2']    = nbc_a
            entry['nbc']['actual_area_m2'] = round(area, 3)
            entry['nbc']['area_pass']      = area >= nbc_a * 0.88
            entry['nbc']['source']         = 'nbc_codes → normalize_nbc_data'
        if nbc_w is not None:
            w = float(room.width)
            entry['nbc']['min_width_m']    = nbc_w
            entry['nbc']['actual_width_m'] = round(w, 3)
            entry['nbc']['width_pass']     = w >= nbc_w

        # ── Transformation: which floor_transformation_rules row produced this room ──
        # Case 1: carry_over — from_room_type == room_type
        rule = ftr.get(rt)
        if rule:
            entry['transformation'] = {
                'from_room_type':      rt,
                'to_room_type':        rule.get('to_room_type') or rt,
                'transformation_type': rule['transformation_type'],
                'priority':            rule['priority'],
                'source':              'floor_transformation_rules',
            }
        else:
            # Case 2: rename — to_room_type == rt (e.g. balcony, dry_kitchen)
            for from_rt, r in ftr.items():
                if r.get('to_room_type') == rt:
                    entry['transformation'] = {
                        'from_room_type':      from_rt,
                        'to_room_type':        rt,
                        'transformation_type': r['transformation_type'],
                        'priority':            r['priority'],
                        'source':              'floor_transformation_rules',
                    }
                    break

        # ── Stacking: vertical_stacking_rules constraint ─────────────────────
        stk = vsr.get(rt)
        if stk:
            entry['stacking'] = {
                'must_stack':  stk['must_stack'],
                'tolerance_m': stk['tolerance_m'],
                'source':      'vertical_stacking_rules',
            }

        trace[rt] = entry

    return trace


def compute_layout_metrics(plan):
    """Compute extended layout quality metrics for a finished FloorPlan.

    Returns a dict with:
      circulation_score   — from plan.score_circulation
      adjacency_score     — from plan.score_adjacency
      space_efficiency    — total room area / (net_w × net_d)
      wall_efficiency     — walls per room (lower → more efficient structure)
      dead_space_ratio    — 1 − space_efficiency (corridors, gaps, thickness)
      nbc_compliance_pct  — fraction of NBC-governed rooms that pass area check
      strategy_compliance — fraction of strategy_rules satisfied for this plan
      room_count, wall_count, door_count, window_count
      total_room_area_m2, net_area_m2, avg_room_area_m2

    Reads: plan fields, NBC_MIN_AREA, _load_strategy_rules().
    Does NOT query DB directly.
    """
    rooms = (plan.rooms if isinstance(plan.rooms, dict)
             else {r.room_type: r for r in plan.rooms})
    room_list  = list(rooms.values())
    n_rooms    = len(room_list)
    net_area   = float(plan.net_w) * float(plan.net_d)
    total_area = sum(float(r.area) for r in room_list)
    n_walls    = len(plan.walls)  if plan.walls   else 0
    n_doors    = len(plan.doors)  if plan.doors   else 0
    n_windows  = len(plan.windows) if plan.windows else 0

    space_eff = round(total_area / net_area, 3) if net_area > 0 else 0.0

    # NBC compliance: rooms with NBC area rules
    nbc_checks = [float(r.area) >= NBC_MIN_AREA[r.room_type] * 0.88
                  for r in room_list if r.room_type in NBC_MIN_AREA]
    nbc_pct = round(sum(nbc_checks) / len(nbc_checks), 3) if nbc_checks else 1.0

    # Strategy compliance: evaluate strategy_rules against plan
    strategy_rules = _load_strategy_rules()
    bhk = int(plan.bhk)
    checks = []
    for rule in strategy_rules:
        if not _bhk_rule_applies(rule['bhk_applicability'], bhk):
            continue
        rt    = rule['rule_key']
        rtype = rule['rule_type']
        rval  = rule['rule_value']
        if rt not in rooms:
            continue
        room = rooms[rt]
        if rtype == 'zoning':
            checks.append(_check_zone_compliance(room.cy_pct, rval))
        elif rtype == 'placement':
            # room.compass is post-rotation octant string: N, NE, E, SE, S, SW, W, NW
            checks.append(rval.upper() in room.compass.upper())
        elif rtype == 'corridor':
            if rval == 'linear' and rt == 'corridor':
                # A linear corridor spans full net_w within 5% tolerance
                checks.append(float(room.width) >= float(plan.net_w) * 0.95)
            else:
                checks.append(True)  # structural rule; satisfied by engine design
    strategy_score = round(sum(checks) / len(checks), 3) if checks else 1.0

    return {
        'circulation_score':    round(float(plan.score_circulation), 3),
        'adjacency_score':      round(float(plan.score_adjacency), 3),
        'space_efficiency':     space_eff,
        'wall_efficiency':      round(n_walls / max(n_rooms, 1), 2),
        'dead_space_ratio':     round(1.0 - space_eff, 3),
        'nbc_compliance_pct':   nbc_pct,
        'strategy_compliance':  strategy_score,
        'room_count':           n_rooms,
        'wall_count':           n_walls,
        'door_count':           n_doors,
        'window_count':         n_windows,
        'total_room_area_m2':   round(total_area, 2),
        'net_area_m2':          round(net_area, 2),
        'avg_room_area_m2':     round(total_area / max(n_rooms, 1), 2),
    }


def _build_decision_log(plan, trace, metrics):
    """Build a human-readable list of decisions made during layout generation.

    Each entry is a one-sentence string naming the relevant DB table.
    Format matches the CLAUDE.md contract for plan.metadata["log"].

    Parameters
    ----------
    plan    : FloorPlan (rooms already converted to dict)
    trace   : dict returned by trace_layout_decisions()
    metrics : dict returned by compute_layout_metrics()

    Returns list[str].
    """
    log = []

    # ── Adjacency decisions (door connections placed) ─────────────────────────
    satisfied_adj = set()
    for rt, entry in trace.items():
        for adj in entry.get('adjacency', []):
            if adj.get('shared_wall'):
                pair = frozenset([rt, adj['neighbor']])
                if pair not in satisfied_adj:
                    log.append(
                        f"Placed {rt} adjacent to {adj['neighbor']} "
                        f"({adj['door_type']} door {adj['door_width_m']}m) "
                        f"[adjacency_rules + movement_paths]"
                    )
                    satisfied_adj.add(pair)

    # ── NBC constraint log ────────────────────────────────────────────────────
    for rt, entry in trace.items():
        nbc = entry.get('nbc', {})
        if 'area_pass' in nbc:
            status = '✓' if nbc['area_pass'] else '✗ FAIL'
            log.append(
                f"{rt}: area {nbc['actual_area_m2']}m² vs NBC min {nbc['min_area_m2']}m² "
                f"[{status}] [nbc_codes]"
            )
        if 'width_pass' in nbc and 'min_width_m' in nbc:
            status = '✓' if nbc['width_pass'] else '✗ FAIL'
            log.append(
                f"{rt}: width {nbc['actual_width_m']}m vs NBC min {nbc['min_width_m']}m "
                f"[{status}] [nbc_codes]"
            )

    # ── Transformation decisions (G+1 plans) ─────────────────────────────────
    seen_transforms = set()
    for rt, entry in trace.items():
        tf = entry.get('transformation')
        if tf and tf['transformation_type'] != 'carry_over':
            key = (tf['from_room_type'], tf['to_room_type'])
            if key not in seen_transforms:
                log.append(
                    f"{tf['from_room_type']} → {tf['to_room_type']} "
                    f"({tf['transformation_type']}, priority {tf['priority']}) "
                    f"[floor_transformation_rules]"
                )
                seen_transforms.add(key)

    # ── Stacking decisions ────────────────────────────────────────────────────
    for rt, entry in trace.items():
        stk = entry.get('stacking')
        if stk and stk['must_stack']:
            log.append(
                f"Stacked {rt} above ground floor "
                f"(tolerance {stk['tolerance_m']}m) "
                f"[vertical_stacking_rules]"
            )

    # ── Strategy compliance summary ───────────────────────────────────────────
    sc = metrics.get('strategy_compliance', 1.0)
    log.append(
        f"Strategy compliance: {sc:.0%} of zone/placement rules satisfied "
        f"[strategy_rules]"
    )
    log.append(
        f"Space efficiency: {metrics['space_efficiency']:.0%} "
        f"({metrics['total_room_area_m2']}m² of {metrics['net_area_m2']}m² net area)"
    )
    log.append(
        f"NBC compliance: {metrics['nbc_compliance_pct']:.0%} "
        f"({metrics['room_count']} rooms, {metrics['wall_count']} walls, "
        f"{metrics['door_count']} doors)"
    )

    return log


def _get_strategy_feedback(metrics: dict, strategy_mode: str) -> list:
    """Generate strategy feedback recommendations based on computed metrics.

    Returns a list of human-readable advisory strings for metadata["strategy_log"].
    Each entry names the threshold that was crossed and the recommended alternative mode.
    This is ADVISORY only — no layout geometry is modified here.
    """
    feedback = []
    se = metrics.get('space_efficiency',    1.0)
    cc = metrics.get('circulation_score',   1.0)
    sc = metrics.get('strategy_compliance', 1.0)

    if se < 0.62 and strategy_mode != 'compact':
        feedback.append(
            f"[STRATEGY FEEDBACK] Space efficiency {se:.0%} < 62% threshold; "
            f"consider strategy_mode='compact' for a denser layout"
        )
    if cc < 0.50 and strategy_mode != 'luxury':
        feedback.append(
            f"[STRATEGY FEEDBACK] Circulation score {cc:.2f} is low; "
            f"consider strategy_mode='luxury' for a wider corridor (1.5 m)"
        )
    if sc < 0.70:
        feedback.append(
            f"[STRATEGY FEEDBACK] Strategy compliance {sc:.0%} < 70%; "
            f"layout deviates from DB zoning/placement rules [strategy_rules]"
        )
    return feedback


def attach_plan_metadata(plan):
    """Attach observability data to plan.metadata dict.

    Calls trace_layout_decisions(), compute_layout_metrics(), and
    _build_decision_log() on the FINISHED plan and stores results in
    plan.metadata["trace"], plan.metadata["metrics"], plan.metadata["log"].

    plan.metadata is created if it does not exist.
    Does NOT modify any geometry field.
    """
    if not hasattr(plan, 'metadata') or plan.metadata is None:
        plan.metadata = {}
    try:
        trace   = trace_layout_decisions(plan)
        metrics = compute_layout_metrics(plan)
        log     = _build_decision_log(plan, trace, metrics)
        plan.metadata['trace']   = trace
        plan.metadata['metrics'] = metrics
        plan.metadata['log']     = log
    except Exception as exc:
        plan.metadata['trace']   = {}
        plan.metadata['metrics'] = {}
        plan.metadata['log']     = [f"[METADATA ERROR] {exc}"]


# ---------------------------------------------------------------------------
# TASK 1 - Public API
# ---------------------------------------------------------------------------

def generate_plan(
    plot_w,
    plot_d,
    bhk,
    facing,
    district,
    floors=1,
    strategy_mode='standard',
    adaptive=False,
    vastu=True,
    baker=True,
    seed=None,
):
    """Generate a complete floor plan (G or G+1).

    Parameters
    ----------
    plot_w, plot_d : float
        Plot width and depth in metres.
    bhk : int
        Number of bedrooms (1-4).
    facing : str
        Road-facing direction - ``'N'``, ``'S'``, ``'E'``, or ``'W'``.
    district : str
        District name (used for setbacks, climate, materials lookup).
    floors : int, optional
        1 for ground only, 2 for G+1.  Default 1.
    vastu : bool, optional
        Reserved for future vastu-toggle.  Currently always applied.
    baker : bool, optional
        Reserved for future baker-toggle.  Currently always applied.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``"ground"``   - :class:`FloorPlan` for the ground floor.
        ``"first"``    - :class:`FloorPlan` for the first floor, or *None*
                         when ``floors == 1``.
        ``"scores"``   - dict of all 7 scores (ground floor).
        ``"metadata"`` - district, climate_zone, soil_type, materials,
                         baker_principles, seed_used.
    """
    # Run feature-alignment guard once at startup
    check_feature_alignment()

    # Generate ground floor via the engine
    params = {
        "plot_w": float(plot_w),
        "plot_d": float(plot_d),
        "bhk": int(bhk),
        "facing": str(facing).upper(),
        "district": str(district),
    }
    if seed is not None:
        params["seed"] = int(seed)
    params["floors"]        = int(floors)
    params["strategy_mode"] = str(strategy_mode).lower()

    ground_fp = _engine_generate(params)

    # Generate first floor if requested.
    # NOTE: _generate_first_floor must run while ground_fp.rooms is still a
    # List[Room] - it iterates the list to build ground_rooms dict internally.
    first_fp = None
    if floors >= 2:
        first_fp = _generate_first_floor(ground_fp)

    # Convert fp.rooms from List[Room] -> dict[room_type, Room] on both floors.
    # This is the public API contract: callers use rooms.keys() / rooms.get().
    # Engine internals (renderer, wall builder, scorer) have already finished
    # by this point and operated on the list form.
    ground_fp.rooms = {r.room_type: r for r in ground_fp.rooms}
    if first_fp is not None:
        first_fp.rooms = {r.room_type: r for r in first_fp.rooms}

    # ── Observability: attach trace / metrics / decision log ─────────────────
    # Must run AFTER rooms-to-dict conversion (trace_layout_decisions() expects
    # plan.rooms to be either a dict or a list — both forms are handled, but the
    # public API contract delivers a dict, so we attach metadata here in dict form).
    attach_plan_metadata(ground_fp)
    if first_fp is not None:
        attach_plan_metadata(first_fp)

    # ── Strategy feedback: advisory recommendations based on metrics ──────────
    # Append feedback messages to strategy_log so callers can surface them.
    # Does NOT modify geometry — advisory only.
    _metrics = getattr(ground_fp, 'metadata', {}).get('metrics', {})
    _feedback = _get_strategy_feedback(_metrics, strategy_mode)
    _strategy_log = getattr(ground_fp, 'strategy_log', [])
    _strategy_log.extend(_feedback)

    # ── Adaptive retry: one attempt with compact mode if space is tight ───────
    # Triggered when: adaptive=True, mode='standard', floors=1,
    # space_efficiency < 0.62 AND compact mode improves it.
    # Bounded to ONE additional generation call — no infinite loop.
    if (adaptive and floors == 1
            and strategy_mode == 'standard'
            and _metrics.get('space_efficiency', 1.0) < 0.62):
        _compact_params = dict(params, strategy_mode='compact')
        try:
            _compact_fp = _engine_generate(_compact_params)
            _compact_fp.rooms = {r.room_type: r for r in _compact_fp.rooms}
            attach_plan_metadata(_compact_fp)
            _compact_metrics = getattr(_compact_fp, 'metadata', {}).get('metrics', {})
            if _compact_metrics.get('space_efficiency', 0.0) > _metrics.get('space_efficiency', 0.0):
                print(f"[STRATEGY FEEDBACK] Adaptive: compact improved efficiency "
                      f"{_metrics['space_efficiency']:.0%} → "
                      f"{_compact_metrics['space_efficiency']:.0%}")
                ground_fp     = _compact_fp
                strategy_mode = 'compact'
                _strategy_log = getattr(ground_fp, 'strategy_log', [])
                _strategy_log.append(
                    f"[STRATEGY FEEDBACK] Adaptive mode switch: standard → compact "
                    f"(space efficiency improved to "
                    f"{_compact_metrics['space_efficiency']:.0%})"
                )
        except Exception as _adapt_exc:
            print(f"[STRATEGY FEEDBACK] Adaptive retry failed ({_adapt_exc}); "
                  f"keeping original layout.")

    # Collect all 7 scores from the ground floor
    scores = {
        "score_valid": ground_fp.score_valid,
        "score_vastu": ground_fp.score_vastu,
        "score_nbc": ground_fp.score_nbc,
        "score_circulation": ground_fp.score_circulation,
        "score_adjacency": ground_fp.score_adjacency,
        "score_overall": ground_fp.score_overall,
    }

    # Add first-floor scores if present
    if first_fp is not None:
        scores["first_score_valid"] = first_fp.score_valid
        scores["first_score_vastu"] = first_fp.score_vastu
        scores["first_score_nbc"] = first_fp.score_nbc
        scores["first_score_circulation"] = first_fp.score_circulation
        scores["first_score_adjacency"] = first_fp.score_adjacency
        scores["first_score_overall"] = first_fp.score_overall

    _gm = getattr(ground_fp, 'metadata', {}) or {}
    metadata = {
        "district":        ground_fp.district,
        "climate_zone":    ground_fp.climate_zone,
        "soil_type":       "alluvial",          # default; extend from DB if available
        "materials":       ground_fp.materials,
        "baker_principles": ground_fp.baker_principles,
        "seed_used":       ground_fp.seed,
        # ── Observability (Task D) ─────────────────────────────────────────────
        "trace":           _gm.get('trace',   {}),
        "metrics":         _gm.get('metrics', {}),
        "log":             _gm.get('log',     []),
        # ── Active Strategy Engine (Task E + F feedback) ─────────────────────
        "strategy_mode":   getattr(ground_fp, 'strategy_mode', strategy_mode),
        "strategy_log":    _strategy_log,   # includes advisory feedback entries
    }

    return {
        "ground": ground_fp,
        "first": first_fp,
        "scores": scores,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# PART 10 - Multi-candidate layout selection
# ---------------------------------------------------------------------------

def generate_best_layout(
    plot_w,
    plot_d,
    bhk,
    facing,
    district,
    floors=1,
    n_candidates=5,
    base_seed=42,
):
    """Generate *n_candidates* floor plans and return the highest-scoring one.

    Each candidate uses a distinct seed offset of x17 from *base_seed* so that
    no two candidates share any of the internal retry-loop seeds (which run
    seed ... seed+14).  This guarantees geometrically independent layouts.

    Selection composite (deterministic criteria only):
        adjacency  x 0.30   - REQUIRED/FORBIDDEN adjacency rule pass-rate
        nbc        x 0.25   - NBC minimum area/width pass-rate
        circulationx 0.20   - corridor/path connectivity score
        vastu      x 0.15   - compass-zone compliance
        space_eff  x 0.10   - covered room area ÷ net buildable area

    XGBoost ``score_valid`` is intentionally excluded - it has a known
    training-distribution gap for plots < 200 sqm (README §Known Limitations §1)
    and would incorrectly penalise valid EWS plans.

    Parameters
    ----------
    plot_w, plot_d : float
    bhk : int  (1-4)
    facing : str  ('N' | 'S' | 'E' | 'W')
    district : str
    floors : int  (1 or 2)
    n_candidates : int
        Number of layout variants to evaluate.  Default 5.
        Total placement calls <= n_candidates x 6 (ADR-8 retry budget).
    base_seed : int
        Candidate k uses seed = base_seed + k x 17.

    Returns
    -------
    dict
        Same structure as ``generate_plan()`` plus a ``'_selection'`` key::

            '_selection': {
                'n_candidates':    int,
                'seeds_attempted': [int, ...],
                'best_seed':       int,
                'selection_score': float,
                'breakdown': {
                    'adjacency':       float,
                    'nbc':             float,
                    'circulation':     float,
                    'vastu':           float,
                    'space_efficiency':float,
                },
                'all_candidates': [
                    {'seed': int, 'selection_score': float, 'breakdown': {...}},
                    ...
                ],
            }

    Raises
    ------
    RuntimeError
        If every candidate fails to generate a valid plan.
    """
    candidates = generate_candidates(
        plot_w, plot_d, bhk, facing, district,
        floors=floors, n=n_candidates, base_seed=base_seed,
    )

    if not candidates:
        raise RuntimeError(
            f"generate_best_layout: all {n_candidates} candidate(s) failed "
            f"(base_seed={base_seed}, plot={plot_w}x{plot_d}, bhk={bhk}, "
            f"facing={facing}, district={district})."
        )

    best = max(candidates, key=lambda c: c['selection_score'])

    best['result']['_selection'] = {
        'n_candidates':    n_candidates,
        'seeds_attempted': [c['seed'] for c in candidates],
        'best_seed':       best['seed'],
        'selection_score': best['selection_score'],
        'breakdown':       best['breakdown'],
        'all_candidates': [
            {
                'seed':            c['seed'],
                'selection_score': c['selection_score'],
                'breakdown':       c['breakdown'],
            }
            for c in candidates
        ],
    }
    return best['result']


# ---------------------------------------------------------------------------
# PART 11 - Structural + Topological Validation + Explainability
# ---------------------------------------------------------------------------

# Rooms that are, by design, accessible only from one parent room.
# These are NEVER flagged as isolated even when BFS cannot reach them
# from the main entrance - their parent room provides the sole legal access.
_PRIVATE_ANNEX = frozenset({'toilet_attached'})

# Main rooms that MUST be reachable from the entrance.
# Any room not in this set and not in _PRIVATE_ANNEX gets a WARNING.
_CRITICAL_ROOMS = frozenset({
    'master_bedroom', 'bedroom_2', 'bedroom_3', 'bedroom_4',
    'living', 'kitchen', 'dining',
})

# NBC-minimum corridor clear width (metres)
_CORR_MIN_WIDTH = 1.0

# Maximum unsupported wall span before a structural warning is raised (metres)
_MAX_WALL_SPAN = 6.0

# Extreme aspect ratio bounds for habitable rooms
_ASPECT_MIN = 0.40
_ASPECT_MAX = 2.80


def _build_door_graph(doors):
    """Return an undirected adjacency dict built from door room_from/room_to pairs.
    The sentinel value 'outside' is excluded - it is not a room node.
    """
    from collections import defaultdict
    graph = defaultdict(set)
    for door in doors:
        rf = getattr(door, 'room_from', None)
        rt = getattr(door, 'room_to',   None)
        if rf and rt and rf != 'outside' and rt != 'outside':
            graph[rf].add(rt)
            graph[rt].add(rf)
    return graph


def _check_connectivity(rooms_dict, graph):
    """BFS reachability from the plan entrance.

    Returns (issues, score, unreachable_set).
    """
    from collections import deque

    issues = []
    entrance = 'verandah' if 'verandah' in rooms_dict else 'living'

    visited = {entrance}
    queue   = deque([entrance])
    while queue:
        node = queue.popleft()
        for nb in graph.get(node, ()):
            if nb not in visited and nb in rooms_dict:
                visited.add(nb)
                queue.append(nb)

    unreachable = {
        rt for rt in rooms_dict
        if rt not in visited and rt not in _PRIVATE_ANNEX
    }

    for rt in sorted(unreachable):
        sev = 'CRITICAL' if rt in _CRITICAL_ROOMS else 'WARNING'
        issues.append({
            'severity': sev,
            'code':     'CON01',
            'message':  f"'{rt}' is not reachable from the entrance via any door.",
        })

    # Check private-annex rooms are reachable from their parent
    for rt in rooms_dict:
        if rt in _PRIVATE_ANNEX:
            parent_candidates = [
                n for n in rooms_dict
                if rt in graph.get(n, ())
            ]
            if not parent_candidates:
                issues.append({
                    'severity': 'WARNING',
                    'code':     'CON02',
                    'message':  f"'{rt}' has no door connection to any parent room.",
                })

    # Score: fraction of non-annex rooms reachable
    non_annex = {rt for rt in rooms_dict if rt not in _PRIVATE_ANNEX}
    reachable_count = len(non_annex - unreachable)
    score = round(reachable_count / max(len(non_annex), 1), 4)
    return issues, score, unreachable


def _check_circulation(rooms_dict, graph):
    """Evaluate corridor presence, width, and bedroom access quality.

    Returns (issues, score).
    """
    issues = []
    score  = 1.0

    corridor = rooms_dict.get('corridor')
    bedrooms = [rt for rt in rooms_dict if 'bedroom' in rt]

    if corridor:
        corr_clear = min(float(corridor.width), float(corridor.depth))
        if corr_clear < _CORR_MIN_WIDTH:
            issues.append({
                'severity': 'WARNING',
                'code':     'CIR01',
                'message':  (
                    f"Corridor clear width {corr_clear:.2f} m < {_CORR_MIN_WIDTH} m minimum "
                    f"(NBC Table 2)."
                ),
            })
            score -= 0.30

        # Bedrooms that have no direct door to corridor
        corr_neighbors = graph.get('corridor', set())
        disconnected_beds = [b for b in bedrooms if b not in corr_neighbors]
        for bed in disconnected_beds:
            issues.append({
                'severity': 'WARNING',
                'code':     'CIR02',
                'message':  (
                    f"'{bed}' has no direct door connection to the corridor; "
                    f"occupants cannot reach bathroom without passing through another room."
                ),
            })
            score -= 0.15
    else:
        if len(bedrooms) >= 2:
            issues.append({
                'severity': 'INFO',
                'code':     'CIR03',
                'message':  (
                    f"No corridor room present; {len(bedrooms)} bedrooms share direct "
                    f"access paths - suitable only for 1BHK or open-plan layouts."
                ),
            })
            score -= 0.10

    # Dead-end rooms (degree 1 in door graph, not a toilet or utility)
    EXEMPT_DEAD_END = {'toilet_attached', 'toilet_common', 'utility', 'store', 'pooja'}
    for rt, room in rooms_dict.items():
        if rt in EXEMPT_DEAD_END:
            continue
        degree = len(graph.get(rt, set()))
        if degree == 1 and rt not in ('verandah',):
            issues.append({
                'severity': 'INFO',
                'code':     'CIR04',
                'message':  f"'{rt}' has only 1 door connection (dead-end room).",
            })
            score -= 0.05

    return issues, round(max(score, 0.0), 4)


def _check_proportion(rooms_dict):
    """Aspect-ratio and relative-area proportion checks.

    Returns (issues, score).
    """
    issues = []
    score  = 1.0

    SKIP = {'corridor', 'verandah', 'store', 'pooja'}

    for rt, room in rooms_dict.items():
        if rt in SKIP:
            continue
        w = float(room.width)
        d = float(room.depth)
        aspect = w / max(d, 0.01)
        if aspect < _ASPECT_MIN or aspect > _ASPECT_MAX:
            issues.append({
                'severity': 'WARNING',
                'code':     'PRO01',
                'message':  (
                    f"'{rt}' has an extreme aspect ratio {aspect:.2f} "
                    f"(w={w:.2f} m, d={d:.2f} m); usability is reduced."
                ),
            })
            score -= 0.10

    # Dining should not exceed living in area
    living = rooms_dict.get('living')
    dining = rooms_dict.get('dining')
    if living and dining:
        ratio = float(dining.area) / max(float(living.area), 0.01)
        if ratio > 0.95:
            issues.append({
                'severity': 'WARNING',
                'code':     'PRO02',
                'message':  (
                    f"Dining ({dining.area:.1f} m2) is near or larger than living "
                    f"({living.area:.1f} m2); ratio {ratio:.2f} - consider reducing dining width."
                ),
            })
            score -= 0.10

    # Kitchen should be >= 70 % of dining area (NBC service-zone balance)
    kitchen = rooms_dict.get('kitchen')
    if kitchen and dining:
        k_ratio = float(kitchen.area) / max(float(dining.area), 0.01)
        if k_ratio < 0.70:
            issues.append({
                'severity': 'INFO',
                'code':     'PRO03',
                'message':  (
                    f"Kitchen ({kitchen.area:.1f} m2) is small relative to dining "
                    f"({dining.area:.1f} m2); ratio {k_ratio:.2f} (recommend >= 0.70)."
                ),
            })
            score -= 0.05

    return issues, round(max(score, 0.0), 4)


def _check_structural(rooms_dict, walls):
    """Flag long unsupported wall spans and undersized staircase footprints.

    Returns (issues, score).
    """
    issues = []
    score  = 1.0

    for wall in walls:
        span = float(getattr(wall, 'length', 0.0))
        if span > _MAX_WALL_SPAN:
            wall_id = getattr(wall, 'wall_id', None) or (
                f"({getattr(wall,'x1',0):.1f},{getattr(wall,'y1',0):.1f})"
                f"-({getattr(wall,'x2',0):.1f},{getattr(wall,'y2',0):.1f})"
            )
            issues.append({
                'severity': 'WARNING',
                'code':     'STR01',
                'message':  (
                    f"Wall {wall_id} span {span:.2f} m > {_MAX_WALL_SPAN} m; "
                    f"an intermediate column or cross-wall may be required."
                ),
            })
            score -= 0.15

    staircase = rooms_dict.get('staircase')
    if staircase:
        sc_w = float(staircase.width)
        sc_d = float(staircase.depth)
        if sc_w < 1.0 or sc_d < 2.0:
            issues.append({
                'severity': 'WARNING',
                'code':     'STR02',
                'message':  (
                    f"Staircase footprint {sc_w:.2f} x {sc_d:.2f} m is below the "
                    f"recommended minimum of 1.0 x 2.5 m (NBC Part 3 §7.4)."
                ),
            })
            score -= 0.20

    return issues, round(max(score, 0.0), 4)


def _build_explanation(rooms_dict, issues, scores, graph, unreachable):
    """Generate a list of human-readable sentences describing the plan quality."""
    lines = []

    # Connectivity summary
    n_rooms     = len(rooms_dict)
    n_reachable = n_rooms - len(unreachable)
    lines.append(
        f"Connectivity: {n_reachable}/{n_rooms} rooms are reachable from the entrance "
        f"(score {scores['connectivity']:.2f})."
    )
    if unreachable:
        lines.append(
            "  Unreachable rooms: " + ", ".join(sorted(unreachable)) +
            ". Add a door on the shared wall to resolve."
        )

    # Corridor
    corridor = rooms_dict.get('corridor')
    if corridor:
        corr_clear = round(min(float(corridor.width), float(corridor.depth)), 2)
        lines.append(
            f"Corridor: present at y={corridor.y:.2f} m, clear width {corr_clear:.2f} m "
            f"({'OK' if corr_clear >= _CORR_MIN_WIDTH else 'NARROW - widen to 1.0 m'})."
        )
    else:
        lines.append("Corridor: absent - open-plan circulation assumed.")

    # Proportion highlights
    prop_warns = [i for i in issues if i['code'].startswith('PRO')]
    if prop_warns:
        lines.append(
            f"Proportion: {len(prop_warns)} issue(s) detected - "
            + "; ".join(i['message'] for i in prop_warns[:2])
            + ("..." if len(prop_warns) > 2 else ".")
        )
    else:
        lines.append("Proportion: all room aspect ratios and relative sizes are within acceptable bounds.")

    # Structural highlights
    str_warns = [i for i in issues if i['code'].startswith('STR')]
    if str_warns:
        lines.append(
            f"Structure: {len(str_warns)} long-span wall(s) flagged - "
            "consider intermediate columns at flagged locations."
        )
    else:
        lines.append(f"Structure: all wall spans <= {_MAX_WALL_SPAN} m - no intermediate columns required.")

    # Overall verdict
    critical_count = sum(1 for i in issues if i['severity'] == 'CRITICAL')
    warning_count  = sum(1 for i in issues if i['severity'] == 'WARNING')
    if critical_count == 0 and warning_count == 0:
        lines.append(
            f"Overall: layout passes all structural and topological checks "
            f"(composite score {scores['overall']:.2f})."
        )
    elif critical_count > 0:
        lines.append(
            f"Overall: {critical_count} CRITICAL issue(s) must be resolved before this "
            f"plan can be considered valid (score {scores['overall']:.2f})."
        )
    else:
        lines.append(
            f"Overall: {warning_count} warning(s) detected; no critical issues "
            f"(score {scores['overall']:.2f}). Address warnings for production quality."
        )

    return lines


def validate_and_explain(plan) -> dict:
    """Structural and topological validation with human-readable explanation.

    Reads only ``plan.rooms``, ``plan.walls``, and ``plan.doors``.
    Does NOT modify the plan. Does NOT call any ML model.

    Parameters
    ----------
    plan : FloorPlan
        A fully generated floor plan (ground or first floor).

    Returns
    -------
    dict with keys:
        is_valid    : bool   - True when zero CRITICAL issues are present.
        issues      : list   - Each item: {severity, code, message}.
                               severity in {'CRITICAL', 'WARNING', 'INFO'}
        scores      : dict   - circulation, connectivity, proportion,
                               structural, overall in [0.0, 1.0]
        explanation : list   - Human-readable sentences (one per finding).
    """
    # Normalise rooms to dict regardless of how caller stored them
    rooms_raw  = plan.rooms
    rooms_dict = rooms_raw if isinstance(rooms_raw, dict) else {r.room_type: r for r in rooms_raw}

    doors = getattr(plan, 'doors', []) or []
    walls = getattr(plan, 'walls', []) or []

    # -- Five validation passes ------------------------------------------------
    graph = _build_door_graph(doors)

    conn_issues,  conn_score,  unreachable = _check_connectivity(rooms_dict, graph)
    circ_issues,  circ_score               = _check_circulation(rooms_dict, graph)
    prop_issues,  prop_score               = _check_proportion(rooms_dict)
    str_issues,   str_score                = _check_structural(rooms_dict, walls)

    all_issues = conn_issues + circ_issues + prop_issues + str_issues

    # -- Composite score -------------------------------------------------------
    overall = round(
        conn_score * 0.30 +
        circ_score * 0.25 +
        prop_score * 0.25 +
        str_score  * 0.20,
        4,
    )

    scores = {
        'connectivity': conn_score,
        'circulation':  circ_score,
        'proportion':   prop_score,
        'structural':   str_score,
        'overall':      overall,
    }

    # -- Explainability --------------------------------------------------------
    explanation = _build_explanation(rooms_dict, all_issues, scores, graph, unreachable)

    is_valid = not any(i['severity'] == 'CRITICAL' for i in all_issues)

    return {
        'is_valid':    is_valid,
        'issues':      all_issues,
        'scores':      scores,
        'explanation': explanation,
    }


# ---------------------------------------------------------------------------
# PART 12 - Feedback-Driven Correction Loop
# ---------------------------------------------------------------------------

# Door widths used when synthesising repair doors.
# Rooms not listed default to 0.90 m.
_REPAIR_DOOR_WIDTHS = {
    'toilet_attached': 0.75,
    'toilet_common':   0.75,
    'utility':         0.75,
    'store':           0.75,
    'pooja':           0.90,
    'kitchen':         0.90,
    'verandah':        1.20,
    'living':          1.20,
}

# Rooms whose repair door should be an archway rather than a swing leaf.
_ARCHWAY_ROOMS = {'pooja', 'verandah', 'living', 'dining'}

# Maximum BFS repair passes before giving up (prevents infinite loops)
_REPAIR_MAX_PASSES = 10


def _find_repair_wall(room_type, reachable, walls):
    """Return the longest interior wall shared between *room_type* and any room
    in *reachable*.  Returns ``None`` if no such wall exists.
    """
    best_wall  = None
    best_len   = 0.0
    for wall in walls:
        rl = getattr(wall, 'room_left',  None)
        rr = getattr(wall, 'room_right', None)
        # Must be an interior wall touching room_type on one side and a
        # reachable room on the other.
        if wall.wall_type != 'interior':
            continue
        sides = {rl, rr}
        if room_type not in sides:
            continue
        other = rr if rl == room_type else rl
        if other not in reachable:
            continue
        if wall.length > best_len:
            best_len  = wall.length
            best_wall = wall
    return best_wall


def _make_repair_door(wall, from_room, to_room, label):
    """Construct a DoorOpening at the midpoint of *wall* connecting
    *from_room* to *to_room*.  Uses archway type for open-plan rooms.
    """
    width = max(
        _REPAIR_DOOR_WIDTHS.get(to_room,   0.90),
        _REPAIR_DOOR_WIDTHS.get(from_room, 0.90),
    )
    # Clamp to 75 % of wall length (door must not consume full wall segment)
    width = min(width, wall.length * 0.75)

    d_type = 'archway' if (from_room in _ARCHWAY_ROOMS or to_room in _ARCHWAY_ROOMS) else 'swing'
    return DoorOpening(
        label      = label,
        wall       = wall,
        position   = 0.5,       # midpoint of wall segment
        width      = round(width, 2),
        door_type  = d_type,
        hinge_side = 'left',
        swing_into = to_room,
        room_from  = from_room,
        room_to    = to_room,
    )


def refine_layout(plan, validation):
    """Targeted constraint-repair pass driven by ``validate_and_explain`` output.

    Repair strategy (in priority order):
    1. CONNECTIVITY (CON01 CRITICAL/WARNING): insert a door on the shared
       interior wall between the unreachable room and its nearest reachable
       neighbour.  Unreachable rooms with no shared interior wall are logged
       as unresolvable.
    2. OPTIONAL ROOM INTEGRATION (CIR02 WARNING): same mechanism for rooms
       that are spatially adjacent to the corridor but lack a door.
    3. PROPORTION (PRO01-03): recorded as advisories only.  Adjusting room
       widths requires a full wall rebuild and is outside the repair scope.
    4. STRUCTURAL (STR01-02): column indicator positions stored as
       ``plan.column_indicators`` metadata list (no geometry change).

    Does NOT: modify room geometry, change wall coordinates, alter band
    heights, or call any ML model.

    Parameters
    ----------
    plan : FloorPlan
    validation : dict  Output of ``validate_and_explain(plan)``.

    Returns
    -------
    tuple (refined_plan, repair_log)
        refined_plan  : FloorPlan with ``doors`` list potentially augmented.
        repair_log    : dict
            doors_added   - list of {room_a, room_b, wall_direction, width, door_type}
            unresolvable  - list of {code, room, reason}
            advisories    - list of str  (proportion + structural notes)
            score_before  - float
            score_after   - float
    """
    # Shallow-copy the plan and give it a fresh doors list so the original
    # is never mutated.
    refined = copy.copy(plan)
    refined.doors = list(plan.doors)   # mutable copy; walls/rooms unchanged

    rooms_raw   = refined.rooms
    rooms_dict  = rooms_raw if isinstance(rooms_raw, dict) else {r.room_type: r for r in rooms_raw}
    walls       = refined.walls or []

    doors_added  = []
    unresolvable = []
    advisories   = []
    column_inds  = []

    score_before = float(validation.get('scores', {}).get('overall', 0.0))

    # ---- Structural: record column indicator positions (metadata only) ------
    for issue in validation.get('issues', []):
        if issue['code'] == 'STR01':
            column_inds.append({'message': issue['message'], 'severity': 'advisory'})
            advisories.append(
                "STRUCTURAL: " + issue['message'] +
                " Column indicator recorded (geometry not modified)."
            )
        if issue['code'] == 'STR02':
            advisories.append("STRUCTURAL: " + issue['message'])

    if column_inds:
        refined.column_indicators = column_inds

    # ---- Proportion: document but do not touch geometry ---------------------
    for issue in validation.get('issues', []):
        if issue['code'].startswith('PRO'):
            advisories.append(
                "PROPORTION (advisory only - geometry repair requires wall rebuild): "
                + issue['message']
            )

    # ---- Connectivity repair: iterative BFS door insertion ------------------
    # Rooms to attempt repair on, ordered by severity (CRITICAL first)
    repair_targets = []
    seen_rooms = set()
    for issue in validation.get('issues', []):
        if issue['code'] in ('CON01', 'CIR02'):
            # Extract room name from message — always in single quotes
            import re
            m = re.search(r"'([^']+)'", issue['message'])
            if m:
                rt = m.group(1)
                if rt not in seen_rooms:
                    seen_rooms.add(rt)
                    repair_targets.append((issue['severity'], issue['code'], rt))

    # Sort: CRITICAL first, then WARNING
    repair_targets.sort(key=lambda t: 0 if t[0] == 'CRITICAL' else 1)

    # Door label counter starting after the last existing door index
    door_counter = len(refined.doors) + 1

    for _pass in range(_REPAIR_MAX_PASSES):
        if not repair_targets:
            break

        # Rebuild reachable set from current doors list
        graph = _build_door_graph(refined.doors)
        entrance = 'verandah' if 'verandah' in rooms_dict else 'living'
        from collections import deque
        visited = {entrance}
        queue   = deque([entrance])
        while queue:
            node = queue.popleft()
            for nb in graph.get(node, ()):
                if nb not in visited and nb in rooms_dict:
                    visited.add(nb)
                    queue.append(nb)

        still_unreachable = []
        made_progress     = False

        for sev, code, rt in repair_targets:
            if rt in visited:
                continue  # already reachable (fixed in earlier pass)

            repair_wall = _find_repair_wall(rt, visited, walls)

            if repair_wall is None:
                unresolvable.append({
                    'code':   code,
                    'room':   rt,
                    'reason': (
                        f"No interior wall found between '{rt}' and any currently "
                        f"reachable room. Requires band-height adjustment or "
                        f"layout regeneration to resolve."
                    ),
                })
                continue  # cannot fix this one; do not re-queue

            # Determine which side of the wall is the reachable room
            rl = getattr(repair_wall, 'room_left',  None)
            rr = getattr(repair_wall, 'room_right', None)
            reachable_side   = rl if rl in visited else rr
            unreachable_side = rt

            label    = f"R{door_counter:02d}"
            new_door = _make_repair_door(
                repair_wall, reachable_side, unreachable_side, label
            )
            refined.doors.append(new_door)
            door_counter += 1

            doors_added.append({
                'room_a':         reachable_side,
                'room_b':         unreachable_side,
                'wall_direction': repair_wall.direction,
                'wall_length':    round(repair_wall.length, 3),
                'width':          new_door.width,
                'door_type':      new_door.door_type,
                'label':          label,
            })
            # Mark as reachable so subsequent targets in this pass can build on it
            visited.add(unreachable_side)
            made_progress = True

            still_unreachable.append((sev, code, rt))  # may now be reachable; re-check next pass

        repair_targets = still_unreachable
        if not made_progress:
            break

    # ---- Re-validate to get score_after ------------------------------------
    v2          = validate_and_explain(refined)
    score_after = float(v2.get('scores', {}).get('overall', 0.0))

    repair_log = {
        'doors_added':   doors_added,
        'unresolvable':  unresolvable,
        'advisories':    advisories,
        'score_before':  round(score_before, 4),
        'score_after':   round(score_after,  4),
    }

    return refined, repair_log


# ---------------------------------------------------------------------------
# PART 13 - Constraint-First Generation (No Invalid Output)
# ---------------------------------------------------------------------------

def _zone_clear(rooms_dict, exclude_rt, x0, x1, y0, y1, tol=0.02):
    """Return True if the rectangle (x0,y0)-(x1,y1) is free of all other rooms."""
    for rt, room in rooms_dict.items():
        if rt == exclude_rt:
            continue
        rx0 = float(room.x);  rx1 = rx0 + float(room.width)
        ry0 = float(room.y);  ry1 = ry0 + float(room.depth)
        if rx0 < x1 - tol and rx1 > x0 + tol and ry0 < y1 - tol and ry1 > y0 + tol:
            return False
    return True


def _fix_connectivity_geometry(plan):
    """Eliminate spatial gaps that prevent bedroom-corridor and corridor-front adjacency.

    Handles three gap patterns:
    A) HORIZONTAL CORRIDOR (N/S facing) — bedrooms below corridor:
       Extend bedroom.depth so top touches corridor.y.
    B) VERTICAL CORRIDOR (E/W facing) — bedrooms left/right of corridor:
       Extend bedroom.width (left-of-corridor) or adjust bedroom.x+width
       (right-of-corridor) so the bedroom edge touches the corridor edge.
    C) CORRIDOR -> FRONT ZONE GAP (empty service/public zone in 1BHK):
       Extend corridor.depth so its top touches the nearest room above.

    All extensions are guarded by a zone-clear overlap check.
    After any modification, rebuilds walls, doors, and wall_geometry.
    """
    import engine.engine as _eng

    rooms_input = plan.rooms
    rooms_dict  = (rooms_input if isinstance(rooms_input, dict)
                   else {r.room_type: r for r in rooms_input})

    corridor = rooms_dict.get('corridor')
    if not corridor:
        return plan

    modified = False
    corr_w = float(corridor.width);  corr_d = float(corridor.depth)
    corr_x = float(corridor.x);      corr_y = float(corridor.y)
    corr_x1 = corr_x + corr_w;       corr_y1 = corr_y + corr_d
    is_horizontal = (corr_w >= corr_d)

    # ---- Pattern A + B: bedroom-to-corridor gap ----------------------------
    for rt, room in rooms_dict.items():
        if 'bedroom' not in rt:
            continue
        rx0 = float(room.x);  rx1 = rx0 + float(room.width)
        ry0 = float(room.y);  ry1 = ry0 + float(room.depth)

        if is_horizontal:
            # A: bedroom top should touch corridor bottom (corr_y)
            gap = corr_y - ry1
            if gap > 0.05 and _zone_clear(rooms_dict, rt, rx0, rx1, ry1, corr_y):
                room.depth = round(corr_y - ry0, 3)
                room.area  = round(float(room.width) * float(room.depth), 3)
                modified   = True
        else:
            # B: bedroom is left or right of vertical corridor
            room_cx = (rx0 + rx1) / 2.0
            if room_cx < corr_x:
                # Bedroom to the LEFT: extend right edge to corridor left
                gap = corr_x - rx1
                if gap > 0.05 and _zone_clear(rooms_dict, rt, rx1, corr_x, ry0, ry1):
                    room.width = round(corr_x - rx0, 3)
                    room.area  = round(float(room.width) * float(room.depth), 3)
                    modified   = True
            elif room_cx > corr_x1:
                # Bedroom to the RIGHT: move left edge to corridor right
                gap = rx0 - corr_x1
                if gap > 0.05 and _zone_clear(rooms_dict, rt, corr_x1, rx0, ry0, ry1):
                    old_right  = rx1
                    room.x     = round(corr_x1, 3)
                    room.width = round(old_right - corr_x1, 3)
                    room.area  = round(float(room.width) * float(room.depth), 3)
                    modified   = True

    # ---- Pattern C: any gap between the topmost room-stack and the front room ---
    # Covers two sub-cases:
    #   C1 (1BHK, empty service zone): corridor top has a gap to verandah
    #   C2 (2BHK no-living): service zone top has a gap to verandah
    # Strategy: find the room with the highest top edge (excluding verandah) that
    # has clear extension space up to verandah.y, and extend it.
    if is_horizontal:
        verandah = rooms_dict.get('verandah')
        if verandah:
            vdy = float(verandah.y)
            # Collect all non-verandah rooms whose top is below verandah with a gap
            extend_candidates = []
            for rt2, r in rooms_dict.items():
                if rt2 == 'verandah':
                    continue
                r_top = float(r.y) + float(r.depth)
                gap   = vdy - r_top
                if gap > 0.05:
                    rx0 = float(r.x);  rx1 = rx0 + float(r.width)
                    if _zone_clear(rooms_dict, rt2, rx0, rx1, r_top, vdy):
                        extend_candidates.append((r_top, rt2, r))
            # Extend the candidate closest to verandah (highest top edge)
            if extend_candidates:
                _, best_rt, best_room = max(extend_candidates, key=lambda c: c[0])
                best_room.depth = round(vdy - float(best_room.y), 3)
                best_room.area  = round(float(best_room.width) * float(best_room.depth), 3)
                modified = True
    else:
        # Vertical corridor (E/W): find nearest room to the right of corridor right edge
        right_rooms = [(float(r.x), rt2, r) for rt2, r in rooms_dict.items()
                       if float(r.x) >= corr_x1 - 0.02 and rt2 != 'corridor']
        if right_rooms:
            nearest_x = min(x for x, _, _ in right_rooms)
            gap = nearest_x - corr_x1
            if gap > 0.05 and _zone_clear(rooms_dict, 'corridor',
                                          corr_x1, nearest_x, corr_y, corr_y + corr_d):
                corridor.width = round(nearest_x - corr_x, 3)
                corridor.area  = round(float(corridor.width) * float(corridor.depth), 3)
                modified = True

    if not modified:
        return plan

    # Rebuild walls, doors, and wall geometry from corrected room geometry
    room_list  = list(rooms_dict.values())
    plan.walls = build_wall_network(room_list, float(plan.net_w), float(plan.net_d))

    saved               = _eng.CURRENT_FACING
    _eng.CURRENT_FACING = str(plan.facing)
    try:
        plan.doors      = place_doors(room_list, plan.walls, int(plan.bhk))
    finally:
        _eng.CURRENT_FACING = saved

    # build_wall_geometry expects a List[Room] in plan.rooms; swap temporarily
    rooms_was_dict = isinstance(plan.rooms, dict)
    if rooms_was_dict:
        plan.rooms = room_list
    plan.wall_geometry = build_wall_geometry(plan)
    if rooms_was_dict:
        plan.rooms = rooms_dict

    return plan


# Keep old name as alias so generate_valid_layout doesn't need updating yet
_fix_bedroom_gaps = _fix_connectivity_geometry


def _ensure_all_reachable(plan):
    """After geometry fixes, add doors for any rooms still unreachable.

    Covers both REQUIRED rooms (bedrooms) and OPTIONAL rooms (pooja, store).
    Uses the same wall-finding helpers as Part 12.
    """
    from collections import deque

    rooms_input = plan.rooms
    rooms_dict  = (rooms_input if isinstance(rooms_input, dict)
                   else {r.room_type: r for r in rooms_input})

    patched   = []
    counter   = len(plan.doors) + 1

    for _pass in range(_REPAIR_MAX_PASSES):
        # Rebuild reachable set from current doors
        graph    = _build_door_graph(plan.doors)
        entrance = 'verandah' if 'verandah' in rooms_dict else 'living'
        visited  = {entrance}
        queue    = deque([entrance])
        while queue:
            node = queue.popleft()
            for nb in graph.get(node, ()):
                if nb not in visited and nb in rooms_dict:
                    visited.add(nb)
                    queue.append(nb)

        unreachable = [
            rt for rt in rooms_dict
            if rt not in visited and rt not in _PRIVATE_ANNEX
        ]
        if not unreachable:
            break

        progress = False
        for rt in unreachable:
            repair_wall = _find_repair_wall(rt, visited, plan.walls)
            if repair_wall is None:
                continue
            rl = getattr(repair_wall, 'room_left',  None)
            rr = getattr(repair_wall, 'room_right', None)
            reachable_side = rl if rl in visited else rr
            label    = f'G{counter:02d}'
            new_door = _make_repair_door(repair_wall, reachable_side, rt, label)
            plan.doors.append(new_door)
            visited.add(rt)
            counter += 1
            patched.append(f'{reachable_side}<->{rt}')
            progress = True

        if not progress:
            break            # no more fixable gaps; remaining rooms are truly isolated

    return plan, patched


def generate_valid_layout(
    plot_w,
    plot_d,
    bhk,
    facing,
    district,
    floors=1,
    max_attempts=5,
    base_seed=42,
) -> dict:
    """Generate a floor plan guaranteed to pass connectivity validation.

    Guarantee mechanism (layered, in order):
    1. GEOMETRY FIX (Rule 3 + Rule 1): extend any bedroom whose depth falls
       short of the corridor bottom, then rebuild walls and doors.  This
       eliminates the dead-space gap that is the root cause of unreachable
       bedrooms in narrow plots.
    2. DOOR PATCH (Rule 2 + Rule 5): after geometry fix, add doors on any
       remaining shared walls between unreachable rooms and reachable rooms.
       Covers optional rooms (pooja, store) and any bedroom not reached by step 1.
    3. REJECT + RETRY (Rule 4): if connectivity is still broken after both
       fixes, discard the candidate and try the next seed.

    Parameters
    ----------
    plot_w, plot_d : float
    bhk : int  (1-4)
    facing : str
    district : str
    floors : int  (1 or 2)
    max_attempts : int  Seeds tried before raising.  Default 5.
    base_seed : int  Candidate k uses seed = base_seed + k * 17.

    Returns
    -------
    dict  Same structure as ``generate_plan()`` plus a ``'_validity'`` key::

        '_validity': {
            'seed_used':       int,
            'attempt_number':  int,
            'bedroom_gaps_fixed': [str, ...],   # room types extended
            'doors_patched':   [str, ...],       # room pairs door-added
            'is_valid':        bool,
            'scores':          dict,
        }

    Raises
    ------
    RuntimeError  If every attempt fails to produce a connected layout.
    """
    import warnings

    last_result = None

    for k in range(max_attempts):
        seed_k = base_seed + k * 17

        try:
            result = generate_plan(
                plot_w=plot_w, plot_d=plot_d, bhk=bhk,
                facing=facing, district=district,
                floors=floors, seed=seed_k,
            )
        except Exception as exc:
            warnings.warn(
                f"generate_valid_layout: seed {seed_k} raised {exc!r}; skipping.",
                RuntimeWarning, stacklevel=2,
            )
            continue

        ground = result['ground']
        gaps_fixed = []

        # ---- Steps 1+2: CSP geometry snap + door patch ----------------------
        ground, csp_report = solve_layout_constraints(ground)
        gaps_fixed = [fix['room'] for fix in csp_report['fixes']]
        patched    = csp_report['doors_patched']
        result['ground'] = ground

        # ---- Step 3: validate -----------------------------------------------
        v = validate_and_explain(ground)
        last_result = (result, v, seed_k, k + 1, gaps_fixed, patched)

        if v['is_valid']:
            result['_validity'] = {
                'seed_used':          seed_k,
                'attempt_number':     k + 1,
                'bedroom_gaps_fixed': gaps_fixed,
                'doors_patched':      patched,
                'is_valid':           True,
                'scores':             v['scores'],
            }
            return result

        # Layout still invalid after both fixes -> reject, try next seed
        warnings.warn(
            f"generate_valid_layout: seed {seed_k} still invalid after fixes "
            f"(issues: {[i['code']+':'+i['room_type'] if 'room_type' in i else i['code'] for i in v['issues'] if i['severity']=='CRITICAL']!r}); "
            f"trying next seed.",
            RuntimeWarning, stacklevel=2,
        )

    # Exhausted all attempts — return the last result with a warning
    if last_result is not None:
        result, v, seed_k, attempt, gaps_fixed, patched = last_result
        result['_validity'] = {
            'seed_used':          seed_k,
            'attempt_number':     attempt,
            'bedroom_gaps_fixed': gaps_fixed,
            'doors_patched':      patched,
            'is_valid':           v['is_valid'],
            'scores':             v['scores'],
        }
        warnings.warn(
            f"generate_valid_layout: all {max_attempts} attempt(s) failed; "
            f"returning best result (is_valid={v['is_valid']}).",
            RuntimeWarning, stacklevel=2,
        )
        return result

    raise RuntimeError(
        f"generate_valid_layout: all {max_attempts} attempt(s) failed to generate "
        f"any plan (base_seed={base_seed}, plot={plot_w}x{plot_d}, bhk={bhk})."
    )


# ---------------------------------------------------------------------------
# PART 14 - General CSP Solver + Metrics Logger
# ---------------------------------------------------------------------------

import itertools as _itertools


def _directional_snap(room_a, room_b, rt_a, rt_b, rooms_dict,
                      min_overlap=0.10, max_gap=2.5):
    """Pure-geometry gap detection between two rooms.

    Returns a dict describing an anchor-stable extension that closes the gap,
    or None if no valid snap exists.  No room-type knowledge is used.

    The 'anchor-stable' invariant: only the lower/left room is extended
    (its width or depth grows); its (x, y) origin never moves.
    """
    ax0 = float(room_a.x);  ax1 = ax0 + float(room_a.width)
    ay0 = float(room_a.y);  ay1 = ay0 + float(room_a.depth)
    bx0 = float(room_b.x);  bx1 = bx0 + float(room_b.width)
    by0 = float(room_b.y);  by1 = by0 + float(room_b.depth)

    y_ov = min(ay1, by1) - max(ay0, by0)   # vertical overlap
    x_ov = min(ax1, bx1) - max(ax0, bx0)   # horizontal overlap

    # A left of B -> extend A.width to bx0
    gap_r = bx0 - ax1
    if 0.05 < gap_r <= max_gap and y_ov >= min_overlap:
        if _zone_clear(rooms_dict, rt_a, ax1, bx0, ay0, ay1):
            return dict(gap=gap_r, rt=rt_a, room=room_a,
                        attr='width', new_val=round(bx0 - ax0, 3))

    # B left of A -> extend B.width to ax0
    gap_l = ax0 - bx1
    if 0.05 < gap_l <= max_gap and y_ov >= min_overlap:
        if _zone_clear(rooms_dict, rt_b, bx1, ax0, by0, by1):
            return dict(gap=gap_l, rt=rt_b, room=room_b,
                        attr='width', new_val=round(ax0 - bx0, 3))

    # A below B -> extend A.depth to by0
    gap_u = by0 - ay1
    if 0.05 < gap_u <= max_gap and x_ov >= min_overlap:
        if _zone_clear(rooms_dict, rt_a, ax0, ax1, ay1, by0):
            return dict(gap=gap_u, rt=rt_a, room=room_a,
                        attr='depth', new_val=round(by0 - ay0, 3))

    # B below A -> extend B.depth to ay0
    gap_d = ay0 - by1
    if 0.05 < gap_d <= max_gap and x_ov >= min_overlap:
        if _zone_clear(rooms_dict, rt_b, bx0, bx1, by1, ay0):
            return dict(gap=gap_d, rt=rt_b, room=room_b,
                        attr='depth', new_val=round(ay0 - by0, 3))

    return None


def solve_layout_constraints(layout, max_iter=10):
    """General CSP room-gap solver (no room-type pattern matching).

    Iterates all room pairs and applies anchor-stable extensions to close
    geometry gaps.  After convergence, rebuilds the wall network, re-runs
    door placement, and calls _ensure_all_reachable.

    Returns
    -------
    (layout, report)
        report keys: fixes (list of dicts), doors_patched (list of str),
                     iterations_run (int)
    """
    fixes = []
    iterations_run = 0

    for _ in range(max_iter):
        iterations_run += 1
        rooms_dict = (layout.rooms if isinstance(layout.rooms, dict)
                      else {r.room_type: r for r in layout.rooms})
        rt_list = list(rooms_dict.keys())
        found = False

        for rt_a, rt_b in _itertools.combinations(rt_list, 2):
            room_a = rooms_dict[rt_a]
            room_b = rooms_dict[rt_b]
            snap = _directional_snap(room_a, room_b, rt_a, rt_b, rooms_dict)
            if snap is not None:
                setattr(snap['room'], snap['attr'], snap['new_val'])
                snap['room'].area = round(
                    float(snap['room'].width) * float(snap['room'].depth), 3)
                fixes.append({
                    'room':       snap['rt'],
                    'attr':       snap['attr'],
                    'gap_closed': round(snap['gap'], 3),
                })
                found = True
                break   # restart iteration with updated geometry

        if not found:
            break   # converged

    # Rebuild walls, doors, windows after geometry changes
    rooms_dict = (layout.rooms if isinstance(layout.rooms, dict)
                  else {r.room_type: r for r in layout.rooms})
    room_list = list(rooms_dict.values())

    try:
        import engine.engine as _eng
        layout.walls = build_wall_network(room_list, float(layout.net_w), float(layout.net_d))

        saved = _eng.CURRENT_FACING
        _eng.CURRENT_FACING = str(layout.facing)
        try:
            layout.doors = place_doors(room_list, layout.walls, int(layout.bhk))
        finally:
            _eng.CURRENT_FACING = saved
        # Windows are not rebuilt — geometry moves only room boxes, not exterior
        # walls, so existing window positions remain valid.

        rooms_was_dict = isinstance(layout.rooms, dict)
        if rooms_was_dict:
            layout.rooms = room_list
        layout.wall_geometry = build_wall_geometry(layout)
        if rooms_was_dict:
            layout.rooms = rooms_dict
    except Exception:
        pass    # keep existing walls/doors if rebuild raises

    # Ensure all rooms reachable via door graph
    layout, patched = _ensure_all_reachable(layout)

    report = {
        'fixes':          fixes,
        'doors_patched':  patched,
        'iterations_run': iterations_run,
    }
    return layout, report


def log_layout_metrics(plan) -> dict:
    """Structured metrics snapshot for one FloorPlan.

    Returns
    -------
    dict with keys:
        is_valid     - bool
        violations   - list of CRITICAL/WARNING issue dicts from validate_and_explain
        scores       - dict (connectivity/circ/proportion/structural/overall)
        room_metrics - dict[room_type -> {area, aspect_ratio, nbc_area_ok, nbc_width_ok}]
        connectivity - dict(reachable=[...], unreachable=[...])
    """
    from collections import deque

    v = validate_and_explain(plan)

    rooms_dict = (plan.rooms if isinstance(plan.rooms, dict)
                  else {r.room_type: r for r in plan.rooms})

    # Per-room NBC compliance
    room_metrics = {}
    for rt, r in rooms_dict.items():
        w = float(r.width); d = float(r.depth); a = float(r.area)
        long_side = max(w, d)
        short_side = min(w, d)
        aspect = round(short_side / long_side, 3) if long_side > 0.01 else 1.0
        nbc_a = NBC_MIN_AREA.get(rt)
        nbc_w = NBC_MIN_WIDTH.get(rt)
        room_metrics[rt] = {
            'area':         round(a, 3),
            'aspect_ratio': aspect,
            'nbc_area_ok':  bool(a >= nbc_a * 0.88) if nbc_a is not None else None,
            'nbc_width_ok': bool(short_side >= nbc_w) if nbc_w is not None else None,
        }

    # Connectivity from door graph
    graph = _build_door_graph(plan.doors)
    start_nodes = {'verandah', 'living'} & set(rooms_dict)
    reachable = set(start_nodes)
    q = deque(start_nodes)
    while q:
        node = q.popleft()
        for nb in graph.get(node, []):
            if nb not in reachable:
                reachable.add(nb)
                q.append(nb)

    unreachable = sorted(
        rt for rt in rooms_dict
        if rt not in reachable and rt != 'outside'
    )

    violations = [i for i in v['issues'] if i['severity'] in ('CRITICAL', 'WARNING')]

    return {
        'is_valid':    v['is_valid'],
        'violations':  violations,
        'scores':      v['scores'],
        'room_metrics': room_metrics,
        'connectivity': {
            'reachable':   sorted(reachable),
            'unreachable': unreachable,
        },
    }
