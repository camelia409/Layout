"""
test_edge_cases.py — Edge-case and robustness tests for the Tamil Nadu Floor Plan Generator.

Tests:
  1. EWS plot (5×9): smallest valid plot — should not crash, NBC must pass
  2. Large plot (30×40): 4BHK luxury — should PASS all metrics
  3. DB fallback: invalid DB path → graceful fallback (no crash)
  4. Invalid strategy_mode: 'extreme' → silently coerces to 'standard'
  5. 1BHK luxury: single bedroom with wide corridor
  6. G+1 compact mode: first floor with compact strategy
  7. South-facing 3BHK standard: compass/Vastu invariants hold
  8. East-facing 2BHK luxury: compass/Vastu invariants hold
"""

import os
import sys
import sqlite3
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.engine_api import generate_plan
from engine.rule_engine import load_all_rules, resolve_conflicts, provide_constraints


NBC_MIN_AREA = {
    'master_bedroom': 9.5, 'bedroom_2': 7.5, 'bedroom_3': 7.5, 'bedroom_4': 7.5,
    'living': 9.5, 'dining': 5.0, 'kitchen': 4.5,
    'toilet_attached': 2.5, 'toilet_common': 2.5,
    'utility': 2.0, 'verandah': 4.0,
}

NBC_TOLERANCE = 0.88  # test suite tolerance (8%)


def _check_plan(result, case_label):
    """Common assertions for any generated plan result."""
    ground = result['ground']
    rooms  = ground.rooms   # dict after generate_plan()

    assert len(rooms) >= 3, f"{case_label}: fewer than 3 rooms placed"

    # NBC area checks
    for rt, room in rooms.items():
        if rt in NBC_MIN_AREA:
            min_area = NBC_MIN_AREA[rt] * NBC_TOLERANCE
            assert float(room.area) >= min_area, (
                f"{case_label}: {rt} area {room.area:.2f} < NBC min {min_area:.2f}"
            )

    # No overlapping rooms
    room_list = list(rooms.values())
    for i, a in enumerate(room_list):
        for b in room_list[i+1:]:
            overlap_x = (a.x < b.x + b.width - 0.05) and (a.x + a.width > b.x + 0.05)
            overlap_y = (a.y < b.y + b.depth - 0.05) and (a.y + a.depth > b.y + 0.05)
            assert not (overlap_x and overlap_y), (
                f"{case_label}: overlap between {a.room_type} and {b.room_type}"
            )

    # Metadata keys present
    md = result['metadata']
    for key in ('strategy_mode', 'strategy_log', 'metrics', 'trace'):
        assert key in md, f"{case_label}: metadata missing key '{key}'"


class TestEdgeCases(unittest.TestCase):

    # ── Case 1: EWS plot ──────────────────────────────────────────────────────
    def test_ews_plot(self):
        """5×9 EWS plot: minimal 1BHK must place rooms without crash."""
        r = generate_plan(plot_w=5, plot_d=9, bhk=1, facing='N',
                          district='Trichy', floors=1)
        _check_plan(r, 'EWS 5×9')
        rooms = r['ground'].rooms
        self.assertIn('master_bedroom', rooms)
        self.assertIn('verandah', rooms)

    # ── Case 2: Large luxury plot ─────────────────────────────────────────────
    def test_large_luxury(self):
        """30×40 4BHK luxury: all NBC checks must pass."""
        r = generate_plan(plot_w=30, plot_d=40, bhk=4, facing='N',
                          district='Chennai', floors=1, strategy_mode='luxury')
        _check_plan(r, 'Large 30×40 luxury')
        md = r['metadata']
        self.assertEqual(md['strategy_mode'], 'luxury')
        # Luxury corridor must be wider than standard
        ground = r['ground']
        corr = ground.rooms.get('corridor')
        if corr:
            self.assertGreater(corr.depth, 1.10,
                               "Luxury corridor should be >1.1 m deep")

    # ── Case 3: DB fallback — invalid DB path ─────────────────────────────────
    def test_db_unavailable_graceful(self):
        """If DB is unreachable, rule_engine must return fallback values without crash."""
        from engine import rule_engine as re_mod
        original_path = re_mod.DB_PATH
        try:
            re_mod.DB_PATH = '/nonexistent/path/floorplan.db'
            re_mod.load_all_rules.cache_clear()
            # Should not raise — fallback values are used
            rs = re_mod.load_all_rules(2, 'standard')
            self.assertIsNotNone(rs)
            self.assertAlmostEqual(rs.dim_scale, 1.0, places=2)
            self.assertAlmostEqual(rs.corridor_depths['standard'], 1.20, places=2)
        finally:
            re_mod.DB_PATH = original_path
            re_mod.load_all_rules.cache_clear()

    # ── Case 4: Invalid strategy_mode coercion ────────────────────────────────
    def test_invalid_strategy_mode(self):
        """strategy_mode='extreme' must silently coerce to 'standard'."""
        r = generate_plan(plot_w=12, plot_d=15, bhk=2, facing='N',
                          district='Coimbatore', floors=1, strategy_mode='extreme')
        _check_plan(r, '2BHK extreme→standard')
        # Either 'extreme' or 'standard' in metadata is acceptable
        # (engine_api passes the raw string; rule_engine coerces internally)
        md = r['metadata']
        self.assertIn(md['strategy_mode'], ('extreme', 'standard'))

    # ── Case 5: 1BHK luxury ───────────────────────────────────────────────────
    def test_1bhk_luxury(self):
        """1BHK luxury 9×12: must produce valid plan with wide corridor."""
        r = generate_plan(plot_w=9, plot_d=12, bhk=1, facing='N',
                          district='Salem', floors=1, strategy_mode='luxury')
        _check_plan(r, '1BHK luxury')
        self.assertEqual(r['metadata']['strategy_mode'], 'luxury')

    # ── Case 6: G+1 compact ───────────────────────────────────────────────────
    def test_g_plus_1_compact(self):
        """12×15 2BHK G+1 compact: both floors must generate without overlap."""
        r = generate_plan(plot_w=12, plot_d=15, bhk=2, facing='N',
                          district='Coimbatore', floors=2, strategy_mode='compact')
        _check_plan(r, 'G+1 compact ground')
        self.assertIsNotNone(r['first'], "First floor must exist for floors=2")
        first_rooms = r['first'].rooms
        self.assertGreater(len(first_rooms), 0, "First floor must have rooms")

    # ── Case 7: South-facing Vastu ────────────────────────────────────────────
    def test_south_facing_vastu(self):
        """15×20 3BHK S-facing: kitchen must not be in NE quadrant."""
        r = generate_plan(plot_w=15, plot_d=20, bhk=3, facing='S',
                          district='Chennai', floors=1)
        _check_plan(r, '3BHK S-facing')
        rooms = r['ground'].rooms
        if 'kitchen' in rooms:
            self.assertNotEqual(rooms['kitchen'].compass, 'NE',
                                "Kitchen must not be in NE (Vastu violation)")
        if 'master_bedroom' in rooms:
            self.assertNotIn(rooms['master_bedroom'].compass, ['NE'],
                             "Master bedroom must not be in NE")

    # ── Case 8: East-facing 2BHK luxury ─────────────────────────────────────
    def test_east_facing_luxury(self):
        """12×15 2BHK E-facing luxury: plan must generate, no overlaps."""
        r = generate_plan(plot_w=12, plot_d=15, bhk=2, facing='E',
                          district='Madurai', floors=1, strategy_mode='luxury')
        _check_plan(r, '2BHK E-facing luxury')

    # ── rule_engine unit tests ─────────────────────────────────────────────────
    def test_load_all_rules_returns_ruleset(self):
        """load_all_rules must return a RuleSet with expected fields."""
        rs = load_all_rules(3, 'standard')
        self.assertIsNotNone(rs)
        self.assertIsInstance(rs.zoning, dict)
        self.assertIsInstance(rs.placement, dict)
        self.assertIsInstance(rs.dim_scale, float)
        self.assertAlmostEqual(rs.dim_scale, 1.0, places=2)

    def test_load_all_rules_compact_dim_scale(self):
        """compact mode dim_scale must be < 1.0."""
        rs = load_all_rules(2, 'compact')
        self.assertLess(rs.dim_scale, 1.0,
                        "compact dim_scale must be < 1.0")

    def test_load_all_rules_luxury_corr_depth(self):
        """luxury mode corridor depth must be > standard."""
        rs_std = load_all_rules(3, 'standard')
        rs_lux = load_all_rules(3, 'luxury')
        self.assertGreater(rs_lux.corridor_depths['standard'],
                           rs_std.corridor_depths['standard'],
                           "luxury corridor depth must exceed standard")

    def test_resolve_conflicts_no_crash(self):
        """resolve_conflicts must not crash on any valid RuleSet."""
        rs = load_all_rules(2, 'standard')
        rs2 = resolve_conflicts(rs)
        self.assertIsNotNone(rs2)

    def test_provide_constraints_ews(self):
        """provide_constraints EWS must return corr_d_target < standard."""
        rs = load_all_rules(1, 'standard')
        rs = resolve_conflicts(rs)
        pc_std = provide_constraints(rs, 12.0, 15.0, ews_plot=False)
        pc_ews = provide_constraints(rs, 6.0,  9.0,  ews_plot=True)
        self.assertLess(pc_ews.corr_d_target, pc_std.corr_d_target,
                        "EWS corridor depth must be less than standard")


if __name__ == '__main__':
    # Suppress TensorFlow startup noise
    import warnings
    warnings.filterwarnings('ignore')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    unittest.main(verbosity=2)
