"""
db_rules.py — SQLite Rule Engine for Tamil Nadu Floor Plan Generator
Provides district-specific setback rules, NBC 2016 room dimensions,
Vastu directional preferences, adjacency rules, and climate data
via cached SQLite queries.
"""

import sqlite3
import os
from functools import lru_cache

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "floorplan.db")


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@lru_cache(maxsize=64)
def get_setbacks(district: str, authority: str = "DTCP"):
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM tn_setbacks WHERE district=? AND authority=?",
            (district, authority),
        ).fetchone()
    return dict(row) if row else {"setback_front_m": 1.5, "setback_rear_m": 1.0, "setback_side_m": 1.0}


@lru_cache(maxsize=32)
def get_nbc_codes(bhk: int):
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM nbc_codes WHERE bhk=?", (bhk,)
        ).fetchall()
    return [dict(r) for r in rows]


@lru_cache(maxsize=16)
def get_vastu_rules():
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM vastu_rules").fetchall()
    return [dict(r) for r in rows]


@lru_cache(maxsize=16)
def get_adjacency_rules():
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM adjacency_rules").fetchall()
    return [dict(r) for r in rows]


@lru_cache(maxsize=16)
def get_climate_data(district: str):
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM climate_data WHERE district=?", (district,)
        ).fetchone()
    return dict(row) if row else {}


@lru_cache(maxsize=8)
def get_strategy_rules(climate_zone: str):
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM strategy_rules WHERE climate_zone=?", (climate_zone,)
        ).fetchall()
    return [dict(r) for r in rows]
