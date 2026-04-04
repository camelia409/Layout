"""
render_mpl.py — Pure-matplotlib floor plan renderer.

Draws wall-centric architectural floor plans directly with matplotlib + Shapely.
No DXF intermediate. Output quality matches reference architectural drawings.
"""
import os, sys, math
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from shapely.geometry import box
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.engine import FloorPlan

# ─── colour palette ────────────────────────────────────────────────────────────
BG         = '#F5F0E8'   # warm cream background
WALL_FC    = '#2C2C2C'   # exterior wall fill
WALL_INT_FC= '#3A3A3A'   # interior wall fill
WALL_EC    = '#1A1A1A'   # wall edge
FURN_FC    = '#D0D0D0'   # furniture fill
FURN_EC    = '#888888'   # furniture edge
DIM_COLOR  = '#333333'
TOL        = 0.08

# ── Renderer zorder stack (DO NOT alter — controls draw layer precedence) ──────
# Any patch or line drawn at a lower zorder is hidden behind a higher one.
Z_BG      = 1    # background cream fill, property boundary dashes
Z_FILL    = 2    # room colour fills (ROOM_FC palette)
Z_WALL    = 5    # wall solid patches — exterior (///) and interior (//)
Z_OPENING = 6    # door leaf/arc lines; window glazing/jamb lines
Z_LABEL   = 7    # door labels; window labels
Z_ANNOT   = 8    # room name text + dimension text inside room
Z_DIM     = 9    # exterior and interior dimension chain arrows + ticks
Z_COMPASS = 10   # north compass rose; legend panel; title block

ROOM_FC = {
    'master_bedroom':  '#B4D2F0',
    'bedroom_2':       '#B4D2F0',
    'bedroom_3':       '#B4D2F0',
    'bedroom_4':       '#B4D2F0',
    'bedroom_5':       '#B4D2F0',
    'bedroom_6':       '#B4D2F0',
    'living':          '#F5F5F5',
    'dining':          '#F0F0F0',
    'kitchen':         '#FFF3B4',
    'toilet_attached': '#B4E6C3',
    'toilet_common':   '#B4E6C3',
    'utility':         '#DCD2BE',
    'verandah':        '#D2F0D2',
    'pooja':           '#FAD2E1',
    'store':           '#EBD4C8',
    'staircase':       '#EEEEEE',
    'balcony':         '#D2F0D2',
    'dry_kitchen':     '#FFF3B4',
    'staircase_head':  '#EEEEEE',
    'corridor':        '#F5F5F5',
}

ROOM_LABEL = {
    'master_bedroom':  'MASTER\nBEDROOM',
    'bedroom_2':       'BEDROOM 2',
    'bedroom_3':       'BEDROOM 3',
    'bedroom_4':       'BEDROOM 4',
    'bedroom_5':       'BEDROOM 5',
    'bedroom_6':       'BEDROOM 6',
    'living':          'LIVING',
    'dining':          'DINING',
    'kitchen':         'KITCHEN',
    'toilet_attached': 'ATTACHED\nTOILET',
    'toilet_common':   'COMMON\nTOILET',
    'utility':         'UTILITY',
    'verandah':        'VERANDAH',
    'pooja':           'POOJA',
    'store':           'STORE',
    'staircase':       'STAIRCASE',
    'balcony':         'BALCONY',
    'dry_kitchen':     'DRY\nKITCHEN',
    'staircase_head':  'STAIRCASE\nHEAD',
    'corridor':        'PASSAGE',
}


# ─── geometry helpers ──────────────────────────────────────────────────────────

def _poly_patch(geom, **kw):
    """Convert Shapely Polygon/MultiPolygon to matplotlib PathPatch."""
    if geom is None or geom.is_empty:
        return None
    polys = [geom] if geom.geom_type == 'Polygon' else [g for g in geom.geoms if not g.is_empty]
    verts, codes = [], []
    for poly in polys:
        ext = list(poly.exterior.coords)
        c = [Path.LINETO] * len(ext)
        c[0] = Path.MOVETO
        verts.extend(ext)
        codes.extend(c)
        for ring in poly.interiors:
            pts = list(ring.coords)
            c = [Path.LINETO] * len(pts)
            c[0] = Path.MOVETO
            verts.extend(pts)
            codes.extend(c)
    if not verts:
        return None
    return PathPatch(Path(verts, codes), **kw)


def _wall_box(wall):
    t2 = wall.thickness / 2.0
    if wall.direction == 'H':
        x0, x1 = sorted((wall.x1, wall.x2))
        return box(x0, wall.y1 - t2, x1, wall.y1 + t2)
    y0, y1 = sorted((wall.y1, wall.y2))
    return box(wall.x1 - t2, y0, wall.x1 + t2, y1)


def _gap_box(wall, width, pos, over=0.06):
    t2 = (wall.thickness + over) / 2.0
    if wall.direction == 'H':
        cx = wall.x1 + pos * (wall.x2 - wall.x1)
        return box(cx - width / 2.0, wall.y1 - t2, cx + width / 2.0, wall.y1 + t2)
    cy = wall.y1 + pos * (wall.y2 - wall.y1)
    return box(wall.x1 - t2, cy - width / 2.0, wall.x1 + t2, cy + width / 2.0)


def _opening_pt(wall, pos):
    if wall.direction == 'H':
        return wall.x1 + pos * (wall.x2 - wall.x1), wall.y1
    return wall.x1, wall.y1 + pos * (wall.y2 - wall.y1)


# ─── draw helpers ─────────────────────────────────────────────────────────────

def _frect(ax, x, y, w, d, fc=FURN_FC, ec=FURN_EC, lw=0.6, zorder=4):
    ax.add_patch(mpatches.Rectangle(
        (x, y), w, d, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder))


def _dim_h(ax, x1, x2, y_ref, text, offset=0.45, fs=6.5):
    """Horizontal dimension arrow between x1 and x2 at y=y_ref+offset."""
    yd = y_ref + offset
    ax.annotate('', xy=(x2, yd), xytext=(x1, yd),
                arrowprops=dict(arrowstyle='<->', color=DIM_COLOR, lw=0.8), zorder=9)
    ax.plot([x1, x1], [y_ref, yd], '-', color='#666666', lw=0.5, zorder=9)
    ax.plot([x2, x2], [y_ref, yd], '-', color='#666666', lw=0.5, zorder=9)
    ax.text((x1 + x2) / 2, yd + 0.07, text, ha='center', va='bottom',
            fontsize=fs, color='#111111', fontweight='semibold', zorder=9)


def _dim_v(ax, y1, y2, x_ref, text, offset=0.55, fs=6.5):
    """Vertical dimension arrow between y1 and y2 at x=x_ref-offset."""
    xd = x_ref - offset
    ax.annotate('', xy=(xd, y2), xytext=(xd, y1),
                arrowprops=dict(arrowstyle='<->', color=DIM_COLOR, lw=0.8), zorder=9)
    ax.plot([x_ref, xd], [y1, y1], '-', color='#666666', lw=0.5, zorder=9)
    ax.plot([x_ref, xd], [y2, y2], '-', color='#666666', lw=0.5, zorder=9)
    ax.text(xd - 0.1, (y1 + y2) / 2, text, ha='right', va='center',
            rotation=90, fontsize=fs, color='#111111', fontweight='semibold', zorder=9)


# ─── drawing sections ─────────────────────────────────────────────────────────

def draw_boundary(ax, fp):
    sb, sr, sf = fp.setback_side, fp.setback_rear, fp.setback_front

    # Outer property boundary (dashed)
    ax.add_patch(mpatches.Rectangle(
        (-sb, -sr), fp.net_w + 2 * sb, fp.net_d + sf + sr,
        lw=1.3, edgecolor='#666666', facecolor='none', linestyle='--', zorder=1))
    ax.text(fp.net_w / 2, fp.net_d + sf + 0.1,
            'Property Boundary', ha='center', va='bottom',
            fontsize=6.5, color='#555555', style='italic', zorder=1)

    # Building envelope (inner dashed)
    ax.add_patch(mpatches.Rectangle(
        (0, 0), fp.net_w, fp.net_d,
        lw=1.0, edgecolor='#999999', facecolor='none', linestyle='--', zorder=1))

    # Setback labels
    kw = dict(fontsize=6, color='#888888', zorder=1)
    if sb > 0.05:
        ax.text(-sb / 2, fp.net_d / 2, f'{sb:.1f}m', ha='center', va='center', rotation=90, **kw)
        ax.text(fp.net_w + sb / 2, fp.net_d / 2, f'{sb:.1f}m', ha='center', va='center', rotation=90, **kw)
    if sr > 0.05:
        ax.text(fp.net_w / 2, -sr / 2, f'Rear {sr:.1f}m', ha='center', va='center', **kw)
    if sf > 0.05:
        ax.text(fp.net_w / 2, fp.net_d + sf / 2, f'Front {sf:.1f}m', ha='center', va='center', **kw)


def draw_room_fills(ax, fp):
    for room in fp.rooms:
        color = ROOM_FC.get(room.room_type, '#FFFFFF')
        ax.add_patch(mpatches.Rectangle(
            (room.x, room.y), room.width, room.depth,
            facecolor=color, edgecolor='none', zorder=2))


def draw_walls(ax, fp):
    """
    Draw pre-built wall geometry. No Shapely operations here.
    All geometry (box creation, unary_union, difference) is done in
    engine.build_wall_polygons() which enforces the three FIX invariants:
      FIX 1: door gaps → interior solid only; window gaps → exterior solid only
      FIX 2: MIN_WALL_LEN = 0.02 m
      FIX 3: floor_index filtering applied first
    """
    from engine.engine import build_wall_polygons
    geom = build_wall_polygons(fp.walls, fp.doors, fp.windows)

    for solid, fc, hatch in [
        (geom.exterior, WALL_FC,      '///'),
        (geom.interior, WALL_INT_FC,  '//'),
    ]:
        if solid is None or solid.is_empty:
            continue
        patch = _poly_patch(solid, facecolor=fc, edgecolor=WALL_EC,
                            linewidth=0.5, hatch=hatch, zorder=5)
        if patch:
            ax.add_patch(patch)


# ── Production opening renderer — Part 7 ────────────────────────────────────

def _draw_door(ax, door, wall, room_index):
    """
    Draw one door: leaf line + quarter-circle arc (swing) or jamb ticks (archway).
    No geometry computation. Reads absolute (x,y) when available (Part 3 schema);
    falls back to fractional position for legacy DoorOpening objects.
    """
    # Resolve absolute centre — Part 3 schema first, legacy fallback
    if hasattr(door, 'x') and door.x is not None:
        cx, cy = float(door.x), float(door.y)
    else:
        cx, cy = _opening_pt(wall, door.position)

    W    = door.width
    is_H = (wall.direction == 'H')

    # ── Archway: two jamb ticks only, no leaf, no arc ────────────────────────
    if door.door_type == 'archway':
        hw = W / 2
        t2 = wall.thickness / 2 + 0.05
        if is_H:
            for dx in (-hw, hw):
                ax.plot([cx + dx, cx + dx], [cy - t2, cy + t2],
                        '-', color='#1A1A1A', lw=0.9, zorder=Z_OPENING)
        else:
            for dy in (-hw, hw):
                ax.plot([cx - t2, cx + t2], [cy + dy, cy + dy],
                        '-', color='#1A1A1A', lw=0.9, zorder=Z_OPENING)
        ax.text(cx, cy, door.label, ha='center', va='center',
                fontsize=4.8, color='#1A1A1A', zorder=Z_LABEL,
                bbox=dict(fc=BG, ec='none', alpha=0.85, pad=0.4, boxstyle='round'))
        return

    # ── Swing door: hinge → leaf line → quarter-circle arc ───────────────────
    swing_room = room_index.get(door.swing_into)

    if is_H:
        if door.hinge_side == 'left':
            hx, hy   = cx - W / 2, wall.y1
            free_x   = cx + W / 2
            start_a  = 0.0
        else:
            hx, hy   = cx + W / 2, wall.y1
            free_x   = cx - W / 2
            start_a  = math.pi
        rmid_y = (swing_room.y + swing_room.depth / 2) if swing_room else (wall.y1 + 1.0)
        sweep  = math.pi / 2 if rmid_y > wall.y1 else -math.pi / 2
        ax.plot([hx, free_x], [hy, hy], '-', color='#1A1A1A', lw=1.6, zorder=Z_OPENING)
        theta = np.linspace(start_a, start_a + sweep, 60)
        ax.plot(hx + W * np.cos(theta), hy + W * np.sin(theta),
                '-', color='#1A1A1A', lw=0.9, zorder=Z_OPENING)
    else:
        if door.hinge_side == 'left':
            hx, hy   = wall.x1, cy - W / 2
            free_y   = cy + W / 2
            start_a  = math.pi / 2
        else:
            hx, hy   = wall.x1, cy + W / 2
            free_y   = cy - W / 2
            start_a  = -math.pi / 2
        rmid_x = (swing_room.x + swing_room.width / 2) if swing_room else (wall.x1 + 1.0)
        sweep  = math.pi / 2 if rmid_x > wall.x1 else -math.pi / 2
        ax.plot([hx, hx], [hy, free_y], '-', color='#1A1A1A', lw=1.6, zorder=Z_OPENING)
        theta = np.linspace(start_a, start_a + sweep, 60)
        ax.plot(hx + W * np.cos(theta), hy + W * np.sin(theta),
                '-', color='#1A1A1A', lw=0.9, zorder=Z_OPENING)

    ax.text(cx, cy, door.label, ha='center', va='center',
            fontsize=4.8, color='#1A1A1A', zorder=Z_LABEL,
            bbox=dict(fc=BG, ec='none', alpha=0.85, pad=0.4, boxstyle='round'))


def _draw_window(ax, win, wall, fp):
    """
    Draw ISO 128-20 window symbol: two glazing lines + two jamb lines + label.
    No geometry computation. Absolute (x,y) or fractional position fallback.
    Only called on exterior walls — guard enforced by draw_openings().
    """
    if hasattr(win, 'x') and win.x is not None:
        cx, cy = float(win.x), float(win.y)
    else:
        cx, cy = _opening_pt(wall, win.position)

    t   = wall.thickness
    hw  = win.width / 2
    is_H = (wall.direction == 'H')

    if is_H:
        # Two parallel glazing lines at ±20% thickness from wall centreline
        for dy in (-t * 0.20, t * 0.20):
            ax.plot([cx - hw, cx + hw], [cy + dy, cy + dy],
                    color='#1A1A1A', lw=1.0, zorder=Z_OPENING)
        # Vertical jamb lines at each end of opening
        for dx in (-hw, hw):
            ax.plot([cx + dx, cx + dx], [cy - t / 2, cy + t / 2],
                    color='#1A1A1A', lw=1.0, zorder=Z_OPENING)
        ly = cy + t / 2 + 0.18 if cy > fp.net_d / 2 else cy - t / 2 - 0.12
        ax.text(cx, ly, win.label, ha='center', va='center',
                fontsize=5.5, color='#333333', zorder=Z_LABEL)
    else:
        for dx in (-t * 0.20, t * 0.20):
            ax.plot([cx + dx, cx + dx], [cy - hw, cy + hw],
                    color='#1A1A1A', lw=1.0, zorder=Z_OPENING)
        for dy in (-hw, hw):
            ax.plot([cx - t / 2, cx + t / 2], [cy + dy, cy + dy],
                    color='#1A1A1A', lw=1.0, zorder=Z_OPENING)
        lx = cx + t / 2 + 0.18 if cx > fp.net_w / 2 else cx - t / 2 - 0.12
        ax.text(lx, cy, win.label, ha='center', va='center',
                fontsize=5.5, color='#333333', zorder=Z_LABEL)


def draw_openings(ax, fp):
    """
    Single public entry point for all opening rendering.
    Resolves wall via wall_id FK (Part 3 schema) → dispatches to _draw_door / _draw_window.
    Falls back to embedded wall object (legacy DoorOpening/WindowOpening schema).
    No geometry. No Shapely. No position arithmetic.
    """
    # FK index — safe when WallSegment has no wall_id yet (getattr fallback)
    wall_index = {getattr(w, 'wall_id', None): w for w in fp.walls
                  if getattr(w, 'wall_id', None) is not None}
    room_index = {r.room_type: r for r in fp.rooms}

    for door in fp.doors:
        # Part 3 schema: door.wall_id FK first | Legacy: door.wall embedded object
        wall = (wall_index.get(getattr(door, 'wall_id', None))
                or getattr(door, 'wall', None))
        if wall is None:
            continue
        _draw_door(ax, door, wall, room_index)

    for win in fp.windows:
        wall = (wall_index.get(getattr(win, 'wall_id', None))
                or getattr(win, 'wall', None))
        if wall is None or wall.wall_type != 'exterior':
            continue            # guard: windows drawn on exterior walls only
        _draw_window(ax, win, wall, fp)


NFZ_DEPTH = 1.20   # 1200 mm no-fly zone depth in front of door
NFZ_HALF  = 0.60   # 600 mm either side of door centre


def _door_nfz_union(fp):
    """Return a Shapely geometry (union) of all 1200 mm × 1200 mm door clearance zones."""
    from shapely.geometry import box as sbox
    room_map = {r.room_type: r for r in fp.rooms}
    zones = []
    for door in fp.doors:
        wall = door.wall
        cx, cy = _opening_pt(wall, door.position)
        W = door.width
        hw = max(W / 2.0, NFZ_HALF)   # at least 600 mm each side
        swing_room = room_map.get(door.swing_into)

        if wall.direction == 'H':
            swings_north = (swing_room.y + swing_room.depth / 2 > wall.y1
                            if swing_room else True)
            if swings_north:
                zones.append(sbox(cx - hw, wall.y1, cx + hw, wall.y1 + NFZ_DEPTH))
            else:
                zones.append(sbox(cx - hw, wall.y1 - NFZ_DEPTH, cx + hw, wall.y1))
        else:
            swings_east = (swing_room.x + swing_room.width / 2 > wall.x1
                           if swing_room else True)
            if swings_east:
                zones.append(sbox(wall.x1, cy - hw, wall.x1 + NFZ_DEPTH, cy + hw))
            else:
                zones.append(sbox(wall.x1 - NFZ_DEPTH, cy - hw, wall.x1, cy + hw))

    return unary_union(zones) if zones else None


def _clip_furn(x, y, w, d, nfz):
    """Return adjusted (x, y, w, d) after subtracting the NFZ, or None to skip.

    Strategy:
      1. If no intersection → return as-is.
      2. Compute remaining area after difference with NFZ.
      3. Bound-box of the largest remaining rectangle must be ≥ 0.20 m in each dim.
      4. If too small → return None (skip this piece).
    """
    if nfz is None or nfz.is_empty:
        return x, y, w, d
    from shapely.geometry import box as sbox
    furn = sbox(x, y, x + w, y + d)
    if not furn.intersects(nfz):
        return x, y, w, d

    remaining = furn.difference(nfz)
    if remaining.is_empty or remaining.area < 0.04:   # < 200 mm × 200 mm
        return None

    # Pick the largest sub-polygon (handles multi-polygon splits)
    if remaining.geom_type == 'MultiPolygon':
        remaining = max(remaining.geoms, key=lambda g: g.area)

    rx0, ry0, rx1, ry1 = remaining.bounds
    rw, rd = rx1 - rx0, ry1 - ry0
    if rw < 0.20 or rd < 0.20:
        return None

    return rx0, ry0, rw, rd


def _door_sides_for_room(room, fp):
    """Return set of wall sides ('top','bottom','left','right') that have doors."""
    x0, y0, w, d = room.x, room.y, room.width, room.depth
    sides = set()
    for door in fp.doors:
        if door.room_from != room.room_type and door.room_to != room.room_type:
            continue
        wl = door.wall
        if wl.direction == 'H':
            if abs(wl.y1 - (y0 + d)) < TOL + 0.15:
                sides.add('top')
            elif abs(wl.y1 - y0) < TOL + 0.15:
                sides.add('bottom')
        else:
            if abs(wl.x1 - x0) < TOL + 0.15:
                sides.add('left')
            elif abs(wl.x1 - (x0 + w)) < TOL + 0.15:
                sides.add('right')
    return sides


def draw_furniture(ax, fp):
    """Draw schematic furniture symbols, respecting 1200mm door no-fly zones."""
    nfz = _door_nfz_union(fp)

    def safe_rect(x, y, w, d, **kw):
        """Place a furniture rect only if it clears all door NFZs."""
        result = _clip_furn(x, y, w, d, nfz)
        if result is None:
            return
        _frect(ax, *result, **kw)

    for room in fp.rooms:
        x0, y0, w, d = room.x, room.y, room.width, room.depth
        t = room.room_type

        if t == 'master_bedroom':
            bw = min(w * 0.65, 1.8)
            bd = min(d * 0.52, 1.95)
            bx = x0 + (w - bw) / 2
            by = y0 + d - bd - 0.12
            safe_rect(bx, by, bw, bd)
            safe_rect(bx, by + bd * 0.75, bw, bd * 0.25, fc='#C0C0C0')  # pillow
            safe_rect(bx - 0.52, by + 0.1, 0.44, 0.44)   # bedside L
            safe_rect(bx + bw + 0.08, by + 0.1, 0.44, 0.44)  # bedside R

        elif t in ('bedroom_2', 'bedroom_3', 'bedroom_4', 'bedroom_5', 'bedroom_6'):
            bw = min(w * 0.65, 1.0)
            bd = min(d * 0.52, 1.9)
            bx = x0 + (w - bw) / 2
            by = y0 + d - bd - 0.12
            safe_rect(bx, by, bw, bd)
            safe_rect(bx, by + bd * 0.75, bw, bd * 0.25, fc='#C0C0C0')

        elif t == 'living':
            sw = min(w * 0.55, 2.0)
            sd = min(d * 0.22, 0.75)
            sx = x0 + (w - sw) / 2
            sy = y0 + 0.15
            safe_rect(sx, sy, sw, sd)                              # main sofa
            safe_rect(sx + sw, sy, min(w * 0.18, 0.65), sd * 1.7) # side sofa
            safe_rect(x0 + (w - 0.75) / 2, y0 + d * 0.42, 0.75, 0.42)  # coffee table

        elif t == 'dining':
            tw = min(1.2, w * 0.55)
            td = min(0.8, d * 0.45)
            tx = x0 + (w - tw) / 2
            ty = y0 + (d - td) / 2
            safe_rect(tx, ty, tw, td)
            cs = 0.28
            for (cx2, cy2) in [
                (tx + tw / 2 - cs / 2, ty - cs - 0.05),
                (tx + tw / 2 - cs / 2, ty + td + 0.05),
                (tx - cs - 0.05,       ty + td / 2 - cs / 2),
                (tx + tw + 0.05,       ty + td / 2 - cs / 2),
            ]:
                safe_rect(cx2, cy2, cs, cs)

        elif t in ('kitchen', 'dry_kitchen'):
            ct = 0.55
            ds = _door_sides_for_room(room, fp)

            back_side = None
            for candidate in ('top', 'bottom'):
                if candidate not in ds:
                    back_side = candidate
                    break

            if back_side == 'top':
                safe_rect(x0 + 0.08, y0 + d - ct - 0.08, max(w - 0.16, 0.3), ct)
                safe_rect(x0 + 0.15, y0 + d - ct + 0.06, 0.38, 0.27, fc='#A8C8E8')
            elif back_side == 'bottom':
                safe_rect(x0 + 0.08, y0 + 0.08, max(w - 0.16, 0.3), ct)
                safe_rect(x0 + 0.15, y0 + 0.08 - 0.27 + ct - 0.06, 0.38, 0.27, fc='#A8C8E8')

            if d > 1.5:
                side_h = d - ct - 0.2 if back_side else d - 0.16
                side_h = max(side_h, 0.3)
                if 'left' not in ds:
                    side_y = (y0 + ct + 0.1) if back_side == 'bottom' else (y0 + 0.08)
                    safe_rect(x0 + 0.08, side_y, ct, side_h)
                elif 'right' not in ds:
                    side_y = (y0 + ct + 0.1) if back_side == 'bottom' else (y0 + 0.08)
                    safe_rect(x0 + w - ct - 0.08, side_y, ct, side_h)

        elif t in ('toilet_attached', 'toilet_common'):
            safe_rect(x0 + (w - 0.38) / 2, y0 + 0.1, 0.38, 0.52)   # WC
            safe_rect(x0 + 0.1, y0 + d - 0.32, 0.32, 0.22)           # basin

        elif t in ('verandah', 'balcony'):
            if w >= 2.5:
                for ix in (0.2, 0.85, 1.5):
                    safe_rect(x0 + ix, y0 + (d - 0.44) / 2, 0.44, 0.44)
            elif w >= 1.5:
                safe_rect(x0 + 0.2, y0 + (d - 0.4) / 2, 0.38, 0.38)
                safe_rect(x0 + 0.75, y0 + (d - 0.38) / 2, 0.38, 0.38)

        elif t == 'pooja':
            pw = min(w * 0.6, 0.7)
            safe_rect(x0 + (w - pw) / 2, y0 + d - 0.25, pw, 0.22)

        elif t == 'store':
            rw = min(w * 0.75, 1.2)
            safe_rect(x0 + (w - rw) / 2, y0 + d - 0.3, rw, 0.27)

        elif t in ('staircase', 'staircase_head'):
            # Draw tread lines at exactly 250 mm intervals (NBC stair tread depth)
            step_h = 0.25
            n_steps = max(int(d / step_h), 4)
            for i in range(1, n_steps + 1):
                ly = y0 + i * step_h
                if ly >= y0 + d:
                    break
                ax.plot([x0 + 0.05, x0 + w - 0.05], [ly, ly],
                        '-', color=FURN_EC, lw=0.7, zorder=4)
            # Direction arrow pointing up (ascending)
            ax.annotate('', xy=(x0 + w / 2, y0 + d - 0.15),
                        xytext=(x0 + w / 2, y0 + 0.15),
                        arrowprops=dict(arrowstyle='->', color='#333333', lw=1.0), zorder=4)


def draw_annotations(ax, fp):
    """Draw room name labels and dimension text inside each room."""
    has_dining = any(r.room_type == 'dining' for r in fp.rooms)
    for room in fp.rooms:
        cx = room.x + room.width / 2
        cy = room.y + room.depth / 2
        label = ROOM_LABEL.get(room.room_type, room.room_type.upper().replace('_', '\n'))
        if room.room_type == 'living' and not has_dining:
            label = 'LIVING/DINING'

        area = room.area
        fs = 8.5 if area >= 12 else 7.5 if area >= 7 else 6.5 if area >= 4 else 5.5

        if room.room_type == 'corridor':
            # Corridor/passage: single centred italic label, no dimension text
            ax.text(cx, cy, label, ha='center', va='center',
                    fontsize=6.0, style='italic', color='#666666',
                    multialignment='center', zorder=8)
        else:
            ax.text(cx, cy + 0.1, label, ha='center', va='center',
                    fontsize=fs, fontweight='bold', color='#111111',
                    multialignment='center', zorder=8)
            ax.text(cx, cy - 0.25, f'{room.width:.1f}m \u00d7 {room.depth:.1f}m',
                    ha='center', va='center', fontsize=fs * 0.78, color='#333333', zorder=8)


def draw_dimensions(ax, fp):
    """Draw external dimension arrows + wall thickness callouts."""
    sb, sr, sf = fp.setback_side, fp.setback_rear, fp.setback_front

    # Overall plot dimensions (outside boundary)
    _dim_h(ax, -sb, fp.net_w + sb, fp.net_d + sf, f'{fp.plot_w:.0f}m', offset=0.6)
    _dim_v(ax, -sr, fp.net_d + sf, fp.net_w + sb, f'{fp.plot_d:.0f}m', offset=0.7)

    # Net buildable dimensions
    _dim_h(ax, 0, fp.net_w, -sr, f'{fp.net_w:.0f}m', offset=-0.5)
    _dim_v(ax, 0, fp.net_d, fp.net_w, f'{fp.net_d:.0f}m', offset=0.45)

    # Side setback ticks
    if sb > 0.05:
        _dim_h(ax, -sb, 0, -sr, f'{sb:.1f}m', offset=-0.5, fs=6.0)
        _dim_h(ax, fp.net_w, fp.net_w + sb, -sr, f'{sb:.1f}m', offset=-0.5, fs=6.0)

    # Wall thickness callouts
    annot_kw = dict(fontsize=5.5, color='#444444', ha='left', va='bottom', zorder=9)
    arrow_kw = dict(arrowstyle='->', color='#666666', lw=0.6)
    for wl in fp.walls:
        if wl.wall_type == 'exterior' and wl.direction == 'H' and abs(wl.y1 - fp.net_d) < TOL:
            px = (wl.x1 + wl.x2) / 2 + 1.0
            ax.annotate('230mm', xy=(px, wl.y1), xytext=(px + 0.3, wl.y1 + 0.85),
                        arrowprops=arrow_kw, **annot_kw)
            break
    for wl in fp.walls:
        if wl.wall_type != 'exterior' and wl.direction == 'V':
            px = wl.x1
            py = (wl.y1 + wl.y2) / 2
            ax.annotate('115mm', xy=(px, py), xytext=(px + 1.0, py + 0.9),
                        arrowprops=arrow_kw, **annot_kw)
            break


def draw_circulation(ax, fp):
    """Draw circulation band between service and private zones.
    Skipped if a real 'corridor' room already occupies that space.
    """
    # If a named corridor/passage room exists, it already renders the zone fill
    if any(r.room_type == 'corridor' for r in fp.rooms):
        return

    brs  = [r for r in fp.rooms
            if r.room_type in ('master_bedroom', 'bedroom_2', 'bedroom_3',
                               'bedroom_4', 'bedroom_5', 'bedroom_6')]
    svcs = [r for r in fp.rooms
            if r.room_type in ('toilet_common', 'kitchen', 'utility')]
    if not brs or not svcs:
        return
    br_top  = max(r.y + r.depth for r in brs)
    svc_bot = min(r.y for r in svcs)
    gap = svc_bot - br_top
    if gap < 0.05:
        return
    ax.add_patch(mpatches.Rectangle(
        (0, br_top), fp.net_w, gap,
        facecolor='#F0F0F0', edgecolor='none', alpha=0.7, zorder=2))
    mx = fp.net_w / 2
    my = (br_top + svc_bot) / 2
    ax.text(mx, my, 'CIRCULATION', ha='center', va='center',
            fontsize=5.5, color='#AAAAAA', style='italic', zorder=3)
    ax.annotate('', xy=(mx + 0.5, svc_bot - 0.08), xytext=(mx + 0.5, br_top + 0.08),
                arrowprops=dict(arrowstyle='->', color='#CCCCCC', lw=0.7), zorder=3)


def draw_north_arrow(ax, x, y, r=0.55):
    """Draw standard architectural compass rose:
    hollow circle, cross-hair, filled north triangle, hollow south triangle,
    N/S/E/W labels outside the circle.
    """
    from matplotlib.patches import Polygon as MplPolygon

    # Outer circle (hollow)
    ax.add_patch(plt.Circle((x, y), r, fill=False, edgecolor='#222222', lw=1.2, zorder=10))

    # Cross-hair lines (H and V through centre)
    ax.plot([x - r, x + r], [y, y], '-', color='#444444', lw=0.7, zorder=10)
    ax.plot([x, x], [y - r, y + r], '-', color='#444444', lw=0.7, zorder=10)

    # North filled triangle (pointing up)
    tri_n = np.array([[x, y + r * 0.95], [x - r * 0.28, y], [x + r * 0.28, y]])
    ax.add_patch(MplPolygon(tri_n, closed=True, facecolor='#111111',
                             edgecolor='#111111', lw=0.8, zorder=11))

    # South hollow triangle (pointing down)
    tri_s = np.array([[x, y - r * 0.95], [x - r * 0.28, y], [x + r * 0.28, y]])
    ax.add_patch(MplPolygon(tri_s, closed=True, facecolor='#FFFFFF',
                             edgecolor='#111111', lw=0.8, zorder=11))

    # Cardinal labels outside the circle
    label_off = r + 0.22
    ax.text(x,       y + label_off, 'N', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#111111', zorder=12)
    ax.text(x,       y - label_off, 'S', ha='center', va='center',
            fontsize=7.5, color='#444444', zorder=12)
    ax.text(x + label_off, y, 'E', ha='center', va='center',
            fontsize=7.5, color='#444444', zorder=12)
    ax.text(x - label_off, y, 'W', ha='center', va='center',
            fontsize=7.5, color='#444444', zorder=12)


def draw_legend(ax, fp, lx, ly_top):
    """Draw room colour legend on the right."""
    ax.text(lx, ly_top, 'LEGEND', ha='left', va='top',
            fontsize=9, fontweight='bold', color='#111111', zorder=10)
    ly = ly_top - 0.55
    seen = []
    for r in fp.rooms:
        if r.room_type not in seen:
            seen.append(r.room_type)
    for rt in seen:
        ax.add_patch(mpatches.Rectangle(
            (lx, ly), 0.38, 0.22,
            facecolor=ROOM_FC.get(rt, '#FFF'), edgecolor='#888888', lw=0.7, zorder=10))
        lbl = ROOM_LABEL.get(rt, rt).replace('\n', ' ')
        ax.text(lx + 0.52, ly + 0.11, lbl, ha='left', va='center',
                fontsize=6.5, color='#111111', zorder=10)
        ly -= 0.34
    ly -= 0.15
    for name, val in [
        ('Vastu',  fp.score_vastu),
        ('NBC',    fp.score_nbc),
        ('Circ.',  fp.score_circulation),
        ('Overall',fp.score_overall),
    ]:
        ax.text(lx, ly, f'{name}: {val:.0%}', ha='left', va='top',
                fontsize=6.5, color='#333333', zorder=10)
        ly -= 0.28


def draw_titleblock(ax, fp, x0, y0, floor_label='ground'):
    """Draw title block below the plan."""
    floor_str = 'GROUND FLOOR' if floor_label == 'ground' else 'FIRST FLOOR (G+1)'
    # Separator line
    ax.plot([x0, x0 + fp.net_w + 2.0], [y0 + 0.9, y0 + 0.9],
            '-', color='#444444', lw=1.8, zorder=9)
    # Circle symbol (plan symbol convention)
    ax.add_patch(plt.Circle((x0 + 0.4, y0 + 0.4), 0.3,
                             fill=False, edgecolor='#333333', lw=0.9, zorder=10))
    tx = x0 + 0.9
    ax.text(tx, y0 + 0.62,
            f'{fp.bhk}BHK RESIDENCE FLOOR PLAN - {floor_str}',
            ha='left', va='center', fontsize=9.5, fontweight='bold', color='#111111', zorder=10)
    ax.text(tx, y0 + 0.33,
            f'{fp.facing}-FACING PLOT: {fp.plot_w:.0f}m × {fp.plot_d:.0f}m',
            ha='left', va='center', fontsize=8, color='#222222', zorder=10)
    ax.text(tx, y0 + 0.08, f'{fp.district}, Tamil Nadu, India',
            ha='left', va='center', fontsize=7.5, color='#333333', zorder=10)
    ax.text(tx, y0 - 0.15,
            f'Climate: {fp.climate_zone}  |  Scale 1:50  |  Net: {fp.net_w:.1f}m × {fp.net_d:.1f}m',
            ha='left', va='center', fontsize=6, color='#555555', zorder=10)


# ─── main entry ───────────────────────────────────────────────────────────────

def render_mpl(fp: FloorPlan, output_dir: str = 'outputs',
               floor_label: str = 'ground',
               filename: str = None) -> str:
    """
    Render a FloorPlan to PNG using pure matplotlib.

    Returns the PNG file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    sb, sr, sf = fp.setback_side, fp.setback_rear, fp.setback_front

    # Data coordinate extents
    x0 = -sb - 1.2           # left: room for vertical dim + margin
    x1 = fp.net_w + sb + 5.0 # right: legend panel (4m) + margin
    y0 = -sr - 2.8           # bottom: title block + dim ticks
    y1 = fp.net_d + sf + 1.5 # top: outer dim + margin

    dw = x1 - x0
    dh = y1 - y0

    # Figure: 20 inches wide, height proportional (min 12)
    FIG_W = 20.0
    FIG_H = max(FIG_W * dh / dw, 12.0)
    DPI   = 150

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # ── draw layers (bottom → top, zorder-safe sequence) ────────────────────
    draw_boundary(ax, fp)       # Z_BG=1    property boundary + setback labels
    draw_room_fills(ax, fp)     # Z_FILL=2  room colour patches
    draw_walls(ax, fp)          # Z_WALL=5  wall solids (build_wall_polygons, no geometry here)
    draw_openings(ax, fp)       # Z_OPENING=6 + Z_LABEL=7  doors + windows via FK dispatch
    draw_furniture(ax, fp)      # Z_OPENING=6  furniture clips to NFZ
    draw_circulation(ax, fp)    # Z_OPENING=6  suppressed when corridor room present
    draw_annotations(ax, fp)    # Z_ANNOT=8    room name + area text
    draw_dimensions(ax, fp)     # Z_DIM=9      exterior + interior dimension chains

    # ── legend, north arrow, title (outside plan area) ──────────────────────
    leg_x  = fp.net_w + sb + 0.6
    leg_y  = fp.net_d + sf - 0.2
    draw_legend(ax, fp, leg_x, leg_y)

    na_x = fp.net_w + sb + 3.5
    na_y = fp.net_d + sf - 0.8
    draw_north_arrow(ax, na_x, na_y)

    draw_titleblock(ax, fp, -sb, -sr - 2.5, floor_label=floor_label)

    # ── save ────────────────────────────────────────────────────────────────
    if filename is None:
        filename = f'plan_{fp.district}_{fp.bhk}BHK_{fp.facing}_{floor_label}.png'
    png_path = os.path.join(output_dir, filename)
    fig.savefig(png_path, dpi=DPI, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'  PNG: {png_path}')
    return png_path


# ─── standalone ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from engine.engine_api import generate_plan
    result = generate_plan(plot_w=12, plot_d=15, bhk=2, facing='N',
                           district='Coimbatore', floors=2)
    ground = result['ground']
    if isinstance(ground.rooms, dict):
        ground.rooms = list(ground.rooms.values())
    render_mpl(ground, 'outputs', 'ground')

    first = result.get('first')
    if first:
        if isinstance(first.rooms, dict):
            first.rooms = list(first.rooms.values())
        render_mpl(first, 'outputs', 'first')
