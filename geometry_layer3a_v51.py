#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stratul 3A v5.1 (Pur Geometric, Low-Level, 0% Muzică):
Metrică locală de grilă cu verificare completă de monotonie și suport,
partiție contiguă a celulelor pe [u_lo, u_hi] în toleranță explicită,
distanță euclidiană EXACTĂ între poligoane convexe (inclusiv cazuri degenerate:
puncte / segmente cu arie zero), clasificare distinctă `collinear_overlap` vs.
`near_collinear_overlap` vs. `parallel_side_by_side`, graf spațial global complet
(inclusiv proximități primitivă-linie și relații linie-linie între grile diferite),
validare completă și audit strict count == 1.

Citește EXCLUSIV JSON-urile Stratului 1 și Stratului 2 (zero pixeli pentru raționament;
imaginea originală este opțională doar ca fundal de debug).

Utilizare:
  python geometry_layer3a.py poza_geometry_layer1.json poza_geometry_layer2.json --out-dir debug_l3a
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SCHEMA_VERSION = "geometry_layer3a.v5.1"
VALID_LAYER1_TYPES = ("banda_dreapta", "banda_curba", "pata_compacta")


class GateViolationError(RuntimeError):
    """Aruncată când oricare poartă matematică (0 -> 1 -> 2 -> 3A) sau configurația este invalidă."""


class AuditViolationError(RuntimeError):
    """Aruncată când auditul strict count == 1 sau partiția celulelor este încălcată."""


# ---------------------------------------------------------------------------
# Configurație centralizată, scalată cu S / w și validată integral
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Layer3AConfig:
    axis_norm_tol: float = 1e-2
    axis_ortho_tol: float = 1e-2

    knot_merge_in_w: float = 1.5
    interval_bridge_in_w: float = 1.0
    max_line_slope: float = 0.15

    bar_angle_min_deg: float = 74.0
    bar_angle_max_deg: float = 106.0
    bar_min_grid_v_overlap_ratio: float = 0.90
    bar_end_overshoot_in_S: float = 0.45
    bar_large_nb_w_in_S: float = 0.50
    bar_large_nb_h_in_S: float = 0.45
    double_bar_max_sep_in_S: float = 1.25
    thick_bar_width_ratio: float = 1.80
    thick_bar_min_w_in_S: float = 0.25
    double_bar_thick_ratio: float = 1.60
    left_edge_bar_margin_in_S: float = 1.50
    min_cell_width_in_S: float = 1.20
    min_trailing_cell_width_in_S: float = 1.80
    cell_continuity_tol_px: float = 1e-2

    grid_assign_u_margin_in_S: float = 1.50
    grid_assign_v_limit_explicit_in_S: float = 3.50
    grid_assign_v_limit_general_in_S: float = 2.60

    parallel_angle_tol_deg: float = 20.0
    perp_angle_min_deg: float = 70.0
    perp_angle_max_deg: float = 110.0

    # Distincție strictă între coliniaritate matematică exactă și coliniaritate aproximativă
    exact_collinear_angle_eps_deg: float = 1e-3
    exact_collinear_axis_eps_in_w: float = 1e-4
    near_collinear_angle_tol_deg: float = 5.0
    near_collinear_axis_tol_in_w: float = 0.25
    parallel_contact_angle_tol_deg: float = 15.0
    parallel_side_min_overlap_ratio: float = 0.15

    near_touch_in_S: float = 0.25
    proximity_radius_in_S: float = 3.00
    bar_proximity_u_in_S: float = 1.50
    bar_proximity_v_in_S: float = 0.75

    end_zone_other_w_mult: float = 1.50
    end_zone_len_frac: float = 0.18
    end_zone_max_in_S: float = 0.45
    end_zone_max_frac: float = 0.35
    blob_end_margin_in_S: float = 0.35
    blob_end_len_frac: float = 0.22
    blob_side_deadzone_in_S: float = 0.08


def _is_finite_num(val: Any) -> bool:
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def _is_2d_finite(val: Any) -> bool:
    return isinstance(val, (list, tuple)) and len(val) == 2 and all(_is_finite_num(x) for x in val)


def validate_config(cfg: Layer3AConfig) -> None:
    """Validează finitudinea, limitele și relațiile logice ale tuturor parametrilor."""
    raw = asdict(cfg)
    for k, v in raw.items():
        if not _is_finite_num(v):
            raise GateViolationError(f"Config invalid `{k}`: valoare ne-finită ({v}).")

    strict_pos = (
        "axis_norm_tol", "axis_ortho_tol", "knot_merge_in_w", "max_line_slope",
        "bar_large_nb_w_in_S", "bar_large_nb_h_in_S", "double_bar_max_sep_in_S",
        "thick_bar_width_ratio", "thick_bar_min_w_in_S", "double_bar_thick_ratio",
        "min_cell_width_in_S", "min_trailing_cell_width_in_S", "cell_continuity_tol_px",
        "grid_assign_v_limit_explicit_in_S", "grid_assign_v_limit_general_in_S",
        "proximity_radius_in_S", "bar_proximity_u_in_S", "end_zone_other_w_mult",
        "end_zone_max_in_S", "blob_end_margin_in_S",
        "exact_collinear_angle_eps_deg", "exact_collinear_axis_eps_in_w",
        "near_collinear_angle_tol_deg", "near_collinear_axis_tol_in_w",
        "parallel_contact_angle_tol_deg",
    )
    for k in strict_pos:
        if float(raw[k]) <= 0.0:
            raise GateViolationError(f"Config invalid `{k}`={raw[k]} (trebuie > 0).")

    non_neg = (
        "interval_bridge_in_w", "bar_end_overshoot_in_S", "left_edge_bar_margin_in_S",
        "grid_assign_u_margin_in_S", "parallel_angle_tol_deg", "near_touch_in_S",
        "bar_proximity_v_in_S", "blob_side_deadzone_in_S",
    )
    for k in non_neg:
        if float(raw[k]) < 0.0:
            raise GateViolationError(f"Config invalid `{k}`={raw[k]} (trebuie >= 0).")

    if not (0.0 < cfg.bar_min_grid_v_overlap_ratio <= 1.0):
        raise GateViolationError("Config invalid: bar_min_grid_v_overlap_ratio trebuie în (0, 1].")
    if not (0.0 < cfg.parallel_side_min_overlap_ratio <= 1.0):
        raise GateViolationError("Config invalid: parallel_side_min_overlap_ratio trebuie în (0, 1].")
    if cfg.exact_collinear_angle_eps_deg > cfg.near_collinear_angle_tol_deg:
        raise GateViolationError("Config invalid: exact_collinear_angle_eps_deg > near_collinear_angle_tol_deg.")
    if cfg.near_collinear_angle_tol_deg > cfg.parallel_contact_angle_tol_deg:
        raise GateViolationError("Config invalid: near_collinear_angle_tol_deg > parallel_contact_angle_tol_deg.")
    if cfg.parallel_contact_angle_tol_deg >= 45.0:
        raise GateViolationError("Config invalid: parallel_contact_angle_tol_deg trebuie < 45.")
    if cfg.exact_collinear_axis_eps_in_w > cfg.near_collinear_axis_tol_in_w:
        raise GateViolationError("Config invalid: exact_collinear_axis_eps_in_w > near_collinear_axis_tol_in_w.")
    if cfg.near_collinear_axis_tol_in_w >= 1.0:
        raise GateViolationError("Config invalid: near_collinear_axis_tol_in_w trebuie < 1.0.")
    if not (0.0 < cfg.end_zone_len_frac <= 0.5 and 0.0 < cfg.end_zone_max_frac <= 0.5):
        raise GateViolationError("Config invalid: end_zone_len_frac / end_zone_max_frac trebuie în (0, 0.5].")
    if cfg.end_zone_len_frac > cfg.end_zone_max_frac:
        raise GateViolationError("Config invalid: end_zone_len_frac > end_zone_max_frac.")
    if not (0.0 < cfg.blob_end_len_frac <= 0.5):
        raise GateViolationError("Config invalid: blob_end_len_frac trebuie în (0, 0.5].")
    if not (0.0 <= cfg.bar_angle_min_deg < cfg.bar_angle_max_deg <= 180.0):
        raise GateViolationError("Config invalid: 0 <= bar_angle_min_deg < bar_angle_max_deg <= 180.")
    if not (0.0 <= cfg.perp_angle_min_deg < cfg.perp_angle_max_deg <= 180.0):
        raise GateViolationError("Config invalid: 0 <= perp_angle_min_deg < perp_angle_max_deg <= 180.")
    if cfg.parallel_angle_tol_deg >= cfg.perp_angle_min_deg:
        raise GateViolationError("Config invalid: parallel_angle_tol_deg se suprapune cu perp_angle_min_deg.")
    if cfg.near_touch_in_S > cfg.proximity_radius_in_S:
        raise GateViolationError("Config invalid: near_touch_in_S > proximity_radius_in_S.")
    if cfg.thick_bar_width_ratio < 1.0 or cfg.double_bar_thick_ratio < 1.0:
        raise GateViolationError("Config invalid: rapoartele de grosime ale barelor trebuie >= 1.0.")


# ---------------------------------------------------------------------------
# Utilitare geometrice exacte (poligoane convexe + tratare exactă cazuri degenerate)
# ---------------------------------------------------------------------------
def _angle_diff_180(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _point_to_segment(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    ab = b - a
    l2 = float(np.dot(ab, ab))
    if l2 < 1e-12:
        return float(np.linalg.norm(pt - a)), 0.0
    t = float(np.clip(np.dot(pt - a, ab) / l2, 0.0, 1.0))
    return float(np.linalg.norm(pt - (a + t * ab))), t


def _segments_intersect(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> Tuple[bool, Optional[float], Optional[float]]:
    da = a1 - a0
    db = b1 - b0
    denom = float(da[0] * db[1] - da[1] * db[0])
    if abs(denom) <= 1e-12:
        return False, None, None
    diff = b0 - a0
    ta = float((diff[0] * db[1] - diff[1] * db[0]) / denom)
    tb = float((diff[0] * da[1] - diff[1] * da[0]) / denom)
    if 0.0 <= ta <= 1.0 and 0.0 <= tb <= 1.0:
        return True, ta, tb
    return False, None, None


def _point_to_aabb(pt: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> float:
    dx = max(x0 - float(pt[0]), 0.0, float(pt[0]) - x1)
    dy = max(y0 - float(pt[1]), 0.0, float(pt[1]) - y1)
    return math.hypot(dx, dy)


def _ribbon_oriented_polygon_xy(p: Dict[str, Any]) -> np.ndarray:
    """Cele 4 colțuri în ordine ciclică ale dreptunghiului orientat al unei benzi drepte."""
    a = np.array(p["start"], dtype=np.float64)
    b = np.array(p["end"], dtype=np.float64)
    hw = 0.5 * float(p["width_px"])
    d = b - a
    L = float(np.linalg.norm(d))
    if L > 1e-9:
        n = np.array([-d[1] / L, d[0] / L], dtype=np.float64)
        return np.array([a + hw * n, a - hw * n, b - hw * n, b + hw * n], dtype=np.float64)
    return np.array([
        [a[0] - hw, a[1] - hw], [a[0] + hw, a[1] - hw],
        [a[0] + hw, a[1] + hw], [a[0] - hw, a[1] + hw],
    ], dtype=np.float64)


def _aabb_polygon_xy(bbox: Sequence[float]) -> np.ndarray:
    x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def _polygon_signed_area2(poly: np.ndarray) -> float:
    m = len(poly)
    if m < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _point_in_convex_poly(pt: np.ndarray, poly: np.ndarray) -> bool:
    """Testează incluziunea exactă a unui punct într-un poligon convex,
    inclusiv cazuri degenerate cu arie zero (punct sau segment coliniar)."""
    m = len(poly)
    if m == 0:
        return False
    eps = 1e-11
    if (pt[0] < float(np.min(poly[:, 0])) - eps or pt[0] > float(np.max(poly[:, 0])) + eps or
            pt[1] < float(np.min(poly[:, 1])) - eps or pt[1] > float(np.max(poly[:, 1])) + eps):
        return False

    if abs(_polygon_signed_area2(poly)) <= 1e-11:
        for i in range(m):
            d, _ = _point_to_segment(pt, poly[i], poly[(i + 1) % m])
            if d <= eps:
                return True
        return False

    pos = False
    neg = False
    for i in range(m):
        v0 = poly[i]
        v1 = poly[(i + 1) % m]
        cross = float((v1[0] - v0[0]) * (pt[1] - v0[1]) - (v1[1] - v0[1]) * (pt[0] - v0[0]))
        if cross > eps:
            pos = True
        elif cross < -eps:
            neg = True
        if pos and neg:
            return False
    return True


def _convex_polygons_exact_distance(poly_a: np.ndarray, poly_b: np.ndarray) -> Tuple[float, bool]:
    """Distanța euclidiană EXACTĂ între două poligoane convexe 2D (inclusiv puncte & segmente degenerate)."""
    na, nb = len(poly_a), len(poly_b)
    for i in range(na):
        a0, a1 = poly_a[i], poly_a[(i + 1) % na]
        for j in range(nb):
            b0, b1 = poly_b[j], poly_b[(j + 1) % nb]
            hit, _, _ = _segments_intersect(a0, a1, b0, b1)
            if hit:
                return 0.0, True

    for i in range(na):
        if _point_in_convex_poly(poly_a[i], poly_b):
            return 0.0, True
    for j in range(nb):
        if _point_in_convex_poly(poly_b[j], poly_a):
            return 0.0, True

    d_min = float("inf")
    for i in range(na):
        pt = poly_a[i]
        for j in range(nb):
            d, _ = _point_to_segment(pt, poly_b[j], poly_b[(j + 1) % nb])
            if d < d_min:
                d_min = d
    for j in range(nb):
        pt = poly_b[j]
        for i in range(na):
            d, _ = _point_to_segment(pt, poly_a[i], poly_a[(i + 1) % na])
            if d < d_min:
                d_min = d
    if d_min <= 1e-11:
        return 0.0, True
    return float(d_min), False


def _prim_xy_box(p: Dict[str, Any]) -> Tuple[float, float, float, float]:
    if p["type"] == "banda_dreapta":
        corners = _ribbon_oriented_polygon_xy(p)
        return (
            float(np.min(corners[:, 0])), float(np.min(corners[:, 1])),
            float(np.max(corners[:, 0])), float(np.max(corners[:, 1])),
        )
    b = p["bbox"]
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def _prim_uv_box(p: Dict[str, Any], u_vec: np.ndarray, v_vec: np.ndarray) -> Tuple[float, float, float, float]:
    corners = _ribbon_oriented_polygon_xy(p) if p["type"] == "banda_dreapta" else _aabb_polygon_xy(p["bbox"])
    us = corners @ u_vec
    vs = corners @ v_vec
    return float(np.min(us)), float(np.min(vs)), float(np.max(us)), float(np.max(vs))


def _prim_uv_center(p: Dict[str, Any], u_vec: np.ndarray, v_vec: np.ndarray) -> Tuple[float, float]:
    c = np.array(p["center"], dtype=np.float64)
    return float(np.dot(c, u_vec)), float(np.dot(c, v_vec))


def _uv_to_xy(u: float, v: float, u_vec: np.ndarray, v_vec: np.ndarray) -> List[float]:
    pt = u * u_vec + v * v_vec
    return [round(float(pt[0]), 2), round(float(pt[1]), 2)]


def _uv_box_gap(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    gu = max(box_b[0] - box_a[2], 0.0, box_a[0] - box_b[2])
    gv = max(box_b[1] - box_a[3], 0.0, box_a[1] - box_b[3])
    return gu, gv, math.hypot(gu, gv)


def _segments_relation_xy(
    p1: Dict[str, Any], p2: Dict[str, Any], S_ref: float, cfg: Layer3AConfig
) -> Dict[str, Any]:
    """Geometria EXACTĂ dintre două benzi drepte:
      - distanță euclidiană exactă între cele două dreptunghiuri orientate;
      - separare strictă între `collinear_overlap`, `near_collinear_overlap`,
        `parallel_side_by_side`, `endpoint_to_endpoint`, `a_endpoint_on_b_interior`,
        `b_endpoint_on_a_interior` și `interior_to_interior_crossing`.
    """
    a0 = np.array(p1["start"], dtype=np.float64)
    a1 = np.array(p1["end"], dtype=np.float64)
    b0 = np.array(p2["start"], dtype=np.float64)
    b1 = np.array(p2["end"], dtype=np.float64)
    w1, w2 = float(p1["width_px"]), float(p2["width_px"])

    poly_a = _ribbon_oriented_polygon_xy(p1)
    poly_b = _ribbon_oriented_polygon_xy(p2)
    exact_poly_gap, polys_intersect = _convex_polygons_exact_distance(poly_a, poly_b)

    axes_intersect, t_a_cross, t_b_cross = _segments_intersect(a0, a1, b0, b1)
    d_a0, t_b_for_a0 = _point_to_segment(a0, b0, b1)
    d_a1, t_b_for_a1 = _point_to_segment(a1, b0, b1)
    d_b0, t_a_for_b0 = _point_to_segment(b0, a0, a1)
    d_b1, t_a_for_b1 = _point_to_segment(b1, a0, a1)

    if axes_intersect and t_a_cross is not None and t_b_cross is not None:
        raw_axis_min, best_ta, best_tb = 0.0, t_a_cross, t_b_cross
    else:
        raw_axis_min, best_ta, best_tb = min(
            [(d_a0, 0.0, t_b_for_a0), (d_a1, 1.0, t_b_for_a1),
             (d_b0, t_a_for_b0, 0.0), (d_b1, t_a_for_b1, 1.0)],
            key=lambda item: item[0],
        )

    da = a1 - a0
    db = b1 - b0
    la = float(np.linalg.norm(da))
    lb = float(np.linalg.norm(db))
    parallel_overlap_px: Optional[float] = None
    parallel_overlap_ratio: Optional[float] = None
    axis_perp_sep_px: Optional[float] = None

    if la > 1e-9 and lb > 1e-9:
        ua = da / la
        ub = db / lb
        if float(np.dot(ua, ub)) < 0.0:
            ub = -ub
        exact_ang_deg = math.degrees(math.acos(float(np.clip(np.dot(ua, ub), -1.0, 1.0))))

        if exact_ang_deg <= cfg.parallel_contact_angle_tol_deg:
            u_sum = la * ua + lb * ub
            u_par = u_sum / max(1e-12, float(np.linalg.norm(u_sum)))
            n_par = np.array([-u_par[1], u_par[0]], dtype=np.float64)

            sa_a0, sa_a1 = float(np.dot(a0, u_par)), float(np.dot(a1, u_par))
            sb_b0, sb_b1 = float(np.dot(b0, u_par)), float(np.dot(b1, u_par))
            sa_lo, sa_hi = (sa_a0, sa_a1) if sa_a0 <= sa_a1 else (sa_a1, sa_a0)
            sb_lo, sb_hi = (sb_b0, sb_b1) if sb_b0 <= sb_b1 else (sb_b1, sb_b0)

            ov_lo = max(sa_lo, sb_lo)
            ov_hi = min(sa_hi, sb_hi)
            ov_px = max(0.0, ov_hi - ov_lo)
            min_proj = max(1e-6, min(sa_hi - sa_lo, sb_hi - sb_lo))
            ov_ratio = ov_px / min_proj
            parallel_overlap_px = round(ov_px, 3)
            parallel_overlap_ratio = round(ov_ratio, 4)

            if ov_px > 1e-6:
                def _pt_at_s(p_start: np.ndarray, p_end: np.ndarray, s0: float, s1: float, s_val: float) -> Tuple[np.ndarray, float]:
                    t = float(np.clip((s_val - s0) / (s1 - s0), 0.0, 1.0)) if abs(s1 - s0) > 1e-9 else 0.5
                    return p_start + t * (p_end - p_start), t

                pa_lo, _ = _pt_at_s(a0, a1, sa_a0, sa_a1, ov_lo)
                pa_hi, _ = _pt_at_s(a0, a1, sa_a0, sa_a1, ov_hi)
                pb_lo, _ = _pt_at_s(b0, b1, sb_b0, sb_b1, ov_lo)
                pb_hi, _ = _pt_at_s(b0, b1, sb_b0, sb_b1, ov_hi)
                perp_lo = abs(float(np.dot(pb_lo - pa_lo, n_par)))
                perp_hi = abs(float(np.dot(pb_hi - pa_hi, n_par)))
                max_perp = max(perp_lo, perp_hi)
                mean_perp = 0.5 * (perp_lo + perp_hi)
                axis_perp_sep_px = round(mean_perp, 3)

                ov_mid = 0.5 * (ov_lo + ov_hi)
                _, t_mid_a = _pt_at_s(a0, a1, sa_a0, sa_a1, ov_mid)
                _, t_mid_b = _pt_at_s(b0, b1, sb_b0, sb_b1, ov_mid)

                exact_axis_eps_px = cfg.exact_collinear_axis_eps_in_w * min(w1, w2)
                near_axis_tol_px = cfg.near_collinear_axis_tol_in_w * min(w1, w2)

                if exact_ang_deg <= cfg.exact_collinear_angle_eps_deg and max_perp <= exact_axis_eps_px:
                    return {
                        "axes_intersect": True,
                        "polygons_intersect": polys_intersect,
                        "angle_between_exact_deg": round(exact_ang_deg, 2),
                        "surface_gap_px": round(exact_poly_gap, 3),
                        "raw_axis_gap_px": round(raw_axis_min, 3),
                        "t_on_a": round(t_mid_a, 4),
                        "t_on_b": round(t_mid_b, 4),
                        "zone_on_a": "collinear_overlap",
                        "zone_on_b": "collinear_overlap",
                        "contact_geometry": "collinear_overlap",
                        "parallel_overlap_px": parallel_overlap_px,
                        "parallel_overlap_ratio": parallel_overlap_ratio,
                        "axis_perp_sep_px": axis_perp_sep_px,
                    }

                if exact_ang_deg <= cfg.near_collinear_angle_tol_deg and max_perp <= near_axis_tol_px:
                    return {
                        "axes_intersect": axes_intersect,
                        "polygons_intersect": polys_intersect,
                        "angle_between_exact_deg": round(exact_ang_deg, 2),
                        "surface_gap_px": round(exact_poly_gap, 3),
                        "raw_axis_gap_px": round(raw_axis_min, 3),
                        "t_on_a": round(t_mid_a, 4),
                        "t_on_b": round(t_mid_b, 4),
                        "zone_on_a": "near_collinear_overlap",
                        "zone_on_b": "near_collinear_overlap",
                        "contact_geometry": "near_collinear_overlap",
                        "parallel_overlap_px": parallel_overlap_px,
                        "parallel_overlap_ratio": parallel_overlap_ratio,
                        "axis_perp_sep_px": axis_perp_sep_px,
                    }

                if (not axes_intersect) and ov_ratio >= cfg.parallel_side_min_overlap_ratio:
                    return {
                        "axes_intersect": False,
                        "polygons_intersect": polys_intersect,
                        "angle_between_exact_deg": round(exact_ang_deg, 2),
                        "surface_gap_px": round(exact_poly_gap, 3),
                        "raw_axis_gap_px": round(raw_axis_min, 3),
                        "t_on_a": round(t_mid_a, 4),
                        "t_on_b": round(t_mid_b, 4),
                        "zone_on_a": "side_overlap",
                        "zone_on_b": "side_overlap",
                        "contact_geometry": "parallel_side_by_side",
                        "parallel_overlap_px": parallel_overlap_px,
                        "parallel_overlap_ratio": parallel_overlap_ratio,
                        "axis_perp_sep_px": axis_perp_sep_px,
                    }
    else:
        exact_ang_deg = _angle_diff_180(float(p1["angle_deg"]), float(p2["angle_deg"]))

    def _zone(t: float, length_px: float, other_w: float) -> str:
        end_margin_px = max(
            cfg.end_zone_other_w_mult * other_w,
            min(cfg.end_zone_len_frac * length_px, cfg.end_zone_max_in_S * S_ref),
        )
        end_frac = min(cfg.end_zone_max_frac, end_margin_px / max(1e-6, length_px))
        if t <= end_frac:
            return "end_A"
        if t >= 1.0 - end_frac:
            return "end_B"
        return "interior"

    zone_a = _zone(best_ta, max(la, float(p1["length_px"])), w2)
    zone_b = _zone(best_tb, max(lb, float(p2["length_px"])), w1)

    if zone_a != "interior" and zone_b != "interior":
        geom = "endpoint_to_endpoint"
    elif zone_a != "interior" and zone_b == "interior":
        geom = "a_endpoint_on_b_interior"
    elif zone_a == "interior" and zone_b != "interior":
        geom = "b_endpoint_on_a_interior"
    else:
        geom = "interior_to_interior_crossing"

    return {
        "axes_intersect": axes_intersect,
        "polygons_intersect": polys_intersect,
        "angle_between_exact_deg": round(exact_ang_deg, 2),
        "surface_gap_px": round(exact_poly_gap, 3),
        "raw_axis_gap_px": round(raw_axis_min, 3),
        "t_on_a": round(best_ta, 4),
        "t_on_b": round(best_tb, 4),
        "zone_on_a": zone_a,
        "zone_on_b": zone_b,
        "contact_geometry": geom,
        "parallel_overlap_px": parallel_overlap_px,
        "parallel_overlap_ratio": parallel_overlap_ratio,
        "axis_perp_sep_px": axis_perp_sep_px,
    }


def _canonical_ribbon_endpoints(
    p: Dict[str, Any], u_vec: np.ndarray, v_vec: np.ndarray
) -> Dict[str, Any]:
    pa = np.array(p["start"], dtype=np.float64)
    pb = np.array(p["end"], dtype=np.float64)
    ua, va = float(np.dot(pa, u_vec)), float(np.dot(pa, v_vec))
    ub, vb = float(np.dot(pb, u_vec)), float(np.dot(pb, v_vec))

    if va <= vb:
        top_xy, bot_xy, top_uv, bot_uv, top_raw, bot_raw = (
            [float(pa[0]), float(pa[1])], [float(pb[0]), float(pb[1])], [ua, va], [ub, vb], "end_A", "end_B"
        )
    else:
        top_xy, bot_xy, top_uv, bot_uv, top_raw, bot_raw = (
            [float(pb[0]), float(pb[1])], [float(pa[0]), float(pa[1])], [ub, vb], [ua, va], "end_B", "end_A"
        )

    if ua <= ub:
        left_xy, right_xy, left_uv, right_uv = [float(pa[0]), float(pa[1])], [float(pb[0]), float(pb[1])], [ua, va], [ub, vb]
    else:
        left_xy, right_xy, left_uv, right_uv = [float(pb[0]), float(pb[1])], [float(pa[0]), float(pa[1])], [ub, vb], [ua, va]

    return {
        "raw_A_xy": [round(float(pa[0]), 2), round(float(pa[1]), 2)],
        "raw_B_xy": [round(float(pb[0]), 2), round(float(pb[1]), 2)],
        "raw_A_uv": [round(ua, 2), round(va, 2)],
        "raw_B_uv": [round(ub, 2), round(vb, 2)],
        "top_xy": [round(top_xy[0], 2), round(top_xy[1], 2)],
        "bot_xy": [round(bot_xy[0], 2), round(bot_xy[1], 2)],
        "top_uv": [round(top_uv[0], 2), round(top_uv[1], 2)],
        "bot_uv": [round(bot_uv[0], 2), round(bot_uv[1], 2)],
        "top_is_raw": top_raw,
        "bot_is_raw": bot_raw,
        "left_xy": [round(left_xy[0], 2), round(left_xy[1], 2)],
        "right_xy": [round(right_xy[0], 2), round(right_xy[1], 2)],
        "left_uv": [round(left_uv[0], 2), round(left_uv[1], 2)],
        "right_uv": [round(right_uv[0], 2), round(right_uv[1], 2)],
    }


def _build_symmetrized_neighbor_graph(
    by_id: Dict[int, Dict[str, Any]]
) -> Tuple[Dict[int, set[int]], int]:
    directed: Dict[int, set[int]] = {pid: set() for pid in by_id}
    sym: Dict[int, set[int]] = {pid: set() for pid in by_id}

    for pid, p in by_id.items():
        for nb_raw in p.get("neighbors", []):
            nb = int(nb_raw)
            directed[pid].add(nb)
            sym[pid].add(nb)
            sym[nb].add(pid)

    asymmetric_count = sum(
        1 for pid, nbs in directed.items() for nb in nbs if pid not in directed[nb]
    )
    return sym, asymmetric_count


# ---------------------------------------------------------------------------
# 0. Validarea STRICTĂ completă (config, geometrie, vecini, referințe, porți)
# ---------------------------------------------------------------------------
def validate_upstream_gates_strict(
    layer1: Dict[str, Any],
    layer2: Dict[str, Any],
    cfg: Layer3AConfig,
) -> Dict[str, Any]:
    validate_config(cfg)

    l1_ver = layer1.get("lossless_verification")
    if not isinstance(l1_ver, dict):
        raise GateViolationError("Stratul 1 nu conține blocul `lossless_verification`.")
    if int(l1_ver.get("layer0_dss_xor_error_pixels", -1)) != 0:
        raise GateViolationError(f"Poarta Strat 0 picată: XOR={l1_ver.get('layer0_dss_xor_error_pixels')}")
    if int(l1_ver.get("layer1_uncovered_ink_pixels", -1)) != 0:
        raise GateViolationError(f"Poarta Strat 1 picată: uncovered={l1_ver.get('layer1_uncovered_ink_pixels')}")
    if l1_ver.get("guaranteed_100pct_coverage") is not True:
        raise GateViolationError("Poarta Strat 1 picată: guaranteed_100pct_coverage != True.")

    raw_prims = layer1.get("primitives")
    if not isinstance(raw_prims, list):
        raise GateViolationError("Stratul 1 nu conține lista `primitives`.")

    l1_id_counts: Counter[int] = Counter()
    prim_by_id: Dict[int, Dict[str, Any]] = {}
    for idx, p in enumerate(raw_prims):
        if not isinstance(p, dict) or "id" not in p or "type" not in p:
            raise GateViolationError(f"Stratul 1: primitiva de la index {idx} nu are `id` sau `type`.")
        pid = int(p["id"])
        ptype = p["type"]
        if ptype not in VALID_LAYER1_TYPES:
            raise GateViolationError(f"Stratul 1: primitiva #{pid} are tip necunoscut `{ptype}`.")
        l1_id_counts[pid] += 1
        prim_by_id[pid] = p

        if not _is_2d_finite(p.get("center")):
            raise GateViolationError(f"Stratul 1: primitiva #{pid} are `center` invalid: {p.get('center')}")
        if not _is_finite_num(p.get("angle_deg")):
            raise GateViolationError(f"Stratul 1: primitiva #{pid} are `angle_deg` invalid: {p.get('angle_deg')}")
        if not _is_finite_num(p.get("area_px")) or float(p["area_px"]) < 0.0:
            raise GateViolationError(f"Stratul 1: primitiva #{pid} are `area_px` invalid: {p.get('area_px')}")
        if not _is_finite_num(p.get("elongation")) or float(p["elongation"]) < 0.0:
            raise GateViolationError(f"Stratul 1: primitiva #{pid} are `elongation` invalid: {p.get('elongation')}")
        if not _is_finite_num(p.get("length_px")) or float(p["length_px"]) < 0.0:
            raise GateViolationError(f"Stratul 1: primitiva #{pid} are `length_px` invalid: {p.get('length_px')}")

        if ptype == "banda_dreapta":
            if not _is_2d_finite(p.get("start")) or not _is_2d_finite(p.get("end")):
                raise GateViolationError(f"Stratul 1: banda_dreapta #{pid} are `start` sau `end` invalid.")
            if not _is_finite_num(p.get("width_px")) or float(p["width_px"]) <= 0.0:
                raise GateViolationError(f"Stratul 1: banda_dreapta #{pid} are `width_px` invalid.")
        else:
            b = p.get("bbox")
            if not isinstance(b, list) or len(b) != 4 or not all(_is_finite_num(x) for x in b):
                raise GateViolationError(f"Stratul 1: primitiva #{pid} are `bbox` invalid.")
            if float(b[2]) < float(b[0]) or float(b[3]) < float(b[1]):
                raise GateViolationError(f"Stratul 1: primitiva #{pid} are `bbox` inversat: {b}")
            if not _is_finite_num(p.get("width_px")) or float(p["width_px"]) < 0.0:
                raise GateViolationError(f"Stratul 1: primitiva #{pid} are `width_px` invalid.")

    l1_dups = sorted(pid for pid, cnt in l1_id_counts.items() if cnt != 1)
    if l1_dups:
        raise GateViolationError(f"Stratul 1 conține ID-uri duplicate în `primitives`: {l1_dups[:15]}")

    for pid, p in prim_by_id.items():
        nbs = p.get("neighbors", [])
        if not isinstance(nbs, list):
            raise GateViolationError(f"Stratul 1: primitiva #{pid} are `neighbors` non-listă.")
        for nb_raw in nbs:
            nb = int(nb_raw)
            if nb == pid:
                raise GateViolationError(f"Stratul 1: primitiva #{pid} conține auto-referință în `neighbors`.")
            if l1_id_counts[nb] != 1:
                raise GateViolationError(
                    f"Stratul 1: primitiva #{pid} referă vecin inexistent #{nb} în `neighbors`."
                )

    l2_ver = layer2.get("accounting_verification")
    if not isinstance(l2_ver, dict):
        raise GateViolationError("Stratul 2 nu conține blocul `accounting_verification`.")
    if int(l2_ver.get("unaccounted_primitives", -1)) != 0:
        raise GateViolationError(f"Poarta Strat 2 picată: unaccounted={l2_ver.get('unaccounted_primitives')}")
    if "duplicate_primitives" in l2_ver:
        dup_val = l2_ver["duplicate_primitives"]
        if (len(dup_val) > 0) if isinstance(dup_val, list) else (int(dup_val) != 0):
            raise GateViolationError(f"Poarta Strat 2 picată: duplicate_primitives={dup_val}")
    if l2_ver.get("guaranteed_100pct_accounting") is not True:
        raise GateViolationError("Poarta Strat 2 picată: guaranteed_100pct_accounting != True.")

    pf = layer2.get("page_frame")
    if not isinstance(pf, dict) or not all(k in pf for k in ("dominant_angle_deg", "u_axis", "v_axis")):
        raise GateViolationError("Stratul 2 nu conține `page_frame` complet.")
    if not _is_finite_num(pf["dominant_angle_deg"]):
        raise GateViolationError(f"`dominant_angle_deg` invalid: {pf['dominant_angle_deg']}")
    if not _is_2d_finite(pf["u_axis"]) or not _is_2d_finite(pf["v_axis"]):
        raise GateViolationError("`page_frame.u_axis` sau `v_axis` nu sunt vectori 2D finiți.")
    u_vec = np.array(pf["u_axis"], dtype=np.float64)
    v_vec = np.array(pf["v_axis"], dtype=np.float64)
    u_norm, v_norm = float(np.linalg.norm(u_vec)), float(np.linalg.norm(v_vec))
    uv_dot = float(np.dot(u_vec, v_vec))
    if abs(u_norm - 1.0) > cfg.axis_norm_tol or abs(v_norm - 1.0) > cfg.axis_norm_tol:
        raise GateViolationError(f"`page_frame` ne-normalizat: |u|={u_norm:.6f}, |v|={v_norm:.6f}")
    if abs(uv_dot) > cfg.axis_ortho_tol:
        raise GateViolationError(f"`page_frame` ne-ortogonal: dot(u, v)={uv_dot:.6f}")

    raw_grids = layer2.get("periodic_grids", [])
    raw_mols = layer2.get("molecules", [])
    if not isinstance(raw_grids, list) or not isinstance(raw_mols, list):
        raise GateViolationError("Stratul 2 nu conține listele `periodic_grids` și `molecules`.")

    l2_grid_counts: Counter[int] = Counter()
    l2_mol_prim_counts: Counter[int] = Counter()
    l2_mol_id_counts: Counter[int] = Counter()
    grid_id_counts: Counter[int] = Counter()

    for g in raw_grids:
        gid = int(g["grid_id"])
        grid_id_counts[gid] += 1
        lc = int(g.get("line_count", 0))
        if lc < 2:
            raise GateViolationError(f"Grila #{gid} are line_count invalid: {lc}")
        step = g.get("step_px")
        if not _is_finite_num(step) or float(step) <= 0.0:
            raise GateViolationError(f"Grila #{gid} are step_px invalid: {step}")
        u_sp = g.get("u_span")
        if not _is_2d_finite(u_sp) or float(u_sp[0]) >= float(u_sp[1]):
            raise GateViolationError(f"Grila #{gid} are u_span invalid: {u_sp}")
        lvs = g.get("line_v_coords")
        if not isinstance(lvs, list) or len(lvs) != lc or not all(_is_finite_num(x) for x in lvs):
            raise GateViolationError(f"Grila #{gid} are line_v_coords invalid (lungime != {lc}).")
        for a, b in zip(lvs, lvs[1:]):
            if float(b) <= float(a):
                raise GateViolationError(f"Grila #{gid} are line_v_coords ne-ordonate strict crescător: {lvs}")
        for pid_raw in g.get("primitive_ids", []):
            pid = int(pid_raw)
            if l1_id_counts[pid] != 1:
                raise GateViolationError(f"Grila #{gid} referă primitive_id inexistent #{pid}.")
            if prim_by_id[pid]["type"] != "banda_dreapta":
                raise GateViolationError(
                    f"Grila #{gid} conține primitiva #{pid} de tip `{prim_by_id[pid]['type']}` (așteptat `banda_dreapta`)."
                )
            l2_grid_counts[pid] += 1

    if any(cnt != 1 for cnt in grid_id_counts.values()):
        raise GateViolationError("Stratul 2 conține `grid_id` duplicate.")
    dup_in_grids = sorted(pid for pid, cnt in l2_grid_counts.items() if cnt != 1)
    if dup_in_grids:
        raise GateViolationError(f"Stratul 2 conține primitive duplicate în `periodic_grids`: {dup_in_grids[:15]}")

    mol_prim_map: Dict[int, List[int]] = {}
    mol_row_id_map: Dict[int, Optional[int]] = {}
    for m in raw_mols:
        mid = int(m["molecule_id"])
        l2_mol_id_counts[mid] += 1
        mpids = [int(pid) for pid in m.get("primitive_ids", [])]
        if not mpids:
            raise GateViolationError(f"Stratul 2: molecula #{mid} are `primitive_ids` vidă.")
        mol_prim_map[mid] = mpids
        for pid in mpids:
            l2_mol_prim_counts[pid] += 1

        mgids_raw = m.get("grid_ids", [])
        if not isinstance(mgids_raw, list):
            raise GateViolationError(f"Molecula #{mid}: `grid_ids` nu este listă.")
        mgid_counts: Counter[int] = Counter(int(x) for x in mgids_raw)
        for g_ref, g_cnt in mgid_counts.items():
            if grid_id_counts[g_ref] != 1:
                raise GateViolationError(f"Molecula #{mid} referă grid_id inexistent #{g_ref}.")
            if g_cnt != 1:
                raise GateViolationError(f"Molecula #{mid} are grid_id duplicat #{g_ref}.")

        rid_val = m.get("row_id")
        mol_row_id_map[mid] = int(rid_val) if rid_val is not None else None

    if any(cnt != 1 for cnt in l2_mol_id_counts.values()):
        raise GateViolationError("Stratul 2 conține `molecule_id` duplicate.")
    dup_in_mols = sorted(pid for pid, cnt in l2_mol_prim_counts.items() if cnt != 1)
    if dup_in_mols:
        raise GateViolationError(f"Stratul 2 conține primitive duplicate în `molecules`: {dup_in_mols[:15]}")

    overlap = sorted(pid for pid in l2_grid_counts if l2_mol_prim_counts[pid] > 0)
    if overlap:
        raise GateViolationError(f"Stratul 2 are primitive simultan în grile și molecule: {overlap[:15]}")

    l2_total = l2_grid_counts + l2_mol_prim_counts
    missing = sorted(pid for pid in l1_id_counts if l2_total[pid] == 0)
    extra = sorted(pid for pid in l2_total if l1_id_counts[pid] == 0)
    multi = sorted(pid for pid, cnt in l2_total.items() if cnt > 1)
    if missing or extra or multi:
        raise GateViolationError(
            f"Bijecția Strat 1 <-> Strat 2 picată: lipsă={len(missing)} ({missing[:8]}), "
            f"extra={len(extra)} ({extra[:8]}), multiple={len(multi)} ({multi[:8]})."
        )

    tbar_prim_counts: Counter[int] = Counter()
    for idx_b, b in enumerate(layer2.get("transverse_grid_bars", [])):
        bpid = int(b["primitive_id"])
        bmid = int(b["molecule_id"])
        tbar_prim_counts[bpid] += 1
        if l1_id_counts[bpid] != 1:
            raise GateViolationError(f"`transverse_grid_bars[{idx_b}]` referă primitive_id inexistent #{bpid}.")
        if prim_by_id[bpid]["type"] != "banda_dreapta":
            raise GateViolationError(f"`transverse_grid_bars[{idx_b}]`: primitiva #{bpid} nu este `banda_dreapta`.")
        if l2_mol_id_counts[bmid] != 1:
            raise GateViolationError(f"`transverse_grid_bars[{idx_b}]` referă molecule_id inexistent #{bmid}.")
        if bpid not in mol_prim_map[bmid]:
            raise GateViolationError(
                f"`transverse_grid_bars[{idx_b}]` inconsecvent: primitiva #{bpid} nu aparține moleculei #{bmid}."
            )
        sgids_raw = b.get("spanned_grid_ids")
        if not isinstance(sgids_raw, list) or len(sgids_raw) == 0:
            raise GateViolationError(f"`transverse_grid_bars[{idx_b}]` are `spanned_grid_ids` vid sau invalid.")
        sg_counts: Counter[int] = Counter(int(x) for x in sgids_raw)
        for sgid, scnt in sg_counts.items():
            if grid_id_counts[sgid] != 1:
                raise GateViolationError(f"`transverse_grid_bars[{idx_b}]` referă grid_id inexistent #{sgid}.")
            if scnt != 1:
                raise GateViolationError(f"`transverse_grid_bars[{idx_b}]` are grid_id duplicat #{sgid}.")

    dup_tbars = sorted(pid for pid, cnt in tbar_prim_counts.items() if cnt != 1)
    if dup_tbars:
        raise GateViolationError(f"`transverse_grid_bars` conține primitive duplicate: {dup_tbars[:10]}")

    row_id_counts: Counter[int] = Counter()
    row_mol_counts: Counter[int] = Counter()
    row_prim_counts: Counter[int] = Counter()
    row_to_mols: Dict[int, set[int]] = {}

    for row in layer2.get("aligned_rows", []):
        rid = int(row["row_id"])
        row_id_counts[rid] += 1
        row_to_mols[rid] = set()
        expected_row_prims: Counter[int] = Counter()
        for mid_raw in row.get("molecule_ids", []):
            mid_i = int(mid_raw)
            if l2_mol_id_counts[mid_i] != 1:
                raise GateViolationError(f"`aligned_rows` #{rid} referă molecule_id inexistent #{mid_i}.")
            row_mol_counts[mid_i] += 1
            row_to_mols[rid].add(mid_i)
            if mol_row_id_map[mid_i] != rid:
                raise GateViolationError(
                    f"Inconsistență bidirecțională rând #{rid}: molecula #{mid_i} are row_id={mol_row_id_map[mid_i]}."
                )
            for p in mol_prim_map[mid_i]:
                expected_row_prims[p] += 1
        actual_row_prims: Counter[int] = Counter()
        for pid_raw in row.get("primitive_ids", []):
            pid_i = int(pid_raw)
            if l1_id_counts[pid_i] != 1:
                raise GateViolationError(f"`aligned_rows` #{rid} referă primitive_id inexistent #{pid_i}.")
            row_prim_counts[pid_i] += 1
            actual_row_prims[pid_i] += 1
        if actual_row_prims != expected_row_prims:
            raise GateViolationError(f"`aligned_rows` #{rid}: `primitive_ids` nu corespunde exact cu `molecule_ids`.")

    if any(cnt != 1 for cnt in row_id_counts.values()):
        raise GateViolationError("Stratul 2 conține `row_id` duplicate în `aligned_rows`.")
    if any(cnt > 1 for cnt in row_mol_counts.values()):
        raise GateViolationError("Stratul 2 are molecule duplicate între rânduri în `aligned_rows`.")
    if any(cnt > 1 for cnt in row_prim_counts.values()):
        raise GateViolationError("Stratul 2 are primitive duplicate între rânduri în `aligned_rows`.")

    for mid_i, rid_val in mol_row_id_map.items():
        if rid_val is not None:
            if row_id_counts[rid_val] != 1:
                raise GateViolationError(
                    f"Inconsistență bidirecțională: molecula #{mid_i} referă row_id={rid_val} inexistent în `aligned_rows`."
                )
            if mid_i not in row_to_mols[rid_val]:
                raise GateViolationError(
                    f"Inconsistență bidirecțională: molecula #{mid_i} declară row_id={rid_val}, dar lipsește din `aligned_rows[{rid_val}]`."
                )

    return {
        "config_valid_ok": True,
        "layer0_xor_ok": True,
        "layer1_coverage_100pct_ok": True,
        "layer1_unique_primitive_ids_ok": True,
        "layer1_geometry_and_neighbors_valid_ok": True,
        "layer2_accounting_100pct_ok": True,
        "layer2_no_duplicate_primitives_ok": True,
        "layer2_no_duplicate_molecules_ok": True,
        "page_frame_orthonormal_ok": True,
        "bidirectional_row_refs_ok": True,
        "secondary_references_ok": True,
        "cross_layer_exact_count_bijection_ok": True,
        "verified_primitive_count": len(l1_id_counts),
        "verified_grid_count": len(grid_id_counts),
        "verified_molecule_count": len(l2_mol_id_counts),
    }


# ---------------------------------------------------------------------------
# 1. Model metric local de grilă CU verificare de monotonie pe întregul domeniu
# ---------------------------------------------------------------------------
class LocalGridModel:
    def __init__(
        self,
        grid_dict: Dict[str, Any],
        by_id: Dict[int, Dict[str, Any]],
        u_vec: np.ndarray,
        v_vec: np.ndarray,
        cfg: Layer3AConfig,
    ) -> None:
        self.grid_id = int(grid_dict["grid_id"])
        self.line_count = int(grid_dict["line_count"])
        self.u_lo = float(grid_dict["u_span"][0])
        self.u_hi = float(grid_dict["u_span"][1])
        self.u_mid = 0.5 * (self.u_lo + self.u_hi)
        self.primitive_ids = sorted(int(x) for x in grid_dict["primitive_ids"])

        base_vs = [float(v) for v in grid_dict["line_v_coords"]]
        line_segments: List[List[Tuple[float, float, float, float, int]]] = [[] for _ in range(self.line_count)]
        line_widths: List[float] = []

        for pid in self.primitive_ids:
            p = by_id[pid]
            pa = np.array(p["start"], dtype=np.float64)
            pb = np.array(p["end"], dtype=np.float64)
            ua, va = float(np.dot(pa, u_vec)), float(np.dot(pa, v_vec))
            ub, vb = float(np.dot(pb, u_vec)), float(np.dot(pb, v_vec))
            if ub < ua:
                ua, ub = ub, ua
                va, vb = vb, va
            k = int(np.argmin([abs(0.5 * (va + vb) - bv) for bv in base_vs]))
            line_segments[k].append((ua, va, ub, vb, pid))
            line_widths.append(float(p["width_px"]))

        self.line_thickness_px = float(np.median(line_widths)) if line_widths else 1.5
        knot_merge_tol = max(1e-6, cfg.knot_merge_in_w * self.line_thickness_px)
        bridge_tol = max(0.0, cfg.interval_bridge_in_w * self.line_thickness_px)

        self.line_models: List[Dict[str, Any]] = []
        slopes: List[float] = []
        span_total = max(1e-6, self.u_hi - self.u_lo)
        critical_u: set[float] = {self.u_lo, self.u_mid, self.u_hi}

        for k in range(self.line_count):
            segs = sorted(line_segments[k], key=lambda s: s[0])

            literal_ivs: List[List[float]] = []
            for ua, _va, ub, _vb, _pid in segs:
                if self.u_lo <= ua <= self.u_hi:
                    critical_u.add(ua)
                if self.u_lo <= ub <= self.u_hi:
                    critical_u.add(ub)
                if literal_ivs and ua <= literal_ivs[-1][1]:
                    literal_ivs[-1][1] = max(literal_ivs[-1][1], ub)
                else:
                    literal_ivs.append([ua, ub])

            bridged_ivs: List[List[float]] = []
            for ua, ub in literal_ivs:
                if bridged_ivs and ua <= bridged_ivs[-1][1] + bridge_tol:
                    bridged_ivs[-1][1] = max(bridged_ivs[-1][1], ub)
                else:
                    bridged_ivs.append([ua, ub])

            clipped_obs_len = 0.0
            for a, b in literal_ivs:
                ca, cb = max(self.u_lo, a), min(self.u_hi, b)
                if cb > ca:
                    clipped_obs_len += (cb - ca)
            cov_ratio = clipped_obs_len / span_total

            pts = sorted([(ua, va) for ua, va, _, _, _ in segs] + [(ub, vb) for _, _, ub, vb, _ in segs], key=lambda t: t[0])
            if len(pts) >= 2:
                us = np.array([t[0] for t in pts], dtype=np.float64)
                vs = np.array([t[1] for t in pts], dtype=np.float64)
                u_ref, v_ref = float(np.mean(us)), float(np.mean(vs))
                du = us - u_ref
                den = float(np.sum(du * du))
                slope = float(np.sum(du * (vs - v_ref)) / den) if den > 1e-6 else 0.0
                slope = max(-cfg.max_line_slope, min(cfg.max_line_slope, slope))
                slopes.append(slope)

                ku, kv = [], []
                bu, bv = [us[0]], [vs[0]]
                for ui, vi in zip(us[1:], vs[1:]):
                    if ui - bu[-1] <= knot_merge_tol:
                        bu.append(ui); bv.append(vi)
                    else:
                        ku.append(float(np.mean(bu))); kv.append(float(np.mean(bv)))
                        bu, bv = [ui], [vi]
                ku.append(float(np.mean(bu))); kv.append(float(np.mean(bv)))
                for ku_val in ku:
                    if self.u_lo <= ku_val <= self.u_hi:
                        critical_u.add(ku_val)

                self.line_models.append({
                    "line_index": k, "u_ref": u_ref, "v_ref": v_ref, "slope": slope,
                    "knots_u": np.asarray(ku, dtype=np.float64),
                    "knots_v": np.asarray(kv, dtype=np.float64),
                    "literal_intervals": literal_ivs, "bridged_intervals": bridged_ivs,
                    "observed_coverage_ratio": cov_ratio,
                    "segment_primitive_ids": [s[4] for s in segs],
                })
            else:
                self.line_models.append({
                    "line_index": k, "u_ref": self.u_mid, "v_ref": base_vs[k], "slope": 0.0,
                    "knots_u": np.array([self.u_lo, self.u_hi], dtype=np.float64),
                    "knots_v": np.array([base_vs[k], base_vs[k]], dtype=np.float64),
                    "literal_intervals": [], "bridged_intervals": [],
                    "observed_coverage_ratio": 0.0, "segment_primitive_ids": [],
                })

        self.global_slope = float(np.median(slopes)) if slopes else 0.0
        mid_vs = [self.line_v_and_support(k, self.u_mid)[0] for k in range(self.line_count)]
        pos_steps = [mid_vs[k + 1] - mid_vs[k] for k in range(self.line_count - 1) if mid_vs[k + 1] > mid_vs[k]]
        self.spacing_px = float(np.median(pos_steps)) if pos_steps else float(grid_dict["step_px"])
        if not math.isfinite(self.spacing_px) or self.spacing_px <= 0.0:
            raise GateViolationError(f"Grila #{self.grid_id}: spacing_px calculat invalid ({self.spacing_px}).")
        self.spacing_std_px = float(np.std(pos_steps)) if len(pos_steps) >= 2 else 0.0
        self.top_v_mid = mid_vs[0]
        self.bot_v_mid = mid_vs[-1]

        self.domain_min_step_px = float("inf")
        raw_crossings: List[float] = []
        for uc in sorted(critical_u):
            vs_c = [self.line_v(k, uc) for k in range(self.line_count)]
            min_s_here = min(vs_c[k + 1] - vs_c[k] for k in range(self.line_count - 1))
            if min_s_here < self.domain_min_step_px:
                self.domain_min_step_px = min_s_here
            if min_s_here <= 0.0:
                raw_crossings.append(round(uc, 2))

        self.crossing_u_samples = sorted(set(raw_crossings))
        self.domain_order_valid = len(self.crossing_u_samples) == 0
        self.geometry_health = "valid" if self.domain_order_valid else "degraded_line_crossing"

    def line_v_and_support(self, k: int, u: float) -> Tuple[float, str, float]:
        m = self.line_models[k]
        ku, kv = m["knots_u"], m["knots_v"]
        lit, brg = m["literal_intervals"], m["bridged_intervals"]

        if len(ku) >= 2:
            if ku[0] <= u <= ku[-1]:
                v_val = float(np.interp(u, ku, kv))
            elif u < ku[0]:
                v_val = float(kv[0] + m["slope"] * (u - ku[0]))
            else:
                v_val = float(kv[-1] + m["slope"] * (u - ku[-1]))
        else:
            v_val = float(m["v_ref"] + m["slope"] * (u - m["u_ref"]))

        if not lit:
            return v_val, "no_direct_support", float(abs(u - self.u_mid))
        if u < lit[0][0]:
            return v_val, "extrapolated_left", float(lit[0][0] - u)
        if u > lit[-1][1]:
            return v_val, "extrapolated_right", float(u - lit[-1][1])

        min_d = float("inf")
        for a, b in lit:
            if a <= u <= b:
                return v_val, "observed", 0.0
            min_d = min(min_d, abs(u - a), abs(u - b))

        for a, b in brg:
            if a <= u <= b:
                return v_val, "bridged_micro_gap", float(min_d)
        return v_val, "interpolated_gap", float(min_d)

    def line_v(self, k: int, u: float) -> float:
        return self.line_v_and_support(k, u)[0]

    def local_profile(self, u: float) -> Dict[str, Any]:
        vs: List[float] = []
        statuses: List[str] = []
        gaps_u: List[float] = []
        for k in range(self.line_count):
            v_val, st, g_u = self.line_v_and_support(k, u)
            vs.append(v_val)
            statuses.append(st)
            gaps_u.append(g_u)

        raw_steps = [vs[k + 1] - vs[k] for k in range(self.line_count - 1)]
        min_step = float(min(raw_steps)) if raw_steps else self.spacing_px
        order_valid = min_step > 0.0
        pos_steps = [st for st in raw_steps if st > 0.0]
        s_loc = float(np.median(pos_steps)) if pos_steps else self.spacing_px

        obs_count = sum(1 for st in statuses if st == "observed")
        unique_st = set(statuses)
        if obs_count == self.line_count:
            grid_support = "fully_observed"
        elif obs_count > 0:
            grid_support = "partially_observed"
        elif unique_st == {"no_direct_support"}:
            grid_support = "no_direct_support"
        elif unique_st == {"extrapolated_left"}:
            grid_support = "extrapolated_left"
        elif unique_st == {"extrapolated_right"}:
            grid_support = "extrapolated_right"
        elif unique_st == {"interpolated_gap"}:
            grid_support = "interpolated_gap"
        elif unique_st == {"bridged_micro_gap"}:
            grid_support = "bridged_micro_gap"
        else:
            grid_support = "mixed_support"

        return {
            "line_vs": vs,
            "line_statuses": statuses,
            "local_spacing_px": s_loc,
            "min_adjacent_step_px": min_step,
            "order_valid": order_valid,
            "observed_line_count": obs_count,
            "support_status": grid_support,
            "max_gap_u_px": float(max(gaps_u)) if gaps_u else 0.0,
            "median_gap_u_px": float(np.median(gaps_u)) if gaps_u else 0.0,
        }

    def v_to_slot_with_support(self, u: float, v: float) -> Dict[str, Any]:
        prof = self.local_profile(u)
        vs = prof["line_vs"]
        s_loc = max(1e-6, float(prof["local_spacing_px"]))
        max_slot = 2.0 * (self.line_count - 1)

        if not prof["order_valid"]:
            slot_f = 2.0 * (v - vs[0]) / s_loc
            v_band = "degraded_crossing"
        elif v <= vs[0]:
            s0 = max(1e-6, vs[1] - vs[0]) if self.line_count >= 2 else s_loc
            slot_f = 2.0 * (v - vs[0]) / s0
            v_band = "above_grid"
        elif v >= vs[-1]:
            sl = max(1e-6, vs[-1] - vs[-2]) if self.line_count >= 2 else s_loc
            slot_f = max_slot + 2.0 * (v - vs[-1]) / sl
            v_band = "below_grid"
        else:
            slot_f = 2.0 * (v - vs[0]) / s_loc
            for k in range(self.line_count - 1):
                if vs[k] <= v <= vs[k + 1]:
                    slot_f = 2.0 * k + 2.0 * (v - vs[k]) / max(1e-6, vs[k + 1] - vs[k])
                    break
            v_band = "inside_grid"

        slot_i = int(round(slot_f))
        return {
            "slot_float": round(slot_f, 4),
            "slot_int": slot_i,
            "slot_error": round(slot_f - slot_i, 4),
            "slot_parity": "on_line" if (slot_i % 2 == 0) else "in_space",
            "v_band": v_band,
            "local_spacing_px": round(s_loc, 3),
            "order_valid": prof["order_valid"],
            "support_status": prof["support_status"],
            "observed_line_count": prof["observed_line_count"],
            "max_gap_u_in_S": round(prof["max_gap_u_px"] / s_loc, 3),
            "median_gap_u_in_S": round(prof["median_gap_u_px"] / s_loc, 3),
            "max_gap_u_px": round(prof["max_gap_u_px"], 2),
            "median_gap_u_px": round(prof["median_gap_u_px"], 2),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_id": self.grid_id,
            "line_count": self.line_count,
            "geometry_health": self.geometry_health,
            "domain_order_valid": self.domain_order_valid,
            "domain_min_step_px": round(self.domain_min_step_px, 3),
            "crossing_u_samples": self.crossing_u_samples,
            "u_span": [round(self.u_lo, 2), round(self.u_hi, 2)],
            "top_v_at_mid": round(self.top_v_mid, 2),
            "bot_v_at_mid": round(self.bot_v_mid, 2),
            "spacing_px": round(self.spacing_px, 3),
            "spacing_std_px": round(self.spacing_std_px, 3),
            "line_thickness_px": round(self.line_thickness_px, 2),
            "global_slope_dv_du": round(self.global_slope, 6),
            "lines": [
                {
                    "line_index": lm["line_index"],
                    "v_at_mid": round(self.line_v(lm["line_index"], self.u_mid), 2),
                    "slope_dv_du": round(lm["slope"], 6),
                    "observed_coverage_ratio_clipped": round(lm["observed_coverage_ratio"], 4),
                    "literal_observed_intervals_u": [[round(a, 2), round(b, 2)] for a, b in lm["literal_intervals"]],
                    "segment_primitive_ids": lm["segment_primitive_ids"],
                }
                for lm in self.line_models
            ],
            "primitive_ids": self.primitive_ids,
        }


# ---------------------------------------------------------------------------
# 2. Bare transversale, Sisteme & Celule (partiție contiguă în toleranță explicită)
# ---------------------------------------------------------------------------
def build_transverse_bars_systems_cells(
    grid_models: Dict[int, LocalGridModel],
    layer2: Dict[str, Any],
    by_id: Dict[int, Dict[str, Any]],
    sym_nb: Dict[int, set[int]],
    uv_boxes: Dict[int, Tuple[float, float, float, float]],
    theta_deg: float,
    cfg: Layer3AConfig,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]], set[int]]:
    grid_ids = sorted(grid_models.keys())
    parent = {gid: gid for gid in grid_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    bars: List[Dict[str, Any]] = []
    bar_pids: set[int] = set()

    prim_to_mol: Dict[int, int] = {}
    mol_by_id: Dict[int, Dict[str, Any]] = {}
    for m in layer2.get("molecules", []):
        mid = int(m["molecule_id"])
        mol_by_id[mid] = m
        for pid in m["primitive_ids"]:
            prim_to_mol[int(pid)] = mid

    for b in layer2.get("transverse_grid_bars", []):
        pid = int(b["primitive_id"])
        spanned = sorted(int(g) for g in b["spanned_grid_ids"])
        p = by_id[pid]
        su0, sv0, su1, sv1 = uv_boxes[pid]
        mid = prim_to_mol[pid]
        bar_pids.add(pid)
        bars.append({
            "bar_id": len(bars) + 1,
            "primitive_id": pid,
            "molecule_id": mid,
            "molecule_primitive_count": len(mol_by_id[mid]["primitive_ids"]),
            "source": "layer2_isolated_bar",
            "status": "confirmed",
            "spanned_grid_ids": spanned,
            "u_pos": round(0.5 * (su0 + su1), 2),
            "v_span": [round(sv0, 2), round(sv1, 2)],
            "length_px": round(float(p["length_px"]), 2),
            "width_px": round(float(p["width_px"]), 2),
            "angle_rel_grid_deg": round(_angle_diff_180(float(p["angle_deg"]), theta_deg), 2),
            "bar_geometry_kind": "single",
        })
        for ga, gb in zip(spanned, spanned[1:]):
            union(ga, gb)

    for m in layer2.get("molecules", []):
        mid = int(m["molecule_id"])
        if m.get("zone") != "pe_grila":
            continue
        for pid_raw in m["primitive_ids"]:
            pid = int(pid_raw)
            if pid in bar_pids:
                continue
            p = by_id[pid]
            if p["type"] != "banda_dreapta":
                continue
            rel_ang = _angle_diff_180(float(p["angle_deg"]), theta_deg)
            if not (cfg.bar_angle_min_deg <= rel_ang <= cfg.bar_angle_max_deg):
                continue
            su0, sv0, su1, sv1 = uv_boxes[pid]
            um = 0.5 * (su0 + su1)
            spanned = []
            for gid in grid_ids:
                gm = grid_models[gid]
                if su1 < gm.u_lo - gm.spacing_px or su0 > gm.u_hi + gm.spacing_px:
                    continue
                gv0, gv1 = gm.line_v(0, um), gm.line_v(gm.line_count - 1, um)
                if (min(sv1, gv1) - max(sv0, gv0)) >= cfg.bar_min_grid_v_overlap_ratio * max(1e-6, gv1 - gv0):
                    spanned.append(gid)
            if not spanned:
                continue
            top_gv0 = min(grid_models[g].line_v(0, um) for g in spanned)
            bot_gv1 = max(grid_models[g].line_v(grid_models[g].line_count - 1, um) for g in spanned)
            s_ref = grid_models[spanned[0]].spacing_px
            if sv0 < top_gv0 - cfg.bar_end_overshoot_in_S * s_ref or sv1 > bot_gv1 + cfg.bar_end_overshoot_in_S * s_ref:
                continue
            has_large_nb = False
            for nb_i in sym_nb[pid]:
                if nb_i not in prim_to_mol:
                    continue
                if by_id[nb_i]["type"] in ("pata_compacta", "banda_curba"):
                    nb_box = uv_boxes[nb_i]
                    if ((nb_box[2] - nb_box[0]) >= cfg.bar_large_nb_w_in_S * s_ref and
                            (nb_box[3] - nb_box[1]) >= cfg.bar_large_nb_h_in_S * s_ref):
                        has_large_nb = True
                        break
            if has_large_nb:
                continue

            bar_pids.add(pid)
            bars.append({
                "bar_id": len(bars) + 1,
                "primitive_id": pid,
                "molecule_id": mid,
                "molecule_primitive_count": len(m["primitive_ids"]),
                "source": "layer3a_attached_spanning_bar",
                "status": "candidate" if len(m["primitive_ids"]) > 1 else "confirmed",
                "spanned_grid_ids": spanned,
                "u_pos": round(um, 2),
                "v_span": [round(sv0, 2), round(sv1, 2)],
                "length_px": round(float(p["length_px"]), 2),
                "width_px": round(float(p["width_px"]), 2),
                "angle_rel_grid_deg": round(rel_ang, 2),
                "bar_geometry_kind": "single",
            })
            for ga, gb in zip(spanned, spanned[1:]):
                union(ga, gb)

    by_grid_bars: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for b in bars:
        for gid in b["spanned_grid_ids"]:
            by_grid_bars[gid].append(b)

    for gid, blist in by_grid_bars.items():
        blist.sort(key=lambda x: x["u_pos"])
        s_ref = grid_models[gid].spacing_px
        med_w = float(np.median([x["width_px"] for x in blist])) if blist else 2.0
        for b in blist:
            if b["width_px"] >= cfg.thick_bar_width_ratio * med_w and b["width_px"] >= cfg.thick_bar_min_w_in_S * s_ref:
                b["bar_geometry_kind"] = "thick_single"
        for a, b in zip(blist, blist[1:]):
            if b["u_pos"] - a["u_pos"] <= cfg.double_bar_max_sep_in_S * s_ref:
                if (b["width_px"] >= cfg.double_bar_thick_ratio * a["width_px"] or
                        a["width_px"] >= cfg.double_bar_thick_ratio * b["width_px"]):
                    a["bar_geometry_kind"] = "double_thin_thick"
                    b["bar_geometry_kind"] = "double_thin_thick"
                else:
                    a["bar_geometry_kind"] = "double_thin"
                    b["bar_geometry_kind"] = "double_thin"

    sys_groups: Dict[int, List[int]] = defaultdict(list)
    for gid in grid_ids:
        sys_groups[find(gid)].append(gid)
    systems = []
    for sid, gids in enumerate(sorted(sys_groups.values(), key=lambda gs: grid_models[gs[0]].top_v_mid), start=1):
        gids.sort(key=lambda g: grid_models[g].top_v_mid)
        systems.append({
            "system_id": sid, "grid_ids": gids, "grid_count": len(gids),
            "u_span": [round(min(grid_models[g].u_lo for g in gids), 2), round(max(grid_models[g].u_hi for g in gids), 2)],
            "v_span": [round(grid_models[gids[0]].top_v_mid, 2), round(grid_models[gids[-1]].bot_v_mid, 2)],
        })

    cells_by_grid: Dict[int, List[Dict[str, Any]]] = {}
    tol_c = cfg.cell_continuity_tol_px
    for gid in grid_ids:
        gm = grid_models[gid]
        s_ref = gm.spacing_px
        blist = sorted(by_grid_bars.get(gid, []), key=lambda x: x["u_pos"])

        seps: List[Dict[str, Any]] = []
        for b in blist:
            if seps and (b["u_pos"] - seps[-1]["u_max"]) <= cfg.double_bar_max_sep_in_S * s_ref:
                seps[-1]["bar_ids"].append(b["bar_id"])
                seps[-1]["u_max"] = max(seps[-1]["u_max"], b["u_pos"])
                seps[-1]["u_mid"] = 0.5 * (seps[-1]["u_min"] + seps[-1]["u_max"])
                seps[-1]["kind"] = b["bar_geometry_kind"]
            else:
                seps.append({
                    "bar_ids": [b["bar_id"]],
                    "u_min": b["u_pos"], "u_max": b["u_pos"], "u_mid": b["u_pos"],
                    "kind": b["bar_geometry_kind"],
                })

        left_edge_cut = gm.u_lo + cfg.left_edge_bar_margin_in_S * s_ref
        cur_u = gm.u_lo
        cur_left = [bid for sp in seps if sp["u_mid"] <= left_edge_cut for bid in sp["bar_ids"]]
        interior_seps = [sp for sp in seps if left_edge_cut < sp["u_mid"] <= gm.u_hi + 0.5 * s_ref]
        cells: List[Dict[str, Any]] = []

        for sp in interior_seps:
            u_end = min(gm.u_hi, sp["u_mid"])
            if u_end <= cur_u + 1e-6:
                if cells:
                    cells[-1]["right_bar_ids"] = sorted(set(cells[-1]["right_bar_ids"] + sp["bar_ids"]))
                    cells[-1]["right_separator_kind"] = sp["kind"]
                continue
            w_s = (u_end - cur_u) / max(1e-6, s_ref)
            ckind = "normal" if w_s >= cfg.min_cell_width_in_S else "narrow_interior"
            cells.append({
                "cell_id": len(cells) + 1, "grid_id": gid,
                "cell_kind": ckind,
                "u_start": round(cur_u, 4), "u_end": round(u_end, 4),
                "width_in_S": round(w_s, 3),
                "left_bar_ids": cur_left, "right_bar_ids": list(sp["bar_ids"]),
                "right_separator_kind": sp["kind"],
            })
            cur_u = u_end
            cur_left = list(sp["bar_ids"])

        if gm.u_hi - cur_u > 1e-6 or not cells:
            w_s = (gm.u_hi - cur_u) / max(1e-6, s_ref)
            ckind = "normal" if (not cells or w_s >= cfg.min_trailing_cell_width_in_S) else "trailing_margin"
            cells.append({
                "cell_id": len(cells) + 1, "grid_id": gid,
                "cell_kind": ckind,
                "u_start": round(cur_u, 4), "u_end": round(gm.u_hi, 4),
                "width_in_S": round(w_s, 3),
                "left_bar_ids": cur_left, "right_bar_ids": [],
                "right_separator_kind": "open_end",
            })

        if not cells:
            raise AuditViolationError(f"Grila #{gid}: zero celule generate.")
        if abs(cells[0]["u_start"] - gm.u_lo) > tol_c:
            raise AuditViolationError(f"Grila #{gid}: gol la începutul celulelor ({cells[0]['u_start']} != {gm.u_lo}).")
        if abs(cells[-1]["u_end"] - gm.u_hi) > tol_c:
            raise AuditViolationError(f"Grila #{gid}: gol la sfârșitul celulelor ({cells[-1]['u_end']} != {gm.u_hi}).")
        for idx_c, c in enumerate(cells):
            if c["u_end"] <= c["u_start"] or c["width_in_S"] <= 0.0:
                raise AuditViolationError(f"Celulă invalidă pe grila #{gid}: {c}")
            if idx_c > 0 and abs(c["u_start"] - cells[idx_c - 1]["u_end"]) > tol_c:
                raise AuditViolationError(
                    f"Discontinuitate/suprapunere între celule pe grila #{gid}: {cells[idx_c-1]} vs {c}"
                )

        cells_by_grid[gid] = cells

    return systems, cells_by_grid, bars, bar_pids


# ---------------------------------------------------------------------------
# 3. Alocare INDIVIDUALĂ a fiecărei primitive
# ---------------------------------------------------------------------------
def _cell_for_u_with_status(cells: Sequence[Dict[str, Any]], u: float) -> Tuple[int, str]:
    eps = 1e-3
    for c in cells:
        if c["u_start"] - eps <= u <= c["u_end"] + eps:
            return int(c["cell_id"]), "inside_cell"
    if u < cells[0]["u_start"]:
        return int(cells[0]["cell_id"]), "outside_grid_left"
    return int(cells[-1]["cell_id"]), "outside_grid_right"


def _assign_primitives_individually(
    layer2: Dict[str, Any],
    grid_models: Dict[int, LocalGridModel],
    by_id: Dict[int, Dict[str, Any]],
    sym_nb: Dict[int, set[int]],
    uv_boxes: Dict[int, Tuple[float, float, float, float]],
    bar_pids: set[int],
    cfg: Layer3AConfig,
) -> Tuple[
    Dict[int, List[int]],
    Dict[int, List[int]],
    List[int],
    Dict[int, int],
    Dict[int, Dict[str, Any]],
]:
    grid_prim_ids: Dict[int, List[int]] = {gid: [] for gid in grid_models}
    corridor_row_prims: Dict[int, List[int]] = defaultdict(list)
    off_grid_prims: List[int] = []
    prim_to_mol: Dict[int, int] = {}
    molecule_summaries: Dict[int, Dict[str, Any]] = {}

    grid_line_to_gid: Dict[int, int] = {}
    for gid, gm in grid_models.items():
        for gpid in gm.primitive_ids:
            grid_line_to_gid[gpid] = gid

    for m in layer2.get("molecules", []):
        mid = int(m["molecule_id"])
        mpids = [int(x) for x in m["primitive_ids"]]
        row_id = m.get("row_id")
        explicit_gids = [int(g) for g in m.get("grid_ids", [])]
        mol_zones: List[str] = []
        mol_grid_set: set[int] = set()

        for pid in mpids:
            prim_to_mol[pid] = mid
            if pid in bar_pids:
                mol_zones.append("transverse_bar")
                continue

            if row_id is not None:
                corridor_row_prims[int(row_id)].append(pid)
                mol_zones.append(f"corridor_row_{int(row_id)}")
                continue

            if not grid_models:
                off_grid_prims.append(pid)
                mol_zones.append("off_grid")
                continue

            u0, v0, u1, v1 = uv_boxes[pid]
            um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)

            touched_gids = sorted({grid_line_to_gid[nb] for nb in sym_nb[pid] if nb in grid_line_to_gid})
            best_gid: Optional[int] = None
            best_score = float("inf")

            if touched_gids:
                for gid in touched_gids:
                    gm = grid_models[gid]
                    g_mid = 0.5 * (gm.line_v(0, um) + gm.line_v(gm.line_count - 1, um))
                    score = abs(vm - g_mid) / max(1e-6, gm.spacing_px)
                    if score < best_score:
                        best_score, best_gid = score, gid
            else:
                for gid, gm in grid_models.items():
                    s = gm.spacing_px
                    if u1 < gm.u_lo - cfg.grid_assign_u_margin_in_S * s or u0 > gm.u_hi + cfg.grid_assign_u_margin_in_S * s:
                        continue
                    gv0 = gm.line_v(0, um)
                    gv1 = gm.line_v(gm.line_count - 1, um)
                    dv = max(gv0 - v1, 0.0, v0 - gv1)
                    ds = dv / max(1e-6, s)
                    limit_s = cfg.grid_assign_v_limit_explicit_in_S if gid in explicit_gids else cfg.grid_assign_v_limit_general_in_S
                    if ds <= limit_s and ds < best_score:
                        best_score, best_gid = ds, gid

            if best_gid is not None:
                grid_prim_ids[best_gid].append(pid)
                mol_grid_set.add(best_gid)
                mol_zones.append(f"grid_{best_gid}")
            else:
                off_grid_prims.append(pid)
                mol_zones.append("off_grid")

        unique_zones = sorted(set(mol_zones))
        molecule_summaries[mid] = {
            "molecule_id": mid,
            "layer2_type": m["type"],
            "primitive_ids": mpids,
            "zones": unique_zones,
            "assigned_grid_ids": sorted(mol_grid_set),
            "spans_multiple_grids": len(mol_grid_set) > 1,
            "contains_transverse_bar": "transverse_bar" in unique_zones,
            "row_id": int(row_id) if row_id is not None else None,
        }

    return grid_prim_ids, dict(corridor_row_prims), off_grid_prims, prim_to_mol, molecule_summaries


# ---------------------------------------------------------------------------
# 4. Evaluatori exacti de contacte & proximități (inclusiv linii de grilă!)
# ---------------------------------------------------------------------------
def _compute_pair_contact(
    pid_a: int,
    pid_b: int,
    pa: Dict[str, Any],
    pb: Dict[str, Any],
    mol_a: Optional[int],
    mol_b: Optional[int],
    grid_a: Optional[int],
    grid_b: Optional[int],
    uv_boxes: Dict[int, Tuple[float, float, float, float]],
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    is_graph_nb: bool,
    S_ref: float,
    cfg: Layer3AConfig,
) -> Optional[Dict[str, Any]]:
    near_touch_px = cfg.near_touch_in_S * max(1e-6, S_ref)
    same_mol = (mol_a is not None) and (mol_a == mol_b)
    same_grd = (grid_a is not None) and (grid_a == grid_b)

    if pa["type"] == "banda_dreapta" and pb["type"] == "banda_dreapta":
        rel = _segments_relation_xy(pa, pb, S_ref, cfg)
        surf_gap = float(rel["surface_gap_px"])
        evidence: List[str] = []
        if is_graph_nb:
            evidence.append("layer1_neighbor_graph")
        if surf_gap <= near_touch_px:
            evidence.append("geometric_near_touch")
        if not evidence:
            return None
        out: Dict[str, Any] = {
            "prim_a": pid_a,
            "prim_b": pid_b,
            "same_molecule": same_mol,
            "same_grid": same_grd,
            "evidence": evidence,
            "distance_precision": "exact_oriented_rectangle_polygon",
            "pair_types": "ribbon_ribbon",
            "angle_between_deg": rel["angle_between_exact_deg"],
            "surface_gap_px": rel["surface_gap_px"],
            "surface_gap_in_S": round(surf_gap / max(1e-6, S_ref), 3),
            "raw_axis_gap_px": rel["raw_axis_gap_px"],
            "axes_intersect": rel["axes_intersect"],
            "polygons_intersect": rel["polygons_intersect"],
            "t_on_a": rel["t_on_a"],
            "t_on_b": rel["t_on_b"],
            "zone_on_a": rel["zone_on_a"],
            "zone_on_b": rel["zone_on_b"],
            "contact_geometry": rel["contact_geometry"],
        }
        if rel["parallel_overlap_px"] is not None:
            out["parallel_overlap_px"] = rel["parallel_overlap_px"]
            out["parallel_overlap_ratio"] = rel["parallel_overlap_ratio"]
        if rel["axis_perp_sep_px"] is not None:
            out["axis_perp_sep_px"] = rel["axis_perp_sep_px"]
        return out

    if (pa["type"] == "banda_dreapta") != (pb["type"] == "banda_dreapta"):
        r_pid, b_pid = (pid_a, pid_b) if pa["type"] == "banda_dreapta" else (pid_b, pid_a)
        pr, pblob = (pa, pb) if pa["type"] == "banda_dreapta" else (pb, pa)
        ep_r = _canonical_ribbon_endpoints(pr, u_vec, v_vec)
        a0 = np.array(pr["start"], dtype=np.float64)
        a1 = np.array(pr["end"], dtype=np.float64)
        bx0, by0, bx1, by1 = [float(v) for v in pblob["bbox"]]

        poly_r = _ribbon_oriented_polygon_xy(pr)
        poly_b = _aabb_polygon_xy(pblob["bbox"])
        surf_gap, polys_intersect = _convex_polygons_exact_distance(poly_r, poly_b)

        evidence = []
        if is_graph_nb:
            evidence.append("layer1_neighbor_graph")
        if surf_gap <= near_touch_px:
            evidence.append("geometric_near_touch")
        if not evidence:
            return None

        pt_top = np.array(ep_r["top_xy"], dtype=np.float64)
        pt_bot = np.array(ep_r["bot_xy"], dtype=np.float64)
        pt_cblob = np.array(pblob["center"], dtype=np.float64)
        d_top = _point_to_aabb(pt_top, bx0, by0, bx1, by1)
        d_bot = _point_to_aabb(pt_bot, bx0, by0, bx1, by1)
        _, t_on_r = _point_to_segment(pt_cblob, a0, a1)

        end_margin = max(cfg.blob_end_margin_in_S * S_ref, cfg.blob_end_len_frac * float(pr["length_px"]))
        if d_top <= d_bot and d_top <= end_margin:
            attach_zone = "top_end"
        elif d_bot < d_top and d_bot <= end_margin:
            attach_zone = "bot_end"
        else:
            attach_zone = "interior"

        r_u_mid = 0.5 * (ep_r["top_uv"][0] + ep_r["bot_uv"][0])
        b_uc, _ = _prim_uv_center(pblob, u_vec, v_vec)
        dz = cfg.blob_side_deadzone_in_S * S_ref
        side_u = "left" if b_uc < r_u_mid - dz else ("right" if b_uc > r_u_mid + dz else "center")

        return {
            "prim_a": pid_a,
            "prim_b": pid_b,
            "ribbon_pid": r_pid,
            "blob_pid": b_pid,
            "same_molecule": same_mol,
            "same_grid": same_grd,
            "evidence": evidence,
            "distance_precision": "exact_oriented_rect_to_bbox_approx_contour",
            "pair_types": f"ribbon_{pblob['type']}",
            "surface_gap_px": round(surf_gap, 3),
            "surface_gap_in_S": round(surf_gap / max(1e-6, S_ref), 3),
            "polygons_intersect": polys_intersect,
            "dist_to_top_end_in_S": round(d_top / max(1e-6, S_ref), 3),
            "dist_to_bot_end_in_S": round(d_bot / max(1e-6, S_ref), 3),
            "t_of_blob_center_on_ribbon": round(float(t_on_r), 4),
            "attach_zone_on_ribbon": attach_zone,
            "blob_side_relative_to_ribbon_u": side_u,
        }

    ax0, ay0, ax1, ay1 = [float(v) for v in pa["bbox"]]
    bx0, by0, bx1, by1 = [float(v) for v in pb["bbox"]]
    dx = max(bx0 - ax1, 0.0, ax0 - bx1)
    dy = max(by0 - ay1, 0.0, ay0 - by1)
    box_gap = math.hypot(dx, dy)
    evidence = []
    if is_graph_nb:
        evidence.append("layer1_neighbor_graph")
    if box_gap <= near_touch_px:
        evidence.append("geometric_near_touch")
    if not evidence:
        return None
    return {
        "prim_a": pid_a,
        "prim_b": pid_b,
        "same_molecule": same_mol,
        "same_grid": same_grd,
        "evidence": evidence,
        "distance_precision": "exact_bbox_to_bbox_approx_contour",
        "pair_types": f"{pa['type']}_{pb['type']}",
        "surface_gap_px": round(box_gap, 3),
        "surface_gap_in_S": round(box_gap / max(1e-6, S_ref), 3),
        "polygons_intersect": box_gap == 0.0,
    }


def _compute_grid_line_contact(
    prim_pid: int,
    grid_id: int,
    grid_line_pid: int,
    line_index: int,
    p_prim: Dict[str, Any],
    p_line: Dict[str, Any],
    is_graph_nb: bool,
    S_ref: float,
    cfg: Layer3AConfig,
) -> Optional[Dict[str, Any]]:
    near_touch_px = cfg.near_touch_in_S * max(1e-6, S_ref)

    if p_prim["type"] == "banda_dreapta":
        rel = _segments_relation_xy(p_prim, p_line, S_ref, cfg)
        surf_gap = float(rel["surface_gap_px"])
        evidence: List[str] = []
        if is_graph_nb:
            evidence.append("layer1_neighbor_graph")
        if surf_gap <= near_touch_px:
            evidence.append("geometric_near_touch")
        if not evidence:
            return None
        out: Dict[str, Any] = {
            "primitive_id": prim_pid,
            "grid_id": grid_id,
            "grid_line_primitive_id": grid_line_pid,
            "grid_line_index": line_index,
            "evidence": evidence,
            "distance_precision": "exact_oriented_rectangle_polygon",
            "pair_types": "ribbon_grid_line",
            "angle_between_deg": rel["angle_between_exact_deg"],
            "surface_gap_px": rel["surface_gap_px"],
            "surface_gap_in_S": round(surf_gap / max(1e-6, S_ref), 3),
            "axes_intersect": rel["axes_intersect"],
            "polygons_intersect": rel["polygons_intersect"],
            "t_on_primitive": rel["t_on_a"],
            "t_on_grid_line": rel["t_on_b"],
            "zone_on_primitive": rel["zone_on_a"],
            "contact_geometry": rel["contact_geometry"],
        }
        if rel["parallel_overlap_px"] is not None:
            out["parallel_overlap_px"] = rel["parallel_overlap_px"]
            out["parallel_overlap_ratio"] = rel["parallel_overlap_ratio"]
        if rel["axis_perp_sep_px"] is not None:
            out["axis_perp_sep_px"] = rel["axis_perp_sep_px"]
        return out

    poly_l = _ribbon_oriented_polygon_xy(p_line)
    poly_p = _aabb_polygon_xy(p_prim["bbox"])
    surf_gap, polys_intersect = _convex_polygons_exact_distance(poly_l, poly_p)
    evidence = []
    if is_graph_nb:
        evidence.append("layer1_neighbor_graph")
    if surf_gap <= near_touch_px:
        evidence.append("geometric_near_touch")
    if not evidence:
        return None
    a0, a1 = np.array(p_line["start"], dtype=np.float64), np.array(p_line["end"], dtype=np.float64)
    _, t_on_line = _point_to_segment(np.array(p_prim["center"], dtype=np.float64), a0, a1)
    return {
        "primitive_id": prim_pid,
        "grid_id": grid_id,
        "grid_line_primitive_id": grid_line_pid,
        "grid_line_index": line_index,
        "evidence": evidence,
        "distance_precision": "exact_oriented_rect_to_bbox_approx_contour",
        "pair_types": f"{p_prim['type']}_grid_line",
        "surface_gap_px": round(surf_gap, 3),
        "surface_gap_in_S": round(surf_gap / max(1e-6, S_ref), 3),
        "polygons_intersect": polys_intersect,
        "t_of_primitive_center_on_grid_line": round(float(t_on_line), 4),
    }


def _compute_grid_line_proximity(
    prim_pid: int,
    grid_id: int,
    grid_line_pid: int,
    line_index: int,
    p_prim: Dict[str, Any],
    p_line: Dict[str, Any],
    box_p: Tuple[float, float, float, float],
    box_l: Tuple[float, float, float, float],
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    S_ref: float,
) -> Dict[str, Any]:
    gu, gv, box_gap = _uv_box_gap(box_p, box_l)
    uc_p, vc_p = _prim_uv_center(p_prim, u_vec, v_vec)
    uc_l, vc_l = _prim_uv_center(p_line, u_vec, v_vec)
    du_s = (uc_p - uc_l) / max(1e-6, S_ref)
    dv_s = (vc_p - vc_l) / max(1e-6, S_ref)
    rel_side = "above" if vc_p < vc_l else ("below" if vc_p > vc_l else "aligned")
    return {
        "primitive_id": prim_pid,
        "grid_id": grid_id,
        "grid_line_primitive_id": grid_line_pid,
        "grid_line_index": line_index,
        "layer1_type": p_prim["type"],
        "du_prim_minus_line_in_S": round(du_s, 3),
        "dv_prim_minus_line_in_S": round(dv_s, 3),
        "box_gap_u_in_S": round(gu / max(1e-6, S_ref), 3),
        "box_gap_v_in_S": round(gv / max(1e-6, S_ref), 3),
        "box_gap_in_S": round(box_gap / max(1e-6, S_ref), 3),
        "primitive_relative_to_line_v": rel_side,
    }


def _describe_grid_primitives_and_graphs(
    gm: LocalGridModel,
    cells: List[Dict[str, Any]],
    assigned_pids: List[int],
    grid_bars: List[Dict[str, Any]],
    by_id: Dict[int, Dict[str, Any]],
    sym_nb: Dict[int, set[int]],
    uv_boxes: Dict[int, Tuple[float, float, float, float]],
    prim_to_mol: Dict[int, int],
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    theta_deg: float,
    cfg: Layer3AConfig,
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    gid = gm.grid_id
    S = gm.spacing_px

    non_bar_pids = sorted(assigned_pids, key=lambda pid: (uv_boxes[pid][0], uv_boxes[pid][1], pid))
    non_bar_set = set(non_bar_pids)

    grid_line_index_of: Dict[int, int] = {}
    for lm in gm.line_models:
        for gpid in lm["segment_primitive_ids"]:
            grid_line_index_of[gpid] = lm["line_index"]
    grid_line_set = set(grid_line_index_of.keys())
    bar_pid_to_id = {b["primitive_id"]: b["bar_id"] for b in grid_bars}

    prim_records: List[Dict[str, Any]] = []
    rec_by_pid: Dict[int, Dict[str, Any]] = {}

    for pid in non_bar_pids:
        p = by_id[pid]
        u0, v0, u1, v1 = uv_boxes[pid]
        uc, vc = _prim_uv_center(p, u_vec, v_vec)
        slot_info = gm.v_to_slot_with_support(uc, vc)
        cell_id, u_containment = _cell_for_u_with_status(cells, uc)
        s_loc = max(1e-6, float(slot_info["local_spacing_px"]))
        w_u, h_v = u1 - u0, v1 - v0
        fill_ratio = float(p["area_px"]) / max(1e-6, w_u * h_v)

        rel_ang = _angle_diff_180(float(p["angle_deg"]), theta_deg)
        if p["type"] == "banda_dreapta":
            if rel_ang <= cfg.parallel_angle_tol_deg:
                orient_class = "parallel_to_grid"
            elif cfg.perp_angle_min_deg <= rel_ang <= cfg.perp_angle_max_deg:
                orient_class = "perpendicular_to_grid"
            else:
                orient_class = "oblique"
            ep = _canonical_ribbon_endpoints(p, u_vec, v_vec)
            top_slot = gm.v_to_slot_with_support(ep["top_uv"][0], ep["top_uv"][1])
            bot_slot = gm.v_to_slot_with_support(ep["bot_uv"][0], ep["bot_uv"][1])
            endpoints_block: Optional[Dict[str, Any]] = {
                **ep,
                "top_slot": top_slot["slot_float"],
                "bot_slot": bot_slot["slot_float"],
                "top_support": top_slot["support_status"],
                "bot_support": bot_slot["support_status"],
            }
        else:
            orient_class = "non_linear"
            endpoints_block = None

        rec = {
            "primitive_id": pid,
            "molecule_id": prim_to_mol[pid],
            "grid_id": gid,
            "cell_id": cell_id,
            "cell_u_containment": u_containment,
            "layer1_type": p["type"],
            "orientation_class": orient_class,
            "angle_rel_grid_deg": round(rel_ang, 2),
            "center_xy": p["center"],
            "center_uv": [round(uc, 2), round(vc, 2)],
            "xy_bbox": [round(x, 2) for x in _prim_xy_box(p)],
            "uv_bbox": [round(u0, 2), round(v0, 2), round(u1, 2), round(v1, 2)],
            "slot": slot_info,
            "metrics_in_S": {
                "length_in_S": round(float(p["length_px"]) / s_loc, 3),
                "width_in_S": round(float(p["width_px"]) / s_loc, 3),
                "w_u_in_S": round(w_u / s_loc, 3),
                "h_v_in_S": round(h_v / s_loc, 3),
                "area_in_S2": round(float(p["area_px"]) / (s_loc * s_loc), 4),
                "elongation": p["elongation"],
                "fill_ratio": round(fill_ratio, 3),
                "has_hole": bool(p.get("has_hole", False)),
            },
            "endpoints": endpoints_block,
            "touches_grid_line_ids": sorted(nb for nb in sym_nb[pid] if nb in grid_line_set),
            "touches_bar_ids": sorted(bar_pid_to_id[nb] for nb in sym_nb[pid] if nb in bar_pid_to_id),
        }
        prim_records.append(rec)
        rec_by_pid[pid] = rec

    direct_contacts: List[Dict[str, Any]] = []
    spatial_proximities: List[Dict[str, Any]] = []
    n = len(non_bar_pids)
    max_prox_px = cfg.proximity_radius_in_S * max(1e-6, S)

    candidate_pairs: set[Tuple[int, int]] = set()
    for i in range(n):
        pid_i = non_bar_pids[i]
        box_i = uv_boxes[pid_i]
        for nb_pid in sym_nb[pid_i]:
            if nb_pid in non_bar_set:
                lo, hi = (pid_i, nb_pid) if pid_i < nb_pid else (nb_pid, pid_i)
                candidate_pairs.add((lo, hi))
        for j in range(i + 1, n):
            pid_j = non_bar_pids[j]
            box_j = uv_boxes[pid_j]
            if box_j[0] - box_i[2] > max_prox_px:
                break
            if _uv_box_gap(box_i, box_j)[2] <= max_prox_px:
                lo, hi = (pid_i, pid_j) if pid_i < pid_j else (pid_j, pid_i)
                candidate_pairs.add((lo, hi))

    for pid_a, pid_b in sorted(candidate_pairs):
        pa, pb = by_id[pid_a], by_id[pid_b]
        ra, rb = rec_by_pid[pid_a], rec_by_pid[pid_b]
        is_graph_nb = pid_b in sym_nb[pid_a]

        contact = _compute_pair_contact(
            pid_a, pid_b, pa, pb, ra["molecule_id"], rb["molecule_id"],
            gid, gid, uv_boxes, u_vec, v_vec, is_graph_nb, S, cfg
        )
        if contact is not None:
            direct_contacts.append(contact)
            continue

        gu, gv, box_gap = _uv_box_gap(uv_boxes[pid_a], uv_boxes[pid_b])
        if box_gap <= max_prox_px:
            uc_a, vc_a = ra["center_uv"]
            uc_b, vc_b = rb["center_uv"]
            slot_a = float(ra["slot"]["slot_float"])
            slot_b = float(rb["slot"]["slot_float"])
            du_s = (uc_b - uc_a) / max(1e-6, S)
            dv_s = (vc_b - vc_a) / max(1e-6, S)
            quadrant = ("right" if du_s >= 0 else "left") if abs(du_s) >= abs(dv_s) else ("below" if dv_s >= 0 else "above")
            spatial_proximities.append({
                "prim_a": pid_a,
                "prim_b": pid_b,
                "same_molecule": ra["molecule_id"] == rb["molecule_id"],
                "type_a": pa["type"],
                "type_b": pb["type"],
                "same_cell": ra["cell_id"] == rb["cell_id"],
                "du_a_to_b_in_S": round(du_s, 3),
                "dv_a_to_b_in_S": round(dv_s, 3),
                "dslot_a_to_b": round(slot_b - slot_a, 3),
                "box_gap_u_in_S": round(gu / max(1e-6, S), 3),
                "box_gap_v_in_S": round(gv / max(1e-6, S), 3),
                "box_gap_in_S": round(box_gap / max(1e-6, S), 3),
                "b_relative_to_a": quadrant,
            })

    # Contacte ȘI proximități intra-grilă între primitive și liniile grilei (până la `proximity_radius_in_S`)
    grid_line_contacts: List[Dict[str, Any]] = []
    grid_line_proximities: List[Dict[str, Any]] = []
    sorted_grid_lines = sorted(grid_line_set)
    for pid in non_bar_pids:
        box_p = uv_boxes[pid]
        for gpid in sorted_grid_lines:
            is_nb = gpid in sym_nb[pid]
            box_l = uv_boxes[gpid]
            _, _, box_gap = _uv_box_gap(box_p, box_l)
            if not is_nb and box_gap > max_prox_px:
                continue
            gl_c = _compute_grid_line_contact(
                pid, gid, gpid, grid_line_index_of[gpid], by_id[pid], by_id[gpid], is_nb, S, cfg
            )
            if gl_c is not None:
                grid_line_contacts.append(gl_c)
            elif box_gap <= max_prox_px:
                grid_line_proximities.append(
                    _compute_grid_line_proximity(
                        pid, gid, gpid, grid_line_index_of[gpid], by_id[pid], by_id[gpid],
                        box_p, box_l, u_vec, v_vec, S
                    )
                )

    bar_proximities: List[Dict[str, Any]] = []
    for b in grid_bars:
        bu = float(b["u_pos"])
        bv0, bv1 = float(b["v_span"][0]), float(b["v_span"][1])
        for rec in prim_records:
            uc, _vc = rec["center_uv"]
            du_s = (uc - bu) / max(1e-6, S)
            if abs(du_s) > cfg.bar_proximity_u_in_S:
                continue
            pv0, pv1 = float(rec["uv_bbox"][1]), float(rec["uv_bbox"][3])
            dv_gap_s = max(bv0 - pv1, 0.0, pv0 - bv1) / max(1e-6, S)
            if dv_gap_s > cfg.bar_proximity_v_in_S:
                continue
            bar_proximities.append({
                "bar_id": b["bar_id"],
                "bar_primitive_id": b["primitive_id"],
                "primitive_id": rec["primitive_id"],
                "layer1_type": rec["layer1_type"],
                "du_from_bar_in_S": round(du_s, 3),
                "dv_gap_from_bar_in_S": round(dv_gap_s, 3),
                "slot_float": rec["slot"]["slot_float"],
            })

    touched_mols = sorted({prim_to_mol[pid] for pid in non_bar_pids})
    grid_out = {
        "grid_id": gid,
        "metrics": gm.to_dict(),
        "cells": cells,
        "bar_ids": [b["bar_id"] for b in grid_bars],
        "molecule_ids_present": touched_mols,
        "counts": {
            "cells": len(cells),
            "bars": len(grid_bars),
            "molecules_present": len(touched_mols),
            "non_bar_primitives": len(prim_records),
            "direct_contacts": len(direct_contacts),
            "grid_line_contacts": len(grid_line_contacts),
            "grid_line_proximities": len(grid_line_proximities),
            "spatial_proximities": len(spatial_proximities),
            "bar_proximities": len(bar_proximities),
        },
        "grid_primitives": prim_records,
        "direct_contacts": direct_contacts,
        "grid_line_contacts": grid_line_contacts,
        "grid_line_proximities": grid_line_proximities,
        "spatial_proximities": spatial_proximities,
        "bar_proximities": bar_proximities,
    }
    return grid_out, rec_by_pid


# ---------------------------------------------------------------------------
# 5. Orchestrator Strat 3A v5.1 + Graf global EXHAUSTIV + Audit STRICT count == 1
# ---------------------------------------------------------------------------

# --- Compatibilitate geometrică pentru orchestratorul actualizat ---
def _prim_polygon_xy(p: Dict[str, Any]) -> np.ndarray:
    if p["type"] == "banda_dreapta":
        return _ribbon_oriented_polygon_xy(p)
    return _aabb_polygon_xy(p["bbox"])


def _poly_to_tuples(poly: Any) -> np.ndarray:
    return np.asarray(poly, dtype=np.float64).reshape((-1, 2))


def _orthonormalize_frame(
    u_raw: np.ndarray, v_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    u_raw = np.asarray(u_raw, dtype=np.float64).reshape(2)
    v_raw = np.asarray(v_raw, dtype=np.float64).reshape(2)
    raw_u_norm = float(np.linalg.norm(u_raw))
    raw_v_norm = float(np.linalg.norm(v_raw))
    raw_dot = float(np.dot(u_raw, v_raw))
    if (not np.all(np.isfinite(u_raw)) or not np.all(np.isfinite(v_raw))
            or raw_u_norm <= 1e-12):
        raise GateViolationError("page_frame: axa U este invalidă sau degenerată.")
    u_vec = u_raw / raw_u_norm
    v_orth = v_raw - float(np.dot(v_raw, u_vec)) * u_vec
    v_norm = float(np.linalg.norm(v_orth))
    if v_norm <= 1e-12 or not math.isfinite(v_norm):
        raise GateViolationError("page_frame: axa V este degenerată după ortogonalizare.")
    v_vec = v_orth / v_norm
    if float(np.dot(v_vec, v_raw)) < 0.0:
        v_vec = -v_vec
    meta = {
        "u_axis_raw": [float(x) for x in u_raw],
        "v_axis_raw": [float(x) for x in v_raw],
        "raw_u_norm": raw_u_norm,
        "raw_v_norm": raw_v_norm,
        "raw_dot_uv": raw_dot,
        "u_axis_orthonormal": [float(x) for x in u_vec],
        "v_axis_orthonormal": [float(x) for x in v_vec],
        "orthonormal_dot_uv": float(np.dot(u_vec, v_vec)),
    }
    return u_vec, v_vec, meta


def _pair_distance_precision(pa: Dict[str, Any], pb: Dict[str, Any]) -> str:
    a_ribbon = pa.get("type") == "banda_dreapta"
    b_ribbon = pb.get("type") == "banda_dreapta"
    if a_ribbon and b_ribbon:
        return "exact_oriented_rectangle_polygon"
    if a_ribbon or b_ribbon:
        return "exact_oriented_rect_to_bbox_approx_contour"
    return "exact_bbox_to_bbox_approx_contour"


def analyze_layer3a(
    layer1: Dict[str, Any],
    layer2: Dict[str, Any],
    config: Optional[Layer3AConfig] = None,
) -> Dict[str, Any]:
    cfg = config if config is not None else Layer3AConfig()
    gate_report = validate_upstream_gates_strict(layer1, layer2, cfg)

    by_id: Dict[int, Dict[str, Any]] = {int(p["id"]): p for p in layer1["primitives"]}
    poly_xy: Dict[int, np.ndarray] = {
        pid: _poly_to_tuples(_prim_polygon_xy(p)) for pid, p in by_id.items()
    }
    sym_nb, asym_edge_count = _build_symmetrized_neighbor_graph(by_id)
    expected_pids = sorted(by_id.keys())
    expected_mids = sorted(int(m["molecule_id"]) for m in layer2.get("molecules", []))

    pf = layer2["page_frame"]
    u_raw = np.array(pf["u_axis"], dtype=np.float64)
    v_raw = np.array(pf["v_axis"], dtype=np.float64)
    u_vec, v_vec, frame_ortho_meta = _orthonormalize_frame(u_raw, v_raw)
    theta_deg = math.degrees(math.atan2(float(u_vec[1]), float(u_vec[0]))) % 180.0
    uv_boxes = {pid: _prim_uv_box(p, u_vec, v_vec) for pid, p in by_id.items()}

    prim_assign_count: Counter[int] = Counter()
    prim_roles: Dict[int, str] = {}

    def record_prim(pid: int, role: str) -> None:
        prim_assign_count[pid] += 1
        if prim_assign_count[pid] > 1:
            raise AuditViolationError(
                f"Alocare multiplă pe primitiva #{pid}: avea '{prim_roles.get(pid)}', încercare nouă '{role}'."
            )
        prim_roles[pid] = role

    grid_models: Dict[int, LocalGridModel] = {}
    all_grid_line_meta: Dict[int, Tuple[int, int]] = {}
    for g in layer2.get("periodic_grids", []):
        gm = LocalGridModel(g, by_id, u_vec, v_vec, cfg)
        grid_models[gm.grid_id] = gm
        for pid in gm.primitive_ids:
            record_prim(pid, f"grid_{gm.grid_id}_line")
        for lm in gm.line_models:
            for gpid in lm["segment_primitive_ids"]:
                all_grid_line_meta[gpid] = (gm.grid_id, lm["line_index"])

    systems, cells_by_grid, bars, bar_pids = build_transverse_bars_systems_cells(
        grid_models, layer2, by_id, sym_nb, uv_boxes, theta_deg, cfg
    )
    for b in bars:
        record_prim(b["primitive_id"], f"transverse_bar_{b['bar_id']}")

    grid_prim_ids, corridor_row_prims, off_grid_prims, prim_to_mol, mol_summaries = _assign_primitives_individually(
        layer2, grid_models, by_id, sym_nb, uv_boxes, bar_pids, cfg
    )

    bars_by_grid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for b in bars:
        for gid in b["spanned_grid_ids"]:
            bars_by_grid[gid].append(b)

    grids_out: List[Dict[str, Any]] = []
    all_grid_prim_recs: Dict[int, Dict[str, Any]] = {}
    for gid in sorted(grid_models.keys()):
        g_out, g_rec_map = _describe_grid_primitives_and_graphs(
            grid_models[gid],
            cells_by_grid[gid],
            grid_prim_ids[gid],
            bars_by_grid[gid],
            by_id,
            sym_nb,
            uv_boxes,
            prim_to_mol,
            u_vec,
            v_vec,
            theta_deg,
            cfg,
        )
        for pid in grid_prim_ids[gid]:
            record_prim(pid, f"grid_{gid}_primitive")
        all_grid_prim_recs.update(g_rec_map)
        grids_out.append(g_out)

    corridor_rows_out: List[Dict[str, Any]] = []
    for r in layer2.get("aligned_rows", []):
        rid = int(r["row_id"])
        corridor_rows_out.append({
            "row_id": rid,
            "molecule_ids": [int(x) for x in r["molecule_ids"]],
            "primitive_ids": [int(x) for x in r["primitive_ids"]],
            "median_height_px": r["median_height_px"],
            "uv_bbox": r["uv_bbox"],
            "xy_bbox": r["xy_bbox"],
        })
    for rid, pids in sorted(corridor_row_prims.items()):
        for pid in pids:
            record_prim(pid, f"corridor_row_{rid}_primitive")

    off_grid_primitives_out: List[Dict[str, Any]] = []
    for pid in sorted(off_grid_prims):
        record_prim(pid, "off_grid_primitive")
        p = by_id[pid]
        off_grid_primitives_out.append({
            "primitive_id": pid,
            "molecule_id": prim_to_mol[pid],
            "layer1_type": p["type"],
            "uv_bbox": [round(x, 2) for x in uv_boxes[pid]],
            "xy_bbox": [round(x, 2) for x in _prim_xy_box(p)],
        })

    missing_prims = [pid for pid in expected_pids if prim_assign_count[pid] == 0]
    multi_prims = [pid for pid, cnt in prim_assign_count.items() if cnt != 1]
    extra_prims = [pid for pid in prim_assign_count if pid not in by_id]
    if missing_prims or multi_prims or extra_prims:
        raise AuditViolationError(
            f"Audit primitive Strat 3A picat: lipsă={len(missing_prims)}, multi={len(multi_prims)}, extra={len(extra_prims)}."
        )

    mol_audit_counts: Counter[int] = Counter(mol_summaries.keys())
    expected_mids_set = set(expected_mids)
    missing_mols = [mid for mid in expected_mids if mol_audit_counts[mid] == 0]
    multi_mols = [mid for mid, cnt in mol_audit_counts.items() if cnt != 1]
    extra_mols = [mid for mid in mol_audit_counts if mid not in expected_mids_set]
    if missing_mols or multi_mols or extra_mols:
        raise AuditViolationError(
            f"Audit molecule Strat 3A picat: lipsă={len(missing_mols)}, multi={len(multi_mols)}, extra={len(extra_mols)}."
        )

    if grid_models:
        page_s_ref = float(np.median([gm.spacing_px for gm in grid_models.values()]))
        max_s_page = max(gm.spacing_px for gm in grid_models.values())
    elif _is_finite_num(pf.get("grid_step_px")) and float(pf["grid_step_px"]) > 0.0:
        page_s_ref = float(pf["grid_step_px"]) / max(1e-12, float(np.linalg.norm(v_raw)))
        max_s_page = page_s_ref
    elif by_id:
        sizes = [max(float(p["length_px"]), float(p["width_px"])) for p in by_id.values() if max(float(p["length_px"]), float(p["width_px"])) > 0.0]
        page_s_ref = float(np.median(sizes)) if sizes else 10.0
        max_s_page = page_s_ref
    else:
        page_s_ref = 10.0
        max_s_page = 10.0

    max_global_prox_px = cfg.proximity_radius_in_S * max(1e-6, max_s_page)

    mol_pids_sorted = sorted(prim_to_mol.keys(), key=lambda pid: (uv_boxes[pid][0], uv_boxes[pid][1], pid))
    global_candidates: set[Tuple[int, int]] = set()
    n_mol_p = len(mol_pids_sorted)

    for i in range(n_mol_p):
        pid_i = mol_pids_sorted[i]
        box_i = uv_boxes[pid_i]
        gid_i = all_grid_prim_recs[pid_i]["grid_id"] if pid_i in all_grid_prim_recs else None

        for nb_pid in sym_nb[pid_i]:
            if nb_pid not in prim_to_mol:
                continue
            gid_nb = all_grid_prim_recs[nb_pid]["grid_id"] if nb_pid in all_grid_prim_recs else None
            if gid_i is not None and gid_i == gid_nb:
                continue
            lo, hi = (pid_i, nb_pid) if pid_i < nb_pid else (nb_pid, pid_i)
            global_candidates.add((lo, hi))

        for j in range(i + 1, n_mol_p):
            pid_j = mol_pids_sorted[j]
            box_j = uv_boxes[pid_j]
            if box_j[0] - box_i[2] > max_global_prox_px + 1e-6:
                break
            gid_j = all_grid_prim_recs[pid_j]["grid_id"] if pid_j in all_grid_prim_recs else None
            if gid_i is not None and gid_i == gid_j:
                continue
            if _uv_box_gap(box_i, box_j)[2] <= max_global_prox_px + 1e-6:
                lo, hi = (pid_i, pid_j) if pid_i < pid_j else (pid_j, pid_i)
                global_candidates.add((lo, hi))

    cross_zone_contacts: List[Dict[str, Any]] = []
    non_grid_internal_contacts: List[Dict[str, Any]] = []
    cross_zone_proximities: List[Dict[str, Any]] = []
    non_grid_internal_proximities: List[Dict[str, Any]] = []

    for pid_a, pid_b in sorted(global_candidates):
        pa, pb = by_id[pid_a], by_id[pid_b]
        gid_a = all_grid_prim_recs[pid_a]["grid_id"] if pid_a in all_grid_prim_recs else None
        gid_b = all_grid_prim_recs[pid_b]["grid_id"] if pid_b in all_grid_prim_recs else None
        if gid_a is not None and gid_b is not None:
            s_pair = 0.5 * (grid_models[gid_a].spacing_px + grid_models[gid_b].spacing_px)
        elif gid_a is not None:
            s_pair = grid_models[gid_a].spacing_px
        elif gid_b is not None:
            s_pair = grid_models[gid_b].spacing_px
        else:
            s_pair = page_s_ref

        is_nb = pid_b in sym_nb[pid_a]
        same_zone = prim_roles[pid_a] == prim_roles[pid_b]
        gu, gv, box_gap = _uv_box_gap(uv_boxes[pid_a], uv_boxes[pid_b])
        near_touch_pair = cfg.near_touch_in_S * max(1e-6, s_pair)

        if is_nb or box_gap <= near_touch_pair + 1e-6:
            contact = _compute_pair_contact(
                pid_a, pid_b, pa, pb, prim_to_mol[pid_a], prim_to_mol[pid_b],
                gid_a, gid_b, uv_boxes, u_vec, v_vec, is_nb, s_pair, cfg
            )
            if contact is not None:
                contact["zone_a"] = prim_roles[pid_a]
                contact["zone_b"] = prim_roles[pid_b]
                contact["involves_transverse_bar"] = (pid_a in bar_pids) or (pid_b in bar_pids)
                if same_zone:
                    non_grid_internal_contacts.append(contact)
                else:
                    cross_zone_contacts.append(contact)
                continue

        exact_gap_px, _ = _convex_polygons_exact_distance(poly_xy[pid_a], poly_xy[pid_b])
        if exact_gap_px <= cfg.proximity_radius_in_S * max(1e-6, s_pair):
            uc_a, vc_a = _prim_uv_center(pa, u_vec, v_vec)
            uc_b, vc_b = _prim_uv_center(pb, u_vec, v_vec)
            du_s = (uc_b - uc_a) / max(1e-6, s_pair)
            dv_s = (vc_b - vc_a) / max(1e-6, s_pair)
            quadrant = ("right" if du_s >= 0 else "left") if abs(du_s) >= abs(dv_s) else ("below" if dv_s >= 0 else "above")
            prox_rec = {
                "prim_a": pid_a,
                "prim_b": pid_b,
                "same_molecule": prim_to_mol[pid_a] == prim_to_mol[pid_b],
                "zone_a": prim_roles[pid_a],
                "zone_b": prim_roles[pid_b],
                "type_a": pa["type"],
                "type_b": pb["type"],
                "distance_precision": _pair_distance_precision(pa, pb),
                "s_ref_px": round(s_pair, 3),
                "du_a_to_b_in_S": round(du_s, 3),
                "dv_a_to_b_in_S": round(dv_s, 3),
                "exact_gap_px": round(exact_gap_px, 3),
                "exact_gap_in_S": round(exact_gap_px / max(1e-6, s_pair), 3),
                "box_gap_u_in_S": round(gu / max(1e-6, s_pair), 3),
                "box_gap_v_in_S": round(gv / max(1e-6, s_pair), 3),
                "box_gap_in_S": round(box_gap / max(1e-6, s_pair), 3),
                "b_relative_to_a": quadrant,
            }
            if same_zone:
                non_grid_internal_proximities.append(prox_rec)
            else:
                cross_zone_proximities.append(prox_rec)

    cross_grid_line_contacts: List[Dict[str, Any]] = []
    cross_grid_line_proximities: List[Dict[str, Any]] = []
    if all_grid_line_meta:
        grid_line_pids_sorted = sorted(all_grid_line_meta.keys())
        for pid in sorted(prim_to_mol.keys()):
            own_gid = all_grid_prim_recs[pid]["grid_id"] if pid in all_grid_prim_recs else None
            box_p = uv_boxes[pid]
            for gpid in grid_line_pids_sorted:
                target_gid, line_idx = all_grid_line_meta[gpid]
                if own_gid is not None and own_gid == target_gid:
                    continue
                s_g = grid_models[target_gid].spacing_px
                is_nb = gpid in sym_nb[pid]
                box_l = uv_boxes[gpid]
                max_g_prox = cfg.proximity_radius_in_S * max(1e-6, s_g)
                _, _, gl_bgap = _uv_box_gap(box_p, box_l)
                if not is_nb and gl_bgap > max_g_prox + 1e-6:
                    continue
                if is_nb or gl_bgap <= cfg.near_touch_in_S * max(1e-6, s_g) + 1e-6:
                    glc = _compute_grid_line_contact(
                        pid, target_gid, gpid, line_idx, by_id[pid], by_id[gpid], is_nb, s_g, cfg
                    )
                    if glc is not None:
                        glc["primitive_zone"] = prim_roles[pid]
                        cross_grid_line_contacts.append(glc)
                        continue
                exact_gap_px, _ = _convex_polygons_exact_distance(poly_xy[pid], poly_xy[gpid])
                if exact_gap_px <= max_g_prox:
                    glp = _compute_grid_line_proximity(
                        pid, target_gid, gpid, line_idx, by_id[pid], by_id[gpid],
                        box_p, box_l, u_vec, v_vec, s_g
                    )
                    glp["distance_precision"] = _pair_distance_precision(by_id[pid], by_id[gpid])
                    glp["exact_gap_px"] = round(exact_gap_px, 3)
                    glp["exact_gap_in_S"] = round(exact_gap_px / max(1e-6, s_g), 3)
                    glp["primitive_zone"] = prim_roles[pid]
                    cross_grid_line_proximities.append(glp)

    cross_grid_line_to_line_contacts: List[Dict[str, Any]] = []
    cross_grid_line_to_line_proximities: List[Dict[str, Any]] = []
    if len(grid_models) >= 2 and all_grid_line_meta:
        gl_sorted_u = sorted(all_grid_line_meta.keys(), key=lambda pid: (uv_boxes[pid][0], uv_boxes[pid][1], pid))
        gl_cands: set[Tuple[int, int]] = set()
        n_gl = len(gl_sorted_u)
        for i in range(n_gl):
            gp_i = gl_sorted_u[i]
            gid_i = all_grid_line_meta[gp_i][0]
            box_i = uv_boxes[gp_i]
            for nb_pid in sym_nb[gp_i]:
                if nb_pid in all_grid_line_meta and all_grid_line_meta[nb_pid][0] != gid_i:
                    lo, hi = (gp_i, nb_pid) if gp_i < nb_pid else (nb_pid, gp_i)
                    gl_cands.add((lo, hi))
            for j in range(i + 1, n_gl):
                gp_j = gl_sorted_u[j]
                box_j = uv_boxes[gp_j]
                if box_j[0] - box_i[2] > max_global_prox_px + 1e-6:
                    break
                if all_grid_line_meta[gp_j][0] == gid_i:
                    continue
                if _uv_box_gap(box_i, box_j)[2] <= max_global_prox_px + 1e-6:
                    lo, hi = (gp_i, gp_j) if gp_i < gp_j else (gp_j, gp_i)
                    gl_cands.add((lo, hi))

        for gp_a, gp_b in sorted(gl_cands):
            gid_a, lidx_a = all_grid_line_meta[gp_a]
            gid_b, lidx_b = all_grid_line_meta[gp_b]
            s_pair = 0.5 * (grid_models[gid_a].spacing_px + grid_models[gid_b].spacing_px)
            is_nb = gp_b in sym_nb[gp_a]
            pa, pb = by_id[gp_a], by_id[gp_b]
            gu, gv, box_gap = _uv_box_gap(uv_boxes[gp_a], uv_boxes[gp_b])

            if is_nb or box_gap <= cfg.near_touch_in_S * max(1e-6, s_pair) + 1e-6:
                contact = _compute_pair_contact(
                    gp_a, gp_b, pa, pb, None, None, gid_a, gid_b,
                    uv_boxes, u_vec, v_vec, is_nb, s_pair, cfg
                )
                if contact is not None:
                    contact["grid_a"] = gid_a
                    contact["line_index_a"] = lidx_a
                    contact["grid_b"] = gid_b
                    contact["line_index_b"] = lidx_b
                    contact["zone_a"] = prim_roles[gp_a]
                    contact["zone_b"] = prim_roles[gp_b]
                    cross_grid_line_to_line_contacts.append(contact)
                    continue

            exact_gap_px, _ = _convex_polygons_exact_distance(poly_xy[gp_a], poly_xy[gp_b])
            if exact_gap_px <= cfg.proximity_radius_in_S * max(1e-6, s_pair):
                uc_a, vc_a = _prim_uv_center(pa, u_vec, v_vec)
                uc_b, vc_b = _prim_uv_center(pb, u_vec, v_vec)
                du_s = (uc_b - uc_a) / max(1e-6, s_pair)
                dv_s = (vc_b - vc_a) / max(1e-6, s_pair)
                quadrant = ("right" if du_s >= 0 else "left") if abs(du_s) >= abs(dv_s) else ("below" if dv_s >= 0 else "above")
                cross_grid_line_to_line_proximities.append({
                    "prim_a": gp_a,
                    "prim_b": gp_b,
                    "grid_a": gid_a,
                    "line_index_a": lidx_a,
                    "grid_b": gid_b,
                    "line_index_b": lidx_b,
                    "zone_a": prim_roles[gp_a],
                    "zone_b": prim_roles[gp_b],
                    "distance_precision": "exact_oriented_rectangle_polygon",
                    "s_ref_px": round(s_pair, 3),
                    "du_a_to_b_in_S": round(du_s, 3),
                    "dv_a_to_b_in_S": round(dv_s, 3),
                    "exact_gap_px": round(exact_gap_px, 3),
                    "exact_gap_in_S": round(exact_gap_px / max(1e-6, s_pair), 3),
                    "box_gap_u_in_S": round(gu / max(1e-6, s_pair), 3),
                    "box_gap_v_in_S": round(gv / max(1e-6, s_pair), 3),
                    "box_gap_in_S": round(box_gap / max(1e-6, s_pair), 3),
                    "b_relative_to_a": quadrant,
                })

    macro_counts: Counter[str] = Counter()
    for role in prim_roles.values():
        if role.endswith("_line"):
            macro_counts["grid_line"] += 1
        elif role.startswith("transverse_bar_"):
            macro_counts["transverse_bar"] += 1
        elif role.startswith("grid_") and role.endswith("_primitive"):
            macro_counts["on_grid_primitive"] += 1
        elif role.startswith("corridor_row_"):
            macro_counts["corridor_row_primitive"] += 1
        else:
            macro_counts["off_grid_primitive"] += 1

    multi_grid_mol_count = sum(1 for ms in mol_summaries.values() if ms["spans_multiple_grids"])
    degraded_grids = [gm.grid_id for gm in grid_models.values() if not gm.domain_order_valid]

    return {
        "schema_version": SCHEMA_VERSION,
        "layer_scope": "pure_geometric_metric_and_relation_graph_zero_music_semantics",
        "graph_contract": {
            "neighbor_graph_symmetrized": True,
            "upstream_unilateral_neighbor_edges_repaired": asym_edge_count,
            "frame_orthonormalized_for_metric": True,
            "proximity_inclusion_metric": "exact_xy_convex_polygon_distance",
            "pair_coverage_scope": {
                "non_line_to_non_line": "all_pairs_with_exact_xy_gap_le_proximity_radius_in_S_plus_all_neighbor_edges",
                "non_line_to_grid_line": "all_intra_and_cross_grid_pairs_with_exact_xy_gap_le_proximity_radius_in_S_plus_all_neighbor_edges",
                "grid_line_to_grid_line_cross_grid": "all_pairs_with_exact_xy_gap_le_proximity_radius_in_S_plus_all_neighbor_edges",
                "grid_line_to_grid_line_same_grid": "modeled_parametrically_in_grid_metrics_with_domain_order_check",
            },
            "pair_ordering": "canonical_undirected_prim_a_lt_prim_b",
            "vector_orientation": "directed_from_prim_a_to_prim_b_inverse_by_sign_flip",
            "ribbon_bbox_model": "exact_oriented_rectangle_perpendicular_width_only",
            "degenerate_bbox_handling": "exact_point_and_segment_polygon_distance",
            "collinearity_taxonomy": [
                "collinear_overlap",
                "shallow_crossing",
                "near_collinear_overlap",
                "parallel_side_by_side",
            ],
            "cell_partition_guarantee": "contiguous_partition_of_grid_u_span_within_tolerance",
            "cell_continuity_tol_px": cfg.cell_continuity_tol_px,
            "distance_precisions": [
                "exact_oriented_rectangle_polygon",
                "exact_oriented_rect_to_bbox_approx_contour",
                "exact_bbox_to_bbox_approx_contour",
            ],
        },
        "config": asdict(cfg),
        "source_image": layer1.get("image"),
        "upstream_gates": gate_report,
        "page_frame": {**pf, "orthonormal_metric_frame": frame_ortho_meta},
        "audit_verification": {
            "total_primitives_expected": len(expected_pids),
            "primitives_assigned_exactly_once": len(prim_assign_count),
            "missing_primitives": 0,
            "duplicate_primitive_assignments": 0,
            "total_molecules_expected": len(expected_mids),
            "molecules_audited_exactly_once": len(mol_summaries),
            "missing_molecules": 0,
            "duplicate_molecule_assignments": 0,
            "multi_grid_molecules": multi_grid_mol_count,
            "degraded_crossing_grid_ids": degraded_grids,
            "primitives_by_macro_category": dict(sorted(macro_counts.items())),
            "guaranteed_exact_once_accounting": True,
        },
        "totals": {
            "systems": len(systems),
            "grids": len(grids_out),
            "cells": sum(g["counts"]["cells"] for g in grids_out),
            "transverse_bars": len(bars),
            "on_grid_non_bar_primitives": sum(g["counts"]["non_bar_primitives"] for g in grids_out),
            "direct_contacts": sum(g["counts"]["direct_contacts"] for g in grids_out),
            "grid_line_contacts": sum(g["counts"]["grid_line_contacts"] for g in grids_out),
            "grid_line_proximities": sum(g["counts"]["grid_line_proximities"] for g in grids_out),
            "cross_zone_contacts": len(cross_zone_contacts),
            "non_grid_internal_contacts": len(non_grid_internal_contacts),
            "cross_zone_proximities": len(cross_zone_proximities),
            "non_grid_internal_proximities": len(non_grid_internal_proximities),
            "cross_grid_line_contacts": len(cross_grid_line_contacts),
            "cross_grid_line_proximities": len(cross_grid_line_proximities),
            "cross_grid_line_to_line_contacts": len(cross_grid_line_to_line_contacts),
            "cross_grid_line_to_line_proximities": len(cross_grid_line_to_line_proximities),
            "spatial_proximities": sum(g["counts"]["spatial_proximities"] for g in grids_out),
            "corridor_rows": len(corridor_rows_out),
            "off_grid_primitives": len(off_grid_primitives_out),
        },
        "systems": systems,
        "transverse_bars": bars,
        "grids": grids_out,
        "cross_zone_contacts": cross_zone_contacts,
        "non_grid_internal_contacts": non_grid_internal_contacts,
        "cross_zone_proximities": cross_zone_proximities,
        "non_grid_internal_proximities": non_grid_internal_proximities,
        "cross_grid_line_contacts": cross_grid_line_contacts,
        "cross_grid_line_proximities": cross_grid_line_proximities,
        "cross_grid_line_to_line_contacts": cross_grid_line_to_line_contacts,
        "cross_grid_line_to_line_proximities": cross_grid_line_to_line_proximities,
        "corridor_rows": corridor_rows_out,
        "off_grid_primitives": off_grid_primitives_out,
        "molecule_summaries": [mol_summaries[mid] for mid in expected_mids],
        "primitive_assignment_ledger": {str(k): prim_roles[k] for k in expected_pids},
    }


def _load_canvas(layer1: Dict[str, Any], image_override: Optional[str]) -> np.ndarray:
    img_path = image_override or layer1.get("image")
    if img_path and Path(img_path).exists():
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is not None:
            return (img.astype(np.float32) * 0.40 + 20.0).astype(np.uint8)
    max_x, max_y = 640, 480
    for p in layer1["primitives"]:
        x0, y0, x1, y1 = _prim_xy_box(p)
        max_x = max(max_x, int(math.ceil(x1)) + 40)
        max_y = max(max_y, int(math.ceil(y1)) + 40)
    return np.zeros((max_y, max_x, 3), dtype=np.uint8)


def _draw_prim(canvas: np.ndarray, p: Dict[str, Any], color: Tuple[int, int, int], thick: int = 2) -> None:
    if p["type"] == "banda_dreapta":
        p0 = (int(round(p["start"][0])), int(round(p["start"][1])))
        p1 = (int(round(p["end"][0])), int(round(p["end"][1])))
        cv2.line(canvas, p0, p1, color, thick, cv2.LINE_AA)
    else:
        x0, y0, x1, y1 = [int(round(v)) for v in p["bbox"]]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, max(1, thick - 1), cv2.LINE_AA)


def draw_debug_layer3a(
    layer1: Dict[str, Any],
    layer3a: Dict[str, Any],
    out_dir: Path,
    image_override: Optional[str] = None,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {int(p["id"]): p for p in layer1["primitives"]}
    base = _load_canvas(layer1, image_override)
    font = cv2.FONT_HERSHEY_SIMPLEX
    ortho = layer3a["page_frame"]["orthonormal_metric_frame"]
    u_vec = np.array(ortho["u_axis_orthonormal"], dtype=np.float64)
    v_vec = np.array(ortho["v_axis_orthonormal"], dtype=np.float64)

    p1 = out_dir / "debug_l3a_1_grid_metric_support.png"
    p2 = out_dir / "debug_l3a_2_contacts_geometry.png"
    p3 = out_dir / "debug_l3a_3_accounting_zones.png"

    v1 = base.copy()
    for g in layer3a["grids"]:
        m = g["metrics"]
        u0, u1 = m["u_span"]
        um = 0.5 * (u0 + u1)
        line_col = (235, 220, 0) if m["domain_order_valid"] else (0, 0, 250)
        for lm in m["lines"]:
            ku = lm.get("knots_u", [])
            kv = lm.get("knots_v", [])
            if len(ku) >= 2 and len(ku) == len(kv):
                pts_curve = np.array(
                    [_uv_to_xy(float(u_k), float(v_k), u_vec, v_vec) for u_k, v_k in zip(ku, kv)],
                    dtype=np.int32,
                ).reshape(-1, 1, 2)
                cv2.polylines(v1, [pts_curve], False, (0, 175, 220), 1, cv2.LINE_AA)
            else:
                lv, sl = lm["v_at_mid"], lm["slope_dv_du"]
                pt0 = _uv_to_xy(u0, lv + sl * (u0 - um), u_vec, v_vec)
                pt1 = _uv_to_xy(u1, lv + sl * (u1 - um), u_vec, v_vec)
                cv2.line(v1, (int(pt0[0]), int(pt0[1])), (int(pt1[0]), int(pt1[1])), (0, 175, 220), 1, cv2.LINE_AA)
            for pid in lm["segment_primitive_ids"]:
                _draw_prim(v1, by_id[pid], line_col, 2)
        for cell in g["cells"]:
            cu = 0.5 * (cell["u_start"] + cell["u_end"])
            cpt = _uv_to_xy(cu, m["top_v_at_mid"] - 0.85 * m["spacing_px"], u_vec, v_vec)
            suffix = "*" if cell["cell_kind"] != "normal" else ""
            cv2.putText(v1, f"G{g['grid_id']}.C{cell['cell_id']}{suffix}", (int(cpt[0]) - 18, max(16, int(cpt[1]))),
                        font, 0.42, (235, 235, 110), 1, cv2.LINE_AA)
    for b in layer3a["transverse_bars"]:
        col = (240, 0, 240) if b["bar_geometry_kind"] == "single" else (0, 120, 255)
        _draw_prim(v1, by_id[b["primitive_id"]], col, 3)
    cv2.imwrite(str(p1), v1)

    v2 = base.copy()
    for g in layer3a["grids"]:
        for pid in g["metrics"]["primitive_ids"]:
            _draw_prim(v2, by_id[pid], (80, 80, 80), 1)
        for rec in g["grid_primitives"]:
            _draw_prim(v2, by_id[rec["primitive_id"]], (180, 220, 180), 1)
        for c in g["direct_contacts"]:
            pa, pb = by_id[c["prim_a"]], by_id[c["prim_b"]]
            ca = (int(round(pa["center"][0])), int(round(pa["center"][1])))
            cb = (int(round(pb["center"][0])), int(round(pb["center"][1])))
            geom = c.get("contact_geometry", c.get("attach_zone_on_ribbon", "contact"))
            if geom in ("interior_to_interior_crossing", "shallow_crossing", "interior"):
                col = (0, 0, 245)
            elif geom in ("collinear_overlap", "near_collinear_overlap"):
                col = (255, 0, 255)
            elif geom == "parallel_side_by_side":
                col = (255, 210, 0)
            else:
                col = (0, 235, 80)
            cv2.line(v2, ca, cb, col, 2, cv2.LINE_AA)
        for glc in g["grid_line_contacts"]:
            pa = by_id[glc["primitive_id"]]
            cv2.circle(v2, (int(round(pa["center"][0])), int(round(pa["center"][1]))), 3, (235, 220, 0), -1)
        for pr in g["spatial_proximities"]:
            if pr["exact_gap_in_S"] <= 1.5:
                pa, pb = by_id[pr["prim_a"]], by_id[pr["prim_b"]]
                cv2.line(v2, (int(round(pa["center"][0])), int(round(pa["center"][1]))),
                         (int(round(pb["center"][0])), int(round(pb["center"][1]))), (210, 90, 210), 1, cv2.LINE_AA)
    for cz in layer3a["cross_zone_contacts"]:
        pa, pb = by_id[cz["prim_a"]], by_id[cz["prim_b"]]
        cv2.line(v2, (int(round(pa["center"][0])), int(round(pa["center"][1]))),
                 (int(round(pb["center"][0])), int(round(pb["center"][1]))), (0, 180, 255), 2, cv2.LINE_AA)
    for cgg in layer3a["cross_grid_line_to_line_contacts"]:
        pa, pb = by_id[cgg["prim_a"]], by_id[cgg["prim_b"]]
        cv2.line(v2, (int(round(pa["center"][0])), int(round(pa["center"][1]))),
                 (int(round(pb["center"][0])), int(round(pb["center"][1]))), (0, 100, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(p2), v2)

    v3 = base.copy()
    for pid_s, role in layer3a["primitive_assignment_ledger"].items():
        p = by_id[int(pid_s)]
        if role.endswith("_line"):
            col = (110, 110, 110)
        elif role.startswith("transverse_bar_"):
            col = (240, 0, 240)
        elif role.startswith("grid_"):
            col = (0, 215, 80)
        elif role.startswith("corridor_row_"):
            col = (235, 195, 40)
        else:
            col = (0, 130, 250)
        _draw_prim(v3, p, col, 2)
    cv2.imwrite(str(p3), v3)

    return {
        "debug_l3a_1_grid_metric_support": str(p1),
        "debug_l3a_2_contacts_geometry": str(p2),
        "debug_l3a_3_accounting_zones": str(p3),
    }


def process_layer3a(
    layer1_json_path: str | Path,
    layer2_json_path: str | Path,
    out_dir: str | Path = "debug_l3a",
    image_override: Optional[str] = None,
    config: Optional[Layer3AConfig] = None,
) -> Dict[str, Any]:
    l1_path = Path(layer1_json_path)
    l2_path = Path(layer2_json_path)
    layer1 = json.loads(l1_path.read_text(encoding="utf-8"))
    layer2 = json.loads(l2_path.read_text(encoding="utf-8"))

    layer3a = analyze_layer3a(layer1, layer2, config=config)
    layer3a["debug_images"] = draw_debug_layer3a(layer1, layer3a, Path(out_dir), image_override)

    stem = l1_path.stem.replace("_geometry_layer1", "")
    out_path = Path(out_dir) / f"{stem}_geometry_layer3a.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(layer3a, ensure_ascii=False, indent=2), encoding="utf-8")
    layer3a["json_file"] = str(out_path)
    return layer3a


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("layer1_json")
    parser.add_argument("layer2_json")
    parser.add_argument("--out-dir", default="debug_l3a")
    parser.add_argument("--image", default=None)
    parser.add_argument("--prox-radius-s", type=float, default=3.0)
    args = parser.parse_args(argv)
    try:
        cfg = Layer3AConfig(proximity_radius_in_S=args.prox_radius_s)
        res = process_layer3a(args.layer1_json, args.layer2_json, args.out_dir, args.image, cfg)
    except GateViolationError as gexc:
        print(f"POARTĂ RESPINSĂ: {gexc}", file=sys.stderr)
        return 3
    except AuditViolationError as aexc:
        print(f"AUDIT PICAT: {aexc}", file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"Eroare: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "schema_version": res["schema_version"],
        "upstream_gates": res["upstream_gates"],
        "audit_verification": res["audit_verification"],
        "totals": res["totals"],
        "debug_images": res["debug_images"],
        "json_file": res["json_file"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
