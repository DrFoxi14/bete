#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometrie pură (Stratul 0 + Stratul 1), 100% orb la domeniu și 100% lossless.

Stratul 0 (Graniță exactă):
  Cerneală -> bucle inter-pixel -> DSS exact (reconstrucție XOR == 0 pixeli).

Stratul 1 (Partiție geometrică completă a cernelii):
  - `banda_dreapta` : pereți DSS opuși peste cerneală continuă + podire coliniară în cerneală.
  - `jonctiune`     : pixeli revendicați simultan de >= 2 benzi drepte (graful de contacte).
  - `banda_curba`   : cerneală rămasă cu gol interior sau alungire de arie A / w^2 >= 1.85.
  - `pata_compacta` : cerneală izotropă / compactă (puncte, ovale, pete de orice mărime).

Garanții matematice verificate automat la fiecare rulare:
  1. reconstructed_from_dss XOR ink == 0
  2. covered_by_primitives  XOR ink == 0  (ZERO pixeli de cerneală pierduți)

Dependențe: numpy, opencv-python, scipy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi


_FIRST_BIT = (0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0)


@dataclass(slots=True)
class DSS:
    dss_id: int
    loop_id: int
    is_hole: bool
    x0: int
    y0: int
    x1: int
    y1: int
    sx: int
    sy: int
    a: int
    b: int
    mu: int
    grid_steps: int


@dataclass
class Wall:
    wall_id: int
    loop_id: int
    is_hole: bool
    dss_ids: List[int]
    points: np.ndarray        # M x 2 float32 (vârfuri de graniță)
    chord_len: float
    arc_len: float
    angle_deg: float          # [0, 180)
    normal: np.ndarray        # normală unitară strict spre interiorul cernelii


@dataclass
class RibbonCandidate:
    cid: int
    points: np.ndarray        # M x 2 float32 (axa centrală)
    width: float
    length: float
    angle_deg: float
    wall_ids: List[int]


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _angle_diff_180(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _angle_diff_360(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _polyline_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _segment_angle(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    c = pts - np.mean(pts, axis=0)
    vals, vecs = np.linalg.eigh(c.T @ c)
    p = vecs[:, int(np.argmax(vals))]
    return float(math.degrees(math.atan2(float(p[1]), float(p[0]))) % 180.0)


def _max_chord_dev(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    p0, p1 = pts[0], pts[-1]
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return float(np.max(np.linalg.norm(pts - p0, axis=1)))
    rel = pts - p0
    return float(np.max(np.abs(rel[:, 0] * v[1] - rel[:, 1] * v[0])) / L)


def _sample_polyline_interior(pts: np.ndarray, n_samples: int = 9) -> np.ndarray:
    if len(pts) < 2:
        return pts
    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(segs)]
    total = float(cum[-1])
    if total < 1e-6:
        return pts[:1]
    lo = min(0.5, 0.10 * total)
    hi = max(lo, total - min(0.5, 0.10 * total))
    ts = np.linspace(lo, hi, n_samples)
    return np.column_stack([
        np.interp(ts, cum, pts[:, 0]),
        np.interp(ts, cum, pts[:, 1]),
    ]).astype(np.float32)


def _ink_fraction_between(p1: np.ndarray, p2: np.ndarray, ink: np.ndarray) -> float:
    h, w = ink.shape
    dist = float(np.linalg.norm(p2 - p1))
    steps = max(3, int(math.ceil(dist * 1.5)))
    ts = np.linspace(0.0, 1.0, steps)
    xs = np.clip(np.floor(p1[0] + ts * (p2[0] - p1[0])).astype(int), 0, w - 1)
    ys = np.clip(np.floor(p1[1] + ts * (p2[1] - p1[1])).astype(int), 0, h - 1)
    return float(np.mean(ink[ys, xs]))


# ---------------------------------------------------------------------------
# Pas 1: Binarizare curată prin scăderea fundalului hârtiei
# ---------------------------------------------------------------------------
def _safe_paper_region(image: np.ndarray) -> Tuple[np.ndarray, str]:
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0].astype(np.float32)

    margin = max(4, int(min(h, w) * 0.03))
    border = np.concatenate([
        light[:margin, :].ravel(),
        light[-margin:, :].ravel(),
        light[:, :margin].ravel(),
        light[:, -margin:].ravel(),
    ])
    center = light[h // 4: 3 * h // 4, w // 4: 3 * w // 4].ravel()
    border_med = float(np.median(border))
    center_med = float(np.median(center))

    if center_med - border_med < 32.0:
        return np.full((h, w), 255, np.uint8), "full_frame_safe"

    cutoff = 0.5 * (border_med + center_med)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    sat_cut = min(
        160.0,
        max(75.0, float(np.percentile(sat[h // 4: 3 * h // 4, w // 4: 3 * w // 4], 95)) + 25.0),
    )
    candidate = ((light >= cutoff) & (sat <= sat_cut)).astype(np.uint8)
    close_size = _odd(int(min(h, w) * 0.025))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate = ndi.binary_fill_holes(candidate > 0).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if count <= 1:
        return np.full((h, w), 255, np.uint8), "full_frame_fallback"
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ratio = float(stats[label, cv2.CC_STAT_AREA]) / float(h * w)
    if ratio < 0.25:
        return np.full((h, w), 255, np.uint8), f"full_frame_small_comp_{ratio:.2f}"

    comp = (labels == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full((h, w), 255, np.uint8), "full_frame_no_contour"
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    paper = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(paper, hull, 255)
    return paper, f"detected_sheet_ratio_{ratio:.2f}"


def binarize_image(image: np.ndarray, scale: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    h, w = image.shape[:2]
    paper, paper_method = _safe_paper_region(image)
    darkest = np.min(image, axis=2)

    scale_bg = max(1, min(h, w) // 420)
    sh, sw = max(16, h // scale_bg), max(16, w // scale_bg)
    small = cv2.resize(darkest, (sw, sh), interpolation=cv2.INTER_AREA)
    ksize = _odd(max(15, int(min(sh, sw) * 0.085)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    bg_small = cv2.GaussianBlur(bg_small, (_odd(ksize // 2), _odd(ksize // 2)), 0)
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)

    diff = np.clip(bg.astype(np.float32) - darkest.astype(np.float32), 0.0, 255.0)
    diff[paper == 0] = 0.0

    paper_vals = diff[paper > 0].astype(np.uint8)
    if paper_vals.size == 0:
        return np.zeros((h, w), dtype=bool), {"paper_method": paper_method, "applied_diff_thresh": 30.0}

    otsu_val, _ = cv2.threshold(paper_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = max(20.0, min(115.0, float(otsu_val) * 0.65))

    if abs(scale - 1.0) > 1e-3:
        ww, wh = int(round(w * scale)), int(round(h * scale))
        diff = cv2.resize(diff, (ww, wh), interpolation=cv2.INTER_CUBIC)
        paper = cv2.resize(paper, (ww, wh), interpolation=cv2.INTER_NEAREST)

    ink_bool = (diff >= thresh) & (paper > 0)
    return ink_bool, {
        "otsu_diff": round(float(otsu_val), 1),
        "applied_diff_thresh": round(thresh, 1),
        "paper_method": paper_method,
        "working_size_px": {"width": int(ink_bool.shape[1]), "height": int(ink_bool.shape[0])},
    }


# ---------------------------------------------------------------------------
# Stratul 0: Contururi inter-pixel -> DSS exact + hartă de fețe (XOR == 0)
# ---------------------------------------------------------------------------
def extract_crack_loops(ink: np.ndarray) -> List[Tuple[List[int], List[int], bool]]:
    h, w = ink.shape
    w1 = w + 1
    out_mask = np.zeros((h + 1, w1), dtype=np.uint8)

    top = ink.copy(); top[1:, :] &= ~ink[:-1, :]
    out_mask[:h, :w][top] |= 1
    right = ink.copy(); right[:, :-1] &= ~ink[:, 1:]
    out_mask[:h, 1:][right] |= 2
    bottom = ink.copy(); bottom[:-1, :] &= ~ink[1:, :]
    out_mask[1:, 1:][bottom] |= 4
    left = ink.copy(); left[:, 1:] &= ~ink[:, :-1]
    out_mask[1:, :w][left] |= 8

    is_saddle = ((out_mask & (out_mask - 1)) != 0).ravel().tobytes()
    mask_buf = bytearray(out_mask.ravel().tobytes())
    start_ys, start_xs = np.nonzero(out_mask)
    dx_tbl = (1, 0, -1, 0)
    dy_tbl = (0, 1, 0, -1)
    didx_tbl = (1, w1, -1, -w1)
    first_bit = _FIRST_BIT
    loops: List[Tuple[List[int], List[int], bool]] = []

    for sy, sx in zip(start_ys.tolist(), start_xs.tolist()):
        start_idx = sy * w1 + sx
        while mask_buf[start_idx]:
            d = first_bit[mask_buf[start_idx]]
            mask_buf[start_idx] ^= (1 << d)
            xs, ys = [sx], [sy]
            x, y = sx + dx_tbl[d], sy + dy_tbl[d]
            idx = start_idx + didx_tbl[d]
            area2 = sx * y - x * sy
            while True:
                m = mask_buf[idx]
                if is_saddle[idx]:
                    nd = (d + 3) & 3
                    if not (m & (1 << nd)):
                        break
                else:
                    if not m:
                        break
                    nd = first_bit[m]
                mask_buf[idx] ^= (1 << nd)
                xs.append(x)
                ys.append(y)
                nx, ny = x + dx_tbl[nd], y + dy_tbl[nd]
                area2 += x * ny - nx * y
                x, y = nx, ny
                idx += didx_tbl[nd]
                d = nd
            loops.append((xs, ys, area2 < 0))
    return loops


def _grow_maximal_dss(
    xs: List[int], ys: List[int], start: int, max_steps: int
) -> Tuple[int, int, int, int, int, int]:
    n = len(xs)
    x0, y0 = xs[start], ys[start]
    i1 = (start + 1) % n
    x1, y1 = xs[i1], ys[i1]
    dx, dy = x1 - x0, y1 - y0

    if dx != 0:
        sx, sy = (1 if dx > 0 else -1), 0
        a, b, mu = 0, 1, 0
        uf_u, uf_v, ul_u, ul_v = 0, 0, 1, 0
        lf_u, lf_v, ll_u, ll_v = 0, 0, 1, 0
    else:
        sx, sy = 0, (1 if dy > 0 else -1)
        a, b, mu = 1, 0, 0
        uf_u, uf_v, ul_u, ul_v = 0, 0, 0, 1
        lf_u, lf_v, ll_u, ll_v = 0, 0, 0, 1

    px, py = x1, y1
    steps = 1
    while steps < max_steps:
        nxt = (start + steps + 1) % n
        cx, cy = xs[nxt], ys[nxt]
        ddx, ddy = cx - px, cy - py
        if ddx != 0:
            if sx == 0:
                sx = 1 if ddx > 0 else -1
            elif ddx != sx:
                break
        if ddy != 0:
            if sy == 0:
                sy = 1 if ddy > 0 else -1
            elif ddy != sy:
                break

        u = (cx - x0) * sx if sx else 0
        v = (cy - y0) * sy if sy else 0
        omega = a + b
        r = a * u - b * v + mu
        if 0 <= r < omega:
            if r == 0:
                ul_u, ul_v = u, v
            if r == omega - 1:
                ll_u, ll_v = u, v
        elif r == -1:
            a, b = v - uf_v, u - uf_u
            mu = -(a * uf_u - b * uf_v)
            ul_u, ul_v = u, v
            lf_u, lf_v = ll_u, ll_v
        elif r == omega:
            a, b = v - lf_v, u - lf_u
            mu = (a + b - 1) - (a * lf_u - b * lf_v)
            ll_u, ll_v = u, v
            uf_u, uf_v = ul_u, ul_v
        else:
            break
        px, py = cx, cy
        steps += 1

    if sx == 0:
        sx = 1
    if sy == 0:
        sy = 1
    return steps, sx, sy, a, b, mu


def decompose_loops_to_dss(
    loops: Sequence[Tuple[List[int], List[int], bool]]
) -> Tuple[List[DSS], List[List[int]]]:
    all_dss: List[DSS] = []
    loop_dss_lists: List[List[int]] = []
    next_id = 0
    for loop_id, (xs, ys, is_hole) in enumerate(loops, start=1):
        n = len(xs)
        first_steps, *_ = _grow_maximal_dss(xs, ys, 0, n)
        offset = first_steps % n
        consumed = 0
        l_ids: List[int] = []
        while consumed < n:
            start = (offset + consumed) % n
            steps, sx, sy, a, b, mu = _grow_maximal_dss(xs, ys, start, n - consumed)
            end = (start + steps) % n
            all_dss.append(DSS(
                next_id, loop_id, is_hole,
                xs[start], ys[start], xs[end], ys[end],
                sx, sy, a, b, mu, steps,
            ))
            l_ids.append(next_id)
            next_id += 1
            consumed += steps
        loop_dss_lists.append(l_ids)
    return all_dss, loop_dss_lists


def build_face_map_and_reconstruct(
    dss_list: Sequence[DSS], shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruiește masca binară EXCLUSIV din DSS și completează face_dss[4, H, W]."""
    h, w = shape
    diff = np.zeros((h, w + 1), dtype=np.int32)
    face_dss = np.full((4, h, w), -1, dtype=np.int32)

    for seg in dss_list:
        sid = seg.dss_id
        x, y = seg.x0, seg.y0
        sx, sy = seg.sx, seg.sy
        a, b = seg.a, seg.b
        r = seg.mu
        u, v = 0, 0
        u_end = (seg.x1 - seg.x0) * sx
        v_end = (seg.y1 - seg.y0) * sy
        while u < u_end or v < v_end:
            if r < b:
                if sx > 0:
                    face_dss[0, y, x] = sid
                    x += 1
                else:
                    x -= 1
                    face_dss[2, y - 1, x] = sid
                u += 1
                r += a
            else:
                if sy > 0:
                    face_dss[1, y, x - 1] = sid
                    diff[y, x] -= 1
                    y += 1
                else:
                    y -= 1
                    face_dss[3, y, x] = sid
                    diff[y, x] += 1
                v += 1
                r -= b
        if u != u_end or v != v_end or x != seg.x1 or y != seg.y1:
            raise AssertionError(f"Eroare decodare DSS #{sid}")

    reconstructed = np.cumsum(diff, axis=1)[:, :w] > 0
    return reconstructed, face_dss


# ---------------------------------------------------------------------------
# Stratul 1A: Pereți coliniari + sondare vectorială prin cerneală
# ---------------------------------------------------------------------------
def build_walls_from_dss(
    dss_list: Sequence[DSS], loop_dss_lists: Sequence[List[int]]
) -> Tuple[List[Wall], np.ndarray]:
    """Grupează DSS-urile consecutive coliniare de pe aceeași buclă în Pereți. Nu aruncă nimic."""
    walls: List[Wall] = []
    dss_to_wall = np.full(len(dss_list), -1, dtype=np.int32)
    next_wid = 0

    def dss_dir_deg(s: DSS) -> float:
        return math.degrees(math.atan2(s.y1 - s.y0, s.x1 - s.x0)) % 360.0

    for l_ids in loop_dss_lists:
        m = len(l_ids)
        if m == 0:
            continue
        start_k = 0
        if m > 1:
            best_turn = -1.0
            for k in range(m):
                a1 = dss_dir_deg(dss_list[l_ids[(k - 1) % m]])
                a2 = dss_dir_deg(dss_list[l_ids[k]])
                t = _angle_diff_360(a1, a2)
                if t > best_turn:
                    best_turn = t
                    start_k = k

        idx = 0
        while idx < m:
            cur_ids = [l_ids[(start_k + idx) % m]]
            s0 = dss_list[cur_ids[0]]
            pts = [[s0.x0, s0.y0], [s0.x1, s0.y1]]
            base_dir = dss_dir_deg(s0)
            prev_dir = base_dir
            idx += 1

            while idx < m:
                cand_id = l_ids[(start_k + idx) % m]
                sc = dss_list[cand_id]
                c_dir = dss_dir_deg(sc)
                if _angle_diff_360(prev_dir, c_dir) > 20.0:
                    break
                if _angle_diff_360(base_dir, c_dir) > 22.0:
                    break
                test_pts = np.asarray(pts + [[sc.x1, sc.y1]], dtype=np.float32)
                chord = float(np.linalg.norm(test_pts[-1] - test_pts[0]))
                if chord < 1e-6 or _max_chord_dev(test_pts) > max(1.05, 0.055 * chord):
                    break
                cur_ids.append(cand_id)
                pts.append([sc.x1, sc.y1])
                prev_dir = c_dir
                idx += 1

            pts_arr = np.asarray(pts, dtype=np.float32)
            v = pts_arr[-1] - pts_arr[0]
            chord_len = float(np.linalg.norm(v))
            arc_len = _polyline_length(pts_arr)
            if chord_len < 1e-6:
                nrm = np.array([0.0, 1.0], dtype=np.float32)
                ang = 0.0
            else:
                t_u = v / chord_len
                nrm = np.array([-t_u[1], t_u[0]], dtype=np.float32)
                ang = float(math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0)

            walls.append(Wall(
                wall_id=next_wid,
                loop_id=s0.loop_id,
                is_hole=s0.is_hole,
                dss_ids=cur_ids,
                points=pts_arr,
                chord_len=chord_len,
                arc_len=arc_len,
                angle_deg=ang,
                normal=nrm,
            ))
            for did in cur_ids:
                dss_to_wall[did] = next_wid
            next_wid += 1

    return walls, dss_to_wall


def collect_ray_starts(
    dss_list: Sequence[DSS], dss_to_wall: np.ndarray, walls: Sequence[Wall]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Puncte de pornire direct pe fețele de graniță (garantat în interiorul cernelii)."""
    ray_wid: List[int] = []
    ray_x0: List[float] = []
    ray_y0: List[float] = []
    ray_nx: List[float] = []
    ray_ny: List[float] = []

    for seg in dss_list:
        wid = int(dss_to_wall[seg.dss_id])
        wl = walls[wid]
        nx, ny = float(wl.normal[0]), float(wl.normal[1])
        stride = max(1, seg.grid_steps // 16)

        x, y = seg.x0, seg.y0
        sx, sy = seg.sx, seg.sy
        a, b = seg.a, seg.b
        r = seg.mu
        u, v = 0, 0
        u_end = (seg.x1 - seg.x0) * sx
        v_end = (seg.y1 - seg.y0) * sy
        step_i = 0

        while u < u_end or v < v_end:
            if r < b:
                if sx > 0:
                    px, py = x + 0.5, y + 0.08
                    x += 1
                else:
                    x -= 1
                    px, py = x + 0.5, y - 0.08
                u += 1
                r += a
            else:
                if sy > 0:
                    px, py = x - 0.08, y + 0.5
                    y += 1
                else:
                    y -= 1
                    px, py = x + 0.08, y + 0.5
                v += 1
                r -= b

            if step_i % stride == 0 or (u == u_end and v == v_end):
                ray_wid.append(wid)
                ray_x0.append(px)
                ray_y0.append(py)
                ray_nx.append(nx)
                ray_ny.append(ny)
            step_i += 1

    return (
        np.asarray(ray_wid, dtype=np.int32),
        np.asarray(ray_x0, dtype=np.float32),
        np.asarray(ray_y0, dtype=np.float32),
        np.asarray(ray_nx, dtype=np.float32),
        np.asarray(ray_ny, dtype=np.float32),
    )


def raycast_walls_across_ink(
    ink: np.ndarray,
    face_dss: np.ndarray,
    dss_to_wall: np.ndarray,
    ray_wid: np.ndarray,
    ray_x0: np.ndarray,
    ray_y0: np.ndarray,
    ray_nx: np.ndarray,
    ray_ny: np.ndarray,
    max_steps: int = 260,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Marș vectorial prin cerneală până la prima ieșire în alb, cu citirea exactă a peretelui opus."""
    h, w = ink.shape
    n_rays = len(ray_wid)
    step_size = 0.5

    cur_x = ray_x0.copy()
    cur_y = ray_y0.copy()
    prev_ix = np.clip(np.floor(cur_x).astype(np.int32), 0, w - 1)
    prev_iy = np.clip(np.floor(cur_y).astype(np.int32), 0, h - 1)

    active = np.arange(n_rays, dtype=np.int32)
    hit_dist = np.full(n_rays, -1.0, dtype=np.float32)
    hit_wall = np.full(n_rays, -1, dtype=np.int32)
    hit_x = np.zeros(n_rays, dtype=np.float32)
    hit_y = np.zeros(n_rays, dtype=np.float32)

    for s in range(1, max_steps + 1):
        if active.size == 0:
            break
        nx_pos = cur_x[active] + step_size * ray_nx[active]
        ny_pos = cur_y[active] + step_size * ray_ny[active]
        ix = np.floor(nx_pos).astype(np.int32)
        iy = np.floor(ny_pos).astype(np.int32)

        oob = (ix < 0) | (ix >= w) | (iy < 0) | (iy >= h)
        in_bounds = ~oob
        exited = oob.copy()
        if np.any(in_bounds):
            ib_idx = np.flatnonzero(in_bounds)
            exited[ib_idx] = ~ink[iy[ib_idx], ix[ib_idx]]

        if np.any(exited):
            ex_local = np.flatnonzero(exited)
            ex_rays = active[ex_local]
            hit_dist[ex_rays] = s * step_size + 0.08
            hit_x[ex_rays] = nx_pos[ex_local]
            hit_y[ex_rays] = ny_pos[ex_local]

            pix = prev_ix[ex_rays]
            piy = prev_iy[ex_rays]
            ddx = ix[ex_local] - pix
            ddy = iy[ex_local] - piy
            rnx = ray_nx[ex_rays]
            rny = ray_ny[ex_rays]

            face_p = np.where(
                np.abs(ddx) >= np.abs(ddy),
                np.where(ddx > 0, 1, 3),
                np.where(ddy > 0, 2, 0),
            )
            did = face_dss[face_p, piy, pix]
            miss = did < 0
            if np.any(miss):
                face_s = np.where(
                    np.abs(rnx[miss]) >= np.abs(rny[miss]),
                    np.where(rnx[miss] > 0, 1, 3),
                    np.where(rny[miss] > 0, 2, 0),
                )
                did[miss] = face_dss[face_s, piy[miss], pix[miss]]
                miss2 = did < 0
                if np.any(miss2):
                    face_t = np.where(
                        np.abs(rnx[miss2]) < np.abs(rny[miss2]),
                        np.where(rnx[miss2] > 0, 1, 3),
                        np.where(rny[miss2] > 0, 2, 0),
                    )
                    did[miss2] = face_dss[face_t, piy[miss2], pix[miss2]]

            valid_d = did >= 0
            if np.any(valid_d):
                hit_wall[ex_rays[valid_d]] = dss_to_wall[did[valid_d]]

        stay_local = ~exited
        stay_rays = active[stay_local]
        cur_x[stay_rays] = nx_pos[stay_local]
        cur_y[stay_rays] = ny_pos[stay_local]
        prev_ix[stay_rays] = ix[stay_local]
        prev_iy[stay_rays] = iy[stay_local]
        active = stay_rays

    return hit_dist, hit_wall, hit_x, hit_y


# ---------------------------------------------------------------------------
# Stratul 1B: Contopire pereți opuși -> Benzi drepte + Podire prin cerneală
# ---------------------------------------------------------------------------
def _project_wall_centerline_exact(
    wl: Wall,
    ray_indices: np.ndarray,
    rx0: np.ndarray,
    ry0: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    w_est: float,
) -> np.ndarray:
    """Proiectează peretele pe axa mediană reală folosind mijloacele exacte ale razelor."""
    mx = 0.5 * (rx0[ray_indices] + hx[ray_indices])
    my = 0.5 * (ry0[ray_indices] + hy[ray_indices])
    mids = np.column_stack((mx, my))
    proj_n = (mids - wl.points[0]) @ wl.normal
    offset = float(np.median(proj_n))
    offset = max(-0.5, min(offset, 0.65 * w_est + 0.5))
    return (wl.points + offset * wl.normal[None, :]).astype(np.float32)


def _merge_centerlines(items: Sequence[Tuple[np.ndarray, float, float]]) -> Tuple[np.ndarray, float]:
    if len(items) == 1:
        return items[0][0].copy(), items[0][1]
    all_pts = np.vstack([it[0] for it in items])
    origin = np.mean(all_pts, axis=0)
    centered = all_pts - origin
    vals, vecs = np.linalg.eigh(centered.T @ centered)
    u = vecs[:, int(np.argmax(vals))]
    v = np.array([-u[1], u[0]], dtype=np.float64)

    s_all = centered @ u
    s_min, s_max = float(np.min(s_all)), float(np.max(s_all))
    span = s_max - s_min
    if span < 1.0:
        best = max(items, key=lambda it: it[2])
        return best[0].copy(), best[1]

    n_bins = max(3, int(math.ceil(span / 1.5)) + 1)
    s_grid = np.linspace(s_min, s_max, n_bins)
    t_sum = np.zeros(n_bins, dtype=np.float64)
    w_sum = np.zeros(n_bins, dtype=np.float64)

    for pts, _mw, weight in items:
        rel = pts - origin
        gs = rel @ u
        gt = rel @ v
        order = np.argsort(gs)
        gs, gt = gs[order], gt[order]
        mask = (s_grid >= gs[0] - 0.5) & (s_grid <= gs[-1] + 0.5)
        if not np.any(mask):
            continue
        t_sum[mask] += max(0.5, weight) * np.interp(s_grid[mask], gs, gt)
        w_sum[mask] += max(0.5, weight)

    valid = w_sum > 0
    pts_out = origin[None, :] + s_grid[valid, None] * u[None, :] + (t_sum[valid] / w_sum[valid])[:, None] * v[None, :]
    mean_w = sum(it[1] * max(0.5, it[2]) for it in items) / max(1e-6, sum(max(0.5, it[2]) for it in items))
    return pts_out.astype(np.float32), float(mean_w)


def extract_straight_ribbons(
    ink: np.ndarray,
    walls: Sequence[Wall],
    rw: np.ndarray,
    rx0: np.ndarray,
    ry0: np.ndarray,
    hdist: np.ndarray,
    hwall: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
) -> List[RibbonCandidate]:
    h, w = ink.shape
    ink_dil1 = ndi.binary_dilation(ink, structure=np.ones((3, 3), dtype=bool))
    wall_normals = np.array([wl.normal for wl in walls], dtype=np.float32)

    wall_ray_indices: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(rw)):
        if hdist[i] > 0:
            wall_ray_indices[int(rw[i])].append(i)

    projected: List[Tuple[int, np.ndarray, float, float, float, set[int]]] = []
    for wl in walls:
        if wl.chord_len < 1.8:
            continue
        r_idxs = wall_ray_indices.get(wl.wall_id)
        if not r_idxs:
            continue
        anti_idxs = [
            i for i in r_idxs
            if hwall[i] >= 0 and hwall[i] != wl.wall_id
            and float(np.dot(wl.normal, wall_normals[hwall[i]])) <= -0.82
        ]
        if len(anti_idxs) < max(1, int(math.ceil(0.25 * len(r_idxs)))):
            continue

        anti_arr = np.asarray(anti_idxs, dtype=np.int32)
        widths = hdist[anti_arr]
        w_est = float(np.percentile(widths, 35))
        if w_est < 0.55:
            continue

        good_mask = np.abs(widths - w_est) <= max(1.5, 0.55 * w_est)
        good_arr = anti_arr[good_mask] if np.any(good_mask) else anti_arr
        center = _project_wall_centerline_exact(wl, good_arr, rx0, ry0, hx, hy, w_est)

        samples = _sample_polyline_interior(center, max(5, min(17, int(wl.chord_len))))
        cx = np.clip(np.floor(samples[:, 0]).astype(int), 0, w - 1)
        cy = np.clip(np.floor(samples[:, 1]).astype(int), 0, h - 1)
        support_mask = ink_dil1 if w_est < 2.0 else ink
        if float(np.mean(support_mask[cy, cx])) < 0.72:
            continue

        opp_ids = {int(hwall[i]) for i in good_arr}
        projected.append((wl.wall_id, center, w_est, wl.chord_len, wl.angle_deg, opp_ids))

    m = len(projected)
    if m == 0:
        return []

    parent = list(range(m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    wid_to_idx = {p[0]: i for i, p in enumerate(projected)}
    bboxes = np.array([
        [float(np.min(p[1][:, 0])), float(np.min(p[1][:, 1])),
         float(np.max(p[1][:, 0])), float(np.max(p[1][:, 1]))]
        for p in projected
    ], dtype=np.float32)
    p_widths = np.array([p[2] for p in projected], dtype=np.float32)
    p_angles = np.array([p[4] for p in projected], dtype=np.float32)

    # 1a) Perechi opuse confirmate direct de raze
    for i, (_wid, ci, wi, _li, ai, opp_ids) in enumerate(projected):
        for owid in opp_ids:
            j = wid_to_idx.get(owid)
            if j is None or j <= i or find(i) == find(j):
                continue
            wj, aj = projected[j][2], projected[j][4]
            if _angle_diff_180(ai, aj) > 18.0:
                continue
            if max(wi, wj) / max(1e-6, min(wi, wj)) > 2.2:
                continue
            mi = _sample_polyline_interior(ci, 3)[1]
            mj = _sample_polyline_interior(projected[j][1], 3)[1]
            check_m = ink_dil1 if max(wi, wj) < 2.0 else ink
            if _ink_fraction_between(mi, mj, check_m) >= 0.85:
                union(i, j)

    # 1b) Axe suprapuse în interiorul cernelii (pre-filtrate vectorizat pentru viteză)
    for i in range(m):
        wi, ai = float(p_widths[i]), float(p_angles[i])
        ci = projected[i][1]
        w_max = np.maximum(wi, p_widths[i + 1:])
        tols = 0.75 * w_max + 1.5
        da = np.abs(ai - p_angles[i + 1:]) % 180.0
        da = np.minimum(da, 180.0 - da)
        cands_j = (
            (da <= 15.0) &
            (bboxes[i, 2] + tols >= bboxes[i + 1:, 0]) &
            (bboxes[i + 1:, 2] + tols >= bboxes[i, 0]) &
            (bboxes[i, 3] + tols >= bboxes[i + 1:, 1]) &
            (bboxes[i + 1:, 3] + tols >= bboxes[i, 1])
        )
        for off in np.flatnonzero(cands_j):
            j = i + 1 + int(off)
            if find(i) == find(j):
                continue
            wj, aj = float(p_widths[j]), float(p_angles[j])
            if max(wi, wj) / max(1e-6, min(wi, wj)) > 2.2:
                continue
            cj = projected[j][1]
            rad = math.radians(ai if projected[i][3] >= projected[j][3] else aj)
            u = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)
            v = np.array([-u[1], u[0]], dtype=np.float32)
            orig = ci[0]
            si, ti = (ci - orig) @ u, (ci - orig) @ v
            sj, tj = (cj - orig) @ u, (cj - orig) @ v
            ov_lo = max(float(np.min(si)), float(np.min(sj)))
            ov_hi = min(float(np.max(si)), float(np.max(sj)))
            if ov_hi - ov_lo < 0.5:
                continue
            qs = np.linspace(ov_lo, ov_hi, 5)
            oi, oj = np.argsort(si), np.argsort(sj)
            qti = np.interp(qs, si[oi], ti[oi])
            qtj = np.interp(qs, sj[oj], tj[oj])
            if float(np.mean(np.abs(qti - qtj))) > 0.65 * max(wi, wj) + 1.2:
                continue
            check_m = ink_dil1 if max(wi, wj) < 2.0 else ink
            if all(
                _ink_fraction_between(orig + sv * u + t1 * v, orig + sv * u + t2 * v, check_m) >= 0.85
                for sv, t1, t2 in zip(qs, qti, qtj)
            ):
                union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(m):
        groups[find(i)].append(i)

    ribbons: List[RibbonCandidate] = []
    for gid, idxs in enumerate(groups.values()):
        items = [(projected[i][1], projected[i][2], projected[i][3]) for i in idxs]
        pts, mw = _merge_centerlines(items)
        ribbons.append(RibbonCandidate(
            cid=gid,
            points=pts,
            width=mw,
            length=_polyline_length(pts),
            angle_deg=_segment_angle(pts),
            wall_ids=[projected[i][0] for i in idxs],
        ))

    # 2) Podire coliniară strict prin cerneală (reunește benzi tăiate de traversări)
    def ep_dir(pts: np.ndarray, at_end: bool) -> np.ndarray:
        v = (pts[-1] - pts[0]) if at_end else (pts[0] - pts[-1])
        nrm = float(np.linalg.norm(v))
        return (v / nrm) if nrm > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)

    for _pass in range(2):
        nr = len(ribbons)
        if nr < 2:
            break
        end_pts = np.zeros((2 * nr, 2), dtype=np.float32)
        end_dirs = np.zeros((2 * nr, 2), dtype=np.float32)
        for i, r in enumerate(ribbons):
            end_pts[2 * i] = r.points[0]
            end_pts[2 * i + 1] = r.points[-1]
            end_dirs[2 * i] = ep_dir(r.points, False)
            end_dirs[2 * i + 1] = ep_dir(r.points, True)

        cands: List[Tuple[float, int, int]] = []
        for a in range(2 * nr):
            ra = ribbons[a // 2]
            if ra.length < 0.85 * ra.width:
                continue
            pa, da = end_pts[a], end_dirs[a]
            diffs = end_pts[a + 1:] - pa
            dists = np.linalg.norm(diffs, axis=1)
            max_g = max(20.0, 14.0 * ra.width)
            for off in np.flatnonzero(dists <= max_g):
                b = a + 1 + int(off)
                if a // 2 == b // 2:
                    continue
                rb = ribbons[b // 2]
                if rb.length < 0.85 * rb.width:
                    continue
                if max(ra.width, rb.width) / max(1e-6, min(ra.width, rb.width)) > 2.0:
                    continue
                if _angle_diff_180(ra.angle_deg, rb.angle_deg) > 15.0:
                    continue
                d = float(dists[off])
                if d > max(20.0, 14.0 * max(ra.width, rb.width)):
                    continue
                if d >= 0.8:
                    u_ab = diffs[off] / d
                    dot_min = 0.88 if d <= 2.5 * max(ra.width, rb.width) else 0.94
                    if float(np.dot(da, u_ab)) < dot_min or float(np.dot(end_dirs[b], -u_ab)) < dot_min:
                        continue
                    perp = abs(float(diffs[off, 0] * (-da[1]) + diffs[off, 1] * da[0]))
                    if perp > max(1.6, 0.90 * max(ra.width, rb.width)):
                        continue
                    if _ink_fraction_between(pa, end_pts[b], ink) < 0.80:
                        continue
                else:
                    if float(np.dot(da, end_dirs[b])) > -0.88:
                        continue
                cands.append((d, a, b))

        if not cands:
            break
        cands.sort(key=lambda x: x[0])
        linked_end: Dict[int, int] = {}
        adj_r: Dict[int, set[int]] = defaultdict(set)
        for _d, a, b in cands:
            if a in linked_end or b in linked_end:
                continue
            linked_end[a] = b
            linked_end[b] = a
            adj_r[a // 2].add(b // 2)
            adj_r[b // 2].add(a // 2)

        visited: set[int] = set()
        bridged: List[RibbonCandidate] = []
        nid = 0
        for idx in range(nr):
            if idx in visited:
                continue
            if idx not in adj_r:
                visited.add(idx)
                r = ribbons[idx]
                r.cid = nid
                nid += 1
                bridged.append(r)
                continue
            comp: set[int] = set()
            st = [idx]
            while st:
                c = st.pop()
                if c in comp:
                    continue
                comp.add(c)
                st.extend(adj_r[c])
            visited.update(comp)
            leaves = [c for c in comp if len(adj_r[c]) <= 1]
            start_r = min(leaves) if leaves else min(comp)
            l0 = (2 * start_r) in linked_end
            l1 = (2 * start_r + 1) in linked_end
            fwd = (not l0) if (l0 != l1) else True
            pts_list = [pt.copy() for pt in (ribbons[start_r].points if fwd else ribbons[start_r].points[::-1])]
            exit_e = (2 * start_r + 1) if fwd else (2 * start_r)
            walked = {start_r}
            while exit_e in linked_end:
                entry_e = linked_end[exit_e]
                nxt_r = entry_e // 2
                if nxt_r in walked or nxt_r not in comp:
                    break
                nfwd = (entry_e % 2 == 0)
                npts = ribbons[nxt_r].points if nfwd else ribbons[nxt_r].points[::-1]
                gap = float(np.linalg.norm(npts[0] - pts_list[-1]))
                steps = int(math.ceil(gap))
                if steps > 1:
                    b0 = pts_list[-1].copy()
                    for t in range(1, steps):
                        pts_list.append((1.0 - t / steps) * b0 + (t / steps) * npts[0])
                pts_list.extend(pt.copy() for pt in npts[1:])
                walked.add(nxt_r)
                exit_e = (2 * nxt_r + 1) if nfwd else (2 * nxt_r)

            mpts = np.asarray(pts_list, dtype=np.float32)
            mw = sum(ribbons[c].width * ribbons[c].length for c in comp) / max(
                1e-6, sum(ribbons[c].length for c in comp)
            )
            bridged.append(RibbonCandidate(
                cid=nid,
                points=mpts,
                width=float(mw),
                length=_polyline_length(mpts),
                angle_deg=_segment_angle(mpts),
                wall_ids=[wid for c in comp for wid in ribbons[c].wall_ids],
            ))
            nid += 1
        ribbons = bridged

    return ribbons


# ---------------------------------------------------------------------------
# Stratul 1C: Partiție 100% fără pierderi a fiecărui pixel de cerneală
# ---------------------------------------------------------------------------
def partition_ink_100pct(
    ink: np.ndarray,
    ribbons: Sequence[RibbonCandidate],
    scale: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], np.ndarray, np.ndarray, int]:
    """Asignează 100% din pixelii de cerneală la primitive geometrice exacte."""
    h, w = ink.shape
    straight = [
        r for r in ribbons
        if (r.length + 0.35 * r.width) / max(0.7, r.width) >= 1.45 and r.length >= 2.0
    ]
    straight.sort(key=lambda r: r.length, reverse=True)

    claim_count = np.zeros((h, w), dtype=np.int32)
    owner_id = np.full((h, w), -1, dtype=np.int32)
    neighbors: Dict[int, set[int]] = defaultdict(set)

    primitives: List[Dict[str, Any]] = []
    pid_counter = 1

    for r in straight:
        pid = pid_counter
        pid_counter += 1
        rad_px = max(1, int(math.ceil(0.5 * r.width + 0.65)))
        x0 = max(0, int(math.floor(np.min(r.points[:, 0]))) - rad_px - 1)
        y0 = max(0, int(math.floor(np.min(r.points[:, 1]))) - rad_px - 1)
        x1 = min(w, int(math.ceil(np.max(r.points[:, 0]))) + rad_px + 2)
        y1 = min(h, int(math.ceil(np.max(r.points[:, 1]))) + rad_px + 2)

        roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        pts_roi = np.rint(r.points - np.array([x0, y0], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        thick = max(1, int(round(r.width + 1.1)))
        cv2.polylines(roi_mask, [pts_roi], False, 1, thick)
        claimed = (roi_mask > 0) & ink[y0:y1, x0:x1]

        prev_owners = owner_id[y0:y1, x0:x1][claimed]
        for other_pid in np.unique(prev_owners):
            if other_pid > 0:
                neighbors[pid].add(int(other_pid))
                neighbors[int(other_pid)].add(pid)

        claim_count[y0:y1, x0:x1][claimed] += 1
        unowned_here = claimed & (owner_id[y0:y1, x0:x1] < 0)
        owner_id[y0:y1, x0:x1][unowned_here] = pid

        area_px = int(np.count_nonzero(claimed))
        pts_orig = r.points / scale
        primitives.append({
            "id": pid,
            "type": "banda_dreapta",
            "start": [round(float(pts_orig[0, 0]), 2), round(float(pts_orig[0, 1]), 2)],
            "end": [round(float(pts_orig[-1, 0]), 2), round(float(pts_orig[-1, 1]), 2)],
            "center": [round(float(np.mean(pts_orig[:, 0])), 2), round(float(np.mean(pts_orig[:, 1])), 2)],
            "length_px": round(r.length / scale, 2),
            "width_px": round(r.width / scale, 2),
            "elongation": round(r.length / max(0.7, r.width), 2),
            "angle_deg": round(r.angle_deg, 2),
            "area_px": round(area_px / (scale * scale), 1),
            "_pts_work": r.points,
        })

    # Joncțiuni (pixeli revendicați simultan de >= 2 benzi drepte)
    junction_mask = (claim_count >= 2) & ink
    j_cnt, j_lbls, j_stats, j_cents = cv2.connectedComponentsWithStats(junction_mask.astype(np.uint8), 8)
    junctions: List[Dict[str, Any]] = []
    for jl in range(1, j_cnt):
        junctions.append({
            "junction_id": jl,
            "center": [round(float(j_cents[jl, 0]) / scale, 2), round(float(j_cents[jl, 1]) / scale, 2)],
            "area_px": round(int(j_stats[jl, cv2.CC_STAT_AREA]) / (scale * scale), 1),
        })

    # Restul de cerneală -> 100% clasificat în `banda_curba`, `pata_compacta` sau capăt absorbit
    rem_ink = ink & (claim_count == 0)
    c_cnt, c_lbls, c_stats, c_cents = cv2.connectedComponentsWithStats(rem_ink.astype(np.uint8), 8)
    absorbed_px = 0

    for cl in range(1, c_cnt):
        area = int(c_stats[cl, cv2.CC_STAT_AREA])
        bx = int(c_stats[cl, cv2.CC_STAT_LEFT])
        by = int(c_stats[cl, cv2.CC_STAT_TOP])
        bw = int(c_stats[cl, cv2.CC_STAT_WIDTH])
        bh = int(c_stats[cl, cv2.CC_STAT_HEIGHT])
        rx0, ry0 = max(0, bx - 1), max(0, by - 1)
        rx1, ry1 = min(w, bx + bw + 1), min(h, by + bh + 1)

        comp_roi = (c_lbls[ry0:ry1, rx0:rx1] == cl)
        dil = ndi.binary_dilation(comp_roi, structure=np.ones((3, 3), dtype=bool))
        touch_owners = owner_id[ry0:ry1, rx0:rx1][dil & ~comp_roi]
        adj_pids = sorted(set(int(v) for v in np.unique(touch_owners) if v > 0))

        # Așchie minusculă lipită direct pe marginea/capătul unei benzi -> absorbită în bandă
        if adj_pids and area <= int(round(14 * scale * scale)):
            d_from_claimed = ndi.distance_transform_edt(~(claim_count[ry0:ry1, rx0:rx1] > 0))
            max_d = float(np.max(d_from_claimed[comp_roi]))
            if area <= int(round(5 * scale * scale)) or max_d <= 1.5 * scale:
                owner_id[ry0:ry1, rx0:rx1][comp_roi] = adj_pids[0]
                absorbed_px += area
                continue

        pid = pid_counter
        pid_counter += 1
        owner_id[ry0:ry1, rx0:rx1][comp_roi] = pid
        for ap in adj_pids:
            neighbors[pid].add(ap)
            neighbors[ap].add(pid)

        # Invariant geometric pur de scară:
        #   w = 2 * r_max (diametrul cercului maxim înscris)
        #   disc -> A / w^2 = pi/4 ≈ 0.79; oval 1.5:1 -> ≈ 1.18; bandă curbă -> L / w >> 1.85
        comp_pad = np.pad(comp_roi.astype(np.uint8), 1)
        r_comp = ndi.distance_transform_edt(comp_pad)[1:-1, 1:-1]
        r_vals = r_comp[comp_roi]
        r_max = float(np.max(r_vals)) if r_vals.size else 0.5
        w_est = max(1.0, 2.0 * r_max)
        elong_area = float(area) / (w_est * w_est)

        filled = ndi.binary_fill_holes(comp_pad)
        has_hole = int(np.count_nonzero(filled & ~comp_pad)) >= max(2, int(round(2 * scale * scale)))

        ys_c, xs_c = np.nonzero(comp_roi)
        if xs_c.size >= 3:
            pts_c = np.column_stack((xs_c, ys_c)).astype(np.float64)
            pts_c -= np.mean(pts_c, axis=0)
            vals_c, vecs_c = np.linalg.eigh(pts_c.T @ pts_c)
            pca_ratio = math.sqrt(max(1e-6, float(vals_c[1])) / max(1e-6, float(vals_c[0])))
            principal = vecs_c[:, 1]
            ang_c = float(math.degrees(math.atan2(float(principal[1]), float(principal[0]))) % 180.0)
        else:
            pca_ratio = 1.0
            ang_c = 0.0

        if has_hole or elong_area >= 1.85:
            ptype = "banda_curba"
            L_est = float(area) / w_est
            elong = max(1.0, L_est / w_est)
        else:
            ptype = "pata_compacta"
            L_est = max(w_est, w_est * pca_ratio)
            elong = max(1.0, pca_ratio)

        primitives.append({
            "id": pid,
            "type": ptype,
            "center": [round(float(c_cents[cl, 0]) / scale, 2), round(float(c_cents[cl, 1]) / scale, 2)],
            "bbox": [round(bx / scale, 1), round(by / scale, 1),
                     round((bx + bw) / scale, 1), round((by + bh) / scale, 1)],
            "length_px": round(L_est / scale, 2),
            "width_px": round(w_est / scale, 2),
            "elongation": round(elong, 2),
            "area_to_w2": round(elong_area, 2),
            "angle_deg": round(ang_c, 2),
            "has_hole": has_hole,
            "area_px": round(area / (scale * scale), 1),
        })

    for p in primitives:
        p["neighbors"] = sorted(neighbors.get(p["id"], set()))

    # Verificare matematică strictă: 100.00% din cerneală este acoperită!
    covered = owner_id > 0
    uncovered = int(np.count_nonzero(ink & ~covered))
    overcovered = int(np.count_nonzero(~ink & covered))
    if uncovered != 0 or overcovered != 0:
        raise RuntimeError(f"Partiția Stratului 1 nu e exactă: neacoperiți={uncovered}, exces={overcovered}")

    return primitives, junctions, owner_id, claim_count, absorbed_px


# ---------------------------------------------------------------------------
# Debug vizual: 4 imagini complementare
# ---------------------------------------------------------------------------
def _elongation_bgr(elong: float) -> Tuple[int, int, int]:
    """Scarǎ continuǎ fǎrǎ prǎpastii: L/w=1 (albastru) -> 3 (cyan) -> 8 (verde) -> 20+ (galben)."""
    t = min(1.0, max(0.0, math.log2(max(1.0, elong)) / 4.5))
    if t < 0.33:
        k = t / 0.33
        return (int(235 - 35 * k), int(70 + 160 * k), int(40 * (1 - k)))
    if t < 0.66:
        k = (t - 0.33) / 0.33
        return (int(200 * (1 - k)), int(230), int(40 * k))
    k = (t - 0.66) / 0.34
    return (0, int(230), int(40 + 215 * k))


def _draw_debug(
    ink: np.ndarray,
    dss_list: Sequence[DSS],
    primitives: Sequence[Dict[str, Any]],
    owner_id: np.ndarray,
    claim_count: np.ndarray,
    out_dir: Path,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "debug_1_dss_exact.png"
    p2 = out_dir / "debug_2_tipuri.png"
    p3 = out_dir / "debug_3_alungire.png"
    p4 = out_dir / "debug_4_verificare_100pct.png"

    h, w = ink.shape

    # 1) Stratul 0: toate DSS-urile exacte pe fond negru
    v1 = np.zeros((h + 1, w + 1, 3), dtype=np.uint8)
    v1[:h, :w][ink] = (45, 45, 45)
    for s in dss_list:
        cv2.line(v1, (s.x0, s.y0), (s.x1, s.y1), (0, 210, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(p1), v1)

    # 2) Stratul 1: partiția completă colorată după tip geometric pur
    #    banda_dreapta = verde, banda_curba = portocaliu, pata_compacta = albastru, jonctiune = roșu
    max_pid = max((p["id"] for p in primitives), default=0)
    lut_type = np.zeros((max_pid + 1, 3), dtype=np.uint8)
    lut_elong = np.zeros((max_pid + 1, 3), dtype=np.uint8)
    type_palette = {
        "banda_dreapta": (0, 200, 0),
        "banda_curba": (0, 155, 255),
        "pata_compacta": (235, 110, 40),
    }
    for p in primitives:
        pid = p["id"]
        lut_type[pid] = type_palette.get(p["type"], (180, 180, 180))
        lut_elong[pid] = _elongation_bgr(float(p["elongation"]))

    safe_owner = np.maximum(0, owner_id)
    v2 = lut_type[safe_owner]
    v2[claim_count >= 2] = (0, 0, 240)  # joncțiuni cu roșu
    # Desenăm axele sub-pixel ale benzilor drepte ușor mai luminos
    for p in primitives:
        if p["type"] == "banda_dreapta" and "_pts_work" in p:
            pts = np.rint(p["_pts_work"]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(v2, [pts], False, (140, 255, 140), 1, cv2.LINE_AA)
    cv2.imwrite(str(p2), v2)

    # 3) Hartă continuă de alungire L/w (fără prăpastie de prag!)
    v3 = lut_elong[safe_owner]
    v3[claim_count >= 2] = (255, 255, 255)  # intersecții cu alb
    cv2.imwrite(str(p3), v3)

    # 4) Dovada vizuală 100% acoperire (verde = acoperit exact, roșu = eroare)
    covered = owner_id > 0
    xor_err = ink != covered
    v4 = np.zeros((h, w, 3), dtype=np.uint8)
    v4[covered & ~xor_err] = (0, 195, 0)
    v4[xor_err] = (0, 0, 255)
    cv2.imwrite(str(p4), v4)

    return {
        "debug_1_dss_exact": str(p1),
        "debug_2_tipuri": str(p2),
        "debug_3_alungire": str(p3),
        "debug_4_verificare_100pct": str(p4),
    }


def process_image(
    image_path: str | Path,
    out_dir: str | Path = "debug_geo",
    scale: float = 1.0,
) -> Dict[str, Any]:
    image_path = Path(image_path)
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")

    ink, prep_meta = binarize_image(original, scale=scale)

    # Stratul 0: DSS exact (XOR == 0)
    loops = extract_crack_loops(ink)
    dss_list, loop_dss_lists = decompose_loops_to_dss(loops)
    reconstructed, face_dss = build_face_map_and_reconstruct(dss_list, ink.shape)
    dss_xor = int(np.count_nonzero(ink != reconstructed))
    if dss_xor != 0:
        raise RuntimeError(f"Stratul 0 nu e lossless: {dss_xor} pixeli diferență")

    # Stratul 1: Geometrie pură (pereți opuși -> benzi / curbe / pete / joncțiuni)
    walls, dss_to_wall = build_walls_from_dss(dss_list, loop_dss_lists)
    rw, rx0, ry0, rnx, rny = collect_ray_starts(dss_list, dss_to_wall, walls)
    hdist, hwall, hx, hy = raycast_walls_across_ink(
        ink, face_dss, dss_to_wall, rw, rx0, ry0, rnx, rny
    )
    ribbons = extract_straight_ribbons(ink, walls, rw, rx0, ry0, hdist, hwall, hx, hy)
    primitives, junctions, owner_id, claim_count, absorbed_px = partition_ink_100pct(ink, ribbons, scale)

    debug_paths = _draw_debug(ink, dss_list, primitives, owner_id, claim_count, Path(out_dir))

    # Curățăm câmpurile interne numpy înainte de salvarea JSON
    json_primitives = [{k: v for k, v in p.items() if not k.startswith("_")} for p in primitives]
    by_type: Dict[str, int] = defaultdict(int)
    for p in json_primitives:
        by_type[p["type"]] += 1

    summary = {
        "image": str(image_path),
        "scale": scale,
        "preprocessing": prep_meta,
        "lossless_verification": {
            "ink_pixels": int(np.count_nonzero(ink)),
            "layer0_dss_xor_error_pixels": dss_xor,
            "layer1_uncovered_ink_pixels": 0,
            "absorbed_endcap_pixels": absorbed_px,
            "guaranteed_100pct_coverage": True,
        },
        "counts": {
            "dss_segments": len(dss_list),
            "walls": len(walls),
            "primitives_total": len(json_primitives),
            "by_type": dict(by_type),
            "junctions": len(junctions),
        },
        "debug_images": debug_paths,
    }

    out_path = Path(out_dir) / f"{image_path.stem}_geometry_layer1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({**summary, "primitives": json_primitives, "junctions": junctions}, ensure_ascii=False),
        encoding="utf-8",
    )
    summary["json_file"] = str(out_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Motor de geometrie pură Stratul 0 + Stratul 1 (100% lossless)")
    parser.add_argument("image", help="Fotografie PNG/JPG")
    parser.add_argument("--out-dir", default="debug_geo", help="Directorul de ieșire")
    parser.add_argument("--scale", type=float, default=1.0, help="Scalare opțională (1.0 nativ, 2.0 sub-pixel)")
    args = parser.parse_args(argv)
    try:
        res = process_image(args.image, args.out_dir, args.scale)
    except Exception as exc:
        print(f"Eroare: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())