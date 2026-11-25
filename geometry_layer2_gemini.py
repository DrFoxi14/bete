from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _angle_diff_180(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _point_to_segment_xy(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> Tuple[float, float]:
    abx = bx - ax
    aby = by - ay
    l2 = abx * abx + aby * aby
    if l2 < 1e-12:
        return math.hypot(px - ax, py - ay), 0.0
    t = ((px - ax) * abx + (py - ay) * aby) / l2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby)), t


def _point_to_segment(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    return _point_to_segment_xy(float(pt[0]), float(pt[1]), float(a[0]), float(a[1]), float(b[0]), float(b[1]))


def _segments_intersect_xy(
    a0x: float, a0y: float, a1x: float, a1y: float,
    b0x: float, b0y: float, b1x: float, b1y: float,
) -> bool:
    dax = a1x - a0x
    day = a1y - a0y
    dbx = b1x - b0x
    dby = b1y - b0y
    denom = dax * dby - day * dbx
    if abs(denom) <= 1e-12:
        return False
    diffx = b0x - a0x
    diffy = b0y - a0y
    ta = (diffx * dby - diffy * dbx) / denom
    if ta < 0.0 or ta > 1.0:
        return False
    tb = (diffx * day - diffy * dax) / denom
    return 0.0 <= tb <= 1.0


def _prim_poly_tuples(p: Dict[str, Any]) -> List[Tuple[float, float]]:
    if p["type"] == "banda_dreapta":
        ax, ay = float(p["start"][0]), float(p["start"][1])
        bx, by = float(p["end"][0]), float(p["end"][1])
        hw = 0.5 * float(p["width_px"])
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy)
        if L > 1e-9:
            nx, ny = -dy / L * hw, dx / L * hw
            return [(ax + nx, ay + ny), (ax - nx, ay - ny), (bx - nx, by - ny), (bx + nx, by + ny)]
        return [(ax - hw, ay - hw), (ax + hw, ay - hw), (ax + hw, ay + hw), (ax - hw, ay + hw)]
    x0, y0, x1, y1 = [float(v) for v in p["bbox"]]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _point_in_poly_xy(px: float, py: float, pts: Sequence[Tuple[float, float]]) -> bool:
    eps = 1e-11
    pos = False
    neg = False
    m = len(pts)
    for i in range(m):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % m]
        cross = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
        if cross > eps:
            pos = True
        elif cross < -eps:
            neg = True
        if pos and neg:
            return False
    return True


def _convex_poly_distance(pa: Sequence[Tuple[float, float]], pb: Sequence[Tuple[float, float]]) -> float:
    na, nb = len(pa), len(pb)
    ax0 = min(p[0] for p in pa)
    ax1 = max(p[0] for p in pa)
    ay0 = min(p[1] for p in pa)
    ay1 = max(p[1] for p in pa)
    bx0 = min(p[0] for p in pb)
    bx1 = max(p[0] for p in pb)
    by0 = min(p[1] for p in pb)
    by1 = max(p[1] for p in pb)
    if not (ax1 < bx0 - 1e-11 or bx1 < ax0 - 1e-11 or ay1 < by0 - 1e-11 or by1 < ay0 - 1e-11):
        for i in range(na):
            a0x, a0y = pa[i]
            a1x, a1y = pa[(i + 1) % na]
            for j in range(nb):
                b0x, b0y = pb[j]
                b1x, b1y = pb[(j + 1) % nb]
                if _segments_intersect_xy(a0x, a0y, a1x, a1y, b0x, b0y, b1x, b1y):
                    return 0.0
        for p in pa:
            if _point_in_poly_xy(p[0], p[1], pb):
                return 0.0
        for p in pb:
            if _point_in_poly_xy(p[0], p[1], pa):
                return 0.0
    d_min = float("inf")
    for px, py in pa:
        for j in range(nb):
            b0x, b0y = pb[j]
            b1x, b1y = pb[(j + 1) % nb]
            d, _ = _point_to_segment_xy(px, py, b0x, b0y, b1x, b1y)
            if d < d_min:
                d_min = d
    for px, py in pb:
        for i in range(na):
            a0x, a0y = pa[i]
            a1x, a1y = pa[(i + 1) % na]
            d, _ = _point_to_segment_xy(px, py, a0x, a0y, a1x, a1y)
            if d < d_min:
                d_min = d
    return 0.0 if d_min <= 1e-11 else d_min


def _prim_uv_box(p: Dict[str, Any], u_vec: np.ndarray, v_vec: np.ndarray) -> Tuple[float, float, float, float]:
    corners = np.array(_prim_poly_tuples(p), dtype=np.float64)
    us = corners @ u_vec
    vs = corners @ v_vec
    return float(np.min(us)), float(np.min(vs)), float(np.max(us)), float(np.max(vs))


def _prim_xy_box(p: Dict[str, Any]) -> Tuple[float, float, float, float]:
    pts = _prim_poly_tuples(p)
    return min(x for x, _ in pts), min(y for _, y in pts), max(x for x, _ in pts), max(y for _, y in pts)


def _prim_dist_to_point(pt: np.ndarray, p: Dict[str, Any]) -> float:
    if p["type"] == "banda_dreapta":
        a = np.array(p["start"], dtype=np.float64)
        b = np.array(p["end"], dtype=np.float64)
        d, _ = _point_to_segment(pt, a, b)
        return max(0.0, d - 0.5 * float(p["width_px"]))
    x0, y0, x1, y1 = [float(v) for v in p["bbox"]]
    dx = max(x0 - float(pt[0]), 0.0, float(pt[0]) - x1)
    dy = max(y0 - float(pt[1]), 0.0, float(pt[1]) - y1)
    return math.hypot(dx, dy)


def _ribbon_touches_near_end(stick: Dict[str, Any], bridge: Dict[str, Any]) -> Tuple[bool, float]:
    pa = np.array(stick["start"], dtype=np.float64)
    pb = np.array(stick["end"], dtype=np.float64)
    ba = np.array(bridge["start"], dtype=np.float64)
    bb = np.array(bridge["end"], dtype=np.float64)
    da, ta = _point_to_segment(pa, ba, bb)
    db, tb = _point_to_segment(pb, ba, bb)
    half_sum = 0.5 * (float(stick["width_px"]) + float(bridge["width_px"]))
    gap_a = max(0.0, da - half_sum)
    gap_b = max(0.0, db - half_sum)
    end_tol = max(2.0 * float(bridge["width_px"]), 0.25 * float(stick["length_px"]))
    if gap_a <= gap_b:
        return gap_a <= end_tol, ta
    return gap_b <= end_tol, tb


def estimate_dominant_axis(primitives: Sequence[Dict[str, Any]]) -> Tuple[float, np.ndarray, np.ndarray]:
    ribbons = [p for p in primitives if p["type"] == "banda_dreapta" and float(p["elongation"]) >= 3.0]
    if not ribbons:
        ribbons = [p for p in primitives if p["type"] == "banda_dreapta"]
    if not ribbons:
        return 0.0, np.array([1.0, 0.0]), np.array([0.0, 1.0])

    bins = np.zeros(180, dtype=np.float64)
    for r in ribbons:
        ang = float(r["angle_deg"]) % 180.0
        w = float(r["length_px"]) ** 2
        bins[int(round(ang)) % 180] += w

    ext = np.concatenate([bins[-4:], bins, bins[:4]])
    smooth = np.convolve(ext, np.array([0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05]), mode="same")[4:-4]
    peak = int(np.argmax(smooth))

    s_sin, s_cos = 0.0, 0.0
    for r in ribbons:
        ang = float(r["angle_deg"]) % 180.0
        if _angle_diff_180(ang, float(peak)) <= 8.0:
            w = float(r["length_px"]) ** 2
            rad2 = math.radians(2.0 * ang)
            s_sin += w * math.sin(rad2)
            s_cos += w * math.cos(rad2)

    theta_deg = (0.5 * math.degrees(math.atan2(s_sin, s_cos))) % 180.0
    rad = math.radians(theta_deg)
    u_vec = np.array([math.cos(rad), math.sin(rad)], dtype=np.float64)
    v_vec = np.array([-math.sin(rad), math.cos(rad)], dtype=np.float64)
    if u_vec[0] < 0 or (abs(u_vec[0]) < 1e-6 and u_vec[1] < 0):
        u_vec = -u_vec
        v_vec = -v_vec
        theta_deg = math.degrees(math.atan2(u_vec[1], u_vec[0])) % 180.0
    return theta_deg, u_vec, v_vec


def _eval_comb_line(gs: Dict[str, Any], line_idx: int, u: float) -> float:
    us, vs = gs["us_v0"], gs["vs_v0"]
    if u <= us[0]:
        sl = (vs[1] - vs[0]) / max(1e-9, us[1] - us[0]) if len(us) >= 2 else 0.0
        sl = max(-0.18, min(0.18, sl))
        v0 = float(vs[0] + sl * (u - us[0]))
    elif u >= us[-1]:
        sl = (vs[-1] - vs[-2]) / max(1e-9, us[-1] - us[-2]) if len(us) >= 2 else 0.0
        sl = max(-0.18, min(0.18, sl))
        v0 = float(vs[-1] + sl * (u - us[-1]))
    else:
        v0 = float(np.interp(u, us, vs))
    return v0 + gs["line_offsets"][line_idx]


def _merged_len(ivs: List[Tuple[float, float]]) -> float:
    if not ivs:
        return 0.0
    s_ivs = sorted(ivs)
    a, b = s_ivs[0]
    tot = 0.0
    for c, d in s_ivs[1:]:
        if c <= b:
            b = max(b, d)
        else:
            tot += b - a
            a, b = c, d
    return tot + (b - a)


def _interval_overlap_len(a0: float, a1: float, ivs: List[Tuple[float, float]]) -> float:
    clipped = [(max(a0, c), min(a1, d)) for c, d in ivs if min(a1, d) > max(a0, c)]
    return _merged_len(clipped)


def _seg_v_at(s: Dict[str, Any], u: float) -> float:
    du = s["ub"] - s["ua"]
    if du < 1e-9:
        return 0.5 * (s["va"] + s["vb"])
    t = max(0.0, min(1.0, (u - s["ua"]) / du))
    return s["va"] + t * (s["vb"] - s["va"])


def _detect_grids_single_scale(
    pool_aligned: Sequence[Dict[str, Any]],
    curved_candidates: Sequence[Dict[str, Any]],
    page_v_span: float,
    theta_deg: float,
) -> Tuple[List[Dict[str, Any]], set[int], List[Dict[str, Any]]]:
    if len(pool_aligned) < 6:
        return [], set(), []

    long_ws = sorted((s["w"], s["len"]) for s in pool_aligned if s["elong"] >= 8.0 and s["ang_diff"] <= 8.0)
    if not long_ws:
        long_ws = sorted((s["w"], s["len"]) for s in pool_aligned if s["elong"] >= 4.0)
    if not long_ws:
        long_ws = sorted((s["w"], s["len"]) for s in pool_aligned)

    tot_lw = sum(l for _, l in long_ws)
    if tot_lw <= 0.0:
        return [], set(), []
    acc = 0.0
    w_line = long_ws[0][0]
    for w_val, l_val in long_ws:
        acc += l_val
        if acc >= 0.5 * tot_lw:
            w_line = w_val
            break

    max_line_w = 1.70 * w_line
    seed_thin = [s for s in pool_aligned if s["w"] <= max_line_w and s["elong"] >= 3.5]
    thin = [s for s in pool_aligned if s["w"] <= max_line_w]
    if len(seed_thin) < 6:
        return [], set(), []

    max_dv_limit = min(28.0 * w_line, max(20.0 * w_line, 0.20 * page_v_span))
    min_dv_limit = 2.2 * w_line

    thin_u = sorted(seed_thin, key=lambda s: s["ua"])
    local_dvs: List[Tuple[float, float]] = []
    n_t = len(thin_u)
    for i in range(n_t):
        sa = thin_u[i]
        if sa["elong"] < 4.0:
            continue
        best_below = float("inf")
        best_w = 0.0
        for j in range(n_t):
            if i == j:
                continue
            sb = thin_u[j]
            if sb["ub"] <= sa["ua"]:
                continue
            if sb["ua"] >= sa["ub"]:
                break
            ov_lo = max(sa["ua"], sb["ua"])
            ov_hi = min(sa["ub"], sb["ub"])
            ov = ov_hi - ov_lo
            if ov < 0.35 * min(sa["len"], sb["len"]):
                continue
            u_m = 0.5 * (ov_lo + ov_hi)
            dv = _seg_v_at(sb, u_m) - _seg_v_at(sa, u_m)
            if min_dv_limit < dv < best_below:
                best_below = dv
                best_w = ov
        if best_below < max_dv_limit:
            local_dvs.append((best_below, best_w))

    if len(local_dvs) < 4:
        return [], set(), []

    local_dvs.sort(key=lambda t: t[0])
    best_score = -1.0
    best_cluster: List[Tuple[float, float]] = []
    for g, _ in local_dvs:
        tol = 0.14 * g
        cl = [(v, wt) for v, wt in local_dvs if abs(v - g) <= tol]
        sc = sum(wt for _, wt in cl)
        if sc > best_score:
            best_score = sc
            best_cluster = cl

    best_cluster.sort(key=lambda t: t[0])
    tot_cw = sum(wt for _, wt in best_cluster)
    acc = 0.0
    step_s = best_cluster[0][0]
    for v_val, wt in best_cluster:
        acc += wt
        if acc >= 0.5 * tot_cw:
            step_s = v_val
            break

    if tot_cw < 4.0 * step_s:
        return [], set(), []

    u_min_all = min(s["ua"] for s in seed_thin)
    u_max_all = max(s["ub"] for s in seed_thin)
    slice_w = 3.5 * step_s
    slice_step = 0.5 * slice_w
    n_slices = max(1, int(math.ceil((u_max_all - u_min_all) / max(1e-9, slice_step))) + 1)
    slice_centers = [u_min_all + m * slice_step for m in range(n_slices)]
    peak_tol = min(0.32 * step_s, 1.35 * w_line)
    step_tol = 0.22 * step_s

    slice_combs: List[List[Dict[str, Any]]] = []
    for m_idx, u_m in enumerate(slice_centers):
        sl_lo = u_m - 0.5 * slice_w
        sl_hi = u_m + 0.5 * slice_w
        samples: List[Tuple[float, float, float, float, float]] = []
        for s in seed_thin:
            if s["ub"] < sl_lo or s["ua"] > sl_hi:
                continue
            ov = min(s["ub"], sl_hi) - max(s["ua"], sl_lo)
            if ov <= 0.0:
                continue
            u_eval = max(s["ua"], min(s["ub"], u_m))
            sl_s = (s["vb"] - s["va"]) / max(1e-9, s["ub"] - s["ua"])
            samples.append((_seg_v_at(s, u_eval), ov, max(s["ua"], sl_lo), min(s["ub"], sl_hi), sl_s))

        if len(samples) < 3:
            slice_combs.append([])
            continue

        samples.sort(key=lambda t: t[0])
        peaks: List[Tuple[float, float, float, float, float]] = []
        cur_pts = [samples[0]]
        for item in samples[1:]:
            cur_mean = sum(x[0] * x[1] for x in cur_pts) / sum(x[1] for x in cur_pts)
            if abs(item[0] - cur_mean) <= peak_tol:
                cur_pts.append(item)
            else:
                tw = sum(x[1] for x in cur_pts)
                peaks.append((
                    sum(x[0] * x[1] for x in cur_pts) / tw, tw,
                    min(x[2] for x in cur_pts), max(x[3] for x in cur_pts),
                    sum(x[4] * x[1] for x in cur_pts) / tw,
                ))
                cur_pts = [item]
        tw = sum(x[1] for x in cur_pts)
        peaks.append((
            sum(x[0] * x[1] for x in cur_pts) / tw, tw,
            min(x[2] for x in cur_pts), max(x[3] for x in cur_pts),
            sum(x[4] * x[1] for x in cur_pts) / tw,
        ))

        combs_here: List[Dict[str, Any]] = []
        used_p: set[int] = set()
        for i in range(len(peaks)):
            if i in used_p:
                continue
            obs: Dict[int, Tuple[float, float]] = {0: (peaks[i][0], peaks[i][1])}
            used_local = [i]
            cur_k = 0
            cur_v = peaks[i][0]
            for j in range(i + 1, len(peaks)):
                if j in used_p:
                    continue
                dv = peaks[j][0] - cur_v
                if dv > 2.35 * step_s:
                    break
                if abs(dv - step_s) <= step_tol:
                    cur_k += 1
                    cur_v = peaks[j][0]
                    obs[cur_k] = (peaks[j][0], peaks[j][1])
                    used_local.append(j)
                elif abs(dv - 2.0 * step_s) <= 1.35 * step_tol:
                    cur_k += 2
                    cur_v = peaks[j][0]
                    obs[cur_k] = (peaks[j][0], peaks[j][1])
                    used_local.append(j)
            if len(obs) >= 3:
                for idx_u in used_local:
                    used_p.add(idx_u)
                v0 = float(np.median([v_k - k * step_s for k, (v_k, _) in obs.items()]))
                loc_sl = max(-0.22, min(0.22, float(np.median([peaks[x][4] for x in used_local]))))
                combs_here.append({
                    "slice_idx": m_idx, "u": u_m,
                    "u_lo": min(peaks[x][2] for x in used_local),
                    "u_hi": max(peaks[x][3] for x in used_local),
                    "v0": v0, "slope": loc_sl, "obs": obs,
                    "weight": sum(wt for _, wt in obs.values()),
                })
        slice_combs.append(combs_here)

    used_comb: set[Tuple[int, int]] = set()
    mid_slice = n_slices // 2
    seed_order = sorted(range(n_slices), key=lambda m: abs(m - mid_slice))
    comb_tracks: List[Dict[str, Any]] = []
    max_slice_gap = 8

    def _extend_direction(tr: Dict[str, Any], direction: int) -> None:
        while True:
            end_pt = tr["pts"][-1] if direction == 1 else tr["pts"][0]
            end_slice, end_u, end_v0, end_sl = end_pt
            best_cand = None
            best_dk = 0
            best_err = float("inf")
            for step_idx in range(1, max_slice_gap + 1):
                m_next = end_slice + direction * step_idx
                if m_next < 0 or m_next >= n_slices:
                    break
                for c_idx, comb in enumerate(slice_combs[m_next]):
                    if (m_next, c_idx) in used_comb:
                        continue
                    du = comb["u"] - end_u
                    pred_v0 = end_v0 + 0.5 * (end_sl + comb["slope"]) * du
                    dk = int(round((comb["v0"] - pred_v0) / step_s))
                    cand_v0 = comb["v0"] - dk * step_s
                    err = abs(cand_v0 - pred_v0)
                    match_tol = min(0.42 * step_s, (0.24 + 0.04 * step_idx) * step_s)
                    if err <= match_tol and err < best_err:
                        g_ks = {k + dk for k in comb["obs"]}
                        if len(g_ks & tr["core_ks"]) >= 2:
                            best_err = err
                            best_cand = (m_next, c_idx, comb, cand_v0)
                            best_dk = dk
                if best_cand is not None:
                    break
            if best_cand is None:
                break
            m_next, c_idx, comb, v0_glob = best_cand
            used_comb.add((m_next, c_idx))
            new_pt = (m_next, comb["u"], v0_glob, comb["slope"])
            if direction == 1:
                tr["pts"].append(new_pt)
            else:
                tr["pts"].insert(0, new_pt)
            tr["u_lo"] = min(tr["u_lo"], comb["u_lo"])
            tr["u_hi"] = max(tr["u_hi"], comb["u_hi"])
            for k_loc, (v_val, wt) in comb["obs"].items():
                k_g = k_loc + best_dk
                tr["line_obs"][k_g].append((comb["u"], v_val, wt))
            if len(tr["pts"]) >= 4:
                n_s = len(tr["pts"])
                core = {k for k, ol in tr["line_obs"].items() if len(ol) >= max(2, int(0.25 * n_s))}
                if len(core) >= 3:
                    tr["core_ks"] = core
            else:
                tr["core_ks"] |= {k + best_dk for k in comb["obs"]}

    for m_seed in seed_order:
        for c_idx, comb in enumerate(slice_combs[m_seed]):
            if (m_seed, c_idx) in used_comb:
                continue
            used_comb.add((m_seed, c_idx))
            line_obs: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
            for k_loc, (v_val, wt) in comb["obs"].items():
                line_obs[k_loc].append((comb["u"], v_val, wt))
            tr = {
                "pts": [(m_seed, comb["u"], comb["v0"], comb["slope"])],
                "u_lo": comb["u_lo"], "u_hi": comb["u_hi"],
                "core_ks": set(comb["obs"].keys()),
                "line_obs": line_obs,
            }
            _extend_direction(tr, +1)
            _extend_direction(tr, -1)
            comb_tracks.append(tr)

    if not comb_tracks:
        return [], set(), []

    valid_tracks = [
        tr for tr in comb_tracks
        if (tr["u_hi"] - tr["u_lo"]) >= 4.0 * step_s and len(tr["pts"]) >= 2
    ]
    if not valid_tracks:
        return [], set(), []

    valid_tracks.sort(key=lambda tr: float(np.median([p[2] for p in tr["pts"]])))

    grid_specs: List[Dict[str, Any]] = []
    for tr in valid_tracks:
        n_tr_slices = len(tr["pts"])
        kept_ks = sorted(
            k for k, obs_l in tr["line_obs"].items()
            if len(obs_l) >= max(2, int(math.ceil(0.18 * n_tr_slices)))
        )
        if len(kept_ks) < 3:
            continue
        full_ks = list(range(min(kept_ks), max(kept_ks) + 1))
        us_v0 = np.array([p[1] for p in tr["pts"]], dtype=np.float64)
        vs_v0 = np.array([p[2] for p in tr["pts"]], dtype=np.float64)
        if len(vs_v0) >= 3:
            vs_sm = vs_v0.copy()
            vs_sm[1:-1] = 0.25 * vs_v0[:-2] + 0.50 * vs_v0[1:-1] + 0.25 * vs_v0[2:]
            vs_v0 = vs_sm

        line_offsets: List[float] = []
        for k in full_ks:
            obs_l = tr["line_obs"].get(k, [])
            if obs_l:
                resids = [v_val - (float(np.interp(u_val, us_v0, vs_v0)) + k * step_s) for u_val, v_val, _ in obs_l]
                dk_off = max(-0.15 * step_s, min(0.15 * step_s, float(np.median(resids))))
            else:
                dk_off = 0.0
            line_offsets.append(k * step_s + dk_off)

        grid_specs.append({
            "us_v0": us_v0,
            "vs_v0": vs_v0,
            "line_offsets": line_offsets,
            "u_lo_comb": float(tr["u_lo"]),
            "u_hi_comb": float(tr["u_hi"]),
            "line_pids": [[] for _ in line_offsets],
            "step_est": step_s,
        })

    if not grid_specs:
        return [], set(), []

    end_tol = min(0.44 * step_s, 2.0 * w_line)
    mean_tol = min(0.34 * step_s, 1.60 * w_line)
    assigned_pids: set[int] = set()

    for s in thin:
        best_match = None
        best_score = float("inf")
        is_long_chord = (s["len"] >= 6.0 * step_s and s["w"] <= 1.50 * w_line)
        s_end_tol = 0.48 * step_s if is_long_chord else end_tol
        s_mean_tol = 0.42 * step_s if is_long_chord else mean_tol
        for gs in grid_specs:
            if s["ub"] < gs["u_lo_comb"] - 0.5 * step_s or s["ua"] > gs["u_hi_comb"] + 0.5 * step_s:
                continue
            if s["ua"] < gs["u_lo_comb"] - 1.5 * step_s or s["ub"] > gs["u_hi_comb"] + 1.5 * step_s:
                continue
            errs_per_line: List[Tuple[float, float, int]] = []
            for li in range(len(gs["line_offsets"])):
                ea = abs(s["va"] - _eval_comb_line(gs, li, s["ua"]))
                eb = abs(s["vb"] - _eval_comb_line(gs, li, s["ub"]))
                pts_err = [ea, eb]
                for frac in (0.25, 0.5, 0.75):
                    uf = s["ua"] + frac * (s["ub"] - s["ua"])
                    vf = s["va"] + frac * (s["vb"] - s["va"])
                    pts_err.append(abs(vf - _eval_comb_line(gs, li, uf)))
                errs_per_line.append((float(np.mean(pts_err)), max(ea, eb), li))
            errs_per_line.sort(key=lambda t: t[0])
            b_mean, b_end_max, b_li = errs_per_line[0]
            sec_mean = errs_per_line[1][0] if len(errs_per_line) > 1 else float("inf")
            if b_end_max <= s_end_tol and b_mean <= s_mean_tol and (sec_mean - b_mean) >= 0.25 * step_s:
                if b_mean < best_score:
                    best_score = b_mean
                    best_match = (gs, b_li)
        if best_match is not None:
            best_match[0]["line_pids"][best_match[1]].append(s["id"])
            assigned_pids.add(s["id"])

    seg_by_id: Dict[int, Dict[str, Any]] = {s["id"]: s for s in thin}

    for cs in curved_candidates:
        if cs["id"] in assigned_pids:
            continue
        if cs["w"] > max_line_w or cs["dv"] > min(0.42 * step_s, 1.85 * w_line):
            continue
        if cs["du"] < 1.2 * step_s or cs["elong"] < 4.0:
            continue
        best_c_match = None
        best_c_err = float("inf")
        u_mid_c = 0.5 * (cs["ua"] + cs["ub"])
        v_mid_c = 0.5 * (cs["va"] + cs["vb"])
        for gs in grid_specs:
            if cs["ub"] < gs["u_lo_comb"] - 0.5 * step_s or cs["ua"] > gs["u_hi_comb"] + 0.5 * step_s:
                continue
            for li in range(len(gs["line_offsets"])):
                lv_a = _eval_comb_line(gs, li, cs["ua"])
                lv_m = _eval_comb_line(gs, li, u_mid_c)
                lv_b = _eval_comb_line(gs, li, cs["ub"])
                lv_lo = min(lv_a, lv_m, lv_b)
                lv_hi = max(lv_a, lv_m, lv_b)
                err_m = abs(v_mid_c - lv_m)
                err_span = max(abs(cs["va"] - lv_lo), abs(cs["vb"] - lv_hi))
                if err_m <= mean_tol and err_span <= end_tol and err_m < best_c_err:
                    existing_ivs = [(seg_by_id[pid]["ua"], seg_by_id[pid]["ub"]) for pid in gs["line_pids"][li]]
                    ov_ratio = _interval_overlap_len(cs["ua"], cs["ub"], existing_ivs) / max(1e-6, cs["du"])
                    if ov_ratio <= 0.25:
                        best_c_err = err_m
                        best_c_match = (gs, li)
        if best_c_match is not None:
            best_c_match[0]["line_pids"][best_c_match[1]].append(cs["id"])
            seg_by_id[cs["id"]] = cs
            assigned_pids.add(cs["id"])

    grids: List[Dict[str, Any]] = []
    kept_specs: List[Dict[str, Any]] = []

    for gs in grid_specs:
        g_span = max(1e-9, gs["u_hi_comb"] - gs["u_lo_comb"])
        line_covs = [
            _merged_len([(seg_by_id[pid]["ua"], seg_by_id[pid]["ub"]) for pid in lp]) / g_span
            for lp in gs["line_pids"]
        ]
        while len(gs["line_pids"]) > 3 and line_covs and line_covs[0] < 0.22:
            for pid in gs["line_pids"].pop(0):
                assigned_pids.discard(pid)
            gs["line_offsets"].pop(0)
            line_covs.pop(0)
        while len(gs["line_pids"]) > 3 and line_covs and line_covs[-1] < 0.22:
            for pid in gs["line_pids"].pop():
                assigned_pids.discard(pid)
            gs["line_offsets"].pop()
            line_covs.pop()

        all_p = sorted(pid for lp in gs["line_pids"] for pid in lp)
        if not all_p:
            continue

        segs_g = sorted([seg_by_id[pid] for pid in all_p], key=lambda s: s["ua"])
        clusters: List[List[Dict[str, Any]]] = [[segs_g[0]]]
        cur_max_u = segs_g[0]["ub"]
        for sg in segs_g[1:]:
            if sg["ua"] - cur_max_u > 2.5 * step_s:
                clusters.append([sg])
                cur_max_u = sg["ub"]
            else:
                clusters[-1].append(sg)
                cur_max_u = max(cur_max_u, sg["ub"])
        if len(clusters) > 1:
            main_cl = max(clusters, key=lambda cl: max(s["ub"] for s in cl) - min(s["ua"] for s in cl))
            keep_ids = {s["id"] for s in main_cl}
            for li in range(len(gs["line_pids"])):
                for pid in gs["line_pids"][li]:
                    if pid not in keep_ids:
                        assigned_pids.discard(pid)
                gs["line_pids"][li] = [pid for pid in gs["line_pids"][li] if pid in keep_ids]
            all_p = sorted(keep_ids)

        u_lo = min(seg_by_id[pid]["ua"] for pid in all_p)
        u_hi = max(seg_by_id[pid]["ub"] for pid in all_p)
        final_span = max(1e-9, u_hi - u_lo)
        final_covs = [
            _merged_len([(seg_by_id[pid]["ua"], seg_by_id[pid]["ub"]) for pid in lp]) / final_span
            for lp in gs["line_pids"]
        ]
        strong_lines = sum(1 for c in final_covs if c >= 0.40)
        tot_line_cov = sum(final_covs)
        if final_span < 4.0 * step_s or strong_lines < 3 or tot_line_cov < 2.2:
            for pid in all_p:
                assigned_pids.discard(pid)
            continue

        all_p_set = set(all_p)
        mid_spaces_hit = 0
        for li in range(len(gs["line_offsets"]) - 1):
            mid_ivs: List[Tuple[float, float]] = []
            for s_cand in pool_aligned:
                if s_cand["id"] in all_p_set or s_cand["w"] > max_line_w:
                    continue
                if s_cand["ub"] < u_lo or s_cand["ua"] > u_hi:
                    continue
                v_ma = 0.5 * (_eval_comb_line(gs, li, s_cand["ua"]) + _eval_comb_line(gs, li + 1, s_cand["ua"]))
                v_mb = 0.5 * (_eval_comb_line(gs, li, s_cand["ub"]) + _eval_comb_line(gs, li + 1, s_cand["ub"]))
                if max(abs(s_cand["va"] - v_ma), abs(s_cand["vb"] - v_mb)) <= 0.22 * step_s:
                    ca, cb = max(u_lo, s_cand["ua"]), min(u_hi, s_cand["ub"])
                    if cb > ca:
                        mid_ivs.append((ca, cb))
            if _merged_len(mid_ivs) / final_span >= 0.35:
                mid_spaces_hit += 1
        if mid_spaces_hit >= 2:
            for pid in all_p:
                assigned_pids.discard(pid)
            continue

        u_mid = 0.5 * (u_lo + u_hi)
        line_vs = [_eval_comb_line(gs, li, u_mid) for li in range(len(gs["line_offsets"]))]
        steps = [line_vs[k + 1] - line_vs[k] for k in range(len(line_vs) - 1)]
        grids.append({
            "grid_id": 0,
            "line_count": len(gs["line_offsets"]),
            "step_px": round(float(np.median(steps)), 2),
            "angle_deg": round(theta_deg, 2),
            "u_span": [round(u_lo, 2), round(u_hi, 2)],
            "v_span": [round(line_vs[0], 2), round(line_vs[-1], 2)],
            "line_v_coords": [round(v, 2) for v in line_vs],
            "line_primitive_ids": [sorted(lp) for lp in gs["line_pids"]],
            "primitive_ids": all_p,
        })
        kept_specs.append(gs)

    return grids, assigned_pids, kept_specs


def detect_periodic_grids(
    primitives: Sequence[Dict[str, Any]],
    theta_deg: float,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
) -> Tuple[List[Dict[str, Any]], set[int], Dict[int, Dict[str, Any]]]:
    raw_aligned: List[Dict[str, Any]] = []
    raw_curved: List[Dict[str, Any]] = []
    all_vs: List[float] = []
    for p in primitives:
        if p["type"] == "banda_dreapta":
            pa = np.array(p["start"], dtype=np.float64)
            pb = np.array(p["end"], dtype=np.float64)
            ua, va = float(np.dot(pa, u_vec)), float(np.dot(pa, v_vec))
            ub, vb = float(np.dot(pb, u_vec)), float(np.dot(pb, v_vec))
            all_vs.extend([va, vb])
            ang_d = _angle_diff_180(float(p["angle_deg"]), theta_deg)
            if ang_d > 18.0 or float(p["elongation"]) < 2.5:
                continue
            if ub < ua:
                ua, ub, va, vb = ub, ua, vb, va
            raw_aligned.append({
                "id": int(p["id"]),
                "ua": ua, "va": va, "ub": ub, "vb": vb,
                "len": float(p["length_px"]), "w": float(p["width_px"]),
                "elong": float(p["elongation"]), "ang_diff": ang_d,
            })
        else:
            u0, v0, u1, v1 = _prim_uv_box(p, u_vec, v_vec)
            all_vs.extend([v0, v1])
            if p["type"] == "banda_curba":
                ang_d = _angle_diff_180(float(p["angle_deg"]), theta_deg)
                if ang_d <= 18.0 and float(p["elongation"]) >= 4.0:
                    raw_curved.append({
                        "id": int(p["id"]),
                        "ua": u0, "va": v0, "ub": u1, "vb": v1,
                        "du": u1 - u0, "dv": v1 - v0,
                        "len": float(p["length_px"]), "w": float(p["width_px"]),
                        "elong": float(p["elongation"]), "ang_diff": ang_d,
                    })

    if len(raw_aligned) < 6 or not all_vs:
        return [], set(), {}

    page_v_span = max(1e-6, max(all_vs) - min(all_vs))
    all_grids: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    assigned_total: set[int] = set()
    remaining = raw_aligned
    remaining_curved = raw_curved

    for _scale_pass in range(4):
        found_grids, found_pids, found_specs = _detect_grids_single_scale(
            remaining, remaining_curved, page_v_span, theta_deg
        )
        if not found_grids:
            break
        for g, sp in zip(found_grids, found_specs):
            all_grids.append((g, sp))
        assigned_total.update(found_pids)
        remaining = [s for s in remaining if s["id"] not in assigned_total]
        remaining_curved = [s for s in remaining_curved if s["id"] not in assigned_total]
        if len(remaining) < 6:
            break

    all_grids.sort(key=lambda pair: (pair[0]["v_span"][0], pair[0]["u_span"][0]))
    grids_out: List[Dict[str, Any]] = []
    grid_curves_by_id: Dict[int, Dict[str, Any]] = {}
    for gid, (g, sp) in enumerate(all_grids, start=1):
        g["grid_id"] = gid
        grids_out.append(g)
        grid_curves_by_id[gid] = sp

    return grids_out, assigned_total, grid_curves_by_id


def analyze_layer2(layer1: Dict[str, Any]) -> Dict[str, Any]:
    l1_ver = layer1.get("lossless_verification")
    if isinstance(l1_ver, dict):
        if int(l1_ver.get("layer0_dss_xor_error_pixels", 0)) != 0:
            raise RuntimeError("Poarta Strat 0 picată: XOR != 0")
        if int(l1_ver.get("layer1_uncovered_ink_pixels", 0)) != 0:
            raise RuntimeError("Poarta Strat 1 picată: uncovered != 0")
        if l1_ver.get("guaranteed_100pct_coverage") is False:
            raise RuntimeError("Poarta Strat 1 picată: guaranteed_100pct_coverage == False")

    primitives: List[Dict[str, Any]] = layer1["primitives"]
    id_counts = Counter(int(p["id"]) for p in primitives)
    dups_l1 = sorted(pid for pid, cnt in id_counts.items() if cnt != 1)
    if dups_l1:
        raise RuntimeError(f"Stratul 1 conține ID-uri duplicate: {dups_l1[:10]}")

    by_id: Dict[int, Dict[str, Any]] = {int(p["id"]): p for p in primitives}
    all_ids = set(by_id.keys())

    theta_deg, u_vec, v_vec = estimate_dominant_axis(primitives)
    uv_boxes = {pid: _prim_uv_box(p, u_vec, v_vec) for pid, p in by_id.items()}
    poly_xy = {pid: _prim_poly_tuples(p) for pid, p in by_id.items()}

    grids, grid_pids, grid_curves = detect_periodic_grids(primitives, theta_deg, u_vec, v_vec)
    page_step = float(np.median([g["step_px"] for g in grids])) if grids else None
    if page_step is not None:
        s_default = page_step
    elif primitives:
        med_dim = float(np.median([max(float(p["length_px"]), float(p["width_px"])) for p in primitives]))
        s_default = max(1e-6, med_dim)
    else:
        s_default = 1.0

    def grid_line_vs_at(g: Dict[str, Any], u_val: float) -> List[float]:
        gs = grid_curves[g["grid_id"]]
        return [_eval_comb_line(gs, li, u_val) for li in range(len(gs["line_offsets"]))]

    def grid_v_span_at(g: Dict[str, Any], u_val: float) -> Tuple[float, float]:
        gs = grid_curves[g["grid_id"]]
        return _eval_comb_line(gs, 0, u_val), _eval_comb_line(gs, len(gs["line_offsets"]) - 1, u_val)

    def prim_overlapping_grids(pid: int) -> List[int]:
        u0, v0, u1, v1 = uv_boxes[pid]
        um = 0.5 * (u0 + u1)
        res = []
        for g in grids:
            gs_step = float(g["step_px"])
            gu0, gu1 = g["u_span"]
            gv0, gv1 = grid_v_span_at(g, um)
            if (u1 >= gu0 - 0.15 * gs_step and u0 <= gu1 + 0.15 * gs_step and
                    v1 >= gv0 - 0.75 * gs_step and v0 <= gv1 + 0.75 * gs_step):
                res.append(g["grid_id"])
        return res

    non_grid_ids = sorted(all_ids - grid_pids)
    adj: Dict[int, set[int]] = {pid: set() for pid in non_grid_ids}
    for pid in non_grid_ids:
        for nb in by_id[pid].get("neighbors", []):
            nb = int(nb)
            if nb in adj and nb != pid:
                adj[pid].add(nb)
                adj[nb].add(pid)

    visited: set[int] = set()
    raw_molecules: List[List[int]] = []
    prim_to_mol_id: Dict[int, int] = {}
    for pid in non_grid_ids:
        if pid in visited:
            continue
        comp: List[int] = []
        stack = [pid]
        visited.add(pid)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        comp.sort()
        mid = len(raw_molecules) + 1
        raw_molecules.append(comp)
        for x in comp:
            prim_to_mol_id[x] = mid

    near_touch_adj: Dict[int, set[int]] = {pid: set(adj[pid]) for pid in non_grid_ids}
    near_touch_tol = 0.25 * s_default
    sorted_ng_u = sorted(non_grid_ids, key=lambda pid: uv_boxes[pid][0])
    n_ng = len(sorted_ng_u)
    for i in range(n_ng):
        pi = sorted_ng_u[i]
        bi = uv_boxes[pi]
        for j in range(i + 1, n_ng):
            pj = sorted_ng_u[j]
            bj = uv_boxes[pj]
            if bj[0] - bi[2] > near_touch_tol:
                break
            if bj[1] - bi[3] > near_touch_tol or bi[1] - bj[3] > near_touch_tol:
                continue
            if pj in near_touch_adj[pi]:
                continue
            if _convex_poly_distance(poly_xy[pi], poly_xy[pj]) <= near_touch_tol:
                near_touch_adj[pi].add(pj)
                near_touch_adj[pj].add(pi)

    def is_blob_like(p_obj: Dict[str, Any], s_ref: float) -> bool:
        if p_obj["type"] in ("pata_compacta", "banda_curba"):
            ub = uv_boxes[int(p_obj["id"])]
            return (ub[2] - ub[0]) >= 0.45 * s_ref and (ub[3] - ub[1]) >= 0.40 * s_ref
        return float(p_obj["width_px"]) >= 0.45 * s_ref and float(p_obj["elongation"]) <= 2.4

    def is_bar_blocker(nb_obj: Dict[str, Any], s_ref: float) -> bool:
        if is_blob_like(nb_obj, s_ref):
            return True
        ub = uv_boxes[int(nb_obj["id"])]
        nb_du, nb_dv = ub[2] - ub[0], ub[3] - ub[1]
        if nb_obj["type"] in ("pata_compacta", "banda_curba"):
            if nb_du >= 0.65 * s_ref and nb_dv >= 0.25 * s_ref:
                return True
        elif nb_obj["type"] == "banda_dreapta":
            ang_r = _angle_diff_180(float(nb_obj["angle_deg"]), theta_deg)
            if ang_r <= 40.0 and float(nb_obj["width_px"]) >= 0.36 * s_ref and float(nb_obj["length_px"]) >= 1.2 * s_ref:
                return True
            if 25.0 <= ang_r <= 65.0 and float(nb_obj["length_px"]) >= 0.60 * s_ref and nb_du >= 0.45 * s_ref:
                return True
        return False

    transverse_bars: List[Dict[str, Any]] = []
    bar_pids_found: set[int] = set()
    bar_pids_by_mol: Dict[int, List[int]] = defaultdict(list)

    if grids:
        max_bar_s = max(float(g["step_px"]) for g in grids)
        verts = []
        for pid in non_grid_ids:
            p = by_id[pid]
            if p["type"] != "banda_dreapta":
                continue
            u0, v0, u1, v1 = uv_boxes[pid]
            du_p, dv_p = u1 - u0, v1 - v0
            ang_r = _angle_diff_180(float(p["angle_deg"]), theta_deg)
            is_anchor = (
                74.0 <= ang_r <= 106.0
                and float(p["elongation"]) >= 2.5
                and float(p["width_px"]) <= 0.55 * max_bar_s
            )
            is_stub = (
                58.0 <= ang_r <= 122.0
                and float(p["elongation"]) >= 1.5
                and float(p["width_px"]) <= 0.40 * max_bar_s
                and du_p <= 0.50 * max_bar_s
                and dv_p >= 0.25 * max_bar_s
            )
            if not (is_anchor or is_stub):
                continue
            verts.append({
                "id": pid,
                "u": 0.5 * (u0 + u1),
                "v0": v0,
                "v1": v1,
                "w": float(p["width_px"]),
                "anchor": is_anchor,
            })

        bar_pid_spanned: Dict[int, set[int]] = defaultdict(set)
        bar_pid_group: Dict[int, int] = {}
        next_bar_grp = 1

        for g in grids:
            gid = g["grid_id"]
            S_g = float(g["step_px"])
            u0_g, u1_g = g["u_span"]
            cands = []
            for vr in verts:
                if vr["w"] > 0.55 * S_g:
                    continue
                if vr["u"] < u0_g - 1.2 * S_g or vr["u"] > u1_g + 1.2 * S_g:
                    continue
                gv0, gv1 = grid_v_span_at(g, vr["u"])
                if vr["v1"] < gv0 - 0.5 * S_g or vr["v0"] > gv1 + 0.5 * S_g:
                    continue
                cands.append(vr)

            cands.sort(key=lambda x: x["u"])
            clusters: List[List[Dict[str, Any]]] = []
            for vr in cands:
                if clusters and abs(vr["u"] - float(np.mean([x["u"] for x in clusters[-1]]))) <= 0.32 * S_g:
                    clusters[-1].append(vr)
                else:
                    clusters.append([vr])

            for cl in clusters:
                if not any(x["anchor"] for x in cl):
                    continue
                u_m = float(np.mean([x["u"] for x in cl]))
                lvs = grid_line_vs_at(g, u_m)
                gv0, gv1 = lvs[0], lvs[-1]
                gh = max(1e-9, gv1 - gv0)
                v_lo = min(x["v0"] for x in cl)
                v_hi = max(x["v1"] for x in cl)

                n_spaces = max(1, len(lvs) - 1)
                spaces_hit = 0
                for sp_i in range(n_spaces):
                    sv0, sv1 = lvs[sp_i], lvs[sp_i + 1]
                    sh = max(1e-6, sv1 - sv0)
                    s_ivs = [
                        (max(sv0, x["v0"]), min(sv1, x["v1"]))
                        for x in cl if min(sv1, x["v1"]) > max(sv0, x["v0"])
                    ]
                    if _merged_len(s_ivs) >= 0.30 * sh:
                        spaces_hit += 1

                ivs = sorted([
                    (max(gv0, x["v0"]), min(gv1, x["v1"]))
                    for x in cl if min(gv1, x["v1"]) > max(gv0, x["v0"])
                ])
                if not ivs:
                    continue
                a, b = ivs[0]
                cov_ink = 0.0
                max_gap = max(0.0, a - gv0)
                for c, d in ivs[1:]:
                    if c <= b:
                        b = max(b, d)
                    else:
                        max_gap = max(max_gap, c - b)
                        cov_ink += b - a
                        a, b = c, d
                cov_ink += b - a
                max_gap = max(max_gap, gv1 - b)

                pids_cl = [x["id"] for x in cl]
                pids_cl_set = set(pids_cl)
                is_system_span = (v_hi - v_lo) > 8.0 * S_g
                has_blocker = False if is_system_span else any(
                    is_bar_blocker(by_id[nb], S_g)
                    for pid in pids_cl
                    for nb in near_touch_adj[pid]
                    if nb not in pids_cl_set
                )
                overshoot = False if is_system_span else (v_lo < gv0 - 0.65 * S_g or v_hi > gv1 + 0.65 * S_g)
                ok_cov = (
                    (cov_ink >= 0.72 * gh and max_gap <= 0.70 * S_g)
                    or (spaces_hit == n_spaces and cov_ink >= 0.50 * gh and max_gap <= 0.85 * S_g)
                )
                if ok_cov and not has_blocker and not overshoot:
                    existing_grps = {bar_pid_group[pid] for pid in pids_cl if pid in bar_pid_group}
                    grp_id = min(existing_grps) if existing_grps else next_bar_grp
                    if not existing_grps:
                        next_bar_grp += 1
                    for pid in pids_cl:
                        bar_pid_spanned[pid].add(gid)
                        bar_pid_group[pid] = grp_id

        for pid in sorted(bar_pid_spanned.keys()):
            bar_pids_found.add(pid)
            mid = prim_to_mol_id[pid]
            bar_pids_by_mol[mid].append(pid)
            su0, sv0, su1, sv1 = uv_boxes[pid]
            s = by_id[pid]
            transverse_bars.append({
                "molecule_id": mid,
                "primitive_id": pid,
                "bar_group_id": bar_pid_group[pid],
                "spanned_grid_ids": sorted(bar_pid_spanned[pid]),
                "u_pos": round(0.5 * (su0 + su1), 2),
                "v_span": [round(sv0, 2), round(sv1, 2)],
                "length_px": s["length_px"],
                "angle_deg": s["angle_deg"],
            })

    mol_zone: Dict[int, str] = {}
    mol_grids: Dict[int, List[int]] = {}
    for mid, comp in enumerate(raw_molecules, start=1):
        touched_grid_prims = {
            int(nb) for i in comp for nb in by_id[i].get("neighbors", []) if int(nb) in grid_pids
        }
        ov_g: set[int] = set()
        for pid in comp:
            ov_g.update(prim_overlapping_grids(pid))
        in_grids = sorted(ov_g)
        mol_grids[mid] = in_grids
        mol_zone[mid] = "pe_grila" if (touched_grid_prims or in_grids or mid in bar_pids_by_mol) else "coridor_liber"

    non_bar_ids = [pid for pid in non_grid_ids if pid not in bar_pids_found]
    touch_adj: Dict[int, set[int]] = {
        pid: set(nb for nb in near_touch_adj[pid] if nb not in bar_pids_found)
        for pid in non_bar_ids
    }

    parallel_bridges: List[Dict[str, Any]] = []
    anchored_sticks: List[Dict[str, Any]] = []
    bridge_ids_global: set[int] = set()
    bridged_stick_ids_global: set[int] = set()
    bridges_by_mol: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    anchors_by_mol: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    min_bridge_len = 1.1 * s_default

    for cid in non_bar_ids:
        cand = by_id[cid]
        if cand["type"] != "banda_dreapta":
            continue
        if float(cand["elongation"]) < 2.8 or float(cand["length_px"]) < min_bridge_len:
            continue
        mid = prim_to_mol_id[cid]
        c_gids = mol_grids[mid]
        s_loc = float(np.median([grids[g - 1]["step_px"] for g in c_gids])) if c_gids else s_default
        nb_hits: List[Tuple[Dict[str, Any], float]] = []
        for nb in touch_adj[cid]:
            nb_obj = by_id[nb]
            if nb_obj["type"] != "banda_dreapta":
                continue
            if float(nb_obj["elongation"]) < 2.5 or float(nb_obj["length_px"]) < 0.45 * s_loc:
                continue
            if _angle_diff_180(float(cand["angle_deg"]), float(nb_obj["angle_deg"])) < 28.0:
                continue
            ok_end, t_on_br = _ribbon_touches_near_end(nb_obj, cand)
            if ok_end:
                nb_hits.append((nb_obj, t_on_br))
        if len(nb_hits) < 2:
            continue
        best_grp: List[Tuple[Dict[str, Any], float]] = []
        for a_i in range(len(nb_hits)):
            grp = [nb_hits[a_i]]
            for b_i in range(len(nb_hits)):
                if a_i != b_i and _angle_diff_180(
                    float(nb_hits[a_i][0]["angle_deg"]), float(nb_hits[b_i][0]["angle_deg"])
                ) <= 22.0:
                    grp.append(nb_hits[b_i])
            if len(grp) > len(best_grp):
                best_grp = grp
        if len(best_grp) >= 2:
            ts = [t_val for _, t_val in best_grp]
            sep_along_px = (max(ts) - min(ts)) * float(cand["length_px"])
            if sep_along_px < 0.50 * s_loc:
                continue
            bridge_ids_global.add(cid)
            p_ids = sorted(int(x[0]["id"]) for x in best_grp)
            bridged_stick_ids_global.update(p_ids)
            b_rec = {
                "molecule_id": mid,
                "bridge_id": cid,
                "bridge_primitive_id": cid,
                "bridge_width_px": cand["width_px"],
                "bridge_angle_deg": cand["angle_deg"],
                "connected_parallel_ids": p_ids,
                "zone": mol_zone[mid],
                "grid_ids": c_gids,
            }
            parallel_bridges.append(b_rec)
            bridges_by_mol[mid].append(b_rec)

    for sid in non_bar_ids:
        if sid in bridge_ids_global:
            continue
        s = by_id[sid]
        if s["type"] != "banda_dreapta" or float(s["elongation"]) < 2.5:
            continue
        mid = prim_to_mol_id[sid]
        s_gids = mol_grids[mid]
        s_loc = float(np.median([grids[g - 1]["step_px"] for g in s_gids])) if s_gids else s_default
        pa = np.array(s["start"], dtype=np.float64)
        pb = np.array(s["end"], dtype=np.float64)
        tol_end = max(2.5 * float(s["width_px"]), 0.35 * float(s["length_px"]))
        end_a_att: List[int] = []
        end_b_att: List[int] = []
        mid_att: List[int] = []
        for nb in touch_adj[sid]:
            np_obj = by_id[nb]
            if not (np_obj["type"] in ("pata_compacta", "banda_curba") or is_blob_like(np_obj, s_loc)):
                continue
            da = _prim_dist_to_point(pa, np_obj)
            db = _prim_dist_to_point(pb, np_obj)
            if da <= db and da <= tol_end:
                end_a_att.append(nb)
            elif db < da and db <= tol_end:
                end_b_att.append(nb)
            else:
                mid_att.append(nb)
        if end_a_att or end_b_att:
            a_rec = {
                "molecule_id": mid,
                "stick_id": sid,
                "start_A": s["start"],
                "end_B": s["end"],
                "length_px": s["length_px"],
                "width_px": s["width_px"],
                "angle_deg": s["angle_deg"],
                "on_bridge": sid in bridged_stick_ids_global,
                "end_A_attachments": sorted(end_a_att),
                "end_B_attachments": sorted(end_b_att),
                "mid_attachments": sorted(mid_att),
                "zone": mol_zone[mid],
                "grid_ids": s_gids,
            }
            anchored_sticks.append(a_rec)
            anchors_by_mol[mid].append(a_rec)

    molecules: List[Dict[str, Any]] = []
    for mid, comp in enumerate(raw_molecules, start=1):
        mu0 = min(uv_boxes[i][0] for i in comp)
        mv0 = min(uv_boxes[i][1] for i in comp)
        mu1 = max(uv_boxes[i][2] for i in comp)
        mv1 = max(uv_boxes[i][3] for i in comp)
        m_w, m_h = mu1 - mu0, mv1 - mv0

        straights = [by_id[i] for i in comp if by_id[i]["type"] == "banda_dreapta"]
        spots = [by_id[i] for i in comp if by_id[i]["type"] == "pata_compacta"]
        curves = [by_id[i] for i in comp if by_id[i]["type"] == "banda_curba"]
        local_bar_pids = bar_pids_by_mol.get(mid, [])
        local_bridges = bridges_by_mol.get(mid, [])
        local_anchored = anchors_by_mol.get(mid, [])

        if len(local_bar_pids) == 1 and len(comp) == 1:
            mtype = "bara_transversala_grila"
        elif local_bridges:
            mtype = "grup_pe_pod"
        elif local_anchored:
            mtype = "bat_ancorat"
        elif local_bar_pids:
            mtype = "bara_cu_atasament"
        elif len(straights) >= 2 and not spots and not curves:
            mtype = "retea_benzi_drepte"
        elif len(comp) == 1:
            mtype = f"{by_id[comp[0]]['type']}_izolata"
        else:
            mtype = "molecula_mixta"

        xy_boxes = [_prim_xy_box(by_id[i]) for i in comp]
        molecules.append({
            "molecule_id": mid,
            "type": mtype,
            "zone": mol_zone[mid],
            "grid_ids": mol_grids[mid],
            "primitive_ids": comp,
            "primitive_count": len(comp),
            "uv_bbox": [round(mu0, 2), round(mv0, 2), round(mu1, 2), round(mv1, 2)],
            "xy_bbox": [
                round(min(b[0] for b in xy_boxes), 1),
                round(min(b[1] for b in xy_boxes), 1),
                round(max(b[2] for b in xy_boxes), 1),
                round(max(b[3] for b in xy_boxes), 1),
            ],
            "u_span_px": round(m_w, 2),
            "v_span_px": round(m_h, 2),
            "row_id": None,
        })

    row_cands = [
        m for m in molecules
        if m["zone"] == "coridor_liber"
        and m["type"] not in ("bara_transversala_grila", "grup_pe_pod")
        and m["v_span_px"] <= 4.0 * s_default
        and m["u_span_px"] <= 12.0 * s_default
    ]
    rows: List[Dict[str, Any]] = []
    if len(row_cands) >= 3:
        cand_hs = [m["v_span_px"] for m in row_cands if m["v_span_px"] > 0.15 * s_default]
        global_med_h = float(np.median(cand_hs)) if cand_hs else s_default
        row_cands.sort(key=lambda m: 0.5 * (m["uv_bbox"][1] + m["uv_bbox"][3]))
        bands: List[List[Dict[str, Any]]] = []
        for m in row_cands:
            mv0, mv1 = m["uv_bbox"][1], m["uv_bbox"][3]
            mvc = 0.5 * (mv0 + mv1)
            mh = max(1e-6, mv1 - mv0)
            best_band = None
            best_dv = float("inf")
            for band in bands:
                b_vcs = [0.5 * (x["uv_bbox"][1] + x["uv_bbox"][3]) for x in band]
                b_hs = [max(1e-6, x["uv_bbox"][3] - x["uv_bbox"][1]) for x in band]
                b_vc = float(np.median(b_vcs))
                b_mh = float(np.median(b_hs))
                ref_h = max(b_mh, min(mh, 2.0 * global_med_h))
                dv = abs(mvc - b_vc)
                if dv <= 0.60 * ref_h + 0.15 * global_med_h and dv < best_dv:
                    best_dv = dv
                    best_band = band
            if best_band is not None:
                best_band.append(m)
            else:
                bands.append([m])

        for band in bands:
            if len(band) < 3:
                continue
            band.sort(key=lambda m: m["uv_bbox"][0])
            heights = [m["v_span_px"] for m in band if m["v_span_px"] >= 0.20 * global_med_h]
            med_h = float(np.median(heights)) if heights else global_med_h
            max_gap = 4.5 * med_h

            runs: List[List[Dict[str, Any]]] = [[band[0]]]
            for m in band[1:]:
                prev_u1 = max(x["uv_bbox"][2] for x in runs[-1])
                if m["uv_bbox"][0] - prev_u1 <= max_gap:
                    runs[-1].append(m)
                else:
                    runs.append([m])

            for run in runs:
                if len(run) < 3:
                    continue
                rid = len(rows) + 1
                for m in run:
                    m["row_id"] = rid
                rows.append({
                    "row_id": rid,
                    "molecule_count": len(run),
                    "molecule_ids": [m["molecule_id"] for m in run],
                    "primitive_ids": sorted(pid for m in run for pid in m["primitive_ids"]),
                    "uv_bbox": [
                        round(min(m["uv_bbox"][0] for m in run), 2),
                        round(min(m["uv_bbox"][1] for m in run), 2),
                        round(max(m["uv_bbox"][2] for m in run), 2),
                        round(max(m["uv_bbox"][3] for m in run), 2),
                    ],
                    "xy_bbox": [
                        round(min(m["xy_bbox"][0] for m in run), 1),
                        round(min(m["xy_bbox"][1] for m in run), 1),
                        round(max(m["xy_bbox"][2] for m in run), 1),
                        round(max(m["xy_bbox"][3] for m in run), 1),
                    ],
                    "median_height_px": round(med_h, 2),
                })

    prim_counts: Counter[int] = Counter()
    for g in grids:
        for pid in g["primitive_ids"]:
            prim_counts[pid] += 1
    for m in molecules:
        for pid in m["primitive_ids"]:
            prim_counts[pid] += 1

    missing = [pid for pid in all_ids if prim_counts[pid] == 0]
    dups = [pid for pid, cnt in prim_counts.items() if cnt != 1]
    if missing or dups:
        raise RuntimeError(f"Audit Strat 2 picat: lipsă={len(missing)}, duplicate={len(dups)}")

    by_mtype: Dict[str, int] = defaultdict(int)
    for m in molecules:
        by_mtype[m["type"]] += 1

    return {
        "source_layer1_image": layer1.get("image"),
        "page_frame": {
            "dominant_angle_deg": round(theta_deg, 3),
            "u_axis": [round(float(u_vec[0]), 6), round(float(u_vec[1]), 6)],
            "v_axis": [round(float(v_vec[0]), 6), round(float(v_vec[1]), 6)],
            "grid_step_px": round(page_step, 2) if page_step is not None else None,
        },
        "accounting_verification": {
            "layer1_primitives_total": len(all_ids),
            "grid_primitives": len(grid_pids),
            "molecule_primitives": len(all_ids) - len(grid_pids),
            "unaccounted_primitives": 0,
            "duplicate_primitives": 0,
            "guaranteed_100pct_accounting": True,
        },
        "counts": {
            "periodic_grids": len(grids),
            "transverse_grid_bars": len(transverse_bars),
            "parallel_bridges": len(parallel_bridges),
            "anchored_sticks": len(anchored_sticks),
            "aligned_rows": len(rows),
            "molecules_total": len(molecules),
            "molecules_by_type": dict(by_mtype),
        },
        "periodic_grids": grids,
        "transverse_grid_bars": transverse_bars,
        "parallel_bridges": parallel_bridges,
        "anchored_sticks": anchored_sticks,
        "aligned_rows": rows,
        "molecules": molecules,
    }


def _load_canvas(layer1: Dict[str, Any], image_override: Optional[str]) -> np.ndarray:
    img_path = image_override or layer1.get("image")
    if img_path and Path(img_path).exists():
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is not None:
            return (img.astype(np.float32) * 0.45 + 25.0).astype(np.uint8)
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


def draw_debug_layer2(
    layer1: Dict[str, Any],
    layer2: Dict[str, Any],
    out_dir: Path,
    image_override: Optional[str] = None,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {int(p["id"]): p for p in layer1["primitives"]}
    base = _load_canvas(layer1, image_override)
    font = cv2.FONT_HERSHEY_SIMPLEX

    p1 = out_dir / "debug_l2_1_grile_si_bare.png"
    p2 = out_dir / "debug_l2_2_poduri_si_ancore.png"
    p3 = out_dir / "debug_l2_3_coridoare_si_siruri.png"

    v1 = base.copy()
    grid_palette = [(230, 210, 0), (235, 160, 40), (210, 235, 80), (230, 120, 120)]
    for idx, g in enumerate(layer2["periodic_grids"]):
        col = grid_palette[idx % len(grid_palette)]
        for pid in g["primitive_ids"]:
            _draw_prim(v1, by_id[pid], col, 2)
        first_p = by_id[g["primitive_ids"][0]]
        fx, fy = (first_p["start"][0], first_p["start"][1]) if first_p["type"] == "banda_dreapta" else (first_p["bbox"][0], first_p["bbox"][1])
        lbl = f"G{g['grid_id']}: {g['line_count']}L, S={g['step_px']}px"
        cv2.putText(v1, lbl, (int(fx) + 4, max(18, int(fy) - 6)), font, 0.50, col, 2, cv2.LINE_AA)
    for bar in layer2["transverse_grid_bars"]:
        _draw_prim(v1, by_id[bar["primitive_id"]], (230, 0, 230), 3)
    cv2.imwrite(str(p1), v1)

    v2 = base.copy()
    for g in layer2["periodic_grids"]:
        for pid in g["primitive_ids"]:
            _draw_prim(v2, by_id[pid], (95, 95, 95), 1)
    bridge_pids = {b["bridge_id"] for b in layer2["parallel_bridges"]}
    bridged_pids = {pid for b in layer2["parallel_bridges"] for pid in b["connected_parallel_ids"]}
    for b in layer2["parallel_bridges"]:
        col_b = (0, 145, 255) if b["zone"] == "pe_grila" else (80, 100, 130)
        col_s = (0, 235, 110) if b["zone"] == "pe_grila" else (80, 120, 90)
        _draw_prim(v2, by_id[b["bridge_id"]], col_b, 3 if b["zone"] == "pe_grila" else 1)
        for sid in b["connected_parallel_ids"]:
            _draw_prim(v2, by_id[sid], col_s, 2 if b["zone"] == "pe_grila" else 1)
    for a in layer2["anchored_sticks"]:
        sid = a["stick_id"]
        on_g = a["zone"] == "pe_grila"
        if sid not in bridged_pids and sid not in bridge_pids:
            _draw_prim(v2, by_id[sid], (0, 210, 0) if on_g else (70, 115, 70), 2 if on_g else 1)
        if on_g:
            if a["end_A_attachments"]:
                pt = (int(round(a["start_A"][0])), int(round(a["start_A"][1])))
                cv2.circle(v2, pt, 4, (0, 0, 245), -1)
            if a["end_B_attachments"]:
                pt = (int(round(a["end_B"][0])), int(round(a["end_B"][1])))
                cv2.circle(v2, pt, 4, (0, 0, 245), -1)
    cv2.imwrite(str(p2), v2)

    v3 = base.copy()
    for g in layer2["periodic_grids"]:
        for pid in g["primitive_ids"]:
            _draw_prim(v3, by_id[pid], (90, 90, 90), 1)
    for m in layer2["molecules"]:
        col = (0, 200, 80) if m["zone"] == "pe_grila" else (235, 190, 40)
        for pid in m["primitive_ids"]:
            _draw_prim(v3, by_id[pid], col, 2)
    for row in layer2["aligned_rows"]:
        x0, y0, x1, y1 = [int(round(v)) for v in row["xy_bbox"]]
        cv2.rectangle(v3, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (0, 225, 255), 2)
        cv2.putText(v3, f"R{row['row_id']} ({row['molecule_count']})", (x0, max(16, y0 - 5)),
                    font, 0.48, (0, 225, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(p3), v3)

    return {
        "debug_l2_1_grile_si_bare": str(p1),
        "debug_l2_2_poduri_si_ancore": str(p2),
        "debug_l2_3_coridoare_si_siruri": str(p3),
    }


def process_layer1_json(
    json_path: str | Path,
    out_dir: str | Path = "debug_l2",
    image_override: Optional[str] = None,
) -> Dict[str, Any]:
    json_path = Path(json_path)
    layer1 = json.loads(json_path.read_text(encoding="utf-8"))
    layer2 = analyze_layer2(layer1)
    debug_paths = draw_debug_layer2(layer1, layer2, Path(out_dir), image_override)
    layer2["debug_images"] = debug_paths

    stem = json_path.stem.replace("_geometry_layer1", "")
    out_path = Path(out_dir) / f"{stem}_geometry_layer2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(layer2, ensure_ascii=False, indent=2), encoding="utf-8")
    layer2["json_file"] = str(out_path)
    return layer2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("layer1_json")
    parser.add_argument("--out-dir", default="debug_l2")
    parser.add_argument("--image", default=None)
    args = parser.parse_args(argv)
    try:
        res = process_layer1_json(args.layer1_json, args.out_dir, args.image)
    except Exception as exc:
        print(f"Eroare: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "page_frame": res["page_frame"],
        "accounting_verification": res["accounting_verification"],
        "counts": res["counts"],
        "debug_images": res["debug_images"],
        "json_file": res["json_file"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
