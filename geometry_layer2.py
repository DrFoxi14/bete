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


def _point_to_segment(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    ab = b - a
    l2 = float(np.dot(ab, ab))
    if l2 < 1e-9:
        return float(np.linalg.norm(pt - a)), 0.0
    t = float(np.clip(np.dot(pt - a, ab) / l2, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(pt - proj)), t


def _prim_uv_box(p: Dict[str, Any], u_vec: np.ndarray, v_vec: np.ndarray) -> Tuple[float, float, float, float]:
    if p["type"] == "banda_dreapta":
        a = np.array(p["start"], dtype=np.float64)
        b = np.array(p["end"], dtype=np.float64)
        hw = 0.5 * float(p["width_px"])
        d = b - a
        L = float(np.linalg.norm(d))
        if L > 1e-9:
            n = np.array([-d[1] / L, d[0] / L], dtype=np.float64)
            corners = np.array([a + hw * n, a - hw * n, b - hw * n, b + hw * n], dtype=np.float64)
        else:
            corners = np.array([
                [a[0] - hw, a[1] - hw], [a[0] + hw, a[1] - hw],
                [a[0] + hw, a[1] + hw], [a[0] - hw, a[1] + hw],
            ], dtype=np.float64)
    else:
        x0, y0, x1, y1 = [float(v) for v in p["bbox"]]
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    us = corners @ u_vec
    vs = corners @ v_vec
    return float(np.min(us)), float(np.min(vs)), float(np.max(us)), float(np.max(vs))


def _prim_xy_box(p: Dict[str, Any]) -> Tuple[float, float, float, float]:
    if p["type"] == "banda_dreapta":
        x0, y0 = float(p["start"][0]), float(p["start"][1])
        x1, y1 = float(p["end"][0]), float(p["end"][1])
        hw = 0.5 * float(p["width_px"])
        return min(x0, x1) - hw, min(y0, y1) - hw, max(x0, x1) + hw, max(y0, y1) + hw
    b = p["bbox"]
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


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


def _ribbon_touches_near_end(stick: Dict[str, Any], bridge: Dict[str, Any]) -> bool:
    pa = np.array(stick["start"], dtype=np.float64)
    pb = np.array(stick["end"], dtype=np.float64)
    ba = np.array(bridge["start"], dtype=np.float64)
    bb = np.array(bridge["end"], dtype=np.float64)
    da, _ = _point_to_segment(pa, ba, bb)
    db, _ = _point_to_segment(pb, ba, bb)
    half_sum = 0.5 * (float(stick["width_px"]) + float(bridge["width_px"]))
    gap_a = max(0.0, da - half_sum)
    gap_b = max(0.0, db - half_sum)
    end_tol = max(4.0, 2.0 * float(bridge["width_px"]), 0.25 * float(stick["length_px"]))
    return min(gap_a, gap_b) <= end_tol


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


def _track_eval_piecewise(members: Sequence[Dict[str, Any]], u_val: float) -> float:
    pts = []
    for m in members:
        pts.append((m["ua"], m["va"]))
        pts.append((m["ub"], m["vb"]))
    pts.sort(key=lambda t: t[0])
    us = np.array([t[0] for t in pts], dtype=np.float64)
    vs = np.array([t[1] for t in pts], dtype=np.float64)
    if u_val <= us[0]:
        first = min(members, key=lambda m: m["ua"])
        sl = (first["vb"] - first["va"]) / max(1e-6, first["ub"] - first["ua"])
        sl = max(-0.08, min(0.08, sl))
        return float(first["va"] + sl * (u_val - first["ua"]))
    if u_val >= us[-1]:
        last = max(members, key=lambda m: m["ub"])
        sl = (last["vb"] - last["va"]) / max(1e-6, last["ub"] - last["ua"])
        sl = max(-0.08, min(0.08, sl))
        return float(last["vb"] + sl * (u_val - last["ub"]))
    return float(np.interp(u_val, us, vs))


def _curve_eval(curve: Dict[str, Any], u_val: float) -> float:
    if curve["kind"] == "track":
        return _track_eval_piecewise(curve["members"], u_val)
    return 0.5 * (_curve_eval(curve["a"], u_val) + _curve_eval(curve["b"], u_val))


def detect_periodic_grids(
    primitives: Sequence[Dict[str, Any]],
    theta_deg: float,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
) -> Tuple[List[Dict[str, Any]], set[int]]:
    aligned: List[Dict[str, Any]] = []
    for p in primitives:
        if p["type"] != "banda_dreapta":
            continue
        if _angle_diff_180(float(p["angle_deg"]), theta_deg) > 10.0:
            continue
        if float(p["elongation"]) < 4.5:
            continue
        pa = np.array(p["start"], dtype=np.float64)
        pb = np.array(p["end"], dtype=np.float64)
        ua, va = float(np.dot(pa, u_vec)), float(np.dot(pa, v_vec))
        ub, vb = float(np.dot(pb, u_vec)), float(np.dot(pb, v_vec))
        if ub < ua:
            ua, ub = ub, ua
            va, vb = vb, va
        aligned.append({
            "id": int(p["id"]),
            "ua": ua, "va": va, "ub": ub, "vb": vb,
            "u_mid": 0.5 * (ua + ub), "v_mid": 0.5 * (va + vb),
            "len": float(p["length_px"]), "w": float(p["width_px"]),
        })

    if len(aligned) < 3:
        return [], set()

    aligned.sort(key=lambda s: (s["ua"], -s["len"]))
    tracks: List[Dict[str, Any]] = []
    for seg in aligned:
        best_tr = None
        best_err = float("inf")
        for tr in tracks:
            last = tr["last"]
            gap_u = seg["ua"] - last["ub"]
            if gap_u < -max(8.0, 0.35 * min(seg["len"], last["len"])):
                continue
            if gap_u > 360.0:
                continue
            w_ref = max(tr["w"], seg["w"])
            if max(tr["w"], seg["w"]) / max(0.5, min(tr["w"], seg["w"])) > 2.4:
                continue
            sl_last = max(-0.08, min(0.08, (last["vb"] - last["va"]) / max(1e-6, last["ub"] - last["ua"])))
            sl_seg = max(-0.08, min(0.08, (seg["vb"] - seg["va"]) / max(1e-6, seg["ub"] - seg["ua"])))
            pred_va = last["vb"] + sl_last * (seg["ua"] - last["ub"])
            pred_last_vb = seg["va"] + sl_seg * (last["ub"] - seg["ua"])
            err = 0.5 * (abs(seg["va"] - pred_va) + abs(last["vb"] - pred_last_vb))
            tol_v = max(2.8, 1.8 * w_ref, min(6.5, 0.025 * max(0.0, gap_u) + 2.0))
            if err <= tol_v and err < best_err:
                best_err = err
                best_tr = tr
        if best_tr is not None:
            best_tr["members"].append(seg)
            best_tr["u0"] = min(best_tr["u0"], seg["ua"])
            best_tr["u1"] = max(best_tr["u1"], seg["ub"])
            best_tr["total_len"] += seg["len"]
            best_tr["w"] = sum(m["w"] * m["len"] for m in best_tr["members"]) / max(1e-6, best_tr["total_len"])
            if seg["ub"] >= best_tr["last"]["ub"]:
                best_tr["last"] = seg
        else:
            tracks.append({
                "members": [seg], "last": seg,
                "u0": seg["ua"], "u1": seg["ub"],
                "total_len": seg["len"], "w": seg["w"],
            })

    spans = np.array([tr["u1"] - tr["u0"] for tr in tracks], dtype=np.float64)
    if spans.size < 3:
        return [], set()
    ref_span = float(np.percentile(spans, 85))
    long_tracks = [
        tr for tr in tracks
        if (tr["u1"] - tr["u0"]) >= max(40.0, 0.30 * ref_span)
        and tr["total_len"] >= 0.25 * (tr["u1"] - tr["u0"])
    ]
    if len(long_tracks) < 3:
        return [], set()

    u_page_mid = float(np.median([0.5 * (tr["u0"] + tr["u1"]) for tr in long_tracks]))
    long_tracks.sort(key=lambda tr: _track_eval_piecewise(tr["members"], u_page_mid))

    def tracks_overlap_and_dv(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[float, float]:
        ov_lo = max(a["u0"], b["u0"])
        ov_hi = min(a["u1"], b["u1"])
        ov = ov_hi - ov_lo
        min_sp = max(1e-6, min(a["u1"] - a["u0"], b["u1"] - b["u0"]))
        if ov > 0:
            us_eval = np.linspace(ov_lo, ov_hi, 5)
            dvs = [_track_eval_piecewise(b["members"], u) - _track_eval_piecewise(a["members"], u) for u in us_eval]
            dv = float(np.median(dvs))
        else:
            dv = _track_eval_piecewise(b["members"], u_page_mid) - _track_eval_piecewise(a["members"], u_page_mid)
        return ov / min_sp, dv

    adj_gaps: List[float] = []
    for a, b in zip(long_tracks, long_tracks[1:]):
        ov_ratio, dv = tracks_overlap_and_dv(a, b)
        if ov_ratio >= 0.30 and dv > 2.0 * max(a["w"], b["w"]):
            adj_gaps.append(dv)
    if not adj_gaps:
        return [], set()

    adj_gaps.sort()
    best_cluster: List[float] = []
    for g in adj_gaps:
        cluster = [x for x in adj_gaps if abs(x - g) <= 0.20 * g + 1.5]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    if len(best_cluster) < 2:
        return [], set()
    step_s = float(np.median(best_cluster))

    grids: List[Dict[str, Any]] = []
    used: set[int] = set()
    tol_dv = max(2.5, 0.25 * step_s)

    for i in range(len(long_tracks)):
        if i in used:
            continue
        chain_indices = [i]
        curves: List[Dict[str, Any]] = [{"kind": "track", "members": long_tracks[i]["members"]}]
        cur = long_tracks[i]
        cur_w = cur["w"]
        for j in range(i + 1, len(long_tracks)):
            if j in used:
                continue
            cand = long_tracks[j]
            ov_ratio, dv = tracks_overlap_and_dv(cur, cand)
            if dv > 2.30 * step_s + 3.0:
                break
            if ov_ratio < 0.30:
                continue
            if max(cur_w, cand["w"]) / max(0.5, min(cur_w, cand["w"])) > 2.6:
                continue
            if abs(dv - step_s) <= tol_dv:
                chain_indices.append(j)
                curves.append({"kind": "track", "members": cand["members"]})
                cur = cand
                cur_w = 0.5 * (cur_w + cand["w"])
            elif abs(dv - 2.0 * step_s) <= 1.35 * tol_dv:
                prev_curve = curves[-1]
                next_curve = {"kind": "track", "members": cand["members"]}
                curves.append({"kind": "interp", "a": prev_curve, "b": next_curve})
                curves.append(next_curve)
                chain_indices.append(j)
                cur = cand
                cur_w = 0.5 * (cur_w + cand["w"])

        if len(chain_indices) >= 3:
            for idx_t in chain_indices:
                used.add(idx_t)
            trs = [long_tracks[idx_t] for idx_t in chain_indices]
            u_lo = min(tr["u0"] for tr in trs)
            u_hi = max(tr["u1"] for tr in trs)
            u_mid = 0.5 * (u_lo + u_hi)
            line_vs = [_curve_eval(cv, u_mid) for cv in curves]
            steps = [line_vs[k + 1] - line_vs[k] for k in range(len(line_vs) - 1)]
            line_w = float(np.median([tr["w"] for tr in trs]))
            line_ids = sorted({m["id"] for tr in trs for m in tr["members"]})
            grids.append({
                "grid_id": len(grids) + 1,
                "line_count": len(curves),
                "step_px": round(float(np.median(steps)), 2),
                "angle_deg": round(theta_deg, 2),
                "u_span": [round(u_lo, 2), round(u_hi, 2)],
                "v_span": [round(line_vs[0], 2), round(line_vs[-1], 2)],
                "line_v_coords": [round(v, 2) for v in line_vs],
                "primitive_ids": line_ids,
                "_curves": curves,
                "_line_w": line_w,
            })

    assigned_grid_pids: set[int] = {pid for g in grids for pid in g["primitive_ids"]}

    def _shifted_eval(cv: Dict[str, Any], delta_v: float, u_val: float) -> float:
        return _curve_eval(cv, u_val) + delta_v

    def _merged_span_len(ivs: List[Tuple[float, float]]) -> float:
        if not ivs:
            return 0.0
        s_ivs = sorted(ivs)
        cur_a, cur_b = s_ivs[0]
        tot = 0.0
        for a, b in s_ivs[1:]:
            if a <= cur_b:
                cur_b = max(cur_b, b)
            else:
                tot += cur_b - cur_a
                cur_a, cur_b = a, b
        return tot + (cur_b - cur_a)

    for g in grids:
        lw = g["_line_w"]
        g_step = float(g["step_px"])
        tol_v = max(2.6, 1.7 * lw)
        for _side_pass in range(2):
            extended = False
            g_u0, g_u1 = g["u_span"]
            g_span = max(1e-6, g_u1 - g_u0)
            for side in ("top", "bot"):
                ref_cv = g["_curves"][0] if side == "top" else g["_curves"][-1]
                delta = -g_step if side == "top" else g_step
                cand_segs: List[Dict[str, Any]] = []
                cand_ivs: List[Tuple[float, float]] = []
                for p in primitives:
                    pid = int(p["id"])
                    if pid in assigned_grid_pids or p["type"] != "banda_dreapta":
                        continue
                    if _angle_diff_180(float(p["angle_deg"]), theta_deg) > 12.0:
                        continue
                    if float(p["width_px"]) > max(3.2, 2.1 * lw):
                        continue
                    pa = np.array(p["start"], dtype=np.float64)
                    pb = np.array(p["end"], dtype=np.float64)
                    ua, va = float(np.dot(pa, u_vec)), float(np.dot(pa, v_vec))
                    ub, vb = float(np.dot(pb, u_vec)), float(np.dot(pb, v_vec))
                    if ub < ua:
                        ua, ub, va, vb = ub, ua, vb, va
                    if ub < g_u0 - 0.5 * g_step or ua > g_u1 + 0.5 * g_step:
                        continue
                    ea = abs(va - _shifted_eval(ref_cv, delta, ua))
                    eb = abs(vb - _shifted_eval(ref_cv, delta, ub))
                    if max(ea, eb) <= tol_v:
                        cand_segs.append({
                            "id": pid, "ua": ua, "va": va, "ub": ub, "vb": vb,
                            "len": float(p["length_px"]), "w": float(p["width_px"]),
                        })
                        ca, cb = max(g_u0, ua), min(g_u1, ub)
                        if cb > ca:
                            cand_ivs.append((ca, cb))
                if cand_segs and _merged_span_len(cand_ivs) >= 0.25 * g_span:
                    new_cv = {"kind": "track", "members": cand_segs}
                    if side == "top":
                        g["_curves"].insert(0, new_cv)
                    else:
                        g["_curves"].append(new_cv)
                    for cs in cand_segs:
                        g["primitive_ids"].append(cs["id"])
                        assigned_grid_pids.add(cs["id"])
                        g["u_span"] = [
                            round(min(g["u_span"][0], cs["ua"]), 2),
                            round(max(g["u_span"][1], cs["ub"]), 2),
                        ]
                    g["line_count"] = len(g["_curves"])
                    u_mid = 0.5 * (g["u_span"][0] + g["u_span"][1])
                    lvs = [_curve_eval(cv, u_mid) for cv in g["_curves"]]
                    g["v_span"] = [round(lvs[0], 2), round(lvs[-1], 2)]
                    g["line_v_coords"] = [round(v, 2) for v in lvs]
                    extended = True
            if not extended:
                break

    for p in primitives:
        pid = int(p["id"])
        if pid in assigned_grid_pids or p["type"] != "banda_dreapta":
            continue
        if _angle_diff_180(float(p["angle_deg"]), theta_deg) > 12.0:
            continue
        pa = np.array(p["start"], dtype=np.float64)
        pb = np.array(p["end"], dtype=np.float64)
        ua, va = float(np.dot(pa, u_vec)), float(np.dot(pa, v_vec))
        ub, vb = float(np.dot(pb, u_vec)), float(np.dot(pb, v_vec))
        if ub < ua:
            ua, ub, va, vb = ub, ua, vb, va
        pw = float(p["width_px"])

        best_g = None
        best_err = float("inf")
        for g in grids:
            lw = g["_line_w"]
            if pw > max(3.2, 2.1 * lw):
                continue
            if ub < g["u_span"][0] - 0.8 * g["step_px"] or ua > g["u_span"][1] + 0.8 * g["step_px"]:
                continue
            tol_v = max(2.4, 1.6 * lw)
            for cv in g["_curves"]:
                ea = abs(va - _curve_eval(cv, ua))
                eb = abs(vb - _curve_eval(cv, ub))
                err = max(ea, eb)
                if err <= tol_v and err < best_err:
                    best_err = err
                    best_g = g
        if best_g is not None:
            best_g["primitive_ids"].append(pid)
            assigned_grid_pids.add(pid)
            best_g["u_span"] = [
                round(min(best_g["u_span"][0], ua), 2),
                round(max(best_g["u_span"][1], ub), 2),
            ]

    for g in grids:
        g["primitive_ids"] = sorted(set(g["primitive_ids"]))
        del g["_curves"]
        del g["_line_w"]

    return grids, assigned_grid_pids


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

    grids, grid_pids = detect_periodic_grids(primitives, theta_deg, u_vec, v_vec)
    page_step = float(np.median([g["step_px"] for g in grids])) if grids else None

    def overlapping_grids(u0: float, v0: float, u1: float, v1: float, margin_v: float = 0.0) -> List[int]:
        res = []
        for g in grids:
            gu0, gu1 = g["u_span"]
            gv0, gv1 = g["v_span"]
            if u1 >= gu0 - margin_v and u0 <= gu1 + margin_v and v1 >= gv0 - margin_v and v0 <= gv1 + margin_v:
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
        raw_molecules.append(comp)

    molecules: List[Dict[str, Any]] = []
    transverse_bars: List[Dict[str, Any]] = []
    parallel_bridges: List[Dict[str, Any]] = []
    anchored_sticks: List[Dict[str, Any]] = []
    bar_pids_found: set[int] = set()

    for mid, comp in enumerate(raw_molecules, start=1):
        mu0 = min(uv_boxes[i][0] for i in comp)
        mv0 = min(uv_boxes[i][1] for i in comp)
        mu1 = max(uv_boxes[i][2] for i in comp)
        mv1 = max(uv_boxes[i][3] for i in comp)
        m_w, m_h = mu1 - mu0, mv1 - mv0

        touched_grid_prims = sorted({
            int(nb) for i in comp for nb in by_id[i].get("neighbors", []) if int(nb) in grid_pids
        })
        grid_margin = 0.75 * page_step if page_step else 0.0
        in_grids = overlapping_grids(mu0, mv0, mu1, mv1, margin_v=grid_margin)
        zone = "pe_grila" if (touched_grid_prims or in_grids) else "coridor_liber"

        straights = [by_id[i] for i in comp if by_id[i]["type"] == "banda_dreapta"]
        spots = [by_id[i] for i in comp if by_id[i]["type"] == "pata_compacta"]
        curves = [by_id[i] for i in comp if by_id[i]["type"] == "banda_curba"]

        local_bars: List[Dict[str, Any]] = []
        if grids and zone == "pe_grila":
            for s in straights:
                sid_bar = int(s["id"])
                if not (70.0 <= _angle_diff_180(float(s["angle_deg"]), theta_deg) <= 110.0):
                    continue
                su0, sv0, su1, sv1 = uv_boxes[sid_bar]
                cand_grids = []
                for g in grids:
                    gu0, gu1 = g["u_span"]
                    gv0, gv1 = g["v_span"]
                    gh = max(1e-6, gv1 - gv0)
                    gs = g["step_px"]
                    if su1 >= gu0 - gs and su0 <= gu1 + gs:
                        ov_v = min(sv1, gv1) - max(sv0, gv0)
                        if ov_v >= 0.80 * gh:
                            cand_grids.append(g)
                if not cand_grids:
                    continue
                top_gv0 = min(g["v_span"][0] for g in cand_grids)
                bot_gv1 = max(g["v_span"][1] for g in cand_grids)
                gs_ref = cand_grids[0]["step_px"]
                if sv0 < top_gv0 - 0.75 * gs_ref or sv1 > bot_gv1 + 0.75 * gs_ref:
                    continue
                has_large_blob_nb = any(
                    by_id[nb]["type"] in ("pata_compacta", "banda_curba")
                    and (uv_boxes[nb][2] - uv_boxes[nb][0]) >= 0.55 * gs_ref
                    and (uv_boxes[nb][3] - uv_boxes[nb][1]) >= 0.45 * gs_ref
                    for nb in adj[sid_bar]
                )
                if has_large_blob_nb:
                    continue
                bar_pids_found.add(sid_bar)
                local_bars.append({
                    "molecule_id": mid,
                    "primitive_id": sid_bar,
                    "spanned_grid_ids": [g["grid_id"] for g in cand_grids],
                    "u_pos": round(0.5 * (su0 + su1), 2),
                    "v_span": [round(sv0, 2), round(sv1, 2)],
                    "length_px": s["length_px"],
                    "angle_deg": s["angle_deg"],
                })

        bridge_ids: set[int] = set()
        bridged_stick_ids: set[int] = set()
        local_bridges: List[Dict[str, Any]] = []
        non_bar_straights = [s for s in straights if int(s["id"]) not in bar_pids_found]
        if len(non_bar_straights) >= 3:
            for cand in non_bar_straights:
                cid = int(cand["id"])
                nb_straights = [
                    by_id[nb] for nb in adj[cid]
                    if nb not in bar_pids_found
                    and by_id[nb]["type"] == "banda_dreapta"
                    and _angle_diff_180(float(cand["angle_deg"]), float(by_id[nb]["angle_deg"])) >= 28.0
                    and _ribbon_touches_near_end(by_id[nb], cand)
                ]
                if len(nb_straights) < 2:
                    continue
                best_grp: List[Dict[str, Any]] = []
                for a_i in range(len(nb_straights)):
                    grp = [nb_straights[a_i]]
                    for b_i in range(len(nb_straights)):
                        if a_i != b_i and _angle_diff_180(
                            float(nb_straights[a_i]["angle_deg"]), float(nb_straights[b_i]["angle_deg"])
                        ) <= 22.0:
                            grp.append(nb_straights[b_i])
                    if len(grp) > len(best_grp):
                        best_grp = grp
                if len(best_grp) >= 2:
                    bridge_ids.add(cid)
                    p_ids = sorted(int(x["id"]) for x in best_grp)
                    bridged_stick_ids.update(p_ids)
                    local_bridges.append({
                        "molecule_id": mid,
                        "bridge_id": cid,
                        "bridge_primitive_id": cid,
                        "bridge_width_px": cand["width_px"],
                        "bridge_angle_deg": cand["angle_deg"],
                        "connected_parallel_ids": p_ids,
                        "zone": zone,
                        "grid_ids": in_grids,
                    })

        local_anchored: List[Dict[str, Any]] = []
        for s in non_bar_straights:
            sid = int(s["id"])
            if sid in bridge_ids:
                continue
            pa = np.array(s["start"], dtype=np.float64)
            pb = np.array(s["end"], dtype=np.float64)
            tol_end = max(4.0, 2.5 * float(s["width_px"]), 0.35 * float(s["length_px"]))
            end_a_att: List[int] = []
            end_b_att: List[int] = []
            mid_att: List[int] = []
            for nb in adj[sid]:
                np_obj = by_id[nb]
                if np_obj["type"] not in ("pata_compacta", "banda_curba"):
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
                local_anchored.append({
                    "molecule_id": mid,
                    "stick_id": sid,
                    "start_A": s["start"],
                    "end_B": s["end"],
                    "length_px": s["length_px"],
                    "width_px": s["width_px"],
                    "angle_deg": s["angle_deg"],
                    "on_bridge": sid in bridged_stick_ids,
                    "end_A_attachments": sorted(end_a_att),
                    "end_B_attachments": sorted(end_b_att),
                    "mid_attachments": sorted(mid_att),
                    "zone": zone,
                    "grid_ids": in_grids,
                })

        transverse_bars.extend(local_bars)
        parallel_bridges.extend(local_bridges)
        anchored_sticks.extend(local_anchored)

        if len(local_bars) == 1 and len(comp) == 1:
            mtype = "bara_transversala_grila"
        elif local_bridges:
            mtype = "grup_pe_pod"
        elif local_anchored:
            mtype = "bat_ancorat"
        elif local_bars:
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
            "zone": zone,
            "grid_ids": in_grids,
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
        if m["zone"] == "coridor_liber" and m["type"] not in ("bara_transversala_grila", "grup_pe_pod")
    ]
    rows: List[Dict[str, Any]] = []
    if len(row_cands) >= 3:
        row_cands.sort(key=lambda m: 0.5 * (m["uv_bbox"][1] + m["uv_bbox"][3]))
        bands: List[List[Dict[str, Any]]] = []
        for m in row_cands:
            mv0, mv1 = m["uv_bbox"][1], m["uv_bbox"][3]
            mh = max(1.0, mv1 - mv0)
            placed = False
            for band in bands:
                bv0 = min(x["uv_bbox"][1] for x in band)
                bv1 = max(x["uv_bbox"][3] for x in band)
                bh = max(1.0, bv1 - bv0)
                tol = 0.35 * max(bh, mh)
                if mv0 <= bv1 + tol and mv1 >= bv0 - tol:
                    b_vc = float(np.median([0.5 * (x["uv_bbox"][1] + x["uv_bbox"][3]) for x in band]))
                    if abs(0.5 * (mv0 + mv1) - b_vc) <= 0.75 * max(bh, mh) + 3.0:
                        band.append(m)
                        placed = True
                        break
            if not placed:
                bands.append([m])

        for band in bands:
            if len(band) < 3:
                continue
            band.sort(key=lambda m: m["uv_bbox"][0])
            heights = [m["v_span_px"] for m in band if m["v_span_px"] >= 3.0]
            med_h = float(np.median(heights)) if heights else 10.0
            max_gap = max(35.0, 4.5 * med_h)

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
        lbl = f"G{g['grid_id']}: {g['line_count']}L, S={g['step_px']}px"
        cv2.putText(v1, lbl, (int(first_p["start"][0]) + 4, max(18, int(first_p["start"][1]) - 6)),
                    font, 0.50, col, 2, cv2.LINE_AA)
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