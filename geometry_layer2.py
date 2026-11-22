#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometrie pură (Stratul 2): Topologie, grile periodice, poduri, molecule și rânduri.

Citește EXCLUSIV JSON-ul produs de geometry_layer1.py. Nu citește masca de pixeli.

Condiție obligatorie de intrare:
  - layer0_dss_xor_error_pixels == 0
  - layer1_uncovered_ink_pixels == 0
  - guaranteed_100pct_coverage == True

Garanție matematică:
  Fiecare primitive_id apare fie într-o singură grilă, fie într-o singură moleculă.
  Dublurile și primitivele necontabilizate opresc analiza.

Utilizare:
  python geometry_layer2.py debug_geo/poza_geometry_layer1.json --out-dir debug_l2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
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
        pts = np.array([p["start"], p["end"]], dtype=np.float64)
        hw = 0.5 * float(p["width_px"])
        us = pts @ u_vec
        vs = pts @ v_vec
        return float(np.min(us)) - hw, float(np.min(vs)) - hw, float(np.max(us)) + hw, float(np.max(vs)) + hw
    x0, y0, x1, y1 = p["bbox"]
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    us = corners @ u_vec
    vs = corners @ v_vec
    return float(np.min(us)), float(np.min(vs)), float(np.max(us)), float(np.max(vs))


def _prim_xy_box(p: Dict[str, Any]) -> Tuple[float, float, float, float]:
    if p["type"] == "banda_dreapta":
        x0, y0 = p["start"]
        x1, y1 = p["end"]
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
    x0, y0, x1, y1 = p["bbox"]
    dx = max(x0 - pt[0], 0.0, pt[0] - x1)
    dy = max(y0 - pt[1], 0.0, pt[1] - y1)
    return math.hypot(dx, dy)


def validate_layer1(layer1: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], set[int]]:
    """Refuză orice intrare care nu conține dovezile complete ale Straturilor 0 și 1."""
    verification = layer1.get("lossless_verification")
    if not isinstance(verification, dict):
        raise ValueError("JSON-ul nu conține lossless_verification; Stratul 2 nu poate porni.")
    if int(verification.get("layer0_dss_xor_error_pixels", -1)) != 0:
        raise ValueError("Stratul 0 nu a confirmat XOR = 0; Stratul 2 a fost blocat.")
    if int(verification.get("layer1_uncovered_ink_pixels", -1)) != 0:
        raise ValueError("Stratul 1 are pixeli de cerneală neacoperiți; Stratul 2 a fost blocat.")
    if verification.get("guaranteed_100pct_coverage") is not True:
        raise ValueError("Stratul 1 nu a confirmat acoperirea 100%; Stratul 2 a fost blocat.")

    primitives = layer1.get("primitives")
    if not isinstance(primitives, list):
        raise ValueError("JSON-ul Stratul 1 nu conține lista primitives.")
    ids: List[int] = []
    for item in primitives:
        if not isinstance(item, dict) or "id" not in item or "type" not in item:
            raise ValueError("Lista primitives conține un element invalid.")
        ids.append(int(item["id"]))
    if len(ids) != len(set(ids)):
        raise ValueError("Stratul 1 conține primitive cu ID-uri duplicate.")
    if len(ids) != int(layer1.get("counts", {}).get("primitives_total", len(ids))):
        raise ValueError("Numărul primitivelor nu corespunde sumarului Stratul 1.")
    return primitives, set(ids)


# ---------------------------------------------------------------------------
# 1. Estimarea axei dominante a paginii (auto-deskew pur geometric)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2. Detecția grilelor paralele periodice (fără număr hardcodat de linii)
# ---------------------------------------------------------------------------
def _track_v_at(tr: Dict[str, Any], u_val: float) -> float:
    return tr["v0"] + tr["slope"] * (u_val - tr["u_ref"])


def _fit_track_line(members: Sequence[Dict[str, Any]]) -> Tuple[float, float, float]:
    """Potrivește v(u) = v0 + slope * (u - u_ref) ponderat cu lungimea segmentelor."""
    us = np.array([x for m in members for x in (m["ua"], m["ub"])], dtype=np.float64)
    vs = np.array([y for m in members for y in (m["va"], m["vb"])], dtype=np.float64)
    ws = np.array([m["len"] for m in members for _ in range(2)], dtype=np.float64)
    w_sum = float(np.sum(ws))
    u_ref = float(np.sum(ws * us) / w_sum)
    v0 = float(np.sum(ws * vs) / w_sum)
    du = us - u_ref
    denom = float(np.sum(ws * du * du))
    slope = float(np.sum(ws * du * (vs - v0)) / denom) if denom > 1e-6 else 0.0
    slope = max(-0.12, min(0.12, slope))
    return u_ref, v0, slope


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
        if _angle_diff_180(float(p["angle_deg"]), theta_deg) > 7.0:
            continue
        if float(p["elongation"]) < 7.0:
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

    aligned.sort(key=lambda s: s["len"], reverse=True)
    tracks: List[Dict[str, Any]] = []
    for seg in aligned:
        best_tr = None
        best_err = float("inf")
        for tr in tracks:
            tol_v = max(2.8, 1.7 * max(tr["w"], seg["w"]))
            err_a = abs(seg["va"] - _track_v_at(tr, seg["ua"]))
            err_b = abs(seg["vb"] - _track_v_at(tr, seg["ub"]))
            err = max(err_a, err_b)
            if err <= tol_v and err < best_err:
                best_err = err
                best_tr = tr
        if best_tr is not None:
            best_tr["members"].append(seg)
            best_tr["u0"] = min(best_tr["u0"], seg["ua"])
            best_tr["u1"] = max(best_tr["u1"], seg["ub"])
            best_tr["total_len"] += seg["len"]
            best_tr["w"] = sum(m["w"] * m["len"] for m in best_tr["members"]) / best_tr["total_len"]
            best_tr["u_ref"], best_tr["v0"], best_tr["slope"] = _fit_track_line(best_tr["members"])
        else:
            u_ref, v0, slope = _fit_track_line([seg])
            tracks.append({
                "members": [seg],
                "u0": seg["ua"], "u1": seg["ub"],
                "total_len": seg["len"], "w": seg["w"],
                "u_ref": u_ref, "v0": v0, "slope": slope,
            })

    spans = np.array([tr["u1"] - tr["u0"] for tr in tracks], dtype=np.float64)
    if spans.size < 3:
        return [], set()
    ref_span = float(np.percentile(spans, 85))
    long_tracks = [
        tr for tr in tracks
        if (tr["u1"] - tr["u0"]) >= max(40.0, 0.35 * ref_span)
        and tr["total_len"] >= 0.40 * (tr["u1"] - tr["u0"])
    ]
    if len(long_tracks) < 3:
        return [], set()

    u_page_mid = float(np.median([0.5 * (tr["u0"] + tr["u1"]) for tr in long_tracks]))
    long_tracks.sort(key=lambda tr: _track_v_at(tr, u_page_mid))

    def tracks_overlap_and_dv(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[float, float]:
        ov_lo = max(a["u0"], b["u0"])
        ov_hi = min(a["u1"], b["u1"])
        ov = ov_hi - ov_lo
        min_sp = max(1e-6, min(a["u1"] - a["u0"], b["u1"] - b["u0"]))
        u_eval = 0.5 * (ov_lo + ov_hi) if ov > 0 else u_page_mid
        dv = _track_v_at(b, u_eval) - _track_v_at(a, u_eval)
        return ov / min_sp, dv

    adj_gaps: List[float] = []
    for a, b in zip(long_tracks, long_tracks[1:]):
        ov_ratio, dv = tracks_overlap_and_dv(a, b)
        if ov_ratio >= 0.35 and dv > 2.0 * max(a["w"], b["w"]):
            adj_gaps.append(dv)
    if not adj_gaps:
        return [], set()

    adj_gaps.sort()
    best_cluster: List[float] = []
    for g in adj_gaps:
        cluster = [x for x in adj_gaps if abs(x - g) <= 0.18 * g + 1.5]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    if len(best_cluster) < 2:
        return [], set()
    step_s = float(np.median(best_cluster))

    grids: List[Dict[str, Any]] = []
    used: set[int] = set()
    for i in range(len(long_tracks)):
        if i in used:
            continue
        chain = [i]
        cur = long_tracks[i]
        for j in range(i + 1, len(long_tracks)):
            if j in used:
                continue
            cand = long_tracks[j]
            ov_ratio, dv = tracks_overlap_and_dv(cur, cand)
            if dv > 1.32 * step_s + 2.5:
                break
            if ov_ratio >= 0.35 and abs(dv - step_s) <= max(2.2, 0.24 * step_s):
                chain.append(j)
                cur = cand
        if len(chain) >= 3:
            for idx_t in chain:
                used.add(idx_t)
            trs = [long_tracks[idx_t] for idx_t in chain]
            u_lo = min(tr["u0"] for tr in trs)
            u_hi = max(tr["u1"] for tr in trs)
            u_mid = 0.5 * (u_lo + u_hi)
            line_vs = [_track_v_at(tr, u_mid) for tr in trs]
            steps = [line_vs[k + 1] - line_vs[k] for k in range(len(line_vs) - 1)]
            line_ids = sorted(m["id"] for tr in trs for m in tr["members"])
            grids.append({
                "grid_id": len(grids) + 1,
                "line_count": len(trs),
                "step_px": round(float(np.median(steps)), 2),
                "angle_deg": round(theta_deg, 2),
                "u_span": [round(u_lo, 2), round(u_hi, 2)],
                "v_span": [round(line_vs[0], 2), round(line_vs[-1], 2)],
                "line_v_coords": [round(v, 2) for v in line_vs],
                "primitive_ids": line_ids,
            })

    grid_pids = {pid for g in grids for pid in g["primitive_ids"]}
    return grids, grid_pids


# ---------------------------------------------------------------------------
# 3. Analiza completă de Strat 2 (molecule, bare, poduri, ancore, rânduri)
# ---------------------------------------------------------------------------
def analyze_layer2(layer1: Dict[str, Any]) -> Dict[str, Any]:
    primitives, all_ids = validate_layer1(layer1)
    by_id: Dict[int, Dict[str, Any]] = {int(p["id"]): p for p in primitives}

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

        is_grid_bar = False
        spanned_grids: List[int] = []
        if len(straights) == 1 and not spots and not curves and grids:
            s = straights[0]
            if 65.0 <= _angle_diff_180(float(s["angle_deg"]), theta_deg) <= 115.0:
                su0, sv0, su1, sv1 = uv_boxes[s["id"]]
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
                if cand_grids:
                    top_gv0 = min(g["v_span"][0] for g in cand_grids)
                    bot_gv1 = max(g["v_span"][1] for g in cand_grids)
                    gs_ref = cand_grids[0]["step_px"]
                    if sv0 >= top_gv0 - 0.85 * gs_ref and sv1 <= bot_gv1 + 0.85 * gs_ref:
                        is_grid_bar = True
                        spanned_grids = [g["grid_id"] for g in cand_grids]

        bridge_ids: set[int] = set()
        bridged_stick_ids: set[int] = set()
        local_bridges: List[Dict[str, Any]] = []
        if len(straights) >= 3:
            for cand in straights:
                cid = int(cand["id"])
                nb_straights = [
                    by_id[nb] for nb in adj[cid]
                    if by_id[nb]["type"] == "banda_dreapta"
                    and _angle_diff_180(float(cand["angle_deg"]), float(by_id[nb]["angle_deg"])) >= 28.0
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
                        "bridge_width_px": cand["width_px"],
                        "bridge_angle_deg": cand["angle_deg"],
                        "connected_parallel_ids": p_ids,
                        "zone": zone,
                        "grid_ids": in_grids,
                    })

        local_anchored: List[Dict[str, Any]] = []
        for s in straights:
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

        if is_grid_bar:
            mtype = "bara_transversala_grila"
            transverse_bars.append({
                "molecule_id": mid,
                "primitive_id": straights[0]["id"],
                "spanned_grid_ids": spanned_grids,
                "u_pos": round(0.5 * (mu0 + mu1), 2),
                "v_span": [round(mv0, 2), round(mv1, 2)],
                "length_px": straights[0]["length_px"],
                "angle_deg": straights[0]["angle_deg"],
            })
        elif local_bridges:
            mtype = "grup_pe_pod"
            parallel_bridges.extend(local_bridges)
        elif local_anchored:
            mtype = "bat_ancorat"
        elif len(straights) >= 2 and not spots and not curves:
            mtype = "retea_benzi_drepte"
        elif len(comp) == 1:
            mtype = f"{by_id[comp[0]]['type']}_izolata"
        else:
            mtype = "molecula_mixta"

        molecules.append({
            "molecule_id": mid,
            "type": mtype,
            "zone": zone,
            "grid_ids": in_grids,
            "primitive_ids": comp,
            "primitive_count": len(comp),
            "uv_bbox": [round(mu0, 2), round(mv0, 2), round(mu1, 2), round(mv1, 2)],
            "xy_bbox": [
                round(min(_prim_xy_box(by_id[i])[0] for i in comp), 1),
                round(min(_prim_xy_box(by_id[i])[1] for i in comp), 1),
                round(max(_prim_xy_box(by_id[i])[2] for i in comp), 1),
                round(max(_prim_xy_box(by_id[i])[3] for i in comp), 1),
            ],
            "u_span_px": round(m_w, 2),
            "v_span_px": round(m_h, 2),
            "row_id": None,
        })

        if is_grid_bar:
            molecules[-1]["spanned_grid_ids"] = spanned_grids

        anchored_sticks.extend(local_anchored)

    # 4. Detecția șirurilor aliniate în coridoare libere (geometrie pură de rând)
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

    # 5. Verificare strictă: fiecare primitivă apare într-o singură grilă sau moleculă.
    grid_occurrences: Dict[int, int] = defaultdict(int)
    molecule_occurrences: Dict[int, int] = defaultdict(int)
    for grid in grids:
        for pid in grid["primitive_ids"]:
            grid_occurrences[int(pid)] += 1
    for molecule in molecules:
        for pid in molecule["primitive_ids"]:
            molecule_occurrences[int(pid)] += 1

    bad_duplicates = sorted(
        pid for pid in all_ids
        if grid_occurrences[pid] + molecule_occurrences[pid] > 1
    )
    missing_ids = sorted(
        pid for pid in all_ids
        if grid_occurrences[pid] + molecule_occurrences[pid] == 0
    )
    if bad_duplicates:
        raise RuntimeError(f"Primitive duplicate între grile/molecule: {bad_duplicates[:12]}")
    if missing_ids:
        raise RuntimeError(f"Contabilitate incompletă în Stratul 2; primitive lipsă: {missing_ids[:12]}")

    accounted_grid = {pid for pid, n in grid_occurrences.items() if n == 1}
    accounted_mol = {pid for pid, n in molecule_occurrences.items() if n == 1}
    if not accounted_grid.isdisjoint(accounted_mol) or (accounted_grid | accounted_mol) != all_ids:
        raise RuntimeError("Auditul final al contabilității Stratul 2 a eșuat.")

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
            "grid_primitives": len(accounted_grid),
            "molecule_primitives": len(accounted_mol),
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


# ---------------------------------------------------------------------------
# 4. Vizualizare debug (3 imagini peste poza originală sau pânză neagră)
# ---------------------------------------------------------------------------
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
        _draw_prim(v2, by_id[b["bridge_id"]], (0, 145, 255), 3)
        for sid in b["connected_parallel_ids"]:
            _draw_prim(v2, by_id[sid], (0, 235, 110), 2)
    for a in layer2["anchored_sticks"]:
        sid = a["stick_id"]
        if sid not in bridged_pids and sid not in bridge_pids:
            _draw_prim(v2, by_id[sid], (0, 210, 0), 2)
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
    # Validarea are loc înainte de orice analiză de topologie.
    validate_layer1(layer1)
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
    parser = argparse.ArgumentParser(description="Motor de geometrie pură Stratul 2 (topologie, grile, molecule)")
    parser.add_argument("layer1_json", help="Fișierul *_geometry_layer1.json produs de Stratul 1")
    parser.add_argument("--out-dir", default="debug_l2", help="Directorul de ieșire")
    parser.add_argument("--image", default=None, help="Opțional: imaginea originală pentru overlay debug")
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
