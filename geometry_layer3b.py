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

SCHEMA_VERSION = "music_layer3b.v1.0"

class Layer3BGateError(RuntimeError):
    pass

@dataclass(frozen=True)
class Layer3BConfig:
    stem_max_width_in_S: float = 0.45
    stem_min_piece_len_in_S: float = 0.45
    stem_min_elongation: float = 2.2
    stem_merge_u_tol_in_S: float = 0.35
    stem_merge_v_gap_in_S: float = 0.75
    stem_min_total_height_in_S: float = 1.65

    nh_min_w_in_S: float = 0.55
    nh_max_w_in_S: float = 2.15
    nh_min_h_in_S: float = 0.50
    nh_max_h_in_S: float = 1.60
    nh_ribbon_max_elong: float = 2.60
    nh_ribbon_min_w_in_S: float = 0.45
    nh_attach_max_gap_in_S: float = 0.25
    nh_attach_u_tol_in_S: float = 1.40
    nh_attach_v_pad_in_S: float = 0.85
    nh_hollow_max_fill_ratio: float = 0.56

    whole_min_w_in_S: float = 0.85
    whole_max_w_in_S: float = 2.30
    whole_min_h_in_S: float = 0.65
    whole_max_h_in_S: float = 1.45

    beam_max_angle_deg: float = 38.0
    beam_min_width_in_S: float = 0.28
    beam_min_len_in_S: float = 1.00
    beam_min_elongation: float = 2.20
    beam_stem_max_gap_in_S: float = 0.32
    beam_free_end_tol_in_S: float = 1.25

    flag_min_len_in_S: float = 0.45
    flag_max_gap_in_S: float = 0.30
    flag_free_end_tol_in_S: float = 1.35

    dot_max_w_in_S: float = 0.48
    dot_max_h_in_S: float = 0.48
    dot_max_area_in_S2: float = 0.22
    dot_max_elongation: float = 2.20
    dot_min_du_in_S: float = 0.15
    dot_max_du_in_S: float = 1.55
    dot_max_dv_in_S: float = 0.60
    dot_max_gap_in_S: float = 1.15

    time_column_u_tol_in_S: float = 0.75

def validate_layer3a_gate(layer3a: Dict[str, Any]) -> None:
    sv = str(layer3a.get("schema_version", ""))
    if not sv.startswith("geometry_layer3a."):
        raise Layer3BGateError(f"Schema Strat 3A invalidă: {sv}.")
    audit = layer3a.get("audit_verification")
    if not isinstance(audit, dict) or audit.get("guaranteed_exact_once_accounting") is not True:
        raise Layer3BGateError("Audit Strat 3A neconfirmat: guaranteed_exact_once_accounting != True.")
    if not isinstance(layer3a.get("grids"), list):
        raise Layer3BGateError("Stratul 3A nu conține lista grids.")

def _duration_from_head_and_subdiv(head_kind: str, subdiv: int, dots: int) -> Tuple[str, float]:
    if head_kind == "whole":
        base_name, base_q = "whole", 4.0
    elif head_kind == "half":
        base_name, base_q = "half", 2.0
    else:
        sub = max(0, min(4, int(subdiv)))
        table = [
            ("quarter", 1.0),
            ("eighth", 0.5),
            ("16th", 0.25),
            ("32nd", 0.125),
            ("64th", 0.0625),
        ]
        base_name, base_q = table[sub]

    d_clamped = max(0, min(2, int(dots)))
    mult = 1.0 if d_clamped == 0 else (1.5 if d_clamped == 1 else 1.75)
    return base_name, round(base_q * mult, 6)

def _is_notehead_candidate(rec: Dict[str, Any], cfg: Layer3BConfig) -> bool:
    m = rec["metrics_in_S"]
    wu, hv = float(m["w_u_in_S"]), float(m["h_v_in_S"])
    if not (cfg.nh_min_w_in_S <= wu <= cfg.nh_max_w_in_S and cfg.nh_min_h_in_S <= hv <= cfg.nh_max_h_in_S):
        return False
    ltype = rec["layer1_type"]
    if ltype == "banda_dreapta":
        if float(m["elongation"]) > cfg.nh_ribbon_max_elong or float(m["width_in_S"]) < cfg.nh_ribbon_min_w_in_S:
            return False
    return True

def _classify_notehead_fill(rec: Dict[str, Any], cfg: Layer3BConfig) -> str:
    m = rec["metrics_in_S"]
    if bool(m.get("has_hole", False)):
        return "hollow"
    if rec["layer1_type"] != "banda_dreapta" and float(m["fill_ratio"]) < cfg.nh_hollow_max_fill_ratio:
        return "hollow"
    return "filled"

def _is_beam_candidate(rec: Dict[str, Any], cfg: Layer3BConfig) -> bool:
    if rec["layer1_type"] != "banda_dreapta":
        return False
    if rec["orientation_class"] not in ("parallel_to_grid", "oblique"):
        return False
    if float(rec["angle_rel_grid_deg"]) > cfg.beam_max_angle_deg:
        return False
    m = rec["metrics_in_S"]
    return (
        float(m["width_in_S"]) >= cfg.beam_min_width_in_S
        and float(m["length_in_S"]) >= cfg.beam_min_len_in_S
        and float(m["elongation"]) >= cfg.beam_min_elongation
    )

def _is_dot_candidate(rec: Dict[str, Any], cfg: Layer3BConfig) -> bool:
    m = rec["metrics_in_S"]
    return (
        float(m["w_u_in_S"]) <= cfg.dot_max_w_in_S
        and float(m["h_v_in_S"]) <= cfg.dot_max_h_in_S
        and float(m["area_in_S2"]) <= cfg.dot_max_area_in_S2
        and float(m["elongation"]) <= cfg.dot_max_elongation
    )

def _analyze_single_grid(
    grid: Dict[str, Any],
    cfg: Layer3BConfig,
    ) -> Dict[str, Any]:
    gid = int(grid["grid_id"])
    S = max(1e-6, float(grid["metrics"]["spacing_px"]))
    prim_recs: List[Dict[str, Any]] = grid.get("grid_primitives", [])
    rec_by_pid: Dict[int, Dict[str, Any]] = {int(r["primitive_id"]): r for r in prim_recs}

    contact_adj: Dict[int, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for c in grid.get("direct_contacts", []):
        pa, pb = int(c["prim_a"]), int(c["prim_b"])
        contact_adj[pa].append((pb, c))
        contact_adj[pb].append((pa, c))

    prox_adj: Dict[int, List[Tuple[int, Dict[str, Any], int]]] = defaultdict(list)
    for pr in grid.get("spatial_proximities", []):
        pa, pb = int(pr["prim_a"]), int(pr["prim_b"])
        prox_adj[pa].append((pb, pr, +1))
        prox_adj[pb].append((pa, pr, -1))

    stem_piece_pids: List[int] = []
    for r in prim_recs:
        if r["layer1_type"] != "banda_dreapta" or r["orientation_class"] != "perpendicular_to_grid":
            continue
        m = r["metrics_in_S"]
        if (float(m["width_in_S"]) <= cfg.stem_max_width_in_S and
                float(m["length_in_S"]) >= cfg.stem_min_piece_len_in_S and
                float(m["elongation"]) >= cfg.stem_min_elongation):
            stem_piece_pids.append(int(r["primitive_id"]))

    stem_piece_pids.sort(key=lambda pid: (rec_by_pid[pid]["center_uv"][0], rec_by_pid[pid]["uv_bbox"][1]))
    raw_stem_clusters: List[List[int]] = []
    for pid in stem_piece_pids:
        r = rec_by_pid[pid]
        uc = float(r["center_uv"][0])
        v0, v1 = float(r["uv_bbox"][1]), float(r["uv_bbox"][3])
        cid = int(r["cell_id"])
        placed = False
        for cl in raw_stem_clusters:
            if int(rec_by_pid[cl[0]]["cell_id"]) != cid:
                continue
            cl_u = float(np.mean([rec_by_pid[x]["center_uv"][0] for x in cl]))
            if abs(uc - cl_u) > cfg.stem_merge_u_tol_in_S * S:
                continue
            cl_v0 = min(float(rec_by_pid[x]["uv_bbox"][1]) for x in cl)
            cl_v1 = max(float(rec_by_pid[x]["uv_bbox"][3]) for x in cl)
            v_gap = max(cl_v0 - v1, 0.0, v0 - cl_v1)
            if v_gap <= cfg.stem_merge_v_gap_in_S * S:
                cl.append(pid)
                placed = True
                break
        if not placed:
            raw_stem_clusters.append([pid])

    stems: List[Dict[str, Any]] = []
    stem_pid_to_sid: Dict[int, int] = {}
    for cl in raw_stem_clusters:
        u_vals = [float(rec_by_pid[x]["center_uv"][0]) for x in cl]
        v_top = min(float(rec_by_pid[x]["uv_bbox"][1]) for x in cl)
        v_bot = max(float(rec_by_pid[x]["uv_bbox"][3]) for x in cl)
        h_in_S = (v_bot - v_top) / S
        if h_in_S < cfg.stem_min_total_height_in_S:
            continue
        sid = len(stems) + 1
        u_pos = float(np.mean(u_vals))
        cid = int(rec_by_pid[cl[0]]["cell_id"])
        stems.append({
            "stem_id": sid,
            "grid_id": gid,
            "cell_id": cid,
            "primitive_ids": sorted(cl),
            "u_pos": round(u_pos, 2),
            "v_top": round(v_top, 2),
            "v_bot": round(v_bot, 2),
            "height_in_S": round(h_in_S, 3),
            "direction": "unknown",
            "notehead_pids": [],
            "beam_ids": [],
            "flag_pids": [],
        })
        for pid in cl:
            stem_pid_to_sid[pid] = sid

    all_stem_pids = set(stem_pid_to_sid.keys())

    beam_cand_pids: set[int] = {
        int(r["primitive_id"]) for r in prim_recs
        if int(r["primitive_id"]) not in all_stem_pids and _is_beam_candidate(r, cfg)
    }

    assigned_nh_pids: set[int] = set()
    noteheads_by_pid: Dict[int, Dict[str, Any]] = {}

    for st in stems:
        sid = st["stem_id"]
        u_st = float(st["u_pos"])
        v_top, v_bot = float(st["v_top"]), float(st["v_bot"])
        v_mid = 0.5 * (v_top + v_bot)

        cand_nhs: set[int] = set()
        for spid in st["primitive_ids"]:
            for nb, c in contact_adj.get(spid, []):
                if nb in all_stem_pids or nb in beam_cand_pids:
                    continue
                if float(c.get("surface_gap_in_S", 99.0)) <= cfg.nh_attach_max_gap_in_S:
                    if _is_notehead_candidate(rec_by_pid[nb], cfg):
                        cand_nhs.add(nb)
            for nb, pr, _sign in prox_adj.get(spid, []):
                if nb in all_stem_pids or nb in beam_cand_pids:
                    continue
                if float(pr.get("exact_gap_in_S", 99.0)) <= cfg.nh_attach_max_gap_in_S:
                    if _is_notehead_candidate(rec_by_pid[nb], cfg):
                        cand_nhs.add(nb)

        valid_nhs: List[int] = []
        for npid in sorted(cand_nhs):
            nr = rec_by_pid[npid]
            nuc, nvc = float(nr["center_uv"][0]), float(nr["center_uv"][1])
            if abs(nuc - u_st) > cfg.nh_attach_u_tol_in_S * S:
                continue
            if nvc < v_top - cfg.nh_attach_v_pad_in_S * S or nvc > v_bot + cfg.nh_attach_v_pad_in_S * S:
                continue
            valid_nhs.append(npid)

        if not valid_nhs:
            continue

        mean_v_nh = float(np.mean([float(rec_by_pid[npid]["center_uv"][1]) for npid in valid_nhs]))
        st["direction"] = "up" if mean_v_nh >= v_mid else "down"
        st["notehead_pids"] = valid_nhs

        for npid in valid_nhs:
            nr = rec_by_pid[npid]
            fill_kind = _classify_notehead_fill(nr, cfg)
            assigned_nh_pids.add(npid)
            noteheads_by_pid[npid] = {
                "primitive_id": npid,
                "stem_id": sid,
                "grid_id": gid,
                "cell_id": int(nr["cell_id"]),
                "fill": fill_kind,
                "slot_int": int(nr["slot"]["slot_int"]),
                "slot_float": float(nr["slot"]["slot_float"]),
                "slot_parity": nr["slot"]["slot_parity"],
                "v_band": nr["slot"]["v_band"],
                "center_uv": nr["center_uv"],
                "center_xy": nr["center_xy"],
                "dot_pids": [],
            }

    valid_stems = [st for st in stems if st["notehead_pids"]]
    valid_stem_by_id = {st["stem_id"]: st for st in valid_stems}
    valid_stem_pid_to_sid = {pid: st["stem_id"] for st in valid_stems for pid in st["primitive_ids"]}

    beam_records: List[Dict[str, Any]] = []
    used_beam_pids: set[int] = set()
    stem_beam_counts: Counter[int] = Counter()

    for bpid in sorted(beam_cand_pids):
        if bpid in assigned_nh_pids:
            continue
        br = rec_by_pid[bpid]
        bv_mid = float(br["center_uv"][1])
        hit_sids: set[int] = set()

        for nb, c in contact_adj.get(bpid, []):
            if nb in valid_stem_pid_to_sid and float(c.get("surface_gap_in_S", 99.0)) <= cfg.beam_stem_max_gap_in_S:
                hit_sids.add(valid_stem_pid_to_sid[nb])
        for nb, pr, _sign in prox_adj.get(bpid, []):
            if nb in valid_stem_pid_to_sid and float(pr.get("exact_gap_in_S", 99.0)) <= cfg.beam_stem_max_gap_in_S:
                hit_sids.add(valid_stem_pid_to_sid[nb])

        free_end_sids: List[int] = []
        for sid in sorted(hit_sids):
            st = valid_stem_by_id[sid]
            free_v = float(st["v_top"]) if st["direction"] == "up" else float(st["v_bot"])
            if abs(bv_mid - free_v) <= cfg.beam_free_end_tol_in_S * S:
                free_end_sids.append(sid)

        if len(free_end_sids) >= 2:
            bid = len(beam_records) + 1
            used_beam_pids.add(bpid)
            beam_records.append({
                "beam_id": bid,
                "primitive_id": bpid,
                "grid_id": gid,
                "cell_id": int(br["cell_id"]),
                "connected_stem_ids": free_end_sids,
                "level": 1,
                "u_span": [br["uv_bbox"][0], br["uv_bbox"][2]],
                "center_xy": br["center_xy"],
            })
            for sid in free_end_sids:
                valid_stem_by_id[sid]["beam_ids"].append(bid)
                stem_beam_counts[sid] += 1

    beamed_sids = {sid for sid, cnt in stem_beam_counts.items() if cnt >= 1}
    for bpid in sorted(beam_cand_pids - used_beam_pids):
        if bpid in assigned_nh_pids:
            continue
        br = rec_by_pid[bpid]
        bv_mid = float(br["center_uv"][1])
        hit_sids = set()
        for nb, c in contact_adj.get(bpid, []):
            if nb in valid_stem_pid_to_sid and float(c.get("surface_gap_in_S", 99.0)) <= cfg.beam_stem_max_gap_in_S:
                hit_sids.add(valid_stem_pid_to_sid[nb])
        for nb, pr, _sign in prox_adj.get(bpid, []):
            if nb in valid_stem_pid_to_sid and float(pr.get("exact_gap_in_S", 99.0)) <= cfg.beam_stem_max_gap_in_S:
                hit_sids.add(valid_stem_pid_to_sid[nb])

        partial_sids = []
        for sid in sorted(hit_sids & beamed_sids):
            st = valid_stem_by_id[sid]
            free_v = float(st["v_top"]) if st["direction"] == "up" else float(st["v_bot"])
            if abs(bv_mid - free_v) <= (cfg.beam_free_end_tol_in_S + 0.6) * S:
                partial_sids.append(sid)

        if partial_sids:
            bid = len(beam_records) + 1
            used_beam_pids.add(bpid)
            beam_records.append({
                "beam_id": bid,
                "primitive_id": bpid,
                "grid_id": gid,
                "cell_id": int(br["cell_id"]),
                "connected_stem_ids": partial_sids,
                "level": 2,
                "u_span": [br["uv_bbox"][0], br["uv_bbox"][2]],
                "center_xy": br["center_xy"],
            })
            for sid in partial_sids:
                valid_stem_by_id[sid]["beam_ids"].append(bid)
                stem_beam_counts[sid] += 1

    used_flag_pids: set[int] = set()
    for st in valid_stems:
        sid = st["stem_id"]
        if stem_beam_counts[sid] > 0:
            continue
        if all(noteheads_by_pid[npid]["fill"] == "hollow" for npid in st["notehead_pids"]):
            continue
        u_st = float(st["u_pos"])
        free_v = float(st["v_top"]) if st["direction"] == "up" else float(st["v_bot"])
        flag_cands: set[int] = set()

        for spid in st["primitive_ids"]:
            for nb, c in contact_adj.get(spid, []):
                if nb in all_stem_pids or nb in assigned_nh_pids or nb in used_beam_pids:
                    continue
                if float(c.get("surface_gap_in_S", 99.0)) <= cfg.flag_max_gap_in_S:
                    flag_cands.add(nb)
            for nb, pr, _sign in prox_adj.get(spid, []):
                if nb in all_stem_pids or nb in assigned_nh_pids or nb in used_beam_pids:
                    continue
                if float(pr.get("exact_gap_in_S", 99.0)) <= cfg.flag_max_gap_in_S:
                    flag_cands.add(nb)

        found_flags: List[int] = []
        for fpid in sorted(flag_cands):
            fr = rec_by_pid[fpid]
            fm = fr["metrics_in_S"]
            if float(fm["length_in_S"]) < cfg.flag_min_len_in_S:
                continue
            if _is_dot_candidate(fr, cfg):
                continue
            fuc, fvc = float(fr["center_uv"][0]), float(fr["center_uv"][1])
            if fuc < u_st - 0.15 * S:
                continue
            if abs(fvc - free_v) > cfg.flag_free_end_tol_in_S * S:
                continue
            found_flags.append(fpid)
            used_flag_pids.add(fpid)

        st["flag_pids"] = found_flags

    whole_note_pids: List[int] = []

    occupied_pids = all_stem_pids | assigned_nh_pids | used_beam_pids | used_flag_pids
    for r in prim_recs:
        pid = int(r["primitive_id"])
        if pid in occupied_pids:
            continue
        if r["layer1_type"] not in ("pata_compacta", "banda_curba"):
            continue

        if not (cfg.whole_min_w_in_S <= wu <= cfg.whole_max_w_in_S and cfg.whole_min_h_in_S <= hv <= cfg.whole_max_h_in_S):
            continue
        if not (bool(m.get("has_hole", False)) or float(m["fill_ratio"]) < cfg.nh_hollow_max_fill_ratio):
            continue
        if any(nb in occupied_pids for nb, _ in contact_adj.get(pid, [])):
            continue
        whole_note_pids.append(pid)
        assigned_nh_pids.add(pid)
        noteheads_by_pid[pid] = {
            "primitive_id": pid,
            "stem_id": None,
            "grid_id": gid,
            "cell_id": int(r["cell_id"]),
            "fill": "whole",
            "slot_int": int(r["slot"]["slot_int"]),
            "slot_float": float(r["slot"]["slot_float"]),
            "slot_parity": r["slot"]["slot_parity"],
            "v_band": r["slot"]["v_band"],
            "center_uv": r["center_uv"],
            "center_xy": r["center_xy"],
            "dot_pids": [],
        }

    occupied_all = all_stem_pids | assigned_nh_pids | used_beam_pids | used_flag_pids
    used_dot_pids: set[int] = set()
    for npid in sorted(noteheads_by_pid.keys()):
        nh = noteheads_by_pid[npid]
        dot_hits: List[Tuple[float, int]] = []
        for nb, pr, sign in prox_adj.get(npid, []):
            if nb in occupied_all or nb in used_dot_pids:
                continue
            if not _is_dot_candidate(rec_by_pid[nb], cfg):
                continue
            du_s = float(pr["du_a_to_b_in_S"]) * sign
            dv_s = float(pr["dv_a_to_b_in_S"]) * sign
            gap_s = float(pr["exact_gap_in_S"])
            if (cfg.dot_min_du_in_S <= du_s <= cfg.dot_max_du_in_S and
                    abs(dv_s) <= cfg.dot_max_dv_in_S and
                    gap_s <= cfg.dot_max_gap_in_S):
                dot_hits.append((du_s, nb))
        dot_hits.sort(key=lambda t: t[0])
        chosen = [nb for _, nb in dot_hits[:2]]
        nh["dot_pids"] = chosen
        used_dot_pids.update(chosen)

    events: List[Dict[str, Any]] = []
    for st in valid_stems:
        sid = st["stem_id"]
        nh_list = [noteheads_by_pid[npid] for npid in st["notehead_pids"]]
        nh_list.sort(key=lambda x: x["slot_int"])
        has_hollow = any(nh["fill"] == "hollow" for nh in nh_list)
        head_kind = "half" if has_hollow else "filled"
        beam_cnt = int(stem_beam_counts[sid])
        flag_cnt = len(st["flag_pids"])
        subdiv = beam_cnt if beam_cnt > 0 else flag_cnt
        dots = max((len(nh["dot_pids"]) for nh in nh_list), default=0)
        dur_name, dur_q = _duration_from_head_and_subdiv(head_kind, subdiv, dots)
        voice = "voice_1" if st["direction"] == "up" else "voice_2"
        u_event = float(np.mean([float(nh["center_uv"][0]) for nh in nh_list]))

        events.append({
            "event_id": len(events) + 1,
            "grid_id": gid,
            "cell_id": int(st["cell_id"]),
            "event_kind": "chord" if len(nh_list) > 1 else "note",
            "voice": voice,
            "stem_id": sid,
            "stem_direction": st["direction"],
            "u_pos": round(u_event, 2),
            "duration_name": dur_name,
            "duration_quarters": dur_q,
            "dots": dots,
            "beam_count": beam_cnt,
            "beam_ids": list(st["beam_ids"]),
            "flag_count": flag_cnt,
            "slots_int": [nh["slot_int"] for nh in nh_list],
            "noteheads": nh_list,
        })

    for wpid in whole_note_pids:
        nh = noteheads_by_pid[wpid]
        dots = len(nh["dot_pids"])
        dur_name, dur_q = _duration_from_head_and_subdiv("whole", 0, dots)
        events.append({
            "event_id": len(events) + 1,
            "grid_id": gid,
            "cell_id": int(nh["cell_id"]),
            "event_kind": "note",
            "voice": "voice_1",
            "stem_id": None,
            "stem_direction": "none",
            "u_pos": round(float(nh["center_uv"][0]), 2),
            "duration_name": dur_name,
            "duration_quarters": dur_q,
            "dots": dots,
            "beam_count": 0,
            "beam_ids": [],
            "flag_count": 0,
            "slots_int": [nh["slot_int"]],
            "noteheads": [nh],
        })

    events.sort(key=lambda e: (e["cell_id"], e["u_pos"], e["voice"], e["event_id"]))
    for idx_e, ev in enumerate(events, start=1):
        ev["event_id"] = idx_e

    events_by_cell: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        events_by_cell[int(ev["cell_id"])].append(ev)

    cells_out: List[Dict[str, Any]] = []
    for cell in grid.get("cells", []):
        cid = int(cell["cell_id"])
        cevs = sorted(events_by_cell.get(cid, []), key=lambda e: (e["u_pos"], e["voice"]))
        columns: List[Dict[str, Any]] = []
        for ev in cevs:
            u_e = float(ev["u_pos"])
            if columns and abs(u_e - float(columns[-1]["u_pos"])) <= cfg.time_column_u_tol_in_S * S:
                columns[-1]["event_ids"].append(ev["event_id"])
                columns[-1]["u_vals"].append(u_e)
                columns[-1]["u_pos"] = round(float(np.mean(columns[-1]["u_vals"])), 2)
            else:
                columns.append({
                    "column_index": len(columns) + 1,
                    "u_pos": round(u_e, 2),
                    "u_vals": [u_e],
                    "event_ids": [ev["event_id"]],
                })
        for col in columns:
            del col["u_vals"]

        dur_by_voice: Dict[str, float] = defaultdict(float)
        for ev in cevs:
            dur_by_voice[ev["voice"]] = round(dur_by_voice[ev["voice"]] + float(ev["duration_quarters"]), 6)

        cells_out.append({
            "cell_id": cid,
            "cell_kind": cell["cell_kind"],
            "u_span": [cell["u_start"], cell["u_end"]],
            "right_separator_kind": cell["right_separator_kind"],
            "event_count": len(cevs),
            "column_count": len(columns),
            "duration_quarters_by_voice": dict(sorted(dur_by_voice.items())),
            "time_columns": columns,
        })

    semantic_used_pids = (
        set(valid_stem_pid_to_sid.keys())
        | assigned_nh_pids
        | used_beam_pids
        | used_flag_pids
        | used_dot_pids
    )
    unclassified_pids = sorted(set(rec_by_pid.keys()) - semantic_used_pids)

    return {
        "grid_id": gid,
        "spacing_px": round(S, 3),
        "counts": {
            "stems": len(valid_stems),
            "noteheads": len(noteheads_by_pid),
            "events": len(events),
            "beams": len(beam_records),
            "flags": len(used_flag_pids),
            "dots": len(used_dot_pids),
            "whole_notes": len(whole_note_pids),
            "unclassified_grid_primitives": len(unclassified_pids),
        },
        "stems": valid_stems,
        "beams": beam_records,
        "events": events,
        "cells": cells_out,
        "unclassified_primitive_ids": unclassified_pids,
    }

def analyze_layer3b(
    layer3a: Dict[str, Any],
    config: Optional[Layer3BConfig] = None,
    ) -> Dict[str, Any]:
    cfg = config if config is not None else Layer3BConfig()
    validate_layer3a_gate(layer3a)

    grids_out = [_analyze_single_grid(g, cfg) for g in layer3a.get("grids", [])]

    return {
        "schema_version": SCHEMA_VERSION,
        "layer_scope": "stem_first_music_semantics_stage1_notes_stems_beams_durations",
        "source_image": layer3a.get("source_image"),
        "upstream_schema_version": layer3a.get("schema_version"),
        "config": asdict(cfg),
        "systems": layer3a.get("systems", []),
        "totals": {
            "grids": len(grids_out),
            "stems": sum(g["counts"]["stems"] for g in grids_out),
            "noteheads": sum(g["counts"]["noteheads"] for g in grids_out),
            "events": sum(g["counts"]["events"] for g in grids_out),
            "beams": sum(g["counts"]["beams"] for g in grids_out),
            "flags": sum(g["counts"]["flags"] for g in grids_out),
            "dots": sum(g["counts"]["dots"] for g in grids_out),
            "whole_notes": sum(g["counts"]["whole_notes"] for g in grids_out),
            "unclassified_grid_primitives": sum(g["counts"]["unclassified_grid_primitives"] for g in grids_out),
        },
        "grids": grids_out,
    }

def _load_canvas_l3b(layer3a: Dict[str, Any], image_override: Optional[str]) -> np.ndarray:
    img_path = image_override or layer3a.get("source_image")
    if img_path and Path(img_path).exists():
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is not None:
            return (img.astype(np.float32) * 0.40 + 20.0).astype(np.uint8)
    max_x, max_y = 640, 480
    for g in layer3a.get("grids", []):
        for r in g.get("grid_primitives", []):
            b = r["xy_bbox"]
            max_x = max(max_x, int(math.ceil(b[2])) + 40)
            max_y = max(max_y, int(math.ceil(b[3])) + 40)
    return np.zeros((max_y, max_x, 3), dtype=np.uint8)

def draw_debug_layer3b(
    layer3a: Dict[str, Any],
    layer3b: Dict[str, Any],
    out_dir: Path,
    image_override: Optional[str] = None,
    ) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = _load_canvas_l3b(layer3a, image_override)
    font = cv2.FONT_HERSHEY_SIMPLEX

    rec_by_pid: Dict[int, Dict[str, Any]] = {}
    for g in layer3a.get("grids", []):
        for r in g.get("grid_primitives", []):
            rec_by_pid[int(r["primitive_id"])] = r

    dur_short = {
        "whole": "1",
        "half": "2",
        "quarter": "4",
        "eighth": "8",
        "16th": "16",
        "32nd": "32",
        "64th": "64",
    }

    for g in layer3b["grids"]:
        for b in g["beams"]:
            r = rec_by_pid[b["primitive_id"]]
            ep = r.get("endpoints")
            if ep:
                p0 = (int(round(ep["raw_A_xy"][0])), int(round(ep["raw_A_xy"][1])))
                p1 = (int(round(ep["raw_B_xy"][0])), int(round(ep["raw_B_xy"][1])))
                cv2.line(canvas, p0, p1, (0, 145, 255), 3, cv2.LINE_AA)

        for st in g["stems"]:
            col = (80, 235, 80) if st["direction"] == "up" else (240, 180, 50)
            for pid in st["primitive_ids"]:
                r = rec_by_pid[pid]
                ep = r.get("endpoints")
                if ep:
                    p0 = (int(round(ep["raw_A_xy"][0])), int(round(ep["raw_A_xy"][1])))
                    p1 = (int(round(ep["raw_B_xy"][0])), int(round(ep["raw_B_xy"][1])))
                    cv2.line(canvas, p0, p1, col, 2, cv2.LINE_AA)
            for fpid in st["flag_pids"]:
                bx = rec_by_pid[fpid]["xy_bbox"]
                cv2.rectangle(canvas, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (240, 0, 240), 2)

        for ev in g["events"]:
            for nh in ev["noteheads"]:
                cx, cy = int(round(nh["center_xy"][0])), int(round(nh["center_xy"][1]))
                if nh["fill"] == "whole":
                    col = (0, 235, 235)
                elif nh["fill"] == "hollow":
                    col = (235, 220, 0)
                else:
                    col = (0, 215, 60)
                cv2.circle(canvas, (cx, cy), 5, col, 2, cv2.LINE_AA)
                for dpid in nh["dot_pids"]:
                    dc = rec_by_pid[dpid]["center_xy"]
                    cv2.circle(canvas, (int(round(dc[0])), int(round(dc[1]))), 3, (0, 0, 250), -1)

            top_nh = min(ev["noteheads"], key=lambda x: x["center_xy"][1])
            lx = int(round(top_nh["center_xy"][0])) - 8
            ly = max(14, int(round(top_nh["center_xy"][1])) - 10)
            lbl = dur_short.get(ev["duration_name"], "?") + ("." * ev["dots"])
            cv2.putText(canvas, lbl, (lx, ly), font, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    out_img = out_dir / "debug_l3b_1_notes_stems_beams.png"
    cv2.imwrite(str(out_img), canvas)
    return {"debug_l3b_1_notes_stems_beams": str(out_img)}

def process_layer3b(
    layer3a_json_path: str | Path,
    out_dir: str | Path = "debug_l3b",
    image_override: Optional[str] = None,
    config: Optional[Layer3BConfig] = None,
    ) -> Dict[str, Any]:
    l3a_path = Path(layer3a_json_path)
    layer3a = json.loads(l3a_path.read_text(encoding="utf-8"))
    layer3b = analyze_layer3b(layer3a, config=config)
    layer3b["debug_images"] = draw_debug_layer3b(layer3a, layer3b, Path(out_dir), image_override)

    stem = l3a_path.stem.replace("_geometry_layer3a", "")
    out_path = Path(out_dir) / f"{stem}_music_layer3b.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(layer3b, ensure_ascii=False, indent=2), encoding="utf-8")
    layer3b["json_file"] = str(out_path)
    return layer3b

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("layer3a_json")
    parser.add_argument("--out-dir", default="debug_l3b")
    parser.add_argument("--image", default=None)
    args = parser.parse_args(argv)
    try:
        res = process_layer3b(args.layer3a_json, args.out_dir, args.image)
    except Layer3BGateError as gexc:
        print(f"POARTĂ RESPINSĂ: {gexc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"Eroare: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "schema_version": res["schema_version"],
        "totals": res["totals"],
        "debug_images": res["debug_images"],
        "json_file": res["json_file"],
    }, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
