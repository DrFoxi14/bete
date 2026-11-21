#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motor geometric determinist (v3) în 2 faze pentru detectarea TUTUROR bețelor.

Reparații față de v2:
  1. Binarizare prin estimarea fundalului hârtiei (nu golește literele groase de titlu).
  2. Separarea petelor compacte (ovale pline lipite de linii subțiri) înainte de
     subțiere + podire coliniară strict peste cerneală (liniile paralele nu se mai
     îndoaie și nu se mai intersectează în interiorul notelor pline ca Fa).
  3. Grupare de joncțiuni pe rază LOCALĂ (nu mediana globală a paginii).
  4. Segmentare pe curbură strict locală + lungime fizică efectivă (păstrează ambele
     jumătăți ale lui `t`, `i`-ul gros din titlu și bețele din versurile mici).

Dependențe: numpy, opencv-python, scipy, scikit-image.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

Point = Tuple[int, int]


@dataclass
class Chain:
    points: np.ndarray              # N x 2 float32 (x, y)
    start_node: Optional[int]
    end_node: Optional[int]
    closed: bool = False


@dataclass
class Piece:
    chain_id: int
    piece_id: int
    points: np.ndarray
    start_index: int
    end_index: int
    mean_width: float
    start_width: float
    end_width: float
    kind: str                       # "line" sau "curve"
    start_node: Optional[int] = None
    end_node: Optional[int] = None


@dataclass
class Stick:
    stick_id: int
    points: np.ndarray              # N x 2 float32 (x, y)
    mean_width: float
    length_px: float
    effective_length_px: float
    angle_deg: float                # [0, 180)
    sagitta_ratio: float
    source_piece_ids: List[int]
    crossed_nodes: set[int] = field(default_factory=set)
    # Câmpuri completate în Faza 2 (relațional, fără distrugere)
    angle_family: int = -1
    family_size: int = 1
    thickness_ratio: float = 1.0
    role: str = "bat_izolat"
    role_reason: str = ""
    l_contacts: List[int] = field(default_factory=list)
    t_touches_trunk: List[int] = field(default_factory=list)
    t_receives_branch: List[int] = field(default_factory=list)
    x_crossings: List[int] = field(default_factory=list)


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _segment_angle(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    centered = points - np.mean(points, axis=0)
    cov = centered.T @ centered
    vals, vecs = np.linalg.eigh(cov)
    principal = vecs[:, int(np.argmax(vals))]
    return float(math.degrees(math.atan2(float(principal[1]), float(principal[0]))) % 180.0)


def _angle_diff_180(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _straightness_metrics(points: np.ndarray) -> Tuple[float, float]:
    """Returnează (sagitta_ratio, tortuosity) față de coarda capăt-capăt."""
    if len(points) < 3:
        return 0.0, 1.0
    p0, p1 = points[0], points[-1]
    chord_vec = p1 - p0
    chord_len = float(np.linalg.norm(chord_vec))
    arc_len = _polyline_length(points)
    if chord_len < 1e-6:
        return 1.0, float("inf")
    rel = points - p0
    cross = np.abs(rel[:, 0] * chord_vec[1] - rel[:, 1] * chord_vec[0])
    max_dev = float(np.max(cross) / chord_len)
    return max_dev / chord_len, arc_len / chord_len


def _end_tangent_turn_deg(points: np.ndarray, mean_width: float) -> float:
    """Unghiul (în grade) dintre direcția de la început și direcția de la sfârșit."""
    n = len(points)
    if n < 5:
        return 0.0
    k = min(max(2, n // 3), max(2, int(round(1.5 * mean_width))))
    v0 = points[k] - points[0]
    v1 = points[-1] - points[-1 - k]
    n0 = float(np.linalg.norm(v0))
    n1 = float(np.linalg.norm(v1))
    if n0 < 1e-6 or n1 < 1e-6:
        return 0.0
    dot = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
    return math.degrees(math.acos(dot))


# ---------------------------------------------------------------------------
# Pas 1: Binarizare prin estimare de fundal (nu golește literele uriașe!)
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


def binarize_ink(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extrage cerneala față de iluminarea estimată a hârtiei (robustă la litere groase și umbre)."""
    h, w = image.shape[:2]
    paper, paper_method = _safe_paper_region(image)
    darkest = np.min(image, axis=2)

    # Estimăm fundalul hârtiei la rezoluție redusă cu o închidere morfologică mare
    # (peste ~7% din latura mică), suficientă să sară peste litere uriașe de titlu.
    scale = max(1, min(h, w) // 420)
    sh, sw = max(16, h // scale), max(16, w // scale)
    small = cv2.resize(darkest, (sw, sh), interpolation=cv2.INTER_AREA)
    ksize = _odd(max(15, int(min(sh, sw) * 0.085)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    bg_small = cv2.GaussianBlur(bg_small, (_odd(ksize // 2), _odd(ksize // 2)), 0)
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)

    diff = np.clip(bg.astype(np.int16) - darkest.astype(np.int16), 0, 255).astype(np.uint8)
    paper_vals = diff[paper > 0]
    if paper_vals.size == 0:
        return np.zeros((h, w), np.uint8), paper, {"threshold_diff": 0.0, "paper_method": paper_method}

    otsu_val, _ = cv2.threshold(paper_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Prag ușor generos ca să păstreze și literele mici / trăsăturile subțiri intacte
    thresh = max(22.0, min(115.0, float(otsu_val) * 0.68))
    ink = (diff.astype(np.float32) >= thresh) & (paper > 0)

    ink_u8 = ink.astype(np.uint8) * 255
    # Fără closing agresiv (ar lipi litere mici sau ar îngroșa portativul); păstrăm fidel cerneala
    ink_u8[paper == 0] = 0
    return ink_u8, paper, {
        "method": "paper_background_subtraction",
        "otsu_diff": round(float(otsu_val), 1),
        "applied_diff_thresh": round(float(thresh), 1),
        "paper_method": paper_method,
    }


# ---------------------------------------------------------------------------
# Pas 2: Separarea petelor compacte (ex. note pline lipite de linii paralele)
# ---------------------------------------------------------------------------
def detect_compact_blobs_on_thin_lines(
    ink: np.ndarray,
    dist_map: np.ndarray,
    raw_skeleton: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Identifică petele ovale/rotunde compacte care au linii subțiri lipite de ele.

    Exemple: capete de note pline pe portativ (nu atinge literele de titlu, nici
    beam-urile alungite). Decuparea lor înainte de subțiere împiedică liniile
    paralele să se îndoaie și să se intersecteze în interiorul petei.
    """
    skel_radii = dist_map[raw_skeleton > 0]
    if skel_radii.size == 0:
        return np.zeros_like(ink), 0

    # Percentila 45 prinde curat raza liniilor subțiri (portative, bețe, versuri mici)
    r_thin = float(np.percentile(skel_radii, 45))
    r_cut = max(2.2, 1.85 * r_thin)
    thick = (dist_map >= r_cut).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thick, 8)
    if count <= 1:
        return np.zeros_like(ink), 0

    h, w = ink.shape
    blob_mask = np.zeros_like(ink)
    blob_count = 0
    thin_ink = (ink > 0) & (dist_map <= max(2.0, 1.65 * r_thin))
    min_core_area = max(5, int(0.45 * math.pi * r_cut * r_cut))
    max_core_area = int(0.015 * h * w)

    for lbl in range(1, count):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_core_area or area > max_core_area:
            continue
        x0 = int(stats[lbl, cv2.CC_STAT_LEFT])
        y0 = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        roi_lbl = (labels[y0:y0 + bh, x0:x0 + bw] == lbl).astype(np.uint8)

        # Aspect ratio prin PCA pe nucleul gros (ovalele au <= 2.35, beam-urile > 2.8)
        ys, xs = np.nonzero(roi_lbl)
        if xs.size < 5:
            continue
        pts = np.column_stack((xs, ys)).astype(np.float64)
        pts -= np.mean(pts, axis=0)
        vals = np.linalg.eigvalsh(pts.T @ pts)
        minor_s = math.sqrt(max(1e-6, float(vals[0])))
        major_s = math.sqrt(max(1e-6, float(vals[1])))
        if major_s / minor_s > 2.35:
            continue

        contours, _ = cv2.findContours(roi_lbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
        if hull_area > 0 and (area / hull_area) < 0.70:
            continue

        r_max = float(np.max(dist_map[y0:y0 + bh, x0:x0 + bw][roi_lbl > 0]))
        pad = int(math.ceil(r_max)) + 3
        rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
        rx1, ry1 = min(w, x0 + bw + pad), min(h, y0 + bh + pad)

        core_pad = np.zeros((ry1 - ry0, rx1 - rx0), np.uint8)
        core_pad[y0 - ry0:y0 - ry0 + bh, x0 - rx0:x0 - rx0 + bw] = roi_lbl

        # Refacem exact corpul petei dilatând nucleul cu raza ei proprie (în interiorul cernelii)
        d_r = max(1, int(round(0.95 * r_max)))
        k_blob = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d_r + 1, 2 * d_r + 1))
        full_blob = cv2.dilate(core_pad, k_blob) & (ink[ry0:ry1, rx0:rx1] > 0).astype(np.uint8)

        # Verificăm dacă are linii SUBȚIRI lipite de margine (deosebește notele de puncte/litere izolate)
        ring = cv2.dilate(full_blob, np.ones((5, 5), np.uint8)) - full_blob
        touching_thin = int(np.count_nonzero((ring > 0) & thin_ink[ry0:ry1, rx0:rx1]))
        if touching_thin < 2:
            continue

        blob_mask[ry0:ry1, rx0:rx1] |= (full_blob * 255)
        blob_count += 1

    return blob_mask, blob_count


# ---------------------------------------------------------------------------
# Pas 3: Graf de schelet cu grupare LOCALĂ a joncțiunilor
# ---------------------------------------------------------------------------
def _pixel_graph(skeleton: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
    ys, xs = np.nonzero(skeleton)
    coords = np.column_stack((xs, ys)).astype(np.int32)
    lookup: Dict[Point, int] = {(int(x), int(y)): i for i, (x, y) in enumerate(coords)}
    adj: List[List[int]] = [[] for _ in range(len(coords))]

    for i, (x0, y0) in enumerate(coords):
        x, y = int(x0), int(y0)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                j = lookup.get((x + dx, y + dy))
                if j is None:
                    continue
                if dx and dy and ((x + dx, y) in lookup or (x, y + dy) in lookup):
                    continue
                adj[i].append(j)
    return coords, adj


def _resample_polyline(points: np.ndarray, count: int = 24) -> np.ndarray:
    if len(points) < 2:
        return np.repeat(points[:1], count, axis=0)
    dists = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if dists[-1] < 1e-6:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, dists[-1], count)
    return np.column_stack([np.interp(targets, dists, points[:, axis]) for axis in range(2)])


def _local_widths(points: np.ndarray, dist_map: np.ndarray) -> np.ndarray:
    xi = np.clip(np.rint(points[:, 0]).astype(int), 0, dist_map.shape[1] - 1)
    yi = np.clip(np.rint(points[:, 1]).astype(int), 0, dist_map.shape[0] - 1)
    return np.maximum(1.0, 2.0 * dist_map[yi, xi])


def trace_and_clean_chains(
    skeleton: np.ndarray,
    dist_map: np.ndarray,
) -> Tuple[List[Chain], List[np.ndarray], Dict[int, np.ndarray]]:
    coords, adj = _pixel_graph(skeleton)
    if len(coords) == 0:
        return [], [], {}

    degrees = np.fromiter((len(a) for a in adj), dtype=np.int32, count=len(coords))
    critical = degrees != 2
    junction = degrees >= 3

    # Grupare de joncțiuni pe RAZA LOCALĂ a fiecărui pixel (merge simultan și pe
    # linii subțiri de 2px și pe litere uriașe de 20px!).
    junction_labels = np.zeros(skeleton.shape, np.int32)
    junction_count = 0
    if np.any(junction):
        grouped = np.zeros(skeleton.shape, np.uint8)
        j_indices = np.flatnonzero(junction)
        for idx in j_indices:
            jx, jy = int(coords[idx, 0]), int(coords[idx, 1])
            r_loc = max(2, min(14, int(round(1.35 * float(dist_map[jy, jx])))))
            cv2.circle(grouped, (jx, jy), r_loc, 1, -1)
        junction_labels, junction_count = ndi.label(grouped > 0, structure=np.ones((3, 3), dtype=np.uint8))

    groups: List[List[int]] = []
    node_map: Dict[int, int] = {}
    node_centers: Dict[int, np.ndarray] = {}

    # Mapare rapidă label -> membri (O(N) în loc de O(N * junction_count))
    label_members: Dict[int, List[int]] = {}
    for idx in np.flatnonzero(junction):
        lbl = int(junction_labels[int(coords[idx, 1]), int(coords[idx, 0])])
        if lbl > 0:
            label_members.setdefault(lbl, []).append(int(idx))

    for lbl in range(1, junction_count + 1):
        members = label_members.get(lbl)
        if members:
            nid = len(groups)
            groups.append(members)
            node_centers[nid] = np.mean(coords[members], axis=0).astype(np.float32)
            for idx in members:
                node_map[idx] = nid

    for i in range(len(coords)):
        if critical[i] and i not in node_map:
            nid = len(groups)
            groups.append([i])
            node_centers[nid] = coords[i].astype(np.float32)
            node_map[i] = nid

    visited: set[Tuple[int, int]] = set()
    chains: List[Chain] = []

    def edge_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    for nid, members in enumerate(groups):
        for start in members:
            for nxt in adj[start]:
                if node_map.get(nxt) == nid:
                    continue
                ek = edge_key(start, nxt)
                if ek in visited:
                    continue
                path = [start, nxt]
                visited.add(ek)
                prev, cur = start, nxt
                guard = 0
                while cur not in node_map and guard <= len(coords):
                    opts = [q for q in adj[cur] if q != prev]
                    if not opts:
                        break
                    following = opts[0]
                    ek2 = edge_key(cur, following)
                    if ek2 in visited:
                        break
                    visited.add(ek2)
                    path.append(following)
                    prev, cur = cur, following
                    guard += 1
                end_nid = node_map.get(cur)
                if end_nid == nid:
                    continue
                xy = coords[np.asarray(path, dtype=np.int32)].astype(np.float32)
                chains.append(Chain(xy, nid, end_nid, False))

    for i in range(len(coords)):
        for nxt in adj[i]:
            ek = edge_key(i, nxt)
            if ek in visited:
                continue
            path = [i, nxt]
            visited.add(ek)
            prev, cur = i, nxt
            guard = 0
            while guard <= len(coords):
                opts = [q for q in adj[cur] if q != prev]
                if not opts:
                    break
                following = opts[0]
                ek2 = edge_key(cur, following)
                if ek2 in visited:
                    break
                visited.add(ek2)
                path.append(following)
                prev, cur = cur, following
                if cur == i:
                    break
                guard += 1
            xy = coords[np.asarray(path, dtype=np.int32)].astype(np.float32)
            is_closed = bool(len(path) >= 4 and path[-1] == path[0])
            if is_closed:
                xy = xy[:-1]
            chains.append(Chain(xy, None, None, is_closed))

    isolated = [coords[i].copy() for i in range(len(coords)) if degrees[i] == 0]

    chains = _deduplicate_chains(chains, dist_map)

    # Eliminare cioturi scurte (spurs) agățate de joncțiuni (inclusiv serifuri minuscule de subțiere)
    changed = True
    while changed:
        changed = False
        node_deg: Dict[int, int] = {}
        for c in chains:
            if c.start_node is not None:
                node_deg[c.start_node] = node_deg.get(c.start_node, 0) + 1
            if c.end_node is not None:
                node_deg[c.end_node] = node_deg.get(c.end_node, 0) + 1

        kept: List[Chain] = []
        for c in chains:
            if c.start_node is not None and c.end_node is not None:
                d_s = node_deg.get(c.start_node, 0)
                d_e = node_deg.get(c.end_node, 0)
                if (d_s == 1 and d_e >= 3) or (d_e == 1 and d_s >= 3):
                    length = _polyline_length(c.points)
                    w = float(np.median(_local_widths(c.points, dist_map)))
                    if length <= max(3.0, 1.45 * w):
                        changed = True
                        continue
            kept.append(c)
        chains = kept

    chains = _contract_degree2_nodes(chains)
    return chains, isolated, node_centers


def _deduplicate_chains(chains: Sequence[Chain], dist_map: np.ndarray) -> List[Chain]:
    grouped: Dict[Tuple[int, int], List[Chain]] = {}
    untouched: List[Chain] = []
    for c in chains:
        if c.start_node is None or c.end_node is None or c.start_node == c.end_node:
            untouched.append(c)
            continue
        key = (min(c.start_node, c.end_node), max(c.start_node, c.end_node))
        norm = Chain(c.points[::-1].copy(), c.end_node, c.start_node, c.closed) if c.start_node > c.end_node else c
        grouped.setdefault(key, []).append(norm)

    result = list(untouched)
    for candidates in grouped.values():
        kept: List[Chain] = []
        candidates.sort(key=lambda ch: _polyline_length(ch.points), reverse=True)
        for cand in candidates:
            clen = _polyline_length(cand.points)
            w = float(np.median(_local_widths(cand.points, dist_map)))
            sample = _resample_polyline(cand.points, 24)
            dup = False
            for prev in kept:
                plen = _polyline_length(prev.points)
                if max(clen, plen) / max(1e-6, min(clen, plen)) > 1.25:
                    continue
                d = np.linalg.norm(sample - _resample_polyline(prev.points, 24), axis=1)
                if float(np.mean(d)) <= 1.5 * w and float(np.max(d)) <= 2.2 * w:
                    dup = True
                    break
            if not dup:
                kept.append(cand)
        result.extend(kept)
    return result


def _contract_degree2_nodes(chains: List[Chain]) -> List[Chain]:
    chains = list(chains)
    while True:
        incident: Dict[int, List[int]] = {}
        for idx, c in enumerate(chains):
            if c.closed:
                continue
            if c.start_node is not None:
                incident.setdefault(c.start_node, []).append(idx)
            if c.end_node is not None:
                incident.setdefault(c.end_node, []).append(idx)

        merged_any = False
        for nid, idxs in incident.items():
            if len(idxs) != 2 or idxs[0] == idxs[1]:
                continue
            i, j = idxs[0], idxs[1]
            ci, cj = chains[i], chains[j]
            pi = ci.points if ci.end_node == nid else ci.points[::-1]
            start_n = ci.start_node if ci.end_node == nid else ci.end_node
            pj = cj.points if cj.start_node == nid else cj.points[::-1]
            end_n = cj.end_node if cj.start_node == nid else cj.start_node

            if start_n is not None and start_n == end_n:
                pts = np.vstack([pi, pj[1:]]) if np.linalg.norm(pi[-1] - pj[0]) < 2.0 else np.vstack([pi, pj])
                new_chain = Chain(pts[:-1].astype(np.float32), None, None, True)
            else:
                pts = np.vstack([pi, pj[1:]]) if np.linalg.norm(pi[-1] - pj[0]) < 2.0 else np.vstack([pi, pj])
                new_chain = Chain(pts.astype(np.float32), start_n, end_n, False)

            for remove_idx in sorted((i, j), reverse=True):
                chains.pop(remove_idx)
            chains.append(new_chain)
            merged_any = True
            break
        if not merged_any:
            break
    return chains


# ---------------------------------------------------------------------------
# Pas 4: Segmentare pe curbură STRICT LOCALĂ (nu mai mănâncă `t` și versurile!)
# ---------------------------------------------------------------------------
def _local_chord_turns(points: np.ndarray, steps: np.ndarray, closed: bool) -> np.ndarray:
    """Unghiul (în grade) dintre coarda intrare și coarda ieșire la pasul local `steps[i]`."""
    n = len(points)
    idx = np.arange(n)
    if closed:
        p_prev = points[(idx - steps) % n]
        p_next = points[(idx + steps) % n]
        valid = np.ones(n, bool)
    else:
        valid = (idx >= steps) & (idx < n - steps)
        p_prev = points[np.clip(idx - steps, 0, n - 1)]
        p_next = points[np.clip(idx + steps, 0, n - 1)]

    v_in = points - p_prev
    v_out = p_next - points
    ni = np.linalg.norm(v_in, axis=1)
    no = np.linalg.norm(v_out, axis=1)
    ok = valid & (ni > 1e-4) & (no > 1e-4)
    dots = np.ones(n, dtype=np.float64)
    dots[ok] = np.sum(v_in[ok] * v_out[ok], axis=1) / (ni[ok] * no[ok])
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def _runs(mask: np.ndarray) -> List[Tuple[int, int, bool]]:
    if len(mask) == 0:
        return []
    changes = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes - 1, len(mask) - 1]
    return [(int(a), int(b), bool(mask[a])) for a, b in zip(starts, ends)]


def split_chains_to_pieces(chains: Sequence[Chain], dist_map: np.ndarray) -> List[Piece]:
    """Separă porțiunile drepte de bucle/cârlige fără să mănânce segmentele drepte vecine."""
    pieces: List[Piece] = []
    next_id = 1

    for chain_id, chain in enumerate(chains, start=1):
        pts = chain.points
        n = len(pts)
        if n < 2:
            continue
        widths = _local_widths(pts, dist_map)
        med_w = max(1.0, float(np.median(widths)))
        mode = "wrap" if chain.closed else "nearest"

        if n >= 5:
            sigma = max(0.55, min(1.6, 0.25 * med_w))
            smoothed = np.column_stack([
                ndi.gaussian_filter1d(pts[:, axis].astype(np.float64), sigma=sigma, mode=mode)
                for axis in range(2)
            ]).astype(np.float32)
            local_w = np.clip(
                ndi.gaussian_filter1d(widths.astype(np.float64), sigma=max(1.0, med_w), mode=mode),
                1.0, 24.0,
            )
        else:
            smoothed = pts.astype(np.float32)
            local_w = widths

        # Etapa A: tăiere STRICT pe colțuri / cârlige ascuțite (pas mic ~1.2w, fără lățire!)
        sharp = np.zeros(n, bool)
        if n >= 7:
            h_sharp = np.clip(np.rint(1.2 * local_w).astype(int), 2, min(10, max(2, n // 3)))
            turns_sharp = _local_chord_turns(smoothed, h_sharp, chain.closed)
            sharp = turns_sharp >= 36.0

        work_smoothed = smoothed
        work_widths = widths
        work_local_w = local_w
        work_sharp = sharp

        if chain.closed:
            if np.all(work_sharp) or np.all(~work_sharp):
                # Buclă închisă uniformă (cerc / O): o testăm direct ca o singură bucată (curbă)
                pieces.append(Piece(
                    chain_id, next_id, work_smoothed.copy(), 0, n - 1,
                    float(np.mean(work_widths)), float(work_widths[0]), float(work_widths[-1]),
                    "curve", None, None,
                ))
                next_id += 1
                continue
            shift = int(np.argmax(work_sharp))
            work_sharp = np.roll(work_sharp, -shift)
            work_smoothed = np.roll(work_smoothed, -shift, axis=0)
            work_widths = np.roll(work_widths, -shift)
            work_local_w = np.roll(work_local_w, -shift)

        for a, b, is_sharp in _runs(work_sharp):
            sub_pts = work_smoothed[a:b + 1].copy()
            sub_w = work_widths[a:b + 1]
            sub_lw = work_local_w[a:b + 1]
            sn = chain.start_node if (not chain.closed and a == 0) else None
            en = chain.end_node if (not chain.closed and b == n - 1) else None
            mw = float(np.mean(sub_w))

            if is_sharp or len(sub_pts) < 2:
                pieces.append(Piece(
                    chain_id, next_id, sub_pts, a, b, mw, float(sub_w[0]), float(sub_w[-1]),
                    "curve", sn, en,
                ))
                next_id += 1
                continue

            # Test direct de rectilinitate pe bucată (fără să pierdem pixeli!)
            sag, tort = _straightness_metrics(sub_pts)
            end_turn = _end_tangent_turn_deg(sub_pts, mw)
            if sag <= 0.13 and tort <= 1.13 and end_turn <= 36.0:
                pieces.append(Piece(
                    chain_id, next_id, sub_pts, a, b, mw, float(sub_w[0]), float(sub_w[-1]),
                    "line", sn, en,
                ))
                next_id += 1
                continue

            # Etapa B: bucata e curbă sau hibridă (ex. `c` -> băț la `d`, sau cârlig moale la `t`).
            # Extragem DOAR sub-intervalele cu adevărat drepte (curbură joasă pe pas mediu).
            m = len(sub_pts)
            if m < 7:
                pieces.append(Piece(
                    chain_id, next_id, sub_pts, a, b, mw, float(sub_w[0]), float(sub_w[-1]),
                    "curve", sn, en,
                ))
                next_id += 1
                continue

            h_mid = np.clip(np.rint(1.7 * sub_lw).astype(int), 2, min(11, max(2, m // 3)))
            turns_mid = _local_chord_turns(sub_pts, h_mid, False)
            # Capetele fără fereastră completă moștenesc vecinii imediați ca să nu tăiem vârfurile drepte
            is_curved_mid = turns_mid >= 15.0
            for idx_e in range(m):
                hm = int(h_mid[idx_e])
                if idx_e < hm and hm < m:
                    is_curved_mid[idx_e] = is_curved_mid[hm]
                elif idx_e >= m - hm and m - hm - 1 >= 0:
                    is_curved_mid[idx_e] = is_curved_mid[m - hm - 1]

            for ra, rb, is_c in _runs(is_curved_mid):
                r_pts = sub_pts[ra:rb + 1].copy()
                r_w = sub_w[ra:rb + 1]
                r_mw = float(np.mean(r_w))
                r_sn = sn if ra == 0 else None
                r_en = en if rb == m - 1 else None
                if not is_c and len(r_pts) >= 3:
                    rsag, rtort = _straightness_metrics(r_pts)
                    rturn = _end_tangent_turn_deg(r_pts, r_mw)
                    if rsag <= 0.11 and rtort <= 1.10 and rturn <= 25.0:
                        kind = "line"
                    else:
                        kind = "curve"
                else:
                    kind = "curve"
                pieces.append(Piece(
                    chain_id, next_id, r_pts, a + ra, a + rb, r_mw, float(r_w[0]), float(r_w[-1]),
                    kind, r_sn, r_en,
                ))
                next_id += 1

    return pieces


# ---------------------------------------------------------------------------
# Pas 5: Unire coliniară ne-distructivă + podire coliniară peste pete (cerneală)
# ---------------------------------------------------------------------------
def _outward_unit(piece: Piece, node_id: int) -> Optional[np.ndarray]:
    p = piece.points
    if len(p) < 2:
        return None
    k = min(len(p) - 1, max(2, int(round(2.5 * piece.mean_width))))
    if piece.start_node == node_id:
        vec = p[k] - p[0]
    elif piece.end_node == node_id:
        vec = p[-1 - k] - p[-1]
    else:
        return None
    norm = float(np.linalg.norm(vec))
    return (vec / norm) if norm > 1e-6 else None


def merge_collinear_non_destructive(
    pieces: Sequence[Piece],
) -> Tuple[List[Stick], List[Piece], List[Piece]]:
    line_pieces = [p for p in pieces if p.kind == "line" and len(p.points) >= 2]
    curve_pieces = [p for p in pieces if p.kind != "line"]
    by_id = {p.piece_id: p for p in line_pieces}

    by_node: Dict[int, List[Tuple[Piece, np.ndarray, str]]] = {}
    for p in line_pieces:
        if p.start_node is not None:
            d = _outward_unit(p, p.start_node)
            if d is not None:
                by_node.setdefault(p.start_node, []).append((p, d, "start"))
        if p.end_node is not None and p.end_node != p.start_node:
            d = _outward_unit(p, p.end_node)
            if d is not None:
                by_node.setdefault(p.end_node, []).append((p, d, "end"))

    endpoint_link: Dict[Tuple[int, str], Tuple[int, str]] = {}
    neighbours: Dict[int, set[int]] = {}
    piece_nodes: Dict[int, set[int]] = {p.piece_id: set() for p in line_pieces}

    for nid, arms in by_node.items():
        if len(arms) < 2:
            continue
        pairs: List[Tuple[float, int, int]] = []
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                pi, di, _ = arms[i]
                pj, dj, _ = arms[j]
                if pi.piece_id == pj.piece_id:
                    continue
                dot = float(np.dot(di, dj))
                if dot > -0.78:
                    continue
                w_pen = 0.12 * abs(math.log(max(1.0, pi.mean_width) / max(1.0, pj.mean_width)))
                pairs.append((dot + w_pen, i, j))
        pairs.sort(key=lambda item: item[0])

        used: set[int] = set()
        for _score, i, j in pairs:
            if i in used or j in used:
                continue
            pi, _, si = arms[i]
            pj, _, sj = arms[j]
            if (pi.piece_id, si) in endpoint_link or (pj.piece_id, sj) in endpoint_link:
                continue
            used.add(i)
            used.add(j)
            endpoint_link[(pi.piece_id, si)] = (pj.piece_id, sj)
            endpoint_link[(pj.piece_id, sj)] = (pi.piece_id, si)
            neighbours.setdefault(pi.piece_id, set()).add(pj.piece_id)
            neighbours.setdefault(pj.piece_id, set()).add(pi.piece_id)
            piece_nodes[pi.piece_id].add(nid)
            piece_nodes[pj.piece_id].add(nid)

    visited: set[int] = set()
    raw_sticks: List[Tuple[np.ndarray, float, float, float, List[int], set[int]]] = []

    for p in line_pieces:
        pid = p.piece_id
        if pid in visited:
            continue
        if pid not in neighbours:
            visited.add(pid)
            raw_sticks.append((p.points.copy(), p.mean_width, p.start_width, p.end_width, [pid], set()))
            continue

        comp: set[int] = set()
        stack = [pid]
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            stack.extend(neighbours.get(cur, ()))
        visited.update(comp)

        leaves = [cid for cid in comp if len(neighbours.get(cid, ())) <= 1]
        start_id = min(leaves) if leaves else min(comp)
        sp = by_id[start_id]
        l_start = (start_id, "start") in endpoint_link
        l_end = (start_id, "end") in endpoint_link
        forward = (not l_start) if (l_start != l_end) else True
        pts_list = [pt.copy() for pt in (sp.points if forward else sp.points[::-1])]
        sw = sp.start_width if forward else sp.end_width
        ew = sp.end_width if forward else sp.start_width
        cur_id = start_id
        exit_side = "end" if forward else "start"
        walked = {start_id}

        while (cur_id, exit_side) in endpoint_link:
            nxt_id, entry_side = endpoint_link[(cur_id, exit_side)]
            if nxt_id in walked or nxt_id not in comp:
                break
            npiece = by_id[nxt_id]
            nxt_forward = (entry_side == "start")
            nxt_pts = npiece.points if nxt_forward else npiece.points[::-1]
            gap = float(np.linalg.norm(nxt_pts[0] - pts_list[-1]))
            steps = int(math.ceil(gap))
            if steps > 1:
                b0 = pts_list[-1].copy()
                for t in range(1, steps):
                    pts_list.append((1.0 - t / steps) * b0 + (t / steps) * nxt_pts[0])
            pts_list.extend(pt.copy() for pt in nxt_pts[1:])
            ew = npiece.end_width if nxt_forward else npiece.start_width
            walked.add(nxt_id)
            cur_id = nxt_id
            exit_side = "end" if nxt_forward else "start"

        merged_pts = np.asarray(pts_list, dtype=np.float32)
        w_num = sum(_polyline_length(by_id[c].points) * by_id[c].mean_width for c in comp)
        w_den = max(1e-6, sum(_polyline_length(by_id[c].points) for c in comp))
        c_nodes: set[int] = set()
        for c in comp:
            c_nodes.update(piece_nodes.get(c, set()))
        raw_sticks.append((merged_pts, w_num / w_den, sw, ew, sorted(comp), c_nodes))

    sticks: List[Stick] = []
    spots: List[Piece] = []
    next_stick_id = 1
    for pts, mw, sw, ew, src_ids, c_nodes in raw_sticks:
        length = _polyline_length(pts)
        # Compensăm capetele mâncate de subțiere (câte 0.5*w la fiecare capăt)
        eff_len = length + 0.5 * sw + 0.5 * ew
        if length < 2.4 or eff_len < 2.2 * max(1.0, mw):
            spots.append(Piece(0, src_ids[0], pts, 0, len(pts) - 1, mw, sw, ew, "spot"))
            continue
        sagitta, tortuosity = _straightness_metrics(pts)
        if sagitta > 0.16 or tortuosity > 1.15:
            curve_pieces.append(Piece(0, src_ids[0], pts, 0, len(pts) - 1, mw, sw, ew, "curve"))
            continue
        sticks.append(Stick(
            stick_id=next_stick_id,
            points=pts,
            mean_width=mw,
            length_px=length,
            effective_length_px=eff_len,
            angle_deg=_segment_angle(pts),
            sagitta_ratio=sagitta,
            source_piece_ids=src_ids,
            crossed_nodes=c_nodes,
        ))
        next_stick_id += 1

    return sticks, curve_pieces, spots


def _endpoint_tangent(points: np.ndarray, at_end: bool, width: float) -> np.ndarray:
    """Vector unitar îndreptat ÎN AFARĂ din capătul segmentului."""
    k = min(len(points) - 1, max(2, int(round(3.0 * width))))
    vec = (points[-1] - points[-1 - k]) if at_end else (points[0] - points[k])
    n = float(np.linalg.norm(vec))
    return (vec / n) if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)


def bridge_collinear_across_ink(sticks: List[Stick], ink: np.ndarray) -> List[Stick]:
    """Unește bețele coliniare tăiate de o pată compactă (ex. linii paralele peste nota Fa).

    Puntea e permisă STRICT dacă >= 75% din pixelii podului stau pe cerneală reală
    (nu sare niciodată peste hârtie albă).
    """
    if len(sticks) < 2:
        return sticks

    h, w = ink.shape
    # Construim lista de capete: (stick_idx, is_end, point, outward_dir, width)
    n = len(sticks)
    end_pts = np.zeros((2 * n, 2), dtype=np.float32)
    end_dirs = np.zeros((2 * n, 2), dtype=np.float32)
    end_w = np.zeros(2 * n, dtype=np.float32)
    for i, s in enumerate(sticks):
        end_pts[2 * i] = s.points[0]
        end_pts[2 * i + 1] = s.points[-1]
        end_dirs[2 * i] = _endpoint_tangent(s.points, False, s.mean_width)
        end_dirs[2 * i + 1] = _endpoint_tangent(s.points, True, s.mean_width)
        end_w[2 * i] = s.mean_width
        end_w[2 * i + 1] = s.mean_width

    max_gap = float(max(45.0, min(140.0, 22.0 * float(np.median(end_w)))))
    candidates: List[Tuple[float, int, int]] = []

    # Căutare rapidă pe perechi apropiate
    for a in range(2 * n):
        si = a // 2
        pa = end_pts[a]
        da = end_dirs[a]
        wa = float(end_w[a])
        # Vectorizat pe restul capetelor
        diffs = end_pts[a + 1:] - pa
        dists = np.linalg.norm(diffs, axis=1)
        near = np.flatnonzero((dists >= 1.5) & (dists <= max_gap))
        for offset in near:
            b = a + 1 + int(offset)
            sj = b // 2
            if si == sj:
                continue
            wb = float(end_w[b])
            if max(wa, wb) / max(1e-6, min(wa, wb)) > 2.2:
                continue
            d = float(dists[offset])
            u_ab = diffs[offset] / d
            # Ambele capete trebuie să arate unul spre celălalt (coliniaritate strânsă <= ~16°)
            if float(np.dot(da, u_ab)) < 0.96:
                continue
            if float(np.dot(end_dirs[b], -u_ab)) < 0.96:
                continue
            # Verificăm dacă podul stă efectiv pe cerneală
            samples = max(4, int(math.ceil(d)))
            ts = np.linspace(0.08, 0.92, samples)
            xs = np.clip(np.rint(pa[0] + ts * diffs[offset, 0]).astype(int), 0, w - 1)
            ys = np.clip(np.rint(pa[1] + ts * diffs[offset, 1]).astype(int), 0, h - 1)
            ink_frac = float(np.mean(ink[ys, xs] > 0))
            if ink_frac < 0.75:
                continue
            candidates.append((d, a, b))

    if not candidates:
        return sticks

    candidates.sort(key=lambda item: item[0])
    linked_end: Dict[int, int] = {}
    adj_sticks: Dict[int, set[int]] = {}
    for _d, a, b in candidates:
        if a in linked_end or b in linked_end:
            continue
        linked_end[a] = b
        linked_end[b] = a
        adj_sticks.setdefault(a // 2, set()).add(b // 2)
        adj_sticks.setdefault(b // 2, set()).add(a // 2)

    visited: set[int] = set()
    out: List[Stick] = []
    next_id = 1

    for idx in range(n):
        if idx in visited:
            continue
        if idx not in adj_sticks:
            visited.add(idx)
            s = sticks[idx]
            s.stick_id = next_id
            next_id += 1
            out.append(s)
            continue

        comp: set[int] = set()
        stack = [idx]
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            stack.extend(adj_sticks.get(cur, ()))
        visited.update(comp)

        leaves = [c for c in comp if len(adj_sticks.get(c, ())) <= 1]
        start_s = min(leaves) if leaves else min(comp)
        start_linked = (2 * start_s) in linked_end
        end_linked = (2 * start_s + 1) in linked_end
        forward = (not start_linked) if (start_linked != end_linked) else True

        pts_list = [pt.copy() for pt in (sticks[start_s].points if forward else sticks[start_s].points[::-1])]
        exit_end = (2 * start_s + 1) if forward else (2 * start_s)
        walked = {start_s}

        while exit_end in linked_end:
            entry_end = linked_end[exit_end]
            nxt_s = entry_end // 2
            if nxt_s in walked or nxt_s not in comp:
                break
            nxt_forward = (entry_end % 2 == 0)
            nxt_pts = sticks[nxt_s].points if nxt_forward else sticks[nxt_s].points[::-1]
            gap = float(np.linalg.norm(nxt_pts[0] - pts_list[-1]))
            steps = int(math.ceil(gap))
            if steps > 1:
                b0 = pts_list[-1].copy()
                for t in range(1, steps):
                    pts_list.append((1.0 - t / steps) * b0 + (t / steps) * nxt_pts[0])
            pts_list.extend(pt.copy() for pt in nxt_pts[1:])
            walked.add(nxt_s)
            exit_end = (2 * nxt_s + 1) if nxt_forward else (2 * nxt_s)

        merged_pts = np.asarray(pts_list, dtype=np.float32)
        w_num = sum(sticks[c].length_px * sticks[c].mean_width for c in comp)
        w_den = max(1e-6, sum(sticks[c].length_px for c in comp))
        mw = w_num / w_den
        length = _polyline_length(merged_pts)
        sag, _ = _straightness_metrics(merged_pts)
        src_ids = sorted(pid for c in comp for pid in sticks[c].source_piece_ids)
        c_nodes: set[int] = set()
        for c in comp:
            c_nodes.update(sticks[c].crossed_nodes)

        out.append(Stick(
            stick_id=next_id,
            points=merged_pts,
            mean_width=mw,
            length_px=length,
            effective_length_px=length + mw,
            angle_deg=_segment_angle(merged_pts),
            sagitta_ratio=sag,
            source_piece_ids=src_ids,
            crossed_nodes=c_nodes,
        ))
        next_id += 1

    return out


# ---------------------------------------------------------------------------
# Faza 2: Graf de relații și clasificare strict din contextul celorlalte bețe
# ---------------------------------------------------------------------------
def _point_to_polyline(pt: np.ndarray, poly: np.ndarray) -> Tuple[float, float]:
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    seg_len2 = np.sum(ab * ab, axis=1)
    ap = pt - a
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(seg_len2 > 1e-9, np.sum(ap * ab, axis=1) / seg_len2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    dists = np.linalg.norm(pt - proj, axis=1)
    best = int(np.argmin(dists))
    seg_lens = np.sqrt(seg_len2)
    total = float(np.sum(seg_lens))
    if total < 1e-6:
        return float(dists[best]), 0.5
    arc_before = float(np.sum(seg_lens[:best])) + float(t[best] * seg_lens[best])
    return float(dists[best]), arc_before / total


def _polylines_cross_interior(pa: np.ndarray, pb: np.ndarray, tol: float) -> bool:
    sa = pa[::max(1, len(pa) // 40)]
    sb = pb[::max(1, len(pb) // 40)]
    diff = sa[:, None, :] - sb[None, :, :]
    return float(np.min(np.linalg.norm(diff, axis=2))) <= tol


def classify_sticks_by_context(sticks: List[Stick]) -> Dict[str, Any]:
    if not sticks:
        return {"median_width_px": 0.0, "median_length_px": 0.0, "angle_families": []}

    bboxes = np.array([
        [float(np.min(s.points[:, 0])), float(np.min(s.points[:, 1])),
         float(np.max(s.points[:, 0])), float(np.max(s.points[:, 1]))]
        for s in sticks
    ], dtype=np.float32)
    widths = np.array([s.mean_width for s in sticks], dtype=np.float32)
    by_id = {s.stick_id: s for s in sticks}
    n = len(sticks)

    # Pre-filtrare vectorizată pe bounding boxes
    for i in range(n):
        si = sticks[i]
        tols = np.maximum(5.0, 2.3 * np.maximum(widths[i], widths[i + 1:]))
        overlap = (
            (bboxes[i, 2] + tols >= bboxes[i + 1:, 0]) &
            (bboxes[i + 1:, 2] + tols >= bboxes[i, 0]) &
            (bboxes[i, 3] + tols >= bboxes[i + 1:, 1]) &
            (bboxes[i + 1:, 3] + tols >= bboxes[i, 1])
        )
        for off in np.flatnonzero(overlap):
            j = i + 1 + int(off)
            sj = sticks[j]
            tol = float(tols[off])

            ends_i = (si.points[0], si.points[-1])
            ends_j = (sj.points[0], sj.points[-1])
            min_ee = min(float(np.linalg.norm(ei - ej)) for ei in ends_i for ej in ends_j)
            if min_ee <= tol:
                si.l_contacts.append(sj.stick_id)
                sj.l_contacts.append(si.stick_id)
                continue

            i_on_j = False
            for ei in ends_i:
                d, frac = _point_to_polyline(ei, sj.points)
                if d <= tol and tol < frac * sj.length_px < (sj.length_px - tol):
                    i_on_j = True
                    break

            j_on_i = False
            for ej in ends_j:
                d, frac = _point_to_polyline(ej, si.points)
                if d <= tol and tol < frac * si.length_px < (si.length_px - tol):
                    j_on_i = True
                    break

            if i_on_j and not j_on_i:
                si.t_touches_trunk.append(sj.stick_id)
                sj.t_receives_branch.append(si.stick_id)
                continue
            if j_on_i and not i_on_j:
                sj.t_touches_trunk.append(si.stick_id)
                si.t_receives_branch.append(sj.stick_id)
                continue

            if (si.crossed_nodes & sj.crossed_nodes) or _polylines_cross_interior(si.points, sj.points, tol):
                si.x_crossings.append(sj.stick_id)
                sj.x_crossings.append(si.stick_id)

    order = sorted(range(n), key=lambda k: sticks[k].length_px, reverse=True)
    families: List[Tuple[float, List[int]]] = []
    for k in order:
        ang = sticks[k].angle_deg
        placed = False
        for f_ang, members in families:
            if _angle_diff_180(ang, f_ang) <= 12.0:
                members.append(k)
                placed = True
                break
        if not placed:
            families.append((ang, [k]))

    families.sort(key=lambda item: len(item[1]), reverse=True)
    family_meta = []
    for f_id, (f_ang, members) in enumerate(families, start=1):
        for k in members:
            sticks[k].angle_family = f_id
            sticks[k].family_size = len(members)
        family_meta.append({"family_id": f_id, "angle_deg": round(f_ang, 1), "count": len(members)})

    med_w = float(np.median(widths))
    med_len = float(np.median([s.length_px for s in sticks]))
    for s in sticks:
        s.thickness_ratio = s.mean_width / max(1e-6, med_w)

    # 1) Linii lungi paralele (ex. linii de portativ)
    for s in sticks:
        if (s.family_size >= 3 and s.length_px >= max(3.0 * med_len, 25.0 * s.mean_width)
                and (len(s.x_crossings) + len(s.t_receives_branch) >= 1 or s.family_size >= 4)):
            s.role = "linie_paralela_lunga"
            s.role_reason = (
                f"familie paralelă ({s.family_size} bețe la ~{s.angle_deg:.0f}°), "
                f"lungime {s.length_px / max(1e-6, med_len):.1f}× mediana"
            )

    # 2) Conectori care leagă >= 2 bețe paralele (ex. beam-uri)
    connector_ids: set[int] = set()
    for s in sticks:
        if s.role != "bat_izolat":
            continue
        attached_ids = list(dict.fromkeys(s.l_contacts + s.t_receives_branch + s.t_touches_trunk))
        transverse = [
            by_id[oid] for oid in attached_ids
            if by_id[oid].role != "linie_paralela_lunga"
            and _angle_diff_180(s.angle_deg, by_id[oid].angle_deg) >= 30.0
        ]
        if len(transverse) >= 2:
            has_parallel_pair = any(
                _angle_diff_180(transverse[a_i].angle_deg, transverse[b_i].angle_deg) <= 22.0
                for a_i in range(len(transverse))
                for b_i in range(a_i + 1, len(transverse))
            )
            if has_parallel_pair:
                s.role = "conector_paralel"
                connector_ids.add(s.stick_id)
                extra = f", grosime {s.thickness_ratio:.2f}× mediana" if s.thickness_ratio >= 1.25 else ""
                s.role_reason = f"leagă {len(transverse)} bețe paralele transversale{extra}"

    # 3) Bețe legate la un conector paralel
    for s in sticks:
        if s.role != "bat_izolat":
            continue
        touching_conns = [
            oid for oid in (s.l_contacts + s.t_touches_trunk + s.t_receives_branch)
            if oid in connector_ids and _angle_diff_180(s.angle_deg, by_id[oid].angle_deg) >= 30.0
        ]
        if touching_conns:
            s.role = "bat_pe_conector"
            s.role_reason = f"băț conectat la conectorul #{touching_conns[0]}"

    # 4) Trunchiuri, ramuri, X, L, izolate
    for s in sticks:
        if s.role != "bat_izolat":
            continue
        non_bg_branches = [oid for oid in s.t_receives_branch if by_id[oid].role != "linie_paralela_lunga"]
        non_bg_trunks = [oid for oid in s.t_touches_trunk if by_id[oid].role != "linie_paralela_lunga"]
        non_bg_x = [oid for oid in s.x_crossings if by_id[oid].role != "linie_paralela_lunga"]
        non_bg_l = [oid for oid in s.l_contacts if by_id[oid].role != "linie_paralela_lunga"]

        if non_bg_branches and not non_bg_trunks:
            s.role = "trunchi"
            s.role_reason = f"trunchi cu {len(non_bg_branches)} ramuri atașate ({non_bg_branches})"
        elif non_bg_trunks:
            longest_trunk = max((by_id[oid].length_px for oid in non_bg_trunks), default=0.0)
            s.role = "ramura"
            s.role_reason = f"se sprijină pe trunchiul #{non_bg_trunks[0]} (raport {s.length_px / max(1e-6, longest_trunk):.2f})"
        elif non_bg_x:
            s.role = "intersectat_X"
            s.role_reason = f"se intersectează interior cu #{non_bg_x}"
        elif non_bg_l:
            other = by_id[non_bg_l[0]]
            ratio = other.length_px / max(1e-6, s.length_px)
            if ratio >= 1.35:
                s.role = "ramura"
                s.role_reason = f"braț mai scurt la colț L cu #{other.stick_id} (1 : {ratio:.2f})"
            elif (1.0 / max(1e-6, ratio)) >= 1.35:
                s.role = "trunchi"
                s.role_reason = f"braț mai lung la colț L cu #{other.stick_id} ({1.0/ratio:.2f} : 1)"
            else:
                s.role = "colt_L"
                s.role_reason = f"colț L echilibrat cu #{other.stick_id}"
        else:
            s.role = "bat_izolat"
            if s.x_crossings:
                s.role_reason = f"băț independent (traversează {len(s.x_crossings)} linii paralele lungi)"
            else:
                s.role_reason = "băț independent, fără contacte"

    return {
        "median_width_px": round(med_w, 2),
        "median_length_px": round(med_len, 2),
        "angle_families": family_meta[:8],
    }


# ---------------------------------------------------------------------------
# Debug vizual
# ---------------------------------------------------------------------------
def _draw_debug(
    original: np.ndarray,
    ink: np.ndarray,
    blob_mask: np.ndarray,
    skeleton: np.ndarray,
    sticks: Sequence[Stick],
    curves: Sequence[Piece],
    node_centers: Dict[int, np.ndarray],
    debug_dir: Path,
) -> Dict[str, str]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    p1 = debug_dir / "debug_1_cerneala.png"
    p2 = debug_dir / "debug_2_schelet.png"
    p3 = debug_dir / "debug_3_toate_betele.png"
    p4 = debug_dir / "debug_4_context.png"

    # Debug 1: cerneala albă + petele compacte decupate (capete pline) în albastru
    ink_vis = cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)
    ink_vis[blob_mask > 0] = (255, 160, 40)
    cv2.imwrite(str(p1), ink_vis)

    skel_vis = np.zeros((*skeleton.shape, 3), np.uint8)
    skel_vis[blob_mask > 0] = (60, 40, 20)
    skel_vis[skeleton > 0] = (230, 230, 230)
    for c in curves:
        poly = np.rint(c.points).astype(np.int32).reshape(-1, 1, 2)
        if len(poly) >= 2:
            cv2.polylines(skel_vis, [poly], False, (200, 70, 200), 2, cv2.LINE_AA)
    for center in node_centers.values():
        cv2.circle(skel_vis, (int(round(center[0])), int(round(center[1]))), 3, (0, 220, 255), -1)
    cv2.imwrite(str(p2), skel_vis)

    font = cv2.FONT_HERSHEY_SIMPLEX

    vis_all = original.copy()
    for c in curves:
        poly = np.rint(c.points).astype(np.int32).reshape(-1, 1, 2)
        if len(poly) >= 2:
            cv2.polylines(vis_all, [poly], False, (175, 175, 175), 1, cv2.LINE_AA)
    for s in sticks:
        poly = np.rint(s.points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis_all, [poly], False, (0, 200, 0), 2, cv2.LINE_AA)
        mid = s.points[len(s.points) // 2]
        cv2.putText(vis_all, str(s.stick_id), (int(mid[0]) + 3, int(mid[1]) - 3),
                    font, 0.42, (0, 140, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(p3), vis_all)

    role_colors = {
        "bat_izolat": (0, 200, 0),
        "bat_pe_conector": (0, 230, 120),
        "conector_paralel": (235, 110, 0),
        "trunchi": (0, 145, 255),
        "ramura": (190, 80, 180),
        "intersectat_X": (0, 0, 235),
        "colt_L": (0, 210, 235),
        "linie_paralela_lunga": (210, 200, 0),
    }
    vis_ctx = original.copy()
    for s in sticks:
        col = role_colors.get(s.role, (150, 150, 150))
        poly = np.rint(s.points).astype(np.int32).reshape(-1, 1, 2)
        thickness = 3 if s.role == "conector_paralel" else 2
        cv2.polylines(vis_ctx, [poly], False, col, thickness, cv2.LINE_AA)
        mid = s.points[len(s.points) // 2]
        cv2.putText(vis_ctx, str(s.stick_id), (int(mid[0]) + 3, int(mid[1]) - 3),
                    font, 0.42, col, 1, cv2.LINE_AA)

    y_leg = 24
    cv2.rectangle(vis_ctx, (8, 6), (260, 12 + 22 * len(role_colors)), (255, 255, 255), -1)
    cv2.rectangle(vis_ctx, (8, 6), (260, 12 + 22 * len(role_colors)), (80, 80, 80), 1)
    for role_name, col in role_colors.items():
        cv2.line(vis_ctx, (16, y_leg - 4), (36, y_leg - 4), col, 3)
        cv2.putText(vis_ctx, role_name, (44, y_leg), font, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
        y_leg += 22
    cv2.imwrite(str(p4), vis_ctx)

    return {
        "debug_1_cerneala": str(p1),
        "debug_2_schelet": str(p2),
        "debug_3_toate_betele": str(p3),
        "debug_4_context": str(p4),
    }


def process_image(
    image_path: str | Path,
    out_dir: str | Path = ".",
    px_per_cm: Optional[float] = None,
) -> Dict[str, Any]:
    image_path = Path(image_path)
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")

    ink, _paper, prep_meta = binarize_ink(original)
    full_dist = ndi.distance_transform_edt(ink > 0)
    raw_skel = skeletonize(ink > 0)

    blob_mask, blob_count = detect_compact_blobs_on_thin_lines(ink, full_dist, raw_skel)
    if blob_count > 0:
        stroke_ink = (ink > 0) & (blob_mask == 0)
        skeleton_bool = skeletonize(stroke_ink)
        dist_map = ndi.distance_transform_edt(stroke_ink)
    else:
        skeleton_bool = raw_skel
        dist_map = full_dist

    skeleton = skeleton_bool.astype(np.uint8) * 255

    chains, isolated, node_centers = trace_and_clean_chains(skeleton_bool, dist_map)
    pieces = split_chains_to_pieces(chains, dist_map)
    sticks, curves, spots = merge_collinear_non_destructive(pieces)
    sticks = bridge_collinear_across_ink(sticks, ink)
    context_meta = classify_sticks_by_context(sticks)

    role_counts: Dict[str, int] = {}
    stick_records: List[Dict[str, Any]] = []
    for s in sticks:
        role_counts[s.role] = role_counts.get(s.role, 0) + 1
        pts = np.rint(s.points).astype(int)
        stick_records.append({
            "id": s.stick_id,
            "start": {"x": int(pts[0, 0]), "y": int(pts[0, 1])},
            "end": {"x": int(pts[-1, 0]), "y": int(pts[-1, 1])},
            "length_px": round(s.length_px, 2),
            "effective_length_px": round(s.effective_length_px, 2),
            "length_cm": round(s.length_px / px_per_cm, 3) if px_per_cm else None,
            "angle_deg": round(s.angle_deg, 2),
            "width_px": round(s.mean_width, 2),
            "thickness_ratio": round(s.thickness_ratio, 2),
            "sagitta_ratio": round(s.sagitta_ratio, 4),
            "angle_family": s.angle_family,
            "family_size": s.family_size,
            "role": s.role,
            "role_reason": s.role_reason,
            "relations": {
                "l_contacts": s.l_contacts,
                "t_touches_trunk": s.t_touches_trunk,
                "t_receives_branch": s.t_receives_branch,
                "x_crossings": s.x_crossings,
            },
        })

    debug_paths = _draw_debug(original, ink, blob_mask, skeleton, sticks, curves, node_centers, Path(out_dir))
    output = {
        "image": str(image_path),
        "image_size_px": {"width": int(original.shape[1]), "height": int(original.shape[0])},
        "preprocessing": {**prep_meta, "compact_blobs_separated": blob_count},
        "context_summary": context_meta,
        "counts": {
            "total_sticks": len(sticks),
            "curves_separated": len(curves),
            "spots_ignored": len(spots) + len(isolated),
            "by_role": role_counts,
        },
        "sticks": stick_records,
        "debug_images": debug_paths,
    }
    out_path = Path(out_dir) / f"{image_path.stem}_bete_v3.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    output["json_file"] = str(out_path)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Detector geometric v3 în 2 faze pentru toate bețele")
    parser.add_argument("image", help="Fotografie PNG/JPG")
    parser.add_argument("--out-dir", default="debug_v3", help="Directorul de ieșire")
    parser.add_argument("--px-per-cm", type=float, default=None, help="Calibrare opțională px/cm")
    args = parser.parse_args(argv)
    try:
        res = process_image(args.image, args.out_dir, args.px_per_cm)
    except Exception as exc:
        print(f"Eroare: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "preprocessing": res["preprocessing"],
        "counts": res["counts"],
        "context_summary": res["context_summary"],
        "debug_images": res["debug_images"],
        "json_file": res["json_file"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())