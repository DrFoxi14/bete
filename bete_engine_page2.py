#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motor geometric determinist în 2 faze pentru detectarea TUTUROR bețelor.

Faza 1 (Extracție ne-distructivă):
  Găsește toate segmentele drepte sau ușor curbate, indiferent de unghi,
  lungime sau intersecții. Nu aplică reguli de F/L/T/X care să șteargă bețe.

Faza 2 (Clasificare relațională din context):
  După ce toate bețele există, le analizează relațiile reciproce (contacte
  la capete, contacte pe corp, traversări, paralelism, grosime relativă) și
  atribuie fiecăruia un rol contextual fără să arunce niciun băț real.

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
from skimage.filters import threshold_sauvola
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
    kind: str                       # "line" sau "curve"
    start_node: Optional[int] = None
    end_node: Optional[int] = None


@dataclass
class Stick:
    stick_id: int
    points: np.ndarray              # N x 2 float32 (x, y)
    mean_width: float
    length_px: float
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
    t_touches_trunk: List[int] = field(default_factory=list)   # acest băț se termină pe trunchiul X
    t_receives_branch: List[int] = field(default_factory=list) # bățul X se termină pe corpul acestui băț
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
    # PCA pentru unghi robust la capete ușor strâmbe
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
    # Distanța perpendiculară a fiecărui punct la dreapta p0-p1
    rel = points - p0
    cross = np.abs(rel[:, 0] * chord_vec[1] - rel[:, 1] * chord_vec[0])
    max_dev = float(np.max(cross) / chord_len)
    return max_dev / chord_len, arc_len / chord_len


# ---------------------------------------------------------------------------
# Pas 1: Binarizare robustă (fără tăierea umbrelor de pe foaie)
# ---------------------------------------------------------------------------
def _safe_paper_region(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """Maschează masa doar dacă marginile cadrului sunt clar mai întunecate decât centrul."""
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

    # Dacă marginile nu sunt clar mai închise decât centrul, foaia umple cadrul:
    # nu tăiem nimic și lăsăm Sauvola să gestioneze umbrele.
    if center_med - border_med < 32.0:
        return np.full((h, w), 255, np.uint8), "full_frame_safe"

    cutoff = 0.5 * (border_med + center_med)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    sat_cut = min(160.0, max(75.0, float(np.percentile(sat[h // 4: 3 * h // 4, w // 4: 3 * w // 4], 95)) + 25.0))
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
    h, w = image.shape[:2]
    paper, paper_method = _safe_paper_region(image)
    darkest = np.min(image, axis=2).astype(np.uint8)

    window = _odd(min(151, max(31, int(min(h, w) * 0.025))))
    window = min(window, _odd(max(3, min(h, w) - 2)))
    local_thresh = threshold_sauvola(darkest, window_size=window, k=0.18, r=128)
    ink = (darkest.astype(np.float32) < local_thresh) & (paper > 0)

    ink_u8 = ink.astype(np.uint8) * 255
    ink_u8 = cv2.morphologyEx(
        ink_u8, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1
    )
    ink_u8[paper == 0] = 0
    return ink_u8, paper, {
        "threshold": "Sauvola",
        "window_px": int(window),
        "paper_method": paper_method,
    }


# ---------------------------------------------------------------------------
# Pas 2: Graf de schelet, curățare cioturi de subțiere, contragere noduri deg=2
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
    """Extrage lanțurile topologice, elimină cioturile de subțiere și contractă nodurile de grad 2."""
    coords, adj = _pixel_graph(skeleton)
    if len(coords) == 0:
        return [], [], {}

    degrees = np.fromiter((len(a) for a in adj), dtype=np.int32, count=len(coords))
    critical = degrees != 2
    junction = degrees >= 3

    junction_labels = np.zeros(skeleton.shape, np.int32)
    junction_count = 0
    if np.any(junction):
        jmask = np.zeros(skeleton.shape, np.uint8)
        jmask[coords[junction, 1], coords[junction, 0]] = 1
        j_radii = dist_map[coords[junction, 1], coords[junction, 0]]
        merge_r = max(2, min(10, int(round(1.6 * float(np.median(j_radii))))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge_r + 1, 2 * merge_r + 1))
        grouped = cv2.dilate(jmask, kernel) > 0
        junction_labels, junction_count = ndi.label(grouped, structure=np.ones((3, 3), dtype=np.uint8))

    groups: List[List[int]] = []
    node_map: Dict[int, int] = {}
    node_centers: Dict[int, np.ndarray] = {}

    for label in range(1, junction_count + 1):
        members = [i for i, (x, y) in enumerate(coords)
                   if junction[i] and junction_labels[int(y), int(x)] == label]
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

    # Bucle închise pure (toți pixelii au grad 2)
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
                xy = xy[:-1]  # eliminăm punctul duplicat de închidere
            chains.append(Chain(xy, None, None, is_closed))

    isolated = [coords[i].copy() for i in range(len(coords)) if degrees[i] == 0]

    # 1) Deduplicare rute paralele din subțiere între aceiași doi noduri
    chains = _deduplicate_chains(chains, dist_map)

    # 2) Eliminare cioturi scurte (spurs) agățate de joncțiuni (artefacte de scheletizare)
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
                d_start = node_deg.get(c.start_node, 0)
                d_end = node_deg.get(c.end_node, 0)
                is_spur = (d_start == 1 and d_end >= 3) or (d_end == 1 and d_start >= 3)
                if is_spur:
                    length = _polyline_length(c.points)
                    width = float(np.median(_local_widths(c.points, dist_map)))
                    if length <= max(3.5, 1.8 * width) and length <= 16.0:
                        changed = True
                        continue
            kept.append(c)
        chains = kept

    # 3) Contragere noduri rămase de grad 2 într-un singur lanț continuu
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
# Pas 3: Segmentare pe curbură locală (lățime adaptivă per pixel!)
# ---------------------------------------------------------------------------
def _curvature_mask_adaptive(
    points: np.ndarray,
    widths: np.ndarray,
    closed: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Separă zonele de curbură mare (bucle, colțuri, ace de păr) de porțiunile drepte.

    Folosește lățimea LOCALĂ a fiecărui pixel, astfel încât un beam gros să nu
    umfle fereastra de tăiere pe un băț subțire atașat de el.
    """
    n = len(points)
    if n < 7:
        return np.zeros(n, bool), points.astype(np.float32)

    med_w = max(1.0, float(np.median(widths)))
    sigma = max(0.7, min(2.2, 0.35 * med_w))
    mode = "wrap" if closed else "nearest"
    smoothed = np.column_stack([
        ndi.gaussian_filter1d(points[:, axis].astype(np.float64), sigma=sigma, mode=mode)
        for axis in range(2)
    ]).astype(np.float32)

    local_w = ndi.gaussian_filter1d(widths.astype(np.float64), sigma=max(1.0, med_w), mode=mode)
    local_w = np.clip(local_w, 1.0, 24.0)

    idx = np.arange(n)
    high = np.zeros(n, bool)

    # 1) Detector strâns de colțuri și întoarceri în ac de păr (ex. vârful lui d, colțul lui F/⊓)
    win = np.clip(np.rint(2.2 * local_w).astype(int), 3, min(18, max(3, n // 3)))
    if closed:
        p_prev = smoothed[(idx - win) % n]
        p_next = smoothed[(idx + win) % n]
        valid_c = np.ones(n, bool)
    else:
        valid_c = (idx >= win) & (idx < n - win)
        p_prev = smoothed[np.clip(idx - win, 0, n - 1)]
        p_next = smoothed[np.clip(idx + win, 0, n - 1)]

    v_in = smoothed - p_prev
    v_out = p_next - smoothed
    ni = np.linalg.norm(v_in, axis=1)
    no = np.linalg.norm(v_out, axis=1)
    ok = valid_c & (ni > 1e-4) & (no > 1e-4)
    dots = np.zeros(n, dtype=np.float64)
    dots[ok] = np.sum(v_in[ok] * v_out[ok], axis=1) / (ni[ok] * no[ok])
    chord_deg = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    high[ok & (chord_deg >= 42.0)] = True

    # 2) Rotație cumulată cu semn pe fereastră medie (prinde tranziția lină c -> băț la d și buclele c/o)
    if closed:
        tangents = np.roll(smoothed, -1, axis=0) - np.roll(smoothed, 1, axis=0)
    else:
        tangents = np.gradient(smoothed, axis=0)
    headings = np.arctan2(tangents[:, 1], tangents[:, 0])
    dh = np.roll(headings, -1) - headings
    dh = (dh + math.pi) % (2.0 * math.pi) - math.pi
    if not closed:
        dh[-1] = 0.0

    rad = np.clip(np.rint(3.5 * local_w).astype(int), 4, min(28, max(4, n // 2)))
    if closed:
        max_r = int(np.max(rad))
        padded = np.concatenate([dh[-max_r:], dh, dh[:max_r]])
        pref = np.concatenate(([0.0], np.cumsum(padded)))
        center = idx + max_r
        total_rot = np.abs(pref[center + rad + 1] - pref[center - rad])
    else:
        pref = np.concatenate(([0.0], np.cumsum(dh)))
        left = np.clip(idx - rad, 0, n - 1)
        right = np.clip(idx + rad, 0, n - 1)
        total_rot = np.abs(pref[right + 1] - pref[left])

    high[total_rot >= math.radians(44.0)] = True

    # Expansiune mică, strict cât să nu lase colțul lipit de segmentul drept
    pad = max(1, min(3, int(round(0.3 * med_w))))
    if closed:
        ext = np.concatenate([high[-pad:], high, high[:pad]])
        ext = ndi.binary_dilation(ext, structure=np.ones(2 * pad + 1, dtype=bool))
        high = ext[pad:-pad]
    else:
        high = ndi.binary_dilation(high, structure=np.ones(2 * pad + 1, dtype=bool))

    return high, smoothed


def _runs(mask: np.ndarray) -> List[Tuple[int, int, bool]]:
    if len(mask) == 0:
        return []
    changes = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes - 1, len(mask) - 1]
    return [(int(a), int(b), bool(mask[a])) for a, b in zip(starts, ends)]


def split_chains_to_pieces(chains: Sequence[Chain], dist_map: np.ndarray) -> List[Piece]:
    pieces: List[Piece] = []
    next_id = 1
    for chain_id, chain in enumerate(chains, start=1):
        pts = chain.points
        if len(pts) < 2:
            continue
        widths = _local_widths(pts, dist_map)
        high, smoothed = _curvature_mask_adaptive(pts, widths, chain.closed)

        if chain.closed:
            if np.all(high):
                pieces.append(Piece(chain_id, next_id, smoothed.copy(), 0, len(pts) - 1,
                                    float(np.mean(widths)), "curve", None, None))
                next_id += 1
                continue
            if np.all(~high):
                # Buclă fără colțuri — tot curbă închisă (cerc mare)
                pieces.append(Piece(chain_id, next_id, smoothed.copy(), 0, len(pts) - 1,
                                    float(np.mean(widths)), "curve", None, None))
                next_id += 1
                continue
            # Rotim ca să începem într-o zonă de curbură și să nu tăiem un băț în două
            shift = int(np.argmax(high))
            high = np.roll(high, -shift)
            smoothed = np.roll(smoothed, -shift, axis=0)
            widths = np.roll(widths, -shift)

        for a, b, is_curve in _runs(high):
            sub = smoothed[a:b + 1].copy()
            mw = float(np.mean(widths[a:b + 1])) if b >= a else float(np.mean(widths))
            sn = chain.start_node if (not chain.closed and a == 0) else None
            en = chain.end_node if (not chain.closed and b == len(pts) - 1) else None
            pieces.append(Piece(chain_id, next_id, sub, a, b, mw,
                                "curve" if is_curve else "line", sn, en))
            next_id += 1
    return pieces


# ---------------------------------------------------------------------------
# Pas 4: Unire coliniară peste joncțiuni (FĂRĂ a șterge celelalte brațe!)
# ---------------------------------------------------------------------------
def _outward_unit(piece: Piece, node_id: int) -> Optional[np.ndarray]:
    p = piece.points
    if len(p) < 2:
        return None
    k = min(len(p) - 1, max(2, int(round(3.0 * piece.mean_width))))
    if piece.start_node == node_id:
        vec = p[k] - p[0]
    elif piece.end_node == node_id:
        vec = p[-1 - k] - p[-1]
    else:
        return None
    norm = float(np.linalg.norm(vec))
    return (vec / norm) if norm > 1e-6 else None


def merge_collinear_non_destructive(pieces: Sequence[Piece]) -> Tuple[List[Stick], List[Piece], List[Piece]]:
    """Unește brațele coliniare la joncțiuni (T, F, X, portativ) păstrând TOATE brațele.

    Orice braț ne-împerecheat rămâne propriul său băț independent.
    """
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

    # La fiecare nod, împerechem greedy perechile coliniare (unghi >= ~141°),
    # preferând grosimi compatibile (băț subțire cu băț subțire, beam cu beam).
    endpoint_link: Dict[Tuple[int, str], Tuple[int, str]] = {}
    neighbours: Dict[int, set[int]] = {}
    piece_nodes: Dict[int, set[int]] = {p.piece_id: set() for p in line_pieces}

    for nid, arms in by_node.items():
        if len(arms) < 2:
            continue
        pairs: List[Tuple[float, int, int]] = []
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                pi, di, si = arms[i]
                pj, dj, sj = arms[j]
                if pi.piece_id == pj.piece_id:
                    continue
                dot = float(np.dot(di, dj))
                if dot > -0.78:  # sub ~141° nu e continuare coliniară
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

    # Asamblăm componentele legate într-un singur băț continuu fiecare
    visited: set[int] = set()
    raw_sticks: List[Tuple[np.ndarray, float, List[int], set[int]]] = []

    for p in line_pieces:
        pid = p.piece_id
        if pid in visited:
            continue
        if pid not in neighbours:
            visited.add(pid)
            raw_sticks.append((p.points.copy(), p.mean_width, [pid], set()))
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
            walked.add(nxt_id)
            cur_id = nxt_id
            exit_side = "end" if nxt_forward else "start"

        merged_pts = np.asarray(pts_list, dtype=np.float32)
        # Media ponderată cu lungimea pentru grosimea bățului reconstruit
        w_num = sum(_polyline_length(by_id[c].points) * by_id[c].mean_width for c in comp)
        w_den = max(1e-6, sum(_polyline_length(by_id[c].points) for c in comp))
        c_nodes: set[int] = set()
        for c in comp:
            c_nodes.update(piece_nodes.get(c, set()))
        raw_sticks.append((merged_pts, w_num / w_den, sorted(comp), c_nodes))

    # Filtrare finală (DUPĂ unirea peste joncțiuni): doar pete minuscule și arce mari
    sticks: List[Stick] = []
    spots: List[Piece] = []
    next_stick_id = 1
    for pts, mw, src_ids, c_nodes in raw_sticks:
        length = _polyline_length(pts)
        if length <= 2.8 * max(1.0, mw):
            spots.append(Piece(0, src_ids[0], pts, 0, len(pts) - 1, mw, "spot"))
            continue
        sagitta, tortuosity = _straightness_metrics(pts)
        if sagitta > 0.16 or tortuosity > 1.15:
            curve_pieces.append(Piece(0, src_ids[0], pts, 0, len(pts) - 1, mw, "curve"))
            continue
        sticks.append(Stick(
            stick_id=next_stick_id,
            points=pts,
            mean_width=mw,
            length_px=length,
            angle_deg=_segment_angle(pts),
            sagitta_ratio=sagitta,
            source_piece_ids=src_ids,
            crossed_nodes=c_nodes,
        ))
        next_stick_id += 1

    return sticks, curve_pieces, spots


# ---------------------------------------------------------------------------
# Faza 2: Graf de relații și clasificare strict din contextul celorlalte bețe
# ---------------------------------------------------------------------------
def _point_to_polyline(pt: np.ndarray, poly: np.ndarray) -> Tuple[float, float]:
    """Returnează (distanța_min_px, fracție_arc_în_[0,1])."""
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
    """Verifică dacă interioarele a două polilinii se ating/intersectează."""
    # Sub-eșantionăm pentru viteză pe linii foarte lungi
    sa = pa[::max(1, len(pa) // 40)]
    sb = pb[::max(1, len(pb) // 40)]
    diff = sa[:, None, :] - sb[None, :, :]
    dmat = np.linalg.norm(diff, axis=2)
    return float(np.min(dmat)) <= tol


def classify_sticks_by_context(sticks: List[Stick]) -> Dict[str, Any]:
    """Construiește relațiile dintre toate bețele și le clasifică contextual."""
    if not sticks:
        return {"median_width_px": 0.0, "median_length_px": 0.0, "angle_families": []}

    # 1) Bounding boxes pentru pre-filtrare rapidă a perechilor
    bboxes = np.array([
        [float(np.min(s.points[:, 0])), float(np.min(s.points[:, 1])),
         float(np.max(s.points[:, 0])), float(np.max(s.points[:, 1]))]
        for s in sticks
    ], dtype=np.float32)

    by_id = {s.stick_id: s for s in sticks}
    n = len(sticks)

    # 2) Graful de contacte (L, T, X)
    for i in range(n):
        si = sticks[i]
        for j in range(i + 1, n):
            sj = sticks[j]
            tol = max(5.0, 2.3 * max(si.mean_width, sj.mean_width))
            if (bboxes[i, 2] + tol < bboxes[j, 0] or bboxes[j, 2] + tol < bboxes[i, 0] or
                    bboxes[i, 3] + tol < bboxes[j, 1] or bboxes[j, 3] + tol < bboxes[i, 1]):
                continue

            # Contact capăt-la-capăt (L)
            ends_i = (si.points[0], si.points[-1])
            ends_j = (sj.points[0], sj.points[-1])
            min_end_end = min(
                float(np.linalg.norm(ei - ej)) for ei in ends_i for ej in ends_j
            )
            if min_end_end <= tol:
                si.l_contacts.append(sj.stick_id)
                sj.l_contacts.append(si.stick_id)
                continue

            # Contact capăt-pe-corp (T)
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

            # Traversare interior-pe-interior (X)
            if (si.crossed_nodes & sj.crossed_nodes) or _polylines_cross_interior(si.points, sj.points, tol):
                si.x_crossings.append(sj.stick_id)
                sj.x_crossings.append(si.stick_id)

    # 3) Familii de unghiuri (clustering determinist ±12°)
    order = sorted(range(n), key=lambda k: sticks[k].length_px, reverse=True)
    families: List[Tuple[float, List[int]]] = []  # (unghi_reprezentativ, membri_index)
    for k in order:
        ang = sticks[k].angle_deg
        placed = False
        for f_idx, (f_ang, members) in enumerate(families):
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
        family_meta.append({
            "family_id": f_id,
            "angle_deg": round(f_ang, 1),
            "count": len(members),
        })

    # 4) Statistici globale de pagină
    med_w = float(np.median([s.mean_width for s in sticks]))
    med_len = float(np.median([s.length_px for s in sticks]))
    for s in sticks:
        s.thickness_ratio = s.mean_width / max(1e-6, med_w)

    # 5) Clasificare contextuală (pur relațională, în ordine semantică)
    # 5a) Linii lungi paralele (ex. linii de portativ / linii de caiet)
    for s in sticks:
        if (s.family_size >= 3 and s.length_px >= max(3.0 * med_len, 25.0 * s.mean_width)
                and (len(s.x_crossings) + len(s.t_receives_branch) >= 1 or s.family_size >= 4)):
            s.role = "linie_paralela_lunga"
            s.role_reason = (
                f"familie paralelă ({s.family_size} bețe la ~{s.angle_deg:.0f}°), "
                f"lungime {s.length_px / max(1e-6, med_len):.1f}× mediana"
            )

    # 5b) Conectori care leagă >= 2 bețe paralele (ex. beam-uri peste optimi/16-imi)
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
            # Verificăm dacă măcar două dintre bețele conectate sunt paralele între ele
            has_parallel_pair = False
            for a_i in range(len(transverse)):
                for b_i in range(a_i + 1, len(transverse)):
                    if _angle_diff_180(transverse[a_i].angle_deg, transverse[b_i].angle_deg) <= 22.0:
                        has_parallel_pair = True
                        break
                if has_parallel_pair:
                    break
            if has_parallel_pair:
                s.role = "conector_paralel"
                connector_ids.add(s.stick_id)
                extra = f", grosime {s.thickness_ratio:.2f}× mediana" if s.thickness_ratio >= 1.25 else ""
                s.role_reason = f"leagă {len(transverse)} bețe paralele transversale{extra}"

    # 5c) Bețe legate la un conector paralel (ex. bețele notelor cu beam din ⊓)
    for s in sticks:
        if s.role != "bat_izolat":
            continue
        touching_conns = [
            oid for oid in (s.l_contacts + s.t_touches_trunk + s.t_receives_branch)
            if oid in connector_ids and _angle_diff_180(s.angle_deg, by_id[oid].angle_deg) >= 30.0
        ]
        if touching_conns:
            s.role = "bat_pe_conector"
            s.role_reason = f"băț conectat la conectorul/beam-ul #{touching_conns[0]}"

    # 5d) Trunchiuri, ramuri, intersecții X, colțuri L, izolate
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
            s.role_reason = f"se sprijină pe trunchiul #{non_bg_trunks[0]} (raport lungime {s.length_px / max(1e-6, longest_trunk):.2f})"
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
# Debug vizual (4 imagini: cerneală, schelet, TOATE bețele, clasificare context)
# ---------------------------------------------------------------------------
def _draw_debug(
    original: np.ndarray,
    ink: np.ndarray,
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

    cv2.imwrite(str(p1), ink)

    skel_vis = np.zeros((*skeleton.shape, 3), np.uint8)
    skel_vis[skeleton > 0] = (230, 230, 230)
    for c in curves:
        poly = np.rint(c.points).astype(np.int32).reshape(-1, 1, 2)
        if len(poly) >= 2:
            cv2.polylines(skel_vis, [poly], False, (200, 70, 200), 2, cv2.LINE_AA)
    for center in node_centers.values():
        cv2.circle(skel_vis, (int(round(center[0])), int(round(center[1]))), 4, (0, 220, 255), -1)
    cv2.imwrite(str(p2), skel_vis)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Debug 3: Faza 1 pură — ABSOLUT TOATE bețele detectate cu verde
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
                    font, 0.45, (0, 150, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(p3), vis_all)

    # Debug 4: Faza 2 — clasificare relațională din context (BGR)
    role_colors = {
        "bat_izolat": (0, 200, 0),           # verde
        "bat_pe_conector": (0, 230, 120),    # verde deschis (bețe cu beam)
        "conector_paralel": (235, 110, 0),   # albastru (beam / conector)
        "trunchi": (0, 145, 255),            # portocaliu
        "ramura": (190, 80, 180),            # mov
        "intersectat_X": (0, 0, 235),        # roșu
        "colt_L": (0, 210, 235),             # galben
        "linie_paralela_lunga": (210, 200, 0), # cyan (linii portativ)
    }
    vis_ctx = original.copy()
    for s in sticks:
        col = role_colors.get(s.role, (150, 150, 150))
        poly = np.rint(s.points).astype(np.int32).reshape(-1, 1, 2)
        thickness = 3 if s.role == "conector_paralel" else 2
        cv2.polylines(vis_ctx, [poly], False, col, thickness, cv2.LINE_AA)
        mid = s.points[len(s.points) // 2]
        cv2.putText(vis_ctx, str(s.stick_id), (int(mid[0]) + 3, int(mid[1]) - 3),
                    font, 0.45, col, 1, cv2.LINE_AA)

    # Legendă compactă în colțul stânga-sus
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
    skeleton_bool = skeletonize(ink > 0)
    skeleton = skeleton_bool.astype(np.uint8) * 255
    dist_map = ndi.distance_transform_edt(ink > 0)

    chains, isolated, node_centers = trace_and_clean_chains(skeleton_bool, dist_map)
    pieces = split_chains_to_pieces(chains, dist_map)
    sticks, curves, spots = merge_collinear_non_destructive(pieces)
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

    debug_paths = _draw_debug(original, ink, skeleton, sticks, curves, node_centers, Path(out_dir))
    output = {
        "image": str(image_path),
        "image_size_px": {"width": int(original.shape[1]), "height": int(original.shape[0])},
        "preprocessing": prep_meta,
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
    out_path = Path(out_dir) / f"{image_path.stem}_bete_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    output["json_file"] = str(out_path)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Detector geometric în 2 faze pentru toate bețele")
    parser.add_argument("image", help="Fotografie PNG/JPG")
    parser.add_argument("--out-dir", default="debug_v2", help="Directorul de ieșire")
    parser.add_argument("--px-per-cm", type=float, default=None, help="Calibrare opțională px/cm")
    args = parser.parse_args(argv)
    try:
        res = process_image(args.image, args.out_dir, args.px_per_cm)
    except Exception as exc:
        print(f"Eroare: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "counts": res["counts"],
        "context_summary": res["context_summary"],
        "debug_images": res["debug_images"],
        "json_file": res["json_file"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())