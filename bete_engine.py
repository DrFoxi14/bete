#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic, geometry-only handwritten stem detector.

Dependencies: numpy, opencv-python, scipy, scikit-image.
No learned models, neural networks, line-transform detectors, or cloud calls.

Usage:
    python bete_engine.py photo.jpg
    python bete_engine.py photo.jpg --out-dir debug --px-per-cm 120

Coordinates in JSON are in the original image's pixel coordinate system.
Without a calibration value, physical lengths cannot be inferred from a photo;
length_cm is therefore null unless --px-per-cm is supplied.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_sauvola
from skimage.morphology import skeletonize

Point = Tuple[int, int]  # (x, y)


@dataclass
class Chain:
    points: np.ndarray              # N x 2, x/y float coordinates
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
    kind: str                       # "line" or "curve"
    start_node: Optional[int] = None
    end_node: Optional[int] = None
    state: str = "certificat"
    reason: str = ""
    source_piece_ids: Optional[List[int]] = None


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _paper_region(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """Estimate the sheet by its low-chroma, relatively bright connected area.

    A convex hull bridges shadow gaps and small occlusions. If the photo does
    not contain a separable sheet/table boundary, the safe fallback is the
    entire frame; ink segmentation still uses a local threshold.
    """
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    light = lab[:, :, 0]
    sat = hsv[:, :, 1]

    # The percentile makes the cutoff adapt to both bright and dim cameras.
    light_floor = max(34.0, float(np.percentile(light, 22)))
    sat_ceiling = min(150.0, max(65.0, float(np.percentile(sat, 82))))
    candidate = ((light >= light_floor) & (sat <= sat_ceiling)).astype(np.uint8)

    close_size = _odd(int(min(h, w) * 0.025))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate = ndi.binary_fill_holes(candidate > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if count <= 1:
        return np.full((h, w), 255, np.uint8), "fallback_full_frame_no_paper_component"

    areas = stats[1:, cv2.CC_STAT_AREA]
    label = 1 + int(np.argmax(areas))
    area = int(stats[label, cv2.CC_STAT_AREA])
    ratio = area / float(h * w)
    if ratio < 0.22 or ratio > 0.985:
        return np.full((h, w), 255, np.uint8), f"fallback_full_frame_component_ratio_{ratio:.3f}"

    component = (labels == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full((h, w), 255, np.uint8), "fallback_full_frame_no_contour"
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    paper = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(paper, hull, 255)
    return paper, f"largest_low_chroma_component_ratio_{ratio:.3f}"


def binarize_ink(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Sauvola binarization on the darkest BGR channel, restricted to paper."""
    h, w = image.shape[:2]
    paper, paper_method = _paper_region(image)
    darkest = np.min(image, axis=2).astype(np.uint8)

    # A scale tied to the photo size handles common phone resolutions. The
    # window remains large enough to estimate illumination under a pen stroke.
    window = _odd(min(151, max(31, int(min(h, w) * 0.025))))
    window = min(window, _odd(max(3, min(h, w) - 2)))
    local_threshold = threshold_sauvola(darkest, window_size=window, k=0.20, r=128)
    ink = (darkest.astype(np.float32) < local_threshold) & (paper > 0)

    # Close only one-pixel gaps; do not open away tiny but legitimate marks.
    ink_u8 = (ink.astype(np.uint8) * 255)
    ink_u8 = cv2.morphologyEx(
        ink_u8, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1
    )
    ink_u8[paper == 0] = 0
    metadata = {
        "threshold": "Sauvola",
        "channel": "minimum(B, G, R)",
        "sauvola_window_px": int(window),
        "paper_region_method": paper_method,
        "paper_pixels": int(np.count_nonzero(paper)),
    }
    return ink_u8, paper, metadata


def _pixel_graph(skeleton: np.ndarray) -> Tuple[np.ndarray, List[List[int]], Dict[Point, int]]:
    """Build an 8-neighbour pixel graph without redundant diagonal shortcuts."""
    ys, xs = np.nonzero(skeleton)
    coords = np.column_stack((xs, ys)).astype(np.int32)
    lookup: Dict[Point, int] = {(int(x), int(y)): i for i, (x, y) in enumerate(coords)}
    adj: List[List[int]] = [[] for _ in range(len(coords))]
    h, w = skeleton.shape

    for i, (x0, y0) in enumerate(coords):
        x, y = int(x0), int(y0)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                j = lookup.get((nx, ny))
                if j is None:
                    continue
                # If an orthogonal route exists, ignore this diagonal edge.
                if dx and dy and ((x + dx, y) in lookup or (x, y + dy) in lookup):
                    continue
                adj[i].append(j)
    return coords, adj, lookup


def trace_chains(skeleton: np.ndarray, stroke_distance: Optional[np.ndarray] = None) -> Tuple[List[Chain], List[np.ndarray], Dict[int, int]]:
    """Compress the skeleton into paths between endpoints/junction clusters."""
    coords, adj, lookup = _pixel_graph(skeleton)
    degrees = np.fromiter((len(a) for a in adj), dtype=np.int32, count=len(adj))
    critical = degrees != 2
    junction = degrees >= 3

    # Group adjacent junction pixels into one topological node. Endpoints and
    # isolated pixels remain individual nodes.
    junction_labels, junction_count = None, 0
    jmask = np.zeros(skeleton.shape, bool)
    if np.any(junction):
        jmask[coords[junction, 1], coords[junction, 0]] = True
        # A thick handwritten crossing can skeletonize into a tiny square
        # cycle with four separate degree-3 pixels. Group junction pixels
        # whose separation is on the order of one local stroke width.
        if stroke_distance is None:
            local_radius = ndi.distance_transform_edt(skeleton > 0)
        else:
            local_radius = stroke_distance
        junction_radii = local_radius[coords[junction, 1], coords[junction, 0]]
        merge_radius = max(2, int(round(2.0 * float(np.median(junction_radii)))))
        merge_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * merge_radius + 1, 2 * merge_radius + 1)
        )
        grouped_junction_mask = cv2.dilate(jmask.astype(np.uint8), merge_kernel) > 0
        expanded_labels, junction_count = ndi.label(
            grouped_junction_mask, structure=np.ones((3, 3), dtype=np.uint8)
        )
        # Only original junction pixels become members; dilated pixels merely
        # connect them into a common topological node.
        junction_labels = expanded_labels
    else:
        junction_labels = np.zeros(skeleton.shape, np.int32)
        junction_count = 0

    groups: List[List[int]] = []
    node_map: Dict[int, int] = {}
    for label in range(1, junction_count + 1):
        members = [i for i, (x, y) in enumerate(coords)
                   if junction[i] and junction_labels[int(y), int(x)] == label]
        if members:
            node_id = len(groups)
            groups.append(members)
            for idx in members:
                node_map[idx] = node_id

    for i in range(len(coords)):
        if critical[i] and i not in node_map:
            node_id = len(groups)
            groups.append([i])
            node_map[i] = node_id

    visited: set[Tuple[int, int]] = set()
    chains: List[Chain] = []

    def edge_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    # Trace every edge leaving a critical node, stopping at the next node.
    for node_id, members in enumerate(groups):
        for start in members:
            for nxt in adj[start]:
                if node_map.get(nxt) == node_id:
                    continue
                first_edge = edge_key(start, nxt)
                if first_edge in visited:
                    continue
                path = [start, nxt]
                visited.add(first_edge)
                previous, current = start, nxt
                guard = 0
                while current not in node_map and guard <= len(coords):
                    options = [q for q in adj[current] if q != previous]
                    if not options:
                        break
                    following = options[0]
                    ek = edge_key(current, following)
                    if ek in visited:
                        break
                    visited.add(ek)
                    path.append(following)
                    previous, current = current, following
                    guard += 1
                end_node = node_map.get(current)
                # A route that leaves and re-enters the same collapsed node
                # belongs to the junction core, not to an independent arm.
                if end_node == node_id:
                    continue
                xy = coords[np.asarray(path, dtype=np.int32)].astype(np.float32)
                chains.append(Chain(xy, node_id, end_node, False))

    # Closed loops have no endpoint/junction node (all pixels have degree 2).
    for i in range(len(coords)):
        for nxt in adj[i]:
            ek = edge_key(i, nxt)
            if ek in visited:
                continue
            path = [i, nxt]
            visited.add(ek)
            previous, current = i, nxt
            guard = 0
            while guard <= len(coords):
                options = [q for q in adj[current] if q != previous]
                if not options:
                    break
                following = options[0]
                ek2 = edge_key(current, following)
                if ek2 in visited:
                    break
                visited.add(ek2)
                path.append(following)
                previous, current = current, following
                if current == i:
                    break
                guard += 1
            xy = coords[np.asarray(path, dtype=np.int32)].astype(np.float32)
            chains.append(Chain(xy, None, None, bool(path[-1] == path[0])))

    # Isolated skeleton pixels are spots, not line candidates.
    isolated = [coords[i].copy() for i in range(len(coords)) if degrees[i] == 0]
    node_arm_counts: Dict[int, int] = {}
    for chain in chains:
        if chain.start_node is not None:
            node_arm_counts[chain.start_node] = node_arm_counts.get(chain.start_node, 0) + 1
        if chain.end_node is not None and chain.end_node != chain.start_node:
            node_arm_counts[chain.end_node] = node_arm_counts.get(chain.end_node, 0) + 1
    return chains, isolated, node_arm_counts



def _resample_polyline(points: np.ndarray, count: int = 32) -> np.ndarray:
    if len(points) < 2:
        return np.repeat(points[:1], count, axis=0)
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if distances[-1] < 1e-6:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, distances[-1], count)
    return np.column_stack([np.interp(targets, distances, points[:, axis]) for axis in range(2)])


def deduplicate_chains(chains: Sequence[Chain], stroke_distance: np.ndarray) -> List[Chain]:
    """Collapse near-identical parallel graph routes created by digital thinning.

    Some one-pixel skeletons contain two nearly coincident routes along one
    thick curved stroke. If they connect the same pair of topological nodes,
    have similar arc length, and remain within one measured pen width, they are
    one geometric path rather than an X or two competing strokes.
    """
    grouped: Dict[Tuple[int, int], List[Chain]] = {}
    untouched: List[Chain] = []
    for chain in chains:
        if chain.start_node is None or chain.end_node is None or chain.start_node == chain.end_node:
            untouched.append(chain)
            continue
        key = tuple(sorted((chain.start_node, chain.end_node)))
        # Normalize orientation so identical routes are directly comparable.
        if chain.start_node > chain.end_node:
            normalized = Chain(chain.points[::-1].copy(), chain.end_node, chain.start_node, chain.closed)
        else:
            normalized = chain
        grouped.setdefault(key, []).append(normalized)

    result = list(untouched)
    for key, candidates in grouped.items():
        kept: List[Chain] = []
        candidates = sorted(candidates, key=lambda c: _polyline_length(c.points), reverse=True)
        for candidate in candidates:
            length = _polyline_length(candidate.points)
            xi = np.clip(np.rint(candidate.points[:, 0]).astype(int), 0, stroke_distance.shape[1] - 1)
            yi = np.clip(np.rint(candidate.points[:, 1]).astype(int), 0, stroke_distance.shape[0] - 1)
            width = max(1.0, 2.0 * float(np.median(stroke_distance[yi, xi])))
            sample = _resample_polyline(candidate.points, 32)
            duplicate = False
            for previous in kept:
                previous_length = _polyline_length(previous.points)
                ratio = max(length, previous_length) / max(1e-6, min(length, previous_length))
                if ratio > 1.25:
                    continue
                other = _resample_polyline(previous.points, 32)
                distances = np.linalg.norm(sample - other, axis=1)
                if float(np.mean(distances)) <= 1.50 * width and float(np.max(distances)) <= 2.00 * width:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        result.extend(kept)
    return result

def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _curvature_mask(points: np.ndarray, widths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return high-curvature positions and a smoothed path.

    Curvature is measured at several scales proportional to local pen width.
    Both the turn between incoming/outgoing chords and accumulated tangent
    rotation are considered. This detects smooth transitions and hairpins,
    not only sharp polygon corners.
    """
    n = len(points)
    if n < 7:
        return np.zeros(n, bool), points.astype(np.float32)

    width = max(1.0, float(np.median(widths)))
    sigma = max(0.65, min(3.0, 0.45 * width))
    smoothed = np.column_stack([
        ndi.gaussian_filter1d(points[:, axis].astype(np.float64), sigma=sigma, mode="nearest")
        for axis in range(2)
    ]).astype(np.float32)

    tangents = np.gradient(smoothed, axis=0)
    headings = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
    signed_turn = np.diff(headings, prepend=headings[0])
    # Keep the sign: opposite bends on either side of a straight stem must not
    # be accumulated together, or an entire d-ascender gets swallowed as curve.
    prefix = np.concatenate(([0.0], np.cumsum(signed_turn)))
    high = np.zeros(n, bool)

    # Width-scaled windows: short catches tight hooks; long catches broad C/O arcs.
    for factor in (3.0, 5.0, 8.0):
        win = max(3, int(round(factor * width)))
        if 2 * win >= n - 1:
            continue
        idx = np.arange(win, n - win)
        incoming = smoothed[idx] - smoothed[idx - win]
        outgoing = smoothed[idx + win] - smoothed[idx]
        ni = np.linalg.norm(incoming, axis=1)
        no = np.linalg.norm(outgoing, axis=1)
        valid = (ni > 1e-4) & (no > 1e-4)
        dot = np.zeros(len(idx), dtype=np.float64)
        dot[valid] = np.sum(incoming[valid] * outgoing[valid], axis=1) / (ni[valid] * no[valid])
        chord_turn = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        # 52° over a width-scaled neighbourhood leaves ordinary hand wobble intact.
        high[idx[valid & (chord_turn >= 48.0)]] = True

    # Integrated absolute rotation catches smooth arcs whose local tangent change
    # is spread over many pixels. This is why the C-to-stem transition is found.
    radius = max(4, int(round(5.0 * width)))
    for i in range(n):
        left, right = max(0, i - radius), min(n - 1, i + radius)
        total = abs(prefix[right + 1] - prefix[left])
        if total >= math.radians(45.0):
            high[i] = True

    # Expand cut zones by approximately one pen width so the cut lies outside
    # the uncertain transition rather than through its centre.
    pad = max(1, int(round(0.5 * width)))
    high = ndi.binary_dilation(high, structure=np.ones(2 * pad + 1, dtype=bool))
    return high, smoothed


def _runs(mask: np.ndarray) -> List[Tuple[int, int, bool]]:
    """Inclusive contiguous runs as (start, end, value)."""
    if len(mask) == 0:
        return []
    changes = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes - 1, len(mask) - 1]
    return [(int(a), int(b), bool(mask[a])) for a, b in zip(starts, ends)]


def split_chains(chains: Sequence[Chain], distance: np.ndarray) -> List[Piece]:
    """Split only by curvature; never reject a straight piece by its width ratio."""
    pieces: List[Piece] = []
    next_piece_id = 1
    for chain_id, chain in enumerate(chains, start=1):
        points = chain.points
        if len(points) < 2:
            continue
        xi = np.clip(np.rint(points[:, 0]).astype(int), 0, distance.shape[1] - 1)
        yi = np.clip(np.rint(points[:, 1]).astype(int), 0, distance.shape[0] - 1)
        widths = np.maximum(1.0, 2.0 * distance[yi, xi])
        high, smoothed = _curvature_mask(points, widths)
        if chain.closed:
            high[:] = True
        for a, b, is_curve in _runs(high):
            part_points = smoothed[a:b + 1].copy()
            mean_width = float(np.mean(widths[a:b + 1])) if b >= a else float(np.mean(widths))
            start_node = chain.start_node if a == 0 else None
            end_node = chain.end_node if b == len(points) - 1 else None
            piece = Piece(
                chain_id, next_piece_id, part_points, a, b, mean_width,
                "curve" if is_curve else "line", start_node, end_node
            )
            if is_curve:
                piece.state = "respins"
                piece.reason = "curbură locală mare / arc / întoarcere; separat prin analiza tangentei"
            else:
                piece.state = "certificat"
                piece.reason = "băț liniar extras; fără prag de lungime/grosime și fără excludere la joncțiuni"
            pieces.append(piece)
            next_piece_id += 1
    return pieces


def _page1_straightness(points: np.ndarray) -> Tuple[float, float]:
    """Return (sagitta/chord, arc/chord), independent of pen thickness."""
    if len(points) < 3:
        return 0.0, 1.0
    chord_vec = points[-1] - points[0]
    chord = float(np.linalg.norm(chord_vec))
    arc = _polyline_length(points)
    if chord < 1e-6:
        return float("inf"), float("inf")
    relative = points - points[0]
    cross = np.abs(relative[:, 0] * chord_vec[1] - relative[:, 1] * chord_vec[0])
    sagitta = float(np.max(cross) / chord)
    return sagitta / chord, arc / chord


def merge_continuous_chain_fragments(pieces: List[Piece]) -> List[Piece]:
    """Rejoin adjacent curvature fragments of the SAME skeleton chain.

    A short high-curvature cut is absorbed only when the combined path remains
    geometrically straight/gently bowed. This repairs false curve cuts such as
    814→815 without joining a real loop or hairpin into a stick.
    """
    by_chain: Dict[int, List[Piece]] = {}
    for piece in pieces:
        by_chain.setdefault(piece.chain_id, []).append(piece)

    output: List[Piece] = []
    for chain_id, chain_pieces in by_chain.items():
        work = sorted(chain_pieces, key=lambda p: (p.start_index, p.end_index))
        changed = True
        while changed and len(work) > 1:
            changed = False
            merged_work: List[Piece] = []
            i = 0
            while i < len(work):
                if i + 1 < len(work):
                    left, right = work[i], work[i + 1]
                    contiguous = right.start_index - left.end_index <= 1
                    if contiguous and right.start_index >= left.start_index:
                        gap = float(np.linalg.norm(left.points[-1] - right.points[0]))
                        if gap <= 2.5:
                            if gap <= 1.5:
                                combined = np.vstack((left.points, right.points[1:]))
                            else:
                                bridge_count = max(1, int(math.ceil(gap)))
                                bridge = [
                                    left.points[-1] + (right.points[0] - left.points[-1]) * (t / bridge_count)
                                    for t in range(1, bridge_count)
                                ]
                                combined = np.vstack((left.points, np.asarray(bridge), right.points))
                            sagitta, tortuosity = _page1_straightness(combined)
                            # These are shape-only tests; no length/width ratio is used.
                            if sagitta <= 0.19 and tortuosity <= 1.15:
                                left_len = _polyline_length(left.points)
                                right_len = _polyline_length(right.points)
                                total = max(1e-6, left_len + right_len)
                                merged_piece = Piece(
                                    chain_id=chain_id,
                                    piece_id=min(left.piece_id, right.piece_id),
                                    points=combined.astype(np.float32),
                                    start_index=min(left.start_index, right.start_index),
                                    end_index=max(left.end_index, right.end_index),
                                    mean_width=(left.mean_width * left_len + right.mean_width * right_len) / total,
                                    kind="line",
                                    start_node=left.start_node,
                                    end_node=right.end_node,
                                    state="certificat",
                                    reason="fragmente consecutive reunite; traseu combinat drept/ușor curbat, fără filtru lungime/grosime",
                                    source_piece_ids=sorted(set(
                                        (left.source_piece_ids or [left.piece_id]) +
                                        (right.source_piece_ids or [right.piece_id])
                                    )),
                                )
                                merged_work.append(merged_piece)
                                i += 2
                                changed = True
                                continue
                merged_work.append(work[i])
                i += 1
            work = sorted(merged_work, key=lambda p: (p.start_index, p.end_index))
        output.extend(work)
    return sorted(output, key=lambda p: (p.chain_id, p.start_index, p.piece_id))


def _piece_outward_direction(piece: Piece, node_id: int) -> Optional[np.ndarray]:
    p = piece.points
    if len(p) < 2:
        return None
    if piece.start_node == node_id:
        anchor = p[0]
        k = min(len(p) - 1, max(3, int(round(4 * piece.mean_width))))
        vector = p[k] - anchor
    elif piece.end_node == node_id:
        anchor = p[-1]
        k = min(len(p) - 1, max(3, int(round(4 * piece.mean_width))))
        vector = p[-1 - k] - anchor
    else:
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else None


def link_collinear_continuations(
    pieces: List[Piece],
) -> List[Tuple[int, str, int, str, int]]:
    """Pair only geometrically opposite arms at each node.

    No glyph labels or T/F/X/L winner rules are used. At a junction, a pair of
    nearly opposite directions is joined into one continuous stick; every
    unpaired straight arm remains its own stick. Thus F yields its stem plus
    both bars, while an X yields two crossing sticks instead of a red winner.
    """
    by_node: Dict[int, List[Tuple[Piece, np.ndarray, str]]] = {}
    for piece in pieces:
        if piece.kind != "line" or len(piece.points) < 2:
            continue
        for node_id, side in ((piece.start_node, "start"), (piece.end_node, "end")):
            if node_id is None:
                continue
            direction = _piece_outward_direction(piece, node_id)
            if direction is not None:
                by_node.setdefault(node_id, []).append((piece, direction, side))

    links: List[Tuple[int, str, int, str, int]] = []
    for node_id, arms in by_node.items():
        pairs: List[Tuple[float, int, int]] = []
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                pi, di, _si = arms[i]
                pj, dj, _sj = arms[j]
                if pi.piece_id == pj.piece_id:
                    continue
                dot = float(np.dot(di, dj))
                # This is a local geometric continuity test only, not a
                # classification of the junction or a rejection of any arm.
                if dot <= -0.70:
                    pairs.append((dot, i, j))

        pairs.sort(key=lambda item: item[0])  # most nearly opposite first
        used: set[int] = set()
        for _dot, i, j in pairs:
            if i in used or j in used:
                continue
            pi, _di, side_i = arms[i]
            pj, _dj, side_j = arms[j]
            used.add(i)
            used.add(j)
            links.append((pi.piece_id, side_i, pj.piece_id, side_j, node_id))

    return links


def merge_collinear_sticks(pieces: List[Piece], links: Sequence[Tuple[int, str, int, str, int]]) -> List[Piece]:
    """Join geometrically continuous arms while preserving every unpaired arm."""
    by_id = {p.piece_id: p for p in pieces}
    endpoint_link: Dict[Tuple[int, str], Tuple[int, str]] = {}
    neighbours: Dict[int, set[int]] = {}
    for a, sa, b, sb, _node in links:
        if a not in by_id or b not in by_id:
            continue
        endpoint_link[(a, sa)] = (b, sb)
        endpoint_link[(b, sb)] = (a, sa)
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)

    visited: set[int] = set()
    merged: List[Piece] = []
    for seed in neighbours:
        if seed in visited:
            continue
        component: set[int] = set()
        stack = [seed]
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(neighbours.get(current, ()))
        visited.update(component)
        if len(component) < 2:
            continue

        # Pick a free end so the merged point sequence can be walked once.
        degree = {pid: len(neighbours.get(pid, ())) for pid in component}
        leaves = [pid for pid, d in degree.items() if d <= 1]
        start_id = min(leaves) if leaves else min(component)
        start_piece = by_id[start_id]
        linked_start = (start_id, "start") in endpoint_link
        linked_end = (start_id, "end") in endpoint_link
        forward = not linked_start if linked_start != linked_end else True
        first_points = start_piece.points.copy() if forward else start_piece.points[::-1].copy()
        assembled = [p.copy() for p in first_points]
        current_id = start_id
        exit_side = "end" if forward else "start"
        walked = {start_id}

        while (current_id, exit_side) in endpoint_link:
            next_id, next_entry_side = endpoint_link[(current_id, exit_side)]
            if next_id in walked or next_id not in component:
                break
            next_piece = by_id[next_id]
            next_forward = next_entry_side == "start"
            next_points = next_piece.points.copy() if next_forward else next_piece.points[::-1].copy()
            # Retain the small junction bridge so the displayed stroke is continuous.
            gap = float(np.linalg.norm(next_points[0] - assembled[-1]))
            steps = int(math.ceil(gap))
            if steps > 1:
                bridge_start = assembled[-1].copy()
                for t in range(1, steps):
                    assembled.append((1.0 - t / steps) * bridge_start + (t / steps) * next_points[0])
            assembled.extend(p.copy() for p in next_points[1:])
            walked.add(next_id)
            current_id = next_id
            exit_side = "end" if next_forward else "start"

        points = np.asarray(assembled, dtype=np.float32)
        source_ids = sorted(component)
        states = {by_id[pid].state for pid in component}
        state = "certificat"
        reason = "segmente coliniare reunite peste joncțiune; toate brațele neîmperecheate sunt păstrate"
        merged_piece = Piece(
            chain_id=min(by_id[pid].chain_id for pid in component),
            piece_id=min(source_ids), points=points,
            start_index=0, end_index=len(points) - 1,
            mean_width=float(np.mean([by_id[pid].mean_width for pid in component])),
            kind="line", state=state, reason=reason, source_piece_ids=source_ids,
        )
        merged.append(merged_piece)

    # Components merged above are removed; all unlinked pieces remain untouched.
    merged_ids = {pid for p in merged for pid in (p.source_piece_ids or [])}
    return [p for p in pieces if p.piece_id not in merged_ids] + merged


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _segment_angle(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    delta = points[-1] - points[0]
    angle = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 180.0
    return float(angle)


def _draw_debug(
    original: np.ndarray,
    ink: np.ndarray,
    skeleton: np.ndarray,
    pieces: Sequence[Piece],
    isolated: Sequence[np.ndarray],
    stroke_distance: np.ndarray,
    debug_dir: Path,
) -> Dict[str, str]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    p1 = debug_dir / "debug_1_cerneala.png"
    p2 = debug_dir / "debug_2_schelet.png"
    p3 = debug_dir / "debug_3_rezultat.png"
    cv2.imwrite(str(p1), ink)

    skel_vis = np.zeros((*skeleton.shape, 3), np.uint8)
    skel_vis[skeleton > 0] = (235, 235, 235)
    # Junction pixels are yellow; path endpoints and cut positions cyan.
    coords, adj, _lookup = _pixel_graph(skeleton > 0)
    if len(coords):
        degrees = np.array([len(a) for a in adj])
        is_junction = degrees >= 3
        if np.any(is_junction):
            jmask = np.zeros(skeleton.shape, np.uint8)
            jmask[coords[is_junction, 1], coords[is_junction, 0]] = 1
            radii = stroke_distance[coords[is_junction, 1], coords[is_junction, 0]]
            radius = max(2, int(round(2.0 * float(np.median(radii)))))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            expanded = cv2.dilate(jmask, kernel) > 0
            labels, count = ndi.label(expanded, structure=np.ones((3, 3), dtype=np.uint8))
            for label in range(1, count + 1):
                members = is_junction & (labels[coords[:, 1], coords[:, 0]] == label)
                if not np.any(members):
                    continue
                centre = np.mean(coords[members], axis=0)
                cv2.circle(skel_vis, (int(round(centre[0])), int(round(centre[1]))),
                           5, (0, 220, 255), -1)
    for piece in pieces:
        for x, y in (piece.points[0], piece.points[-1]):
            cv2.circle(skel_vis, (int(round(x)), int(round(y))), 3, (255, 220, 0), -1)
    for xy in isolated:
        cv2.circle(skel_vis, (int(xy[0]), int(xy[1])), 3, (0, 0, 255), -1)
    cv2.imwrite(str(p2), skel_vis)

    result_vis = original.copy()
    palette = {
        "certificat": (0, 210, 0),
        "ambiguu": (0, 0, 240),
        "respins": (145, 145, 145),
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    for piece in pieces:
        color = palette.get(piece.state, (0, 165, 255))
        poly = np.rint(piece.points).astype(np.int32).reshape(-1, 1, 2)
        if len(poly) >= 2:
            cv2.polylines(result_vis, [poly], False, color, 3, cv2.LINE_AA)
        mid = piece.points[len(piece.points) // 2]
        label = f"{piece.piece_id}"
        cv2.putText(result_vis, label, (int(mid[0]) + 4, int(mid[1]) - 4),
                    font, 0.55, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(p3), result_vis)
    return {"debug_1_cerneala": str(p1), "debug_2_schelet": str(p2), "debug_3_rezultat": str(p3)}


def process_image(
    image_path: str | Path,
    out_dir: str | Path = ".",
    px_per_cm: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the complete detector and save three debug images plus JSON."""
    image_path = Path(image_path)
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")
    if px_per_cm is not None and px_per_cm <= 0:
        raise ValueError("px_per_cm trebuie să fie pozitiv")

    ink, paper, prep_meta = binarize_ink(original)
    skeleton_bool = skeletonize(ink > 0)
    skeleton = (skeleton_bool.astype(np.uint8) * 255)
    distance = ndi.distance_transform_edt(ink > 0)

    chains, isolated, _node_arm_counts = trace_chains(skeleton_bool, distance)
    chains = deduplicate_chains(chains, distance)
    pieces = split_chains(chains, distance)
    pieces = merge_continuous_chain_fragments(pieces)
    junction_links = link_collinear_continuations(pieces)
    pieces = merge_collinear_sticks(pieces, junction_links)

    results: List[Dict[str, Any]] = []
    for piece in pieces:
        points = np.rint(piece.points).astype(int)
        length_px = _polyline_length(piece.points)
        width = max(1.0, piece.mean_width)
        x1, y1 = map(int, points[0])
        x2, y2 = map(int, points[-1])
        results.append({
            "id": int(piece.piece_id),
            "chain_id": int(piece.chain_id),
            "source_piece_ids": piece.source_piece_ids or [int(piece.piece_id)],
            "coordinates": {"start": {"x": x1, "y": y1}, "end": {"x": x2, "y": y2}},
            "points_xy": [{"x": int(x), "y": int(y)} for x, y in points],
            "length_px": round(length_px, 2),
            "length_cm": round(length_px / px_per_cm, 3) if px_per_cm else None,
            "angle_deg": round(_segment_angle(piece.points), 2),
            "local_stroke_width_px": round(width, 2),
            "state": piece.state,
            "reason": piece.reason,
            "geometry": piece.kind,
        })

    # Standalone isolated pixels are explicitly reported as spots.
    for xy in isolated:
        pid = len(results) + 1
        x, y = int(xy[0]), int(xy[1])
        results.append({
            "id": pid, "chain_id": None,
            "coordinates": {"start": {"x": x, "y": y}, "end": {"x": x, "y": y}},
            "points_xy": [{"x": x, "y": y}], "length_px": 0.0, "length_cm": 0.0 if px_per_cm else None,
            "angle_deg": 0.0, "local_stroke_width_px": 0.0,
            "state": "respins", "reason": "pixel izolat; pată, nu segment", "geometry": "spot",
        })

    debug_paths = _draw_debug(original, ink, skeleton, pieces, isolated, distance, Path(out_dir))
    output = {
        "image": str(image_path),
        "image_size_px": {"width": int(original.shape[1]), "height": int(original.shape[0])},
        "calibration": {"px_per_cm": px_per_cm, "note": "length_cm este null fără calibrare"},
        "preprocessing": prep_meta,
        "summary": {
            "certified": sum(r["state"] == "certificat" for r in results),
            "ambiguous": sum(r["state"] == "ambiguu" for r in results),
            "rejected": sum(r["state"] == "respins" for r in results),
            "skeleton_chains": len(chains),
        },
        "segments": results,
        "debug_images": debug_paths,
        "limitations": [
            "Fără scară/calibrare în fotografie, distanțele fizice în cm nu pot fi deduse.",
            "Brațele necoliniare de la joncțiuni sunt păstrate separat; traseele coliniare sunt reunite geometric.",
            "Pragurile de curbură sunt raportate la geometria traseului, nu la un raport lungime/grosime.",
        ],
    }
    out_path = Path(out_dir) / f"{image_path.stem}_rezultate.json"
    output["json_file"] = str(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Motor geometric determinist pentru detectarea bețelor")
    parser.add_argument("image", help="Fotografie PNG/JPG")
    parser.add_argument("--out-dir", default=".", help="Directorul pentru debug și JSON (implicit: .)")
    parser.add_argument("--px-per-cm", type=float, default=None,
                        help="Calibrare opțională: pixeli pentru un centimetru")
    args = parser.parse_args(argv)
    try:
        result = process_image(args.image, args.out_dir, args.px_per_cm)
    except Exception as exc:
        print(f"Eroare: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
