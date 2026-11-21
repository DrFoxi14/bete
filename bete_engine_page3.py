#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extractor 100% LOSSLESS: Cerneală -> Contururi inter-pixel -> DSS exact.

Principii stricte:
  - ZERO praguri de lungime, ZERO filtru de „pată”, ZERO filtru de „curbă”.
  - Orice succesiune de cel puțin 2 puncte (1 pas) devine un Digital Straight
    Segment (DSS) exact în aritmetică întreagă: 0 <= a*u - b*v + mu < a + b.
  - Verificare matematică la fiecare rulare: reconstrucția binară exclusiv din
    lista de DSS are EXACT 0 pixeli diferență (XOR == 0) față de masca binară.

Dependențe: numpy, opencv-python, scipy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi


# Tabel lookup: cel mai mic bit setat pentru măști 0..15 (direcții 0=R, 1=D, 2=L, 3=U)
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

    @property
    def length_px(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.y1 - self.y0, self.x1 - self.x0)) % 180.0


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


# ---------------------------------------------------------------------------
# Pas 1: Binarizare curată (fără scobirea literelor groase și fără tăierea umbrelor)
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
        return np.zeros((h, w), dtype=bool), {"paper_method": paper_method, "thresh": 30.0}

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
# Pas 2: Extragere exactă a tuturor buclelor de graniță inter-pixel (cracks)
# ---------------------------------------------------------------------------
def extract_crack_loops(ink: np.ndarray) -> List[Tuple[List[int], List[int], bool]]:
    """Extrage toate contururile exacte pe grila (W+1) x (H+1).

    Fiecare pixel de cerneală (x, y) ocupă pătratul [x, x+1] x [y, y+1].
    Granița este orientată cu cerneala pe partea DREAPTĂ:
      - contur exterior (cerneală): arie semnată > 0 (is_hole = False)
      - contur interior (gol alb, ex. ochiul lui `e`/`o`): arie semnată < 0 (is_hole = True)
    La punctele șa (diagonală în tablă de șah), virăm la STÂNGA (cerneală 8-conexă),
    păstrând liniile subțiri diagonale ca un singur contur continuu.
    """
    h, w = ink.shape
    w1 = w + 1
    out_mask = np.zeros((h + 1, w1), dtype=np.uint8)

    # Muchia de SUS (0 = Right): (x, y) -> (x+1, y)
    top = ink.copy()
    top[1:, :] &= ~ink[:-1, :]
    out_mask[:h, :w][top] |= 1

    # Muchia din DREAPTA (1 = Down): (x+1, y) -> (x+1, y+1)
    right = ink.copy()
    right[:, :-1] &= ~ink[:, 1:]
    out_mask[:h, 1:][right] |= 2

    # Muchia de JOS (2 = Left): (x+1, y+1) -> (x, y+1)
    bottom = ink.copy()
    bottom[:-1, :] &= ~ink[1:, :]
    out_mask[1:, 1:][bottom] |= 4

    # Muchia din STÂNGA (3 = Up): (x, y+1) -> (x, y)
    left = ink.copy()
    left[:, 1:] &= ~ink[:, :-1]
    out_mask[1:, :w][left] |= 8

    # Identificăm nodurile șa (exact 2 muchii de ieșire) înainte de a consuma masca
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

            xs = [sx]
            ys = [sy]
            x = sx + dx_tbl[d]
            y = sy + dy_tbl[d]
            idx = start_idx + didx_tbl[d]
            area2 = sx * y - x * sy

            while True:
                m = mask_buf[idx]
                if is_saddle[idx]:
                    nd = (d + 3) & 3  # viraj stânga -> 8-conex
                    if not (m & (1 << nd)):
                        break
                else:
                    if not m:
                        break
                    nd = first_bit[m]

                mask_buf[idx] ^= (1 << nd)
                xs.append(x)
                ys.append(y)
                nx = x + dx_tbl[nd]
                ny = y + dy_tbl[nd]
                area2 += x * ny - nx * y
                x, y = nx, ny
                idx += didx_tbl[nd]
                d = nd

            # Bucla este închisă: ultimul pas a revenit exact în (sx, sy)
            loops.append((xs, ys, area2 < 0))

    return loops


# ---------------------------------------------------------------------------
# Pas 3: Recunoaștere exactă 4-conexă DSS (Debled-Rennesson / Reveillès)
# ---------------------------------------------------------------------------
def _grow_maximal_dss(
    xs: List[int],
    ys: List[int],
    start: int,
    max_steps: int,
) -> Tuple[int, int, int, int, int, int]:
    """Crește cel mai lung DSS 4-conex exact pornind din `start`.

    Returnează (steps, sx, sy, a, b, mu) cu 1 <= steps <= max_steps.
    """
    n = len(xs)
    x0 = xs[start]
    y0 = ys[start]
    i1 = (start + 1) % n
    x1 = xs[i1]
    y1 = ys[i1]
    dx = x1 - x0
    dy = y1 - y0

    if dx != 0:
        sx = 1 if dx > 0 else -1
        sy = 0
        a, b, mu = 0, 1, 0
        uf_u, uf_v = 0, 0
        ul_u, ul_v = 1, 0
        lf_u, lf_v = 0, 0
        ll_u, ll_v = 1, 0
    else:
        sx = 0
        sy = 1 if dy > 0 else -1
        a, b, mu = 1, 0, 0
        uf_u, uf_v = 0, 0
        ul_u, ul_v = 0, 1
        lf_u, lf_v = 0, 0
        ll_u, ll_v = 0, 1

    px, py = x1, y1
    steps = 1

    while steps < max_steps:
        nxt = (start + steps + 1) % n
        cx, cy = xs[nxt], ys[nxt]
        ddx = cx - px
        ddy = cy - py

        # Monotonicitate obligatorie într-un singur cadran
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
            a = v - uf_v
            b = u - uf_u
            mu = -(a * uf_u - b * uf_v)
            ul_u, ul_v = u, v
            lf_u, lf_v = ll_u, ll_v
        elif r == omega:
            a = v - lf_v
            b = u - lf_u
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
    loops: Sequence[Tuple[List[int], List[int], bool]],
) -> List[DSS]:
    """Descompune 100% fără pierderi toate buclele în segmente drepte digitale (DSS)."""
    all_dss: List[DSS] = []
    next_id = 1

    for loop_id, (xs, ys, is_hole) in enumerate(loops, start=1):
        n = len(xs)
        # Găsim un punct natural de tăiere (sfârșitul primului DSS maximal)
        # ca să nu rupem artificial un perete drept chiar la punctul arbitrar de pornire.
        first_steps, *_ = _grow_maximal_dss(xs, ys, 0, n)
        offset = first_steps % n

        consumed = 0
        while consumed < n:
            start = (offset + consumed) % n
            rem = n - consumed
            steps, sx, sy, a, b, mu = _grow_maximal_dss(xs, ys, start, rem)
            end = (start + steps) % n
            all_dss.append(DSS(
                dss_id=next_id,
                loop_id=loop_id,
                is_hole=is_hole,
                x0=xs[start],
                y0=ys[start],
                x1=xs[end],
                y1=ys[end],
                sx=sx,
                sy=sy,
                a=a,
                b=b,
                mu=mu,
                grid_steps=steps,
            ))
            next_id += 1
            consumed += steps

    return all_dss


# ---------------------------------------------------------------------------
# Pas 4: Reconstrucție EXCLUSIV din DSS + verificare matematică XOR == 0
# ---------------------------------------------------------------------------
def reconstruct_from_dss(dss_list: Sequence[DSS], shape: Tuple[int, int]) -> np.ndarray:
    """Reconstruiește imaginea binară EXCLUSIV din cele 8 numere ale fiecărui DSS."""
    h, w = shape
    diff = np.zeros((h, w + 1), dtype=np.int32)

    for seg in dss_list:
        x, y = seg.x0, seg.y0
        sx, sy = seg.sx, seg.sy
        a, b = seg.a, seg.b
        r = seg.mu
        u, v = 0, 0
        u_end = (seg.x1 - seg.x0) * sx
        v_end = (seg.y1 - seg.y0) * sy

        while u < u_end or v < v_end:
            if r < b:
                x += sx
                u += 1
                r += a
            else:
                if sy > 0:
                    diff[y, x] -= 1
                    y += 1
                else:
                    y -= 1
                    diff[y, x] += 1
                v += 1
                r -= b

        if u != u_end or v != v_end or x != seg.x1 or y != seg.y1:
            raise AssertionError(f"Eroare internă de decodare pe DSS #{seg.dss_id}")

    return np.cumsum(diff, axis=1)[:, :w] > 0


# ---------------------------------------------------------------------------
# Debug vizual (100% din segmente desenate, zero segmente ascunse)
# ---------------------------------------------------------------------------
def _draw_debug(
    ink: np.ndarray,
    reconstructed: np.ndarray,
    dss_list: Sequence[DSS],
    out_dir: Path,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "debug_1_binar_exact.png"
    p2 = out_dir / "debug_2_dss_toate.png"
    p3 = out_dir / "debug_3_verificare_xor.png"

    h, w = ink.shape
    cv2.imwrite(str(p1), ink.astype(np.uint8) * 255)

    # Debug 2: TOATE segmentele DSS (de la 1 px în sus), colorate după lungime
    # ca să vezi imediat cu ochiul cum liniile drepte ies natural fără niciun filtru:
    #   < 3 px   : mov închis
    #   3..7 px  : portocaliu (liniuțe scurte / curbe / versuri mici)
    #   7..20 px : galben-verde
    #   >= 20 px : verde aprins / cyan (pereți lungi: bețe, portative, bare)
    vis = np.zeros((h + 1, w + 1, 3), dtype=np.uint8)
    vis[:h, :w][ink] = (40, 40, 40)

    for seg in dss_list:
        L = seg.length_px
        if L < 3.0:
            col = (190, 70, 170)      # mov
        elif L < 7.0:
            col = (0, 155, 255)       # portocaliu
        elif L < 20.0:
            col = (0, 235, 210)       # galben-verde
        else:
            col = (0, 235, 0)         # verde aprins
        cv2.line(vis, (seg.x0, seg.y0), (seg.x1, seg.y1), col, 1, cv2.LINE_AA)

    cv2.imwrite(str(p2), vis)

    # Debug 3: dovada vizuală XOR (verde = identic 100%, roșu = diferență)
    xor_mask = ink != reconstructed
    xor_vis = np.zeros((h, w, 3), dtype=np.uint8)
    xor_vis[reconstructed & ~xor_mask] = (0, 190, 0)
    xor_vis[xor_mask] = (0, 0, 255)
    cv2.imwrite(str(p3), xor_vis)

    return {
        "debug_1_binar_exact": str(p1),
        "debug_2_dss_toate": str(p2),
        "debug_3_verificare_xor": str(p3),
    }


def process_image(
    image_path: str | Path,
    out_dir: str | Path = "debug_dss",
    scale: float = 1.0,
) -> Dict[str, Any]:
    image_path = Path(image_path)
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")

    ink, prep_meta = binarize_image(original, scale=scale)
    loops = extract_crack_loops(ink)
    dss_list = decompose_loops_to_dss(loops)

    # Verificare matematică strictă: reconstruim imaginea DOAR din DSS
    reconstructed = reconstruct_from_dss(dss_list, ink.shape)
    xor_errors = int(np.count_nonzero(ink != reconstructed))
    if xor_errors != 0:
        raise RuntimeError(f"Reconstrucția DSS nu este lossless: {xor_errors} pixeli diferență!")

    total_ink_px = int(np.count_nonzero(ink))
    total_crack_edges = sum(len(xs) for xs, _, _ in loops)
    sum_dss_steps = sum(s.grid_steps for s in dss_list)

    debug_paths = _draw_debug(ink, reconstructed, dss_list, Path(out_dir))

    # Salvăm JSON compact (toate segmentele, fără nicio excepție)
    segments_json = [
        {
            "id": s.dss_id,
            "loop": s.loop_id,
            "hole": s.is_hole,
            "start": [s.x0, s.y0],
            "end": [s.x1, s.y1],
            "dss": {"sx": s.sx, "sy": s.sy, "a": s.a, "b": s.b, "mu": s.mu},
            "steps": s.grid_steps,
            "len": round(s.length_px, 2),
            "ang": round(s.angle_deg, 2),
        }
        for s in dss_list
    ]

    summary = {
        "image": str(image_path),
        "scale": scale,
        "preprocessing": prep_meta,
        "lossless_verification": {
            "ink_pixels_original": total_ink_px,
            "crack_edges_total": total_crack_edges,
            "dss_steps_sum": sum_dss_steps,
            "reconstructed_xor_error_pixels": xor_errors,
            "guaranteed_lossless": (xor_errors == 0 and total_crack_edges == sum_dss_steps),
        },
        "counts": {
            "closed_loops": len(loops),
            "outer_loops": sum(1 for _, _, h in loops if not h),
            "hole_loops": sum(1 for _, _, h in loops if h),
            "total_dss_segments": len(dss_list),
            "compression_edges_per_dss": round(total_crack_edges / max(1, len(dss_list)), 2),
        },
        "debug_images": debug_paths,
    }

    out_path = Path(out_dir) / f"{image_path.stem}_dss_lossless.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({**summary, "segments": segments_json}, ensure_ascii=False),
        encoding="utf-8",
    )
    summary["json_file"] = str(out_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extractor 100% lossless Contur -> DSS exact")
    parser.add_argument("image", help="Fotografie PNG/JPG")
    parser.add_argument("--out-dir", default="debug_dss", help="Directorul de ieșire")
    parser.add_argument("--scale", type=float, default=1.0, help="Scalare opțională (1.0 = nativ, 2.0 = sub-pixel)")
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