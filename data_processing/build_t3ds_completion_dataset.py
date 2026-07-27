#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teeth Completion Data Builder (Union L∞ Normalization Only, Pre-Coloring, Nine-Grid, Missing Export)

Simplified as requested:
- **Only one normalization**: after alignment, we take the union of {target tooth, 2 neighbors, k gingiva},
  compute **s_final = max(max(|qx|), max(|qy|), max(|qz|))** and normalize all selected points by s_final.
  This guarantees each coordinate component is in **[-1, 1]**.
- **+Y=lingual anchoring**: Z is jaw axis; X is neighbor axis (projected ⟂ Z). We flip (X,Y) if +Y points buccal, so that +Y
  always points lingual. This makes 3×3 partition semantically stable (left-right vs occlusal-cervical).
- **Pre-Coloring**: first color the full target tooth in normalized frame, then do nine-grid cutting. Colors reveal
  anatomical loss at a glance.
- **Exports** (all in normalized local coords, restorable by meta):
  - complete/{tooth}/{case}.pcd             (XYZ, target tooth only, s_final)
  - gt/{tooth}/{case}.pcd                   (XYZRGB, full tooth + neighbors + gingiva)
  - partial/{tooth}/{case}/{vi:02d}.pcd     (XYZRGB, nine-grid missing on tooth; keep neighbors + gingiva)
  - gt_missing/{tooth}/{case}/{vi:02d}.pcd  (XYZRGB, removed tooth region only)

Restore formula (all files):
    X_global = R_align.T @ (X_norm * scale_used) + center
Where center is the target tooth center in world; R_align rows are (x_hat, y_hat, z_hat) in world.
"""

import argparse
import os
import json
import hashlib
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm

# ---------------- I/O ----------------
def load_labels(json_path):
    with open(json_path, "r") as f:
        return np.array(json.load(f)["labels"], dtype=int)


def save_pcd_xyzrgb(path, pts, colors=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pts = np.asarray(pts, dtype=np.float32)
    pcd.points = o3d.utility.Vector3dVector(
        pts if pts.size else np.zeros((0, 3), dtype=np.float32)
    )
    if colors is not None:
        cols = np.asarray(colors, dtype=np.float32)
        if cols.ndim == 1:
            cols = np.tile(cols.reshape(1, 3), (pts.shape[0], 1))
        assert cols.shape == (pts.shape[0], 3)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0.0, 1.0))
    o3d.io.write_point_cloud(path, pcd)


def save_meta(path, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------- geometry utils ----------------
def fps_idx(pts, m, rng):
    N = pts.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=np.int64)
    if m >= N:
        return np.arange(N, dtype=np.int64)
    chosen = np.empty(m, dtype=np.int64)
    farthest = rng.randint(0, N)
    dist = np.full(N, np.inf, dtype=np.float64)
    for i in range(m):
        chosen[i] = farthest
        p = pts[farthest]
        d = np.sum((pts - p) ** 2, axis=1)
        dist = np.minimum(dist, d)
        farthest = int(np.argmax(dist))
    return chosen


def fix_num_points_with_index(pts, target=2048, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    pts = np.asarray(pts, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((target, 3), dtype=np.float32), -np.ones(
            (target,), dtype=np.int64
        )
    N = pts.shape[0]
    if N == target:
        return pts, np.arange(N, dtype=np.int64)
    if N > target:
        sel = fps_idx(pts, target, rng)
        return pts[sel], sel
    reps = target // N
    rem = target - reps * N
    base = np.repeat(np.arange(N, dtype=np.int64), reps)
    if rem > 0:
        extra = rng.choice(N, size=rem, replace=True).astype(np.int64)
        sel = np.concatenate([base, extra], axis=0)
    else:
        sel = base
    return pts[sel], sel


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def orthonormal_to(a):
    a = unit(a)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t = np.cross(a, ref)
    return unit(t)


# ---------------- neighbor & gingiva ----------------
def pick_neighbor_teeth(tooth_centers, target_label, k_neighbor=2):
    if target_label not in tooth_centers:
        return []
    c_t = tooth_centers[target_label]
    arr = []
    for l, c in tooth_centers.items():
        if l == target_label:
            continue
        arr.append((l, float(np.linalg.norm(c - c_t))))
    arr.sort(key=lambda x: x[1])
    return [l for l, _ in arr[:k_neighbor]]


def select_nearest_gingiva(all_pts, gingiva_idx, center, k):
    if gingiva_idx.size == 0 or k <= 0:
        return np.zeros((0,), dtype=np.int64)
    gpts = all_pts[gingiva_idx]
    d = np.linalg.norm(gpts - center[None, :], axis=1)
    order = np.argsort(d)
    sel = order[: min(len(order), k)]
    return gingiva_idx[sel]


# ---------------- orientation normalization ----------------
def build_alignment_matrix(c_t, a_axis, neighbor_centers):
    """
    Build R_align to map world -> normalized frame where:
      - a_axis (jaw) -> +Z
      - neighbor line direction (projected onto plane ⟂ a_axis) -> +X
        *If two neighbors exist*, use line n1−n2 (projected). If one neighbor, use n1−c_t (projected).
        If no neighbor, pick any axis ⟂ Z.
      - Y = Z × X (right-handed)
    Return: R_align (3x3), (x_hat, y_hat, z_hat).
    """
    z_hat = unit(a_axis)

    if len(neighbor_centers) >= 2:
        n1, n2 = neighbor_centers[0], neighbor_centers[1]
        v_line = n1 - n2
        v_line = v_line - np.dot(v_line, z_hat) * z_hat
        if np.linalg.norm(v_line) < 1e-9:
            v_line = orthonormal_to(z_hat)
        x_hat = unit(v_line)
    elif len(neighbor_centers) == 1:
        v = neighbor_centers[0] - c_t
        v = v - np.dot(v, z_hat) * z_hat
        x_hat = unit(v) if np.linalg.norm(v) >= 1e-9 else orthonormal_to(z_hat)
    else:
        x_hat = orthonormal_to(z_hat)

    y_hat = unit(np.cross(z_hat, x_hat))
    x_hat = unit(np.cross(y_hat, z_hat))  # re-orthogonalize

    R_world_to_align = np.vstack([x_hat, y_hat, z_hat])
    return R_world_to_align, x_hat, y_hat, z_hat


# ---------------- nine-grid partition in normalized frame ----------------
def nine_grid_cells(pts_world, c_t, R_align):
    """Return per-point cell_id in 1..9 using normalized coords q = R_align @ (x - c_t).
    Grid along qz (growth) and qx (neighbor axis), 3 equal bins each.
    """
    q = (pts_world - c_t) @ R_align.T  # (N,3)
    u = q[:, 2]  # growth (Z)
    v = q[:, 0]  # neighbor (X)
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    u_c1 = u_min + (u_max - u_min) / 3.0
    u_c2 = u_min + 2.0 * (u_max - u_min) / 3.0
    v_c1 = v_min + (v_max - v_min) / 3.0
    v_c2 = v_min + 2.0 * (v_max - v_min) / 3.0
    # row: 0=top(u high),1=mid,2=bottom(u low)
    row = np.zeros_like(u, dtype=np.int32)
    row[u < u_c1] = 2
    row[(u >= u_c1) & (u < u_c2)] = 1
    row[u >= u_c2] = 0
    # col: 0=left(v low),1=mid,2=right(v high)
    col = np.zeros_like(v, dtype=np.int32)
    col[v < v_c1] = 0
    col[(v >= v_c1) & (v < v_c2)] = 1
    col[v >= v_c2] = 2
    cell_ids = (row * 3 + col + 1).astype(np.int32)
    cuts = {
        "u": [float(u_c1), float(u_c2)],
        "v": [float(v_c1), float(v_c2)],
        "u_min": float(u_min),
        "u_max": float(u_max),
        "v_min": float(v_min),
        "v_max": float(v_max),
        "numbering": "1..3 top(u high,left->right),4..6 mid,7..9 bottom",
    }
    return cell_ids, cuts, q


# ---------------- stable pick ----------------
def stable_choice_idx(key, n):
    if n <= 1:
        return 0
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h, 16) % n


# ---------------- main per-case ----------------
def process_case(
    obj_path,
    json_path,
    out_root,
    split,
    case_id,
    # neighbor / gingiva
    k_neighbor_teeth=2,
    k_gingiva=2000,
    # quotas (gt vs partial)
    quota_gt_tooth=1152,
    quota_gt_neigh=384 * 2,
    quota_gt_gingiva=512,
    quota_pt_tooth=1152,
    quota_pt_neigh=384 * 2,
    quota_pt_gingiva=512,
    target_n=2048,
    # also export missing-only colored tooth
    export_missing=True,
    target_missing_n=2048,
    # axes
    tax_id=None,
    use_jaw_z_axis=True,
    upper_axis_sign=-1,
    lower_axis_sign=+1,
    # colors
    color_gt=True,
    color_partial=True,
    color_neighbor=(0.65, 0.65, 0.65),
    color_gingiva=(1.00, 0.70, 0.82),
    tooth_color_mode="tri-gradient",  # map (qx,qy,qz) → (R,G,B)
    # nine-grid variants
    nine_missing_variants=None,
    eval_pick_one=True,
    seed=42,
):
    rng = np.random.RandomState(seed)

    mesh = trimesh.load(obj_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    all_pts = mesh.vertices
    labels = load_labels(json_path)

    if labels.shape[0] != all_pts.shape[0]:
        n = min(labels.shape[0], all_pts.shape[0])
        labels = labels[:n]
        all_pts = all_pts[:n]

    tooth_labels = sorted({int(l) for l in labels if l > 0})
    gingiva_idx = np.where(labels <= 0)[0]

    # centers & raw tooth radii
    label_to_indices, tooth_centers, tooth_scales = {}, {}, {}
    for t in tooth_labels:
        idx = np.where(labels == t)[0]
        if idx.size == 0:
            continue
        pts = all_pts[idx]
        c = pts.mean(axis=0)
        r = float(np.max(np.linalg.norm(pts - c[None, :], axis=1)))
        r = 1.0 if r < 1e-12 else r
        label_to_indices[t] = idx
        tooth_centers[t] = c
        tooth_scales[t] = r

    # jaw center for lingual/buccal sign anchoring
    jaw_center = (
        np.mean(np.stack(list(tooth_centers.values()), axis=0), axis=0)
        if len(tooth_centers)
        else np.mean(all_pts, axis=0)
    )

    # variants default: 1..9 single cells
    if not nine_missing_variants:
        variants = [[i] for i in range(1, 10)]
        variant_strategy = "single_cells_all"
    else:
        variants = []
        for vs in nine_missing_variants:
            sset = sorted({int(v) for v in vs if 1 <= int(v) <= 9})
            if sset:
                variants.append(sset)
        variant_strategy = "provided_list"
    available_variants = [{"missing_cells": v} for v in variants]

    # per tooth: align, union-L∞ normalize, export
    for t in tooth_labels:
        idx_t = label_to_indices.get(t)
        if idx_t is None:
            continue
        pts_t = all_pts[idx_t]
        c_t = tooth_centers[t]
        s_t = tooth_scales[t]  # for reporting only

        # jaw axis
        if use_jaw_z_axis:
            if str(tax_id).lower() == "upper":
                a_axis = np.array([0.0, 0.0, float(upper_axis_sign)], dtype=np.float64)
            elif str(tax_id).lower() == "lower":
                a_axis = np.array([0.0, 0.0, float(lower_axis_sign)], dtype=np.float64)
            else:
                a_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            a_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # neighbors
        neigh_lbls = pick_neighbor_teeth(tooth_centers, t, k_neighbor=k_neighbor_teeth)
        neigh_lbls = sorted(neigh_lbls)
        neigh_idx_list = [label_to_indices[nl] for nl in neigh_lbls if nl in label_to_indices]
        neigh_idx = (
            np.concatenate(neigh_idx_list, axis=0)
            if len(neigh_idx_list) > 0
            else np.zeros((0,), dtype=np.int64)
        )
        neigh_pts = all_pts[neigh_idx]
        neigh_centers = [tooth_centers[nl] for nl in neigh_lbls if nl in tooth_centers]

        # alignment
        R_align, x_hat, y_hat, z_hat = build_alignment_matrix(c_t, a_axis, neigh_centers)
        # +Y should be lingual; flip (X,Y) if +Y points buccal (outward of arch)
        r_out = c_t - jaw_center
        r_out = r_out - np.dot(r_out, z_hat) * z_hat
        if np.linalg.norm(r_out) >= 1e-9 and np.dot(y_hat, r_out) > 0:
            y_hat = -y_hat
            x_hat = -x_hat
            R_align = np.vstack([x_hat, y_hat, z_hat])

        # align coords (before scaling)
        q_tooth = (pts_t - c_t) @ R_align.T
        g_sel = select_nearest_gingiva(all_pts, gingiva_idx, c_t, k_gingiva)
        g_pts = all_pts[g_sel] if g_sel.size > 0 else np.zeros((0, 3), dtype=np.float32)
        q_neigh = (neigh_pts - c_t) @ R_align.T
        q_gingi = (g_pts   - c_t) @ R_align.T

        # union L∞ scale (box fit) – ensures per-axis in [-1,1]
        if q_tooth.size or q_neigh.size or q_gingi.size:
            Q_all = np.concatenate([arr for arr in (q_tooth, q_neigh, q_gingi) if arr.size], axis=0)
            aabs = np.max(np.abs(Q_all), axis=0)
            s_linf = float(np.max(aabs))
        else:
            s_linf = 1.0
        # union L2 (for reporting only)
        r_tooth = float(np.max(np.linalg.norm(q_tooth, axis=1))) if q_tooth.size else 1.0
        r_neigh = float(np.max(np.linalg.norm(q_neigh, axis=1))) if q_neigh.size else 0.0
        r_gingi = float(np.max(np.linalg.norm(q_gingi, axis=1))) if q_gingi.size else 0.0
        s_l2 = max(r_tooth, r_neigh, r_gingi, 1e-12)

        s_final = s_linf  # single mode

        # normalize
        tooth_norm_full = q_tooth / s_final
        neigh_norm_full = q_neigh / s_final
        gingi_norm_full = q_gingi / s_final

        # --------- export COMPLETE (target tooth only, s_final) ---------
        cd = os.path.join(out_root, split, "complete", str(t))
        tooth_complete_2048, _ = fix_num_points_with_index(tooth_norm_full, target=target_n, rng=rng)
        save_pcd_xyzrgb(os.path.join(cd, f"{case_id}.pcd"), tooth_complete_2048, colors=None)
        meta_c = {
            "type": "complete_single_tooth_sfinal",
            "tax_id": tax_id,
            "case_id": case_id,
            "tooth_id": int(t),
            "transform": {
                "center_global_xyz": c_t.tolist(),
                "scale_used": float(s_final),
                "scale_mode": "linf",
                "scale_components": {
                    "tooth_radius_l2": float(r_tooth),
                    "union_radius_l2": float(s_l2),
                    "union_radius_linf": float(s_linf),
                },
                "R_align_rows_xyz": np.vstack([x_hat, y_hat, z_hat]).tolist(),
                "jaw_center_global_xyz": jaw_center.tolist(),
                "y_positive_is_lingual": True,
                "buccal_outward_reference": "proj((c_t - jaw_center), plane ⟂ Z)",
                "restore_formula": "X_global = R_align.T @ (X_norm * scale_used) + center",
            },
            "raw_paths": {"obj": obj_path, "label_json": json_path},
            "target_points": int(target_n),
        }
        save_meta(os.path.join(cd, f"{case_id}.meta.json"), meta_c)

        # ---------- pre-color FULL tooth (before cutting) ----------
        def pos2color_full(q):
            if q.shape[0] == 0:
                return np.zeros((0, 3), dtype=np.float32)
            tmin = np.min(tooth_norm_full, axis=0)
            tmax = np.max(tooth_norm_full, axis=0)
            span = np.maximum(tmax - tmin, 1e-6)
            u = (q - tmin) / span
            return np.clip(u, 0.0, 1.0).astype(np.float32)

        tooth_colors_full = pos2color_full(tooth_norm_full)

        # ---------- GT (XYZRGB): full tooth (colored) + neighbors + gingiva ----------
        gt_t_sub, sel_t = fix_num_points_with_index(tooth_norm_full, target=quota_gt_tooth, rng=rng)
        cols_t = tooth_colors_full[sel_t] if color_gt else None
        gt_n_sub, _ = fix_num_points_with_index(neigh_norm_full, target=quota_gt_neigh, rng=rng)
        gt_g_sub, _ = fix_num_points_with_index(gingi_norm_full, target=quota_gt_gingiva, rng=rng)
        if color_gt:
            cols_n = np.tile(np.array(color_neighbor, dtype=np.float32).reshape(1, 3), (gt_n_sub.shape[0], 1))
            cols_g = np.tile(np.array(color_gingiva, dtype=np.float32).reshape(1, 3), (gt_g_sub.shape[0], 1))
            gt_cols_full = np.concatenate([cols_t, cols_n, cols_g], axis=0)
        else:
            gt_cols_full = None
        gt_concat = np.concatenate([gt_t_sub, gt_n_sub, gt_g_sub], axis=0)
        gt_2048, sel_gt = fix_num_points_with_index(gt_concat, target=target_n, rng=rng)
        gt_cols = gt_cols_full[sel_gt] if color_gt else None
        gd = os.path.join(out_root, split, "gt", str(t))
        save_pcd_xyzrgb(os.path.join(gd, f"{case_id}.pcd"), gt_2048, colors=gt_cols)
        meta_gt = {
            "type": "gt_xyzrgb_precolored",
            "tax_id": tax_id,
            "case_id": case_id,
            "tooth_id": int(t),
            "neighbors": [int(x) for x in neigh_lbls],
            "k_gingiva": int(k_gingiva),
            "quotas": {
                "tooth": int(quota_gt_tooth),
                "neighbor": int(quota_gt_neigh),
                "gingiva": int(quota_gt_gingiva),
                "final_target": int(target_n),
            },
            "transform": {
                "center_global_xyz": c_t.tolist(),
                "scale_used": float(s_final),
                "scale_mode": "linf",
                "scale_components": {
                    "tooth_radius_l2": float(r_tooth),
                    "union_radius_l2": float(s_l2),
                    "union_radius_linf": float(s_linf),
                },
                "R_align_rows_xyz": np.vstack([x_hat, y_hat, z_hat]).tolist(),
                "jaw_center_global_xyz": jaw_center.tolist(),
                "y_positive_is_lingual": True,
                "buccal_outward_reference": "proj((c_t - jaw_center), plane ⟂ Z)",
                "restore_formula": "X_global = R_align.T @ (X_norm * scale_used) + center",
            },
            "coloring": (
                {
                    "stage": "precolor_before_cut",
                    "mode": tooth_color_mode,
                    "tooth_norm_min": np.min(tooth_norm_full, axis=0).tolist(),
                    "tooth_norm_max": np.max(tooth_norm_full, axis=0).tolist(),
                    "neighbor_rgb": tuple(map(float, color_neighbor)),
                    "gingiva_rgb": tuple(map(float, color_gingiva)),
                }
                if color_gt
                else None
            ),
            "note": "Grid is defined in normalized frame (qx along neighbors, qz along growth)",
        }
        save_meta(os.path.join(gd, f"{case_id}.meta.json"), meta_gt)

        # ---------- PARTIAL & GT_MISSING ----------
        cell_ids, cuts, _ = nine_grid_cells(pts_t, c_t, R_align)

        # choose variants
        if eval_pick_one and str(split).lower() in ("val", "validation", "test"):
            key = f"{tax_id}|{case_id}|{t}"
            idx_var = stable_choice_idx(key, len(variants))
            chosen_variants = [variants[idx_var]]
            chosen_indices = [idx_var]
            strat_eff = (
                "provided_list" if variant_strategy == "provided_list" else "single_cells_all"
            ) + "+eval_pick_one(hash)"
        else:
            chosen_variants = variants
            chosen_indices = list(range(len(variants)))
            strat_eff = variant_strategy

        for local_i, miss_cells in enumerate(chosen_variants):
            vi = local_i
            miss_mask = np.isin(cell_ids, np.array(miss_cells, dtype=np.int32))
            keep_mask = ~miss_mask

            # kept tooth and missing tooth (pre-colored already)
            tooth_keep_norm = tooth_norm_full[keep_mask]
            tooth_keep_cols = tooth_colors_full[keep_mask] if color_partial else None
            tooth_miss_norm = tooth_norm_full[miss_mask]
            tooth_miss_cols = tooth_colors_full[miss_mask] if color_partial else None

            # neighbors & gingiva unchanged
            part_neigh_norm = neigh_norm_full
            part_gingi_norm = gingi_norm_full

            # sample quotas
            t_sub, sel_ts = fix_num_points_with_index(tooth_keep_norm, target=quota_pt_tooth, rng=rng)
            n_sub, _ = fix_num_points_with_index(part_neigh_norm, target=quota_pt_neigh, rng=rng)
            g_sub, _ = fix_num_points_with_index(part_gingi_norm, target=quota_pt_gingiva, rng=rng)

            if color_partial:
                cols_t = tooth_keep_cols[sel_ts]
                cols_n = np.tile(np.array(color_neighbor, dtype=np.float32).reshape(1, 3), (n_sub.shape[0], 1))
                cols_g = np.tile(np.array(color_gingiva, dtype=np.float32).reshape(1, 3), (g_sub.shape[0], 1))
                cols_full = np.concatenate([cols_t, cols_n, cols_g], axis=0)
            else:
                cols_full = None

            partial = np.concatenate([t_sub, n_sub, g_sub], axis=0)
            p_2048, sel = fix_num_points_with_index(partial, target=target_n, rng=rng)
            cols = cols_full[sel] if color_partial else None

            pd = os.path.join(out_root, split, "partial", str(t), case_id)
            save_pcd_xyzrgb(os.path.join(pd, f"{vi:02d}.pcd"), p_2048, colors=cols)

            meta_p = {
                "type": "partial_xyzrgb_nine_missing_normed_precolored",
                "tax_id": tax_id,
                "case_id": case_id,
                "tooth_id": int(t),
                "variant_id": int(vi),
                "variant_strategy": strat_eff,
                "available_variants": available_variants,
                "chosen_variant_index_in_available": int(chosen_indices[local_i]),
                "missing_cells": [int(x) for x in miss_cells],
                "nine_grid": {
                    "cuts": cuts,
                    "axes_note": "+Y=lingual, −Y=buccal; +X neighbor; +Z growth",
                },
                "neighbors": [int(x) for x in neigh_lbls],
                "k_gingiva": int(k_gingiva),
                "quotas": {
                    "tooth_keep": int(quota_pt_tooth),
                    "neighbor_total": int(quota_pt_neigh),
                    "gingiva": int(quota_pt_gingiva),
                    "final_target": int(target_n),
                },
                "transform": {
                    "center_global_xyz": c_t.tolist(),
                    "scale_used": float(s_final),
                    "scale_mode": "linf",
                    "scale_components": {
                        "tooth_radius_l2": float(r_tooth),
                        "union_radius_l2": float(s_l2),
                        "union_radius_linf": float(s_linf),
                    },
                    "R_align_rows_xyz": np.vstack([x_hat, y_hat, z_hat]).tolist(),
                    "jaw_center_global_xyz": jaw_center.tolist(),
                    "y_positive_is_lingual": True,
                    "buccal_outward_reference": "proj((c_t - jaw_center), plane ⟂ Z)",
                    "restore_formula": "X_global = R_align.T @ (X_norm * scale_used) + center",
                },
                "coloring": (
                    {
                        "stage": "precolor_before_cut",
                        "mode": tooth_color_mode,
                        "tooth_norm_min": np.min(tooth_norm_full, axis=0).tolist(),
                        "tooth_norm_max": np.max(tooth_norm_full, axis=0).tolist(),
                        "neighbor_rgb": tuple(map(float, color_neighbor)),
                        "gingiva_rgb": tuple(map(float, color_gingiva)),
                    }
                    if color_partial
                    else None
                ),
                "raw_paths": {"obj": obj_path, "label_json": json_path},
            }
            save_meta(os.path.join(pd, f"{vi:02d}.meta.json"), meta_p)

            # ----- export GT_MISSING (colored tooth-only) -----
            # Deployment prototypes are exported later into a separate root;
            # never conflate them with this true missing-region target.
            if export_missing:
                miss_dir = os.path.join(out_root, split, "gt_missing", str(t), case_id)
                miss_pts, miss_sel = fix_num_points_with_index(tooth_miss_norm, target=target_missing_n, rng=rng)
                miss_cols = tooth_miss_cols[miss_sel] if color_partial else None
                save_pcd_xyzrgb(os.path.join(miss_dir, f"{vi:02d}.pcd"), miss_pts, colors=miss_cols)

                unique_cells, counts = np.unique(cell_ids[miss_mask], return_counts=True)
                miss_hist = {int(k): int(v) for k, v in zip(unique_cells.tolist(), counts.tolist())}
                meta_miss = {
                    "type": "missing_region_xyzrgb",
                    "tax_id": tax_id,
                    "case_id": case_id,
                    "tooth_id": int(t),
                    "variant_id": int(vi),
                    "missing_cells": [int(x) for x in miss_cells],
                    "missing_hist": miss_hist,
                    "nine_grid": {
                        "cuts": cuts,
                        "axes_note": "+Y=lingual, −Y=buccal; +X neighbor; +Z growth",
                    },
                    "target_points": int(target_missing_n),
                    "transform": {
                        "center_global_xyz": c_t.tolist(),
                        "scale_used": float(s_final),
                        "scale_mode": "linf",
                        "scale_components": {
                            "tooth_radius_l2": float(r_tooth),
                            "union_radius_l2": float(s_l2),
                            "union_radius_linf": float(s_linf),
                        },
                        "R_align_rows_xyz": np.vstack([x_hat, y_hat, z_hat]).tolist(),
                        "jaw_center_global_xyz": jaw_center.tolist(),
                        "y_positive_is_lingual": True,
                        "buccal_outward_reference": "proj((c_t - jaw_center), plane ⟂ Z)",
                        "restore_formula": "X_global = R_align.T @ (X_norm * scale_used) + center",
                    },
                    "coloring": (
                        {
                            "stage": "precolor_before_cut",
                            "mode": tooth_color_mode,
                            "tooth_norm_min": np.min(tooth_norm_full, axis=0).tolist(),
                            "tooth_norm_max": np.max(tooth_norm_full, axis=0).tolist(),
                        }
                        if color_partial
                        else None
                    ),
                }
                save_meta(os.path.join(miss_dir, f"{vi:02d}.meta.json"), meta_miss)

    print(
        f"{case_id} | {split}: done (union L∞ normalization, pre-colored tooth, partial & missing; 2048 pts)"
    )


# ---------------- batch ----------------
def batch_process(
    json_split_path,
    data_root,
    out_root,
    k_neighbor_teeth=2,
    k_gingiva=2000,
    quota_gt_tooth=1152,
    quota_gt_neigh=384 * 2,
    quota_gt_gingiva=512,
    quota_pt_tooth=1152,
    quota_pt_neigh=384 * 2,
    quota_pt_gingiva=512,
    target_n=2048,
    export_missing=True,
    target_missing_n=2048,
    use_jaw_z_axis=True,
    upper_axis_sign=-1,
    lower_axis_sign=+1,
    color_gt=True,
    color_partial=True,
    color_neighbor=(0.65, 0.65, 0.65),
    color_gingiva=(1.00, 0.70, 0.82),
    tooth_color_mode="tri-gradient",
    nine_missing_variants=None,
    eval_pick_one=True,
    seed=42,
):
    with open(json_split_path, "r") as f:
        splits = json.load(f)
    for cat in splits:
        tax_id = cat["taxonomy_id"]
        for split in ["train", "val", "test"]:
            cases = cat.get(split, [])
            for case in tqdm(cases, desc=f"{tax_id}-{split}"):
                case_id = case.replace(f"_{tax_id}", "")
                obj_path = os.path.join(
                    data_root, tax_id, case_id, f"{case_id}_{tax_id}.obj"
                )
                json_path = os.path.join(
                    data_root, tax_id, case_id, f"{case_id}_{tax_id}.json"
                )
                if not (os.path.exists(obj_path) and os.path.exists(json_path)):
                    print("[Missing]", obj_path, "|", json_path)
                    continue
                try:
                    process_case(
                        obj_path=obj_path,
                        json_path=json_path,
                        out_root=out_root,
                        split=split,
                        case_id=case_id,
                        k_neighbor_teeth=k_neighbor_teeth,
                        k_gingiva=k_gingiva,
                        quota_gt_tooth=quota_gt_tooth,
                        quota_gt_neigh=quota_gt_neigh,
                        quota_gt_gingiva=quota_gt_gingiva,
                        quota_pt_tooth=quota_pt_tooth,
                        quota_pt_neigh=quota_pt_neigh,
                        quota_pt_gingiva=quota_pt_gingiva,
                        target_n=target_n,
                        export_missing=export_missing,
                        target_missing_n=target_missing_n,
                        tax_id=tax_id,
                        use_jaw_z_axis=use_jaw_z_axis,
                        upper_axis_sign=upper_axis_sign,
                        lower_axis_sign=lower_axis_sign,
                        color_gt=color_gt,
                        color_partial=color_partial,
                        color_neighbor=color_neighbor,
                        color_gingiva=color_gingiva,
                        tooth_color_mode=tooth_color_mode,
                        nine_missing_variants=nine_missing_variants,
                        eval_pick_one=eval_pick_one,
                        seed=seed,
                    )
                except Exception as e:
                    print(f"[Error] {case} ({tax_id}-{split}): {e}")
                    continue


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", required=True, help="JSON list with upper/lower train and test case IDs")
    parser.add_argument("--raw-root", required=True, help="Root containing <jaw>/<case>/<case>_<jaw>.obj/.json")
    parser.add_argument("--output-root", required=True, help="Empty output root for the derived benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-variants", type=int, default=8, help="Number of listed masks used for train cases")
    parser.add_argument("--target-points", type=int, default=2048)
    parser.add_argument("--missing-points", type=int, default=2048)
    parser.add_argument("--k-gingiva", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if os.path.exists(args.output_root):
        raise FileExistsError(f"Refusing to overwrite output root: {args.output_root}")
    if args.train_variants < 1:
        raise ValueError("--train-variants must be positive")

    # These eight masks reproduce the derived T3DS protocol: four individual
    # cells and four contiguous three-cell strips. Test cases receive one
    # deterministic hash-selected mask to avoid leaking a test case through
    # multiple correlated variants.
    candidate_masks = [[1, 4, 7], [1, 2, 3], [3, 6, 9], [1], [3], [4], [6], [2]]
    if args.train_variants > len(candidate_masks):
        raise ValueError(f"At most {len(candidate_masks)} public protocol masks are available")
    batch_process(
        json_split_path=args.split_json,
        data_root=args.raw_root,
        out_root=args.output_root,
        k_neighbor_teeth=2,
        k_gingiva=args.k_gingiva,
        quota_gt_tooth=1152,
        quota_gt_neigh=384 * 2,
        quota_gt_gingiva=512,
        quota_pt_tooth=1152,
        quota_pt_neigh=384 * 2,
        quota_pt_gingiva=512,
        target_n=args.target_points,
        export_missing=True,
        target_missing_n=args.missing_points,
        use_jaw_z_axis=True,
        upper_axis_sign=+1,
        lower_axis_sign=+1,
        color_gt=True,
        color_partial=True,
        color_neighbor=(0.65, 0.65, 0.65),
        color_gingiva=(1.00, 0.70, 0.82),
        tooth_color_mode="tri-gradient",
        nine_missing_variants=candidate_masks[:args.train_variants],
        eval_pick_one=True,
        seed=args.seed,
    )
