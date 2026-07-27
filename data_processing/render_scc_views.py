#!/usr/bin/env python3
"""Render coordinate-colored SCC views for a derived point-cloud cohort.

This follows the repository's established ``-y``, ``y``, and ``z`` camera
convention. Each normalized point coordinate is encoded as vertex RGB:
``(x, y, z) in [-1, 1]^3 -> ((x + 1) / 2, (y + 1) / 2, (z + 1) / 2)``.
It reads only derived point clouds; source files are never accessed here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d


VIEWS = {
    "-y": (np.array([0.0, -1.0, 0.0]), 90.0),
    "y": (np.array([0.0, 1.0, 0.0]), -90.0),
    "z": (np.array([0.0, 0.0, 1.0]), 0.0),
}


def xyz_to_rgb(points: np.ndarray) -> np.ndarray:
    """Encode the shared normalized XYZ frame as RGB, with clipping for safety."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point coordinates with shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains non-finite coordinates")
    return np.clip((points + 1.0) * 0.5, 0.0, 1.0)


def choose_up(direction: np.ndarray) -> np.ndarray:
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    up = np.array([0.0, 1.0, 0.0])
    return np.array([1.0, 0.0, 0.0]) if abs(float(direction @ up)) > 0.95 else up


def rotate(vector: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    radians = math.radians(degrees)
    return vector * math.cos(radians) + np.cross(axis, vector) * math.sin(radians) + axis * (axis @ vector) * (1.0 - math.cos(radians))


def render_pcd(renderer: o3d.visualization.rendering.OffscreenRenderer, source: Path, destination: Path, width: int, height: int) -> None:
    point_cloud = o3d.io.read_point_cloud(str(source))
    if point_cloud.is_empty():
        raise ValueError(f"empty derived point cloud: {source}")
    bounds = point_cloud.get_axis_aligned_bounding_box()
    center = np.asarray(bounds.get_center(), dtype=float)
    extent = np.asarray(bounds.get_max_bound()) - np.asarray(bounds.get_min_bound())
    diagonal = float(np.linalg.norm(extent))
    distance = (0.5 * diagonal / math.tan(math.radians(25.0))) * 1.25 + 1e-6
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 8.0
    # The renderer must preserve geometry position rather than use a uniform
    # gray color: RGB is the sample's normalized XYZ coordinate encoding.
    point_cloud.colors = o3d.utility.Vector3dVector(xyz_to_rgb(np.asarray(point_cloud.points)))
    geometry_name = "derived_clinical_partial"
    renderer.scene.add_geometry(geometry_name, point_cloud, material)
    metadata: dict[str, object] = {
        "view_convention": "repository_scc_v1",
        "image_size": [width, height],
        "fov_y_deg": 50.0,
        "color_encoding": {
            "name": "normalized_xyz_to_rgb_v1",
            "input_range": "normalized XYZ, expected [-1, 1] per axis",
            "formula": "rgb = clip((xyz + 1.0) / 2.0, 0.0, 1.0)",
            "channels": {"R": "x", "G": "y", "B": "z"},
        },
    }
    try:
        for suffix, (direction, roll) in VIEWS.items():
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            up = rotate(choose_up(direction), direction, roll)
            eye = center + direction * distance
            renderer.scene.camera.set_projection(
                50.0, width / float(height), max(1e-3, distance * 0.05), distance + diagonal * 3.0,
                o3d.visualization.rendering.Camera.FovType.Vertical,
            )
            renderer.scene.camera.look_at(center, eye, up)
            image = renderer.render_to_image()
            output = destination / f"{source.stem}_{suffix}.png"
            if not o3d.io.write_image(str(output), image, quality=9):
                raise RuntimeError(f"failed to write SCC rendering: {output}")
            metadata[suffix] = {"roll_deg": roll}
    finally:
        renderer.scene.remove_geometry(geometry_name)
    (destination / f"{source.stem}_cameras.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    args = parser.parse_args()
    if args.image_root.exists():
        raise FileExistsError(f"refusing to overwrite image root: {args.image_root}")
    sources = sorted(args.pcd_root.glob("*/partial/*/*/*.pcd"))
    if not sources:
        raise FileNotFoundError("no derived partial PCD files found")
    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    renderer.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    try:
        for index, source in enumerate(sources, 1):
            relative = source.relative_to(args.pcd_root)
            target = args.image_root / relative.parent
            target.mkdir(parents=True, exist_ok=True)
            render_pcd(renderer, source, target, args.width, args.height)
            if index % 25 == 0 or index == len(sources):
                print(f"rendered={index}/{len(sources)}", flush=True)
    finally:
        try:
            renderer.scene.clear_geometry()
        except Exception:
            pass
    manifest = {
        "images": len(sources) * len(VIEWS),
        "point_clouds": len(sources),
        "view_suffixes": list(VIEWS),
        "source": "derived anonymous partial point clouds only",
        "color_encoding": "normalized_xyz_to_rgb_v1: rgb=clip((xyz+1)/2,0,1), channels=(x,y,z)",
    }
    (args.image_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
