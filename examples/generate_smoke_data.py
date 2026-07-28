"""Generate the tiny, non-clinical fixture used by the repository smoke test.

The points are sampled from analytic ellipsoids.  They are not derived from
Teeth3DS, a patient scan, or any other external dataset.  The output mirrors
the Teeth3DS directory contract closely enough to exercise the real loader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


FIXTURE_VERSION = 1
SPLIT_IDS = {
    "train": ("synthetic_train_00", "synthetic_train_01"),
    "test": ("synthetic_test_00",),
}
VIEW_SUFFIXES = ("-y", "y", "z")


def _sample_ellipsoid_surface(
    rng: np.random.Generator,
    count: int,
    center: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    directions = rng.normal(size=(count, 3)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-8)
    # A mild crown-like taper makes the fixture less spherical while remaining analytic.
    points = directions * radii[None, :]
    points[:, 0] *= 1.0 - 0.12 * np.clip(points[:, 2] / radii[2], 0.0, 1.0)
    return points + center[None, :]


def _resample(rng: np.random.Generator, points: np.ndarray, count: int) -> np.ndarray:
    indices = rng.choice(len(points), size=count, replace=len(points) < count)
    return points[indices].astype(np.float32, copy=True)


def _farthest_point_sample(points: np.ndarray, count: int) -> np.ndarray:
    selected = np.empty(count, dtype=np.int64)
    distances = np.full(len(points), np.inf, dtype=np.float32)
    current = int(np.argmax(np.sum(points * points, axis=1)))
    for index in range(count):
        selected[index] = current
        distance = np.sum((points - points[current]) ** 2, axis=1)
        distances = np.minimum(distances, distance)
        current = int(np.argmax(distances))
    return points[selected].astype(np.float32, copy=True)


def _make_case(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    full_parts = []
    missing_parts = []
    x_centers = np.linspace(-0.62, 0.62, 5, dtype=np.float32)
    for tooth_index, x_center in enumerate(x_centers):
        center = np.array(
            [x_center, 0.12 + 0.24 * x_center * x_center, 0.015 * np.cos(tooth_index)],
            dtype=np.float32,
        )
        center += rng.normal(scale=0.008, size=3).astype(np.float32)
        radii = np.array([0.19, 0.25, 0.27], dtype=np.float32)
        radii *= rng.uniform(0.94, 1.06, size=3).astype(np.float32)
        local = _sample_ellipsoid_surface(rng, 4096, center, radii)
        full_parts.append(local)
        if tooth_index == 2:
            # A contiguous cap on the central crown is the synthetic missing support.
            missing_parts.append(local[(local[:, 2] - center[2] > 0.10) & (local[:, 1] < center[1] + 0.15)])

    full_pool = np.concatenate(full_parts, axis=0)
    missing_pool = np.concatenate(missing_parts, axis=0)
    missing_center = missing_pool.mean(axis=0)
    missing_radius = np.linalg.norm(missing_pool - missing_center, axis=1).max()
    keep = np.linalg.norm(full_pool - missing_center, axis=1) > missing_radius * 0.92
    partial_pool = full_pool[keep]

    # The repository contract uses normalized local coordinates.
    scale = float(np.abs(full_pool).max()) / 0.88
    full_pool /= scale
    partial_pool /= scale
    missing_pool /= scale

    complete = _resample(rng, full_pool, 2048)
    partial = _resample(rng, partial_pool, 2048)
    gt_missing = _resample(rng, missing_pool, 256)
    prototype = _farthest_point_sample(gt_missing, 30)
    return partial, complete, gt_missing, prototype


def _write_ascii_pcd(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# .PCD v0.7 - synthetic VGP smoke-test fixture\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\n"
        "DATA ascii\n"
    )
    rows = "\n".join(f"{x:.7f} {y:.7f} {z:.7f} 0" for x, y, z in points)
    path.write_text(header + rows + "\n", encoding="ascii")


def _render_scc(points: np.ndarray, suffix: str, size: int = 96) -> Image.Image:
    if suffix == "-y":
        horizontal, vertical, depth = points[:, 0], points[:, 2], -points[:, 1]
    elif suffix == "y":
        horizontal, vertical, depth = -points[:, 0], points[:, 2], points[:, 1]
    elif suffix == "z":
        horizontal, vertical, depth = points[:, 0], points[:, 1], points[:, 2]
    else:
        raise ValueError(f"Unsupported view suffix: {suffix}")

    pixels_x = np.clip(((horizontal + 1.0) * 0.5 * (size - 9) + 4).round(), 0, size - 1).astype(int)
    pixels_y = np.clip(((1.0 - (vertical + 1.0) * 0.5) * (size - 9) + 4).round(), 0, size - 1).astype(int)
    colors = np.clip((points + 1.0) * 127.5, 0, 255).astype(np.uint8)
    canvas = Image.new("RGB", (size, size), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    for index in np.argsort(depth):
        x, y = int(pixels_x[index]), int(pixels_y[index])
        color = tuple(int(channel) for channel in colors[index])
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    return canvas


def expected_files(root: Path) -> list[Path]:
    files = [root / "Tooth_smoke.json", root / "MANIFEST.json", root / "README.md"]
    for split, model_ids in SPLIT_IDS.items():
        for model_id in model_ids:
            files.extend(
                [
                    root / "points" / split / "gt" / "11" / f"{model_id}.pcd",
                    root / "points" / split / "partial" / "11" / model_id / "00.pcd",
                    root / "points" / split / "gt_missing" / "11" / model_id / "00.pcd",
                    root / "points" / split / "missing" / "11" / model_id / "00.pcd",
                ]
            )
            files.extend(
                root / "scc" / split / "partial" / "11" / model_id / f"00_{suffix}.png"
                for suffix in VIEW_SUFFIXES
            )
    return files


def ensure_smoke_data(root: Path, force: bool = False) -> Path:
    root = root.resolve()
    if not force and all(path.is_file() for path in expected_files(root)):
        return root

    for split_index, (split, model_ids) in enumerate(SPLIT_IDS.items()):
        for case_index, model_id in enumerate(model_ids):
            partial, complete, gt_missing, prototype = _make_case(2027 + split_index * 100 + case_index)
            _write_ascii_pcd(root / "points" / split / "gt" / "11" / f"{model_id}.pcd", complete)
            _write_ascii_pcd(root / "points" / split / "partial" / "11" / model_id / "00.pcd", partial)
            _write_ascii_pcd(root / "points" / split / "gt_missing" / "11" / model_id / "00.pcd", gt_missing)
            _write_ascii_pcd(root / "points" / split / "missing" / "11" / model_id / "00.pcd", prototype)
            image_dir = root / "scc" / split / "partial" / "11" / model_id
            image_dir.mkdir(parents=True, exist_ok=True)
            for suffix in VIEW_SUFFIXES:
                _render_scc(partial, suffix).save(image_dir / f"00_{suffix}.png", optimize=True)

    category = [
        {
            "taxonomy_id": "11",
            "taxonomy_name": "Synthetic tooth-like patch",
            "train": list(SPLIT_IDS["train"]),
            "test": list(SPLIT_IDS["test"]),
        }
    ]
    root.mkdir(parents=True, exist_ok=True)
    (root / "Tooth_smoke.json").write_text(json.dumps(category, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "source": "Procedurally generated analytic ellipsoid surfaces",
        "external_or_patient_data": False,
        "train_cases": len(SPLIT_IDS["train"]),
        "test_cases": len(SPLIT_IDS["test"]),
        "variants_per_case": 1,
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# Synthetic smoke-test fixture\n\n"
        "These tooth-like point sets and SCC-style views are generated from analytic "
        "ellipsoid surfaces by `examples/generate_smoke_data.py`. They contain no "
        "Teeth3DS or patient-derived content and are intended only to verify software "
        "execution, tensor shapes, and data interfaces. They are not benchmark examples "
        "and must not be used to infer model accuracy.\n",
        encoding="utf-8",
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "synthetic_t3ds_smoke",
        help="Output fixture directory (default: examples/synthetic_t3ds_smoke).",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate the known fixture files.")
    args = parser.parse_args()
    output = ensure_smoke_data(args.output, force=args.force)
    print(f"Synthetic smoke-test fixture ready: {output}")


if __name__ == "__main__":
    main()
