"""Dataset for MissingGT prototype predictors and optional DINO token caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

VIEW_SUFFIXES = ("-y", "y", "z")


def collate_distill_batch(samples: list[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | list[str]]:
    """Avoid PyTorch shared-storage resizing for Open3D-backed worker tensors."""
    tensor_keys = (
        "partial",
        "views",
        "teacher",
        "gt_missing",
        "gt_prototype",
        "reference_prototype",
        "reference_view_tokens",
    )
    batch: Dict[str, torch.Tensor | list[str]] = {
        key: torch.stack([sample[key] for sample in samples], dim=0).contiguous()  # type: ignore[arg-type]
        for key in tensor_keys
    }
    batch["sample_id"] = [str(sample["sample_id"]) for sample in samples]
    return batch


def farthest_point_sample_numpy(points: np.ndarray, n_points: int) -> np.ndarray:
    """Deterministic FPS for the GT set; the start point is the farthest origin."""
    if len(points) <= n_points:
        repeats = int(np.ceil(n_points / max(len(points), 1)))
        return np.tile(points, (repeats, 1))[:n_points]

    selected = np.empty(n_points, dtype=np.int64)
    distances = np.full(len(points), np.inf, dtype=np.float32)
    current = int(np.argmax(np.sum(points * points, axis=1)))
    for idx in range(n_points):
        selected[idx] = current
        distance = np.sum((points - points[current]) ** 2, axis=1)
        distances = np.minimum(distances, distance)
        current = int(np.argmax(distances))
    return points[selected]


def read_pcd_xyz(path: Path) -> np.ndarray:
    """Read XYZ from the repository's binary PCD files without Open3D.

    All benchmark PCDs store four float32 fields (x, y, z, rgb). Keeping this
    tiny reader makes the DINO experiment self-contained in its own runtime.
    """
    with path.open("rb") as handle:
        header = bytearray()
        while b"DATA " not in header or not header.endswith(b"\n"):
            chunk = handle.readline()
            if not chunk:
                raise ValueError(f"Malformed PCD header: {path}")
            header.extend(chunk)
        header_text = header.decode("ascii")
        fields_line = next(line for line in header_text.splitlines() if line.startswith("FIELDS "))
        fields = fields_line.split()[1:]
        if fields[:3] != ["x", "y", "z"]:
            raise ValueError(f"Expected xyz-leading PCD fields, got {fields}: {path}")
        points_line = next(line for line in header_text.splitlines() if line.startswith("POINTS "))
        n_points = int(points_line.split()[1])
        data_line = next(line for line in header_text.splitlines() if line.startswith("DATA "))
        if data_line == "DATA ascii":
            values = np.loadtxt(handle, dtype=np.float32).reshape(n_points, len(fields))
        elif data_line == "DATA binary":
            values = np.fromfile(handle, dtype=np.float32, count=n_points * len(fields))
            if values.size != n_points * len(fields):
                raise ValueError(f"Unexpected binary PCD length: {path}")
            values = values.reshape(n_points, len(fields))
        else:
            raise ValueError(f"Unsupported PCD encoding '{data_line}': {path}")
    return values[:, :3].copy()


class LiteVCPDistillDataset(Dataset):
    """Load partial clouds, SCC views, and dense true missing-region targets."""

    def __init__(
        self,
        dataset_root: str | Path,
        image_root: str | Path,
        category_file: str | Path,
        split: str,
        image_size: int = 160,
        num_prototype_points: int = 30,
        load_views: bool = True,
        load_teacher: bool = True,
        taxonomy_id: str = "11",
        image_taxonomy_id: str | None = None,
        reference_root: str | Path | None = None,
        load_gt_prototype: bool = True,
        train_variants: int = 8,
        test_variants: int = 1,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.dataset_root = Path(dataset_root)
        self.image_root = Path(image_root)
        self.split = split
        self.image_size = image_size
        self.num_prototype_points = num_prototype_points
        self.load_views = bool(load_views)
        self.load_teacher = bool(load_teacher)
        self.taxonomy_id = str(taxonomy_id)
        # This is normally identical to the point-cloud taxonomy. Keeping it
        # separately configurable also supports datasets whose renderings are
        # indexed by a different source taxonomy.
        self.image_taxonomy_id = str(image_taxonomy_id or taxonomy_id)
        self.reference_root = Path(reference_root) if reference_root else None
        self.load_gt_prototype = bool(load_gt_prototype)
        self.train_variants = int(train_variants)
        self.test_variants = int(test_variants)
        if self.train_variants < 1 or self.test_variants < 1:
            raise ValueError("train_variants and test_variants must be positive")
        self.npoints = 2048
        self.image_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self.image_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
        categories = json.loads(Path(category_file).read_text(encoding="utf-8"))
        tooth_category = next(category for category in categories if category["taxonomy_id"] == self.taxonomy_id)
        n_variants = self.train_variants if split == "train" else self.test_variants
        self.samples: List[Dict[str, object]] = [
            {"model_id": model_id, "variant": variant}
            for model_id in tooth_category[split]
            for variant in range(n_variants)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_points(self, kind: str, model_id: str, variant: int | None = None) -> np.ndarray:
        if kind == "gt":
            path = self.dataset_root / self.split / kind / self.taxonomy_id / f"{model_id}.pcd"
        else:
            assert variant is not None
            path = self.dataset_root / self.split / kind / self.taxonomy_id / model_id / f"{variant:02d}.pcd"
        return read_pcd_xyz(path)

    def _load_views(self, model_id: str, variant: int) -> torch.Tensor:
        views = []
        for suffix in VIEW_SUFFIXES:
            path = self.image_root / self.split / "partial" / self.image_taxonomy_id / model_id / f"{variant:02d}_{suffix}.png"
            with Image.open(path) as image:
                image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32) / 255.0
            # ImageNet normalization stabilizes training while retaining SCC colors.
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensor = (tensor - self.image_mean) / self.image_std
            views.append(tensor)
        return torch.stack(views, dim=0)

    def _load_reference(self, model_id: str, variant: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load optional legacy prototype and current DINO view-token caches."""
        if self.reference_root is None:
            return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
        path = self.reference_root / self.split / "reference" / self.taxonomy_id / model_id / f"{variant:02d}.npz"
        with np.load(path, allow_pickle=False) as reference:
            # C-VGP caches deliberately contain only frozen DINO visual tokens.
            # Keep support for archived caches that also stored prototypes.
            prototype = (
                np.asarray(reference["prototype"], dtype=np.float32).copy()
                if "prototype" in reference.files
                else np.empty((0, 3), dtype=np.float32)
            )
            view_tokens = np.asarray(reference["view_tokens"], dtype=np.float32).copy()
        return torch.from_numpy(prototype).clone(), torch.from_numpy(view_tokens).clone()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        model_id = str(sample["model_id"])
        variant = int(sample["variant"])
        partial = self._load_points("partial", model_id, variant)
        # Kept only for archived SetKD compatibility. The MissingGT route
        # passes load_teacher=False and never opens deployment prototypes here.
        teacher = self._load_points("missing", model_id, variant) if self.load_teacher else np.empty((0, 3), dtype=np.float32)
        gt_missing = self._load_points("gt_missing", model_id, variant)
        reference_prototype, reference_view_tokens = self._load_reference(model_id, variant)
        # Legacy teacher files can contain fewer than 30 points. The current
        # MissingGT protocol disables this branch entirely.
        if len(teacher):
            teacher = farthest_point_sample_numpy(teacher, self.num_prototype_points)
        # The missing_gt route has no FPS-30 surrogate target.  Preserve this
        # field only for archived SetKD checkpoints and diagnostics.
        gt_prototype = (
            farthest_point_sample_numpy(gt_missing, self.num_prototype_points)
            if self.load_gt_prototype
            else np.empty((0, 3), dtype=np.float32)
        )

        # Every source patch already has 2,048 points. Random permutation is a
        # point-order augmentation and cannot affect the permutation-invariant encoder.
        if self.split == "train":
            partial = partial[np.random.permutation(len(partial))]
        # Clone each tensor so DataLoader workers expose independently
        # resizable storage during batched collation.
        return {
            "partial": torch.from_numpy(partial.copy()).clone(),
            # Point-only ablations must not pay image I/O for features they
            # deliberately do not consume. The empty tensor still collates
            # cleanly through the shared student training interface.
            "views": (
                self._load_views(model_id, variant).contiguous().clone()
                if self.load_views else torch.empty(0, dtype=torch.float32)
            ),
            "teacher": torch.from_numpy(teacher.copy()).clone(),
            "gt_missing": torch.from_numpy(gt_missing.copy()).clone(),
            "gt_prototype": torch.from_numpy(gt_prototype.copy()).clone(),
            "reference_prototype": reference_prototype,
            "reference_view_tokens": reference_view_tokens,
            "sample_id": f"{model_id}_{variant:02d}",
        }
