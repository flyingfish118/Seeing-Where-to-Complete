"""Export fixed deployment prototypes from a trained DINO-VGP or C-VGP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from litevcp.data import LiteVCPDistillDataset, collate_distill_batch, read_pcd_xyz
from litevcp.losses import chamfer_l2_per_sample, d2p_field, prototype_cdmiss_per_sample
from litevcp.train import build_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def valid_export_entry(
    point_path: Path,
    meta_path: Path,
    sample_id: str,
    source_name: str,
    num_points: int,
    config_sha256: str,
    checkpoint_sha256: str,
) -> bool:
    """Reuse only a complete PCD/metadata pair after an interrupted export."""
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            metadata.get("id") != sample_id
            or metadata.get("source") != source_name
            or int(metadata.get("num_points", -1)) != int(num_points)
            or metadata.get("entry_schema_version") != 2
            or metadata.get("producer_config_sha256") != config_sha256
            or metadata.get("producer_checkpoint_sha256") != checkpoint_sha256
        ):
            return False
        points = read_pcd_xyz(point_path)
        return points.shape == (int(num_points), 3) and bool(torch.isfinite(torch.from_numpy(points)).all())
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None, help="Smoke-test limit; omit for full export")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing point/meta pairs and write only missing prototype files.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return _expand_environment(yaml.safe_load(handle))


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    return value


def write_ascii_pcd(path: Path, points: torch.Tensor) -> None:
    """Write the predicted VGP set in a simple XYZ PCD representation."""
    points = points.detach().cpu().float().numpy()
    lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {len(points)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(points)}",
        "DATA ascii",
    ]
    lines.extend("{:.8f} {:.8f} {:.8f}".format(*point) for point in points)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="ascii")
    os.replace(temporary, path)


def load_student(config: dict[str, Any], checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = [name for name in incompatible.unexpected_keys if not name.startswith("view_encoder.model.")]
    missing = [name for name in incompatible.missing_keys if not name.startswith("view_encoder.model.")]
    if unexpected or missing:
        raise RuntimeError(f"Checkpoint/model mismatch; missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


@torch.no_grad()
def export_split(
    model: torch.nn.Module,
    config: dict[str, Any],
    split: str,
    output_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_samples: int | None,
    overwrite: bool,
    resume: bool,
    config_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, float | int | str]:
    dataset = LiteVCPDistillDataset(
        dataset_root=config["data"]["dataset_root"],
        image_root=config["data"]["image_root"],
        category_file=config["data"]["category_file"],
        split=split,
        image_size=config["data"]["image_size"],
        num_prototype_points=config["model"]["num_prototype_points"],
        load_views=bool(config["data"].get("load_views", True)),
        load_teacher=bool(config["data"].get("load_teacher", True)),
        taxonomy_id=str(config["data"].get("taxonomy_id", "11")),
        image_taxonomy_id=str(config["data"].get("image_taxonomy_id", config["data"].get("taxonomy_id", "11"))),
        reference_root=config["data"].get("reference_root"),
        load_gt_prototype=bool(config["data"].get("load_gt_prototype", True)),
        train_variants=int(config["data"].get("train_variants", 8)),
        test_variants=int(config["data"].get("test_variants", 1)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_distill_batch,
    )
    total_proto_gt = total_missing_gt = total_cdmiss1 = total_cdmiss10 = total_d2p_gt = 0.0
    has_gt_prototype = False
    total = 0
    reused = 0
    model_type = str(config["model"].get("type", "lite")).lower()
    source_name = str(config["model"].get(
        "export_source",
        "dino_vgp" if model_type == "dinov3" else "c_vgp" if model_type == "litedino" else "legacy_setkd",
    ))
    for batch in loader:
        partial = batch["partial"].to(device, non_blocking=True)
        views = batch["views"].to(device, non_blocking=True)
        gt_prototype = batch["gt_prototype"].to(device, non_blocking=True)
        gt_missing = batch["gt_missing"].to(device, non_blocking=True)
        point_paths = []
        meta_paths = []
        for sample_id in batch["sample_id"]:
            model_id, variant = str(sample_id).rsplit("_", 1)
            taxonomy_id = str(config["data"].get("taxonomy_id", "11"))
            case_dir = output_root / split / "missing" / taxonomy_id / model_id
            point_paths.append(case_dir / f"{int(variant):02d}.pcd")
            meta_paths.append(case_dir / f"{int(variant):02d}.meta.json")
        # Even after an interrupted export, evaluate every sample so the
        # manifest's aggregate metrics remain complete and comparable.
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(partial, views)
        prediction = prediction.float()
        if gt_prototype.numel():
            total_proto_gt += float(chamfer_l2_per_sample(prediction, gt_prototype).sum())
            has_gt_prototype = True
        total_missing_gt += float(chamfer_l2_per_sample(prediction, gt_missing).sum())
        total_cdmiss1 += float(prototype_cdmiss_per_sample(prediction, gt_missing, k_pred=1).sum())
        total_cdmiss10 += float(prototype_cdmiss_per_sample(prediction, gt_missing).sum())
        total_d2p_gt += float((d2p_field(partial.float(), prediction, config["loss"]["tau"]) - d2p_field(
            partial.float(), gt_missing, config["loss"]["tau"]
        )).abs().sum(dim=1).sum())

        for index, (point_path, meta_path) in enumerate(zip(point_paths, meta_paths)):
            case_dir = point_path.parent
            case_dir.mkdir(parents=True, exist_ok=True)
            sample_id = str(batch["sample_id"][index])
            if resume and valid_export_entry(
                point_path,
                meta_path,
                sample_id,
                source_name,
                int(prediction.shape[1]),
                config_sha256,
                checkpoint_sha256,
            ):
                reused += 1
                continue
            if (point_path.exists() or meta_path.exists()) and not (overwrite or resume):
                raise FileExistsError(f"Refusing to overwrite existing prototype entry: {point_path}")
            write_ascii_pcd(point_path, prediction[index])
            write_json_atomic(
                meta_path,
                {
                    "entry_schema_version": 2,
                    "id": sample_id,
                    "model_id": sample_id.rsplit("_", 1)[0],
                    "taxonomy_id": str(config["data"].get("taxonomy_id", "11")),
                    "split": split,
                    "num_points": int(prediction.shape[1]),
                    "source": source_name,
                    "producer_config_sha256": config_sha256,
                    "producer_checkpoint_sha256": checkpoint_sha256,
                },
            )
        total += prediction.shape[0]
        if max_samples is not None and total >= max_samples:
            break

    result = {
        "split": split,
        "samples": total,
        "reused": reused,
        "prototype_cdl2_to_missing_gt": total_missing_gt / total,
        "prototype_cdmiss1_to_missing_gt": total_cdmiss1 / total,
        "prototype_cdmiss10_to_missing_gt": total_cdmiss10 / total,
        "d2p_mae_to_gt": total_d2p_gt / (total * dataset.npoints),
    }
    if has_gt_prototype:
        result["prototype_cdl2_to_gt"] = total_proto_gt / total
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.ckpt).resolve()
    config_sha256 = sha256(config_path)
    checkpoint_sha256 = sha256(checkpoint_path)
    device = torch.device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model = load_student(config, str(checkpoint_path), device)
    batch_size = args.batch_size or config["train"]["eval_batch_size"]
    results = [
        export_split(
            model,
            config,
            split,
            output_root,
            batch_size,
            args.num_workers,
            device,
            args.max_samples,
            args.overwrite,
            args.resume,
            config_sha256,
            checkpoint_sha256,
        )
        for split in args.splits
    ]
    manifest = {
        "schema_version": 2,
        "student_config": str(config_path),
        "student_config_sha256": config_sha256,
        "student_checkpoint": str(checkpoint_path),
        "student_checkpoint_sha256": checkpoint_sha256,
        "output_root": str(output_root.resolve()),
        "max_samples": args.max_samples,
        "num_prototype_points": int(config["model"]["num_prototype_points"]),
        "source": str(config["model"].get("export_source", "unknown")),
        "results": results,
    }
    write_json_atomic(output_root / "export_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
