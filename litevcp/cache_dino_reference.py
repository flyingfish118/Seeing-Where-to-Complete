"""Cache frozen DINO per-SCC-view tokens for C-VGP feature alignment.

The cache is made once from the DINO visual-geometric predictor. C-VGP
training then has no DINO dependency and never reads another model's prototype.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from litevcp.data import LiteVCPDistillDataset, collate_distill_batch
from litevcp.losses import chamfer_l2_per_sample, d2p_field
from litevcp.export_prototypes import load_student


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


def valid_reference_entry(path: Path, tokens_only: bool) -> bool:
    """Accept an interrupted-cache entry only when its required array is intact."""
    try:
        with np.load(path, allow_pickle=False) as payload:
            if "view_tokens" not in payload.files:
                return False
            tokens = payload["view_tokens"]
            if tokens.ndim != 2 or tokens.shape[0] != 3 or tokens.shape[1] < 1:
                return False
            if not np.isfinite(tokens).all():
                return False
            return tokens_only or "prototype" in payload.files
    except (OSError, ValueError, KeyError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="GT-only DINO reference configuration")
    parser.add_argument("--ckpt", required=True, help="Selected GT-only DINO checkpoint")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test limit; omit for the full cache")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--tokens-only",
        action="store_true",
        help="Write only DINO view tokens; use this for the missing_gt four-predictor route.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep already valid cache entries and write only missing entries.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return _expand_environment(yaml.safe_load(handle))


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    return value


@torch.no_grad()
def cache_split(
    model: torch.nn.Module,
    config: dict[str, Any],
    split: str,
    output_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    overwrite: bool,
    max_samples: int | None,
    tokens_only: bool,
    resume: bool,
) -> dict[str, float | int | str]:
    dataset = LiteVCPDistillDataset(
        dataset_root=config["data"]["dataset_root"],
        image_root=config["data"]["image_root"],
        category_file=config["data"]["category_file"],
        split=split,
        image_size=config["data"]["image_size"],
        num_prototype_points=config["model"]["num_prototype_points"],
        load_views=True,
        load_teacher=False,
        taxonomy_id=str(config["data"].get("taxonomy_id", "11")),
        image_taxonomy_id=str(config["data"].get("image_taxonomy_id", config["data"].get("taxonomy_id", "11"))),
        train_variants=int(config["data"].get("train_variants", 8)),
        test_variants=int(config["data"].get("test_variants", 1)),
        load_gt_prototype=bool(config["data"].get("load_gt_prototype", True)),
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
    total = 0
    reused = 0
    total_proto_gt = total_missing_gt = total_d2p_gt = 0.0
    has_gt_prototype = False
    taxonomy_id = str(config["data"].get("taxonomy_id", "11"))
    for batch in loader:
        partial = batch["partial"].to(device, non_blocking=True)
        views = batch["views"].to(device, non_blocking=True)
        gt_prototype = batch["gt_prototype"].to(device, non_blocking=True)
        gt_missing = batch["gt_missing"].to(device, non_blocking=True)
        # Keep metric aggregation deterministic even when a resumed cache has
        # already written some files. Existing valid entries are skipped only
        # at write time, while the reference forward still covers every sample.
        paths = []
        for sample_id in batch["sample_id"]:
            model_id, variant = str(sample_id).rsplit("_", 1)
            paths.append(output_root / split / "reference" / taxonomy_id / model_id / f"{int(variant):02d}.npz")
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prototype, view_tokens = model.forward_with_view_tokens(partial, views)
        prototype = prototype.float()
        view_tokens = view_tokens.float()
        if gt_prototype.numel():
            total_proto_gt += float(chamfer_l2_per_sample(prototype, gt_prototype).sum())
            has_gt_prototype = True
        total_missing_gt += float(chamfer_l2_per_sample(prototype, gt_missing).sum())
        total_d2p_gt += float((d2p_field(partial.float(), prototype, config["loss"]["tau"]) - d2p_field(
            partial.float(), gt_missing, config["loss"]["tau"]
        )).abs().sum(dim=1).sum())
        for index, path in enumerate(paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if resume and valid_reference_entry(path, tokens_only):
                    reused += 1
                    continue
                if not (overwrite or resume):
                    raise FileExistsError(f"Refusing to overwrite reference: {path}")
            # Float16 is sufficient for the frozen visual target and keeps the
            # full cache compact.  The new route requests tokens only, whereas
            # archived PrototypeDistill experiments retain their old cache.
            payload = {"view_tokens": view_tokens[index].detach().cpu().numpy().astype(np.float16)}
            if not tokens_only:
                payload["prototype"] = prototype[index].detach().cpu().numpy().astype(np.float32)
            temporary = path.with_name(path.name + ".tmp")
            # Passing an open handle prevents NumPy from silently appending
            # another suffix to ``.npz.tmp``.
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **payload)
            os.replace(temporary, path)
        total += prototype.shape[0]
        if max_samples is not None and total >= max_samples:
            break
    result = {
        "split": split,
        "samples": total,
        "reused": reused,
        "reference_cdl2_to_missing_gt": total_missing_gt / total,
        "reference_d2p_mae_to_gt": total_d2p_gt / (total * dataset.npoints),
    }
    if has_gt_prototype:
        result["reference_proto_cdl2_to_gt"] = total_proto_gt / total
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if str(config["model"].get("type", "")).lower() != "dinov3":
        raise ValueError("The DINO reference cache must be built from a dinov3 model")
    device = torch.device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model = load_student(config, args.ckpt, device)
    results = [
        cache_split(
            model, config, split, output_root, args.batch_size, args.num_workers, device, args.overwrite,
            args.max_samples, args.tokens_only, args.resume,
        )
        for split in args.splits
    ]
    manifest = {
        "schema_version": 2,
        "reference_type": "frozen_gt_supervised_dinov3",
        "reference_config": str(Path(args.config).resolve()),
        "reference_config_sha256": sha256(Path(args.config).resolve()),
        "reference_checkpoint": str(Path(args.ckpt).resolve()),
        "reference_checkpoint_sha256": sha256(Path(args.ckpt).resolve()),
        "output_root": str(output_root.resolve()),
        "tokens_only": bool(args.tokens_only),
        "results": results,
    }
    write_json_atomic(output_root / "reference_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
