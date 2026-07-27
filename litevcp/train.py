"""Train and evaluate missing-region prototype predictors."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from litevcp.data import LiteVCPDistillDataset, collate_distill_batch
from litevcp.losses import (
    chamfer_l2_per_sample,
    d2p_field,
    direct_missing_gt_loss,
    feature_distill_missing_gt_loss,
    litedino_loss,
    prototype_cdmiss_per_sample,
    setkd_loss,
)
from litevcp.model import DINOv3SetKD, LiteDINOPrototypeStudent, LiteVCPSetKD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="cfgs/Tooth_models/C_VGP.yaml")
    parser.add_argument("--exp_name", default="c_vgp")
    parser.add_argument("--output_root", default="experiments/C_VGP")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epoch count")
    parser.add_argument("--num_workers", type=int, default=None, help="Override configured loader workers")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume or evaluate")
    parser.add_argument("--eval", action="store_true", help="Evaluate --resume without training")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return _expand_environment(yaml.safe_load(handle))


def _expand_environment(value: Any) -> Any:
    """Keep local datasets and foundation weights outside the source release."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def mean_dict(items: list[dict[str, float]]) -> dict[str, float]:
    keys = items[0].keys()
    return {key: float(np.mean([item[key] for item in items])) for key in keys}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tau: float,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    results: list[dict[str, float]] = []
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    for batch in loader:
        partial = batch["partial"].to(device, non_blocking=True)
        views = batch["views"].to(device, non_blocking=True)
        teacher = batch["teacher"].to(device, non_blocking=True)
        reference_prototype = batch["reference_prototype"].to(device, non_blocking=True)
        gt_missing = batch["gt_missing"].to(device, non_blocking=True)
        gt_prototype = batch["gt_prototype"].to(device, non_blocking=True)
        with torch.autocast(device_type=amp_device, enabled=use_amp):
            prediction = model(partial, views)

        # The new protocol evaluates the full missing region.  The FPS-30
        # statistic remains available only when an archived configuration asks
        # the dataset to construct it.
        student_missing_gt = chamfer_l2_per_sample(prediction.float(), gt_missing.float()).mean()
        student_field = d2p_field(partial.float(), prediction.float(), tau)
        gt_field = d2p_field(partial.float(), gt_missing.float(), tau)
        result = {
            "student_missing_cdl2": float(student_missing_gt),
            "student_missing_cdmiss1": float(prototype_cdmiss_per_sample(prediction, gt_missing, k_pred=1).mean()),
            "student_missing_cdmiss10": float(prototype_cdmiss_per_sample(prediction, gt_missing).mean()),
            "student_d2p_mae_gt": float((student_field - gt_field).abs().mean()),
        }
        if gt_prototype.numel():
            result["student_proto_cdl2"] = float(
                chamfer_l2_per_sample(prediction.float(), gt_prototype).mean()
            )
        # The reference is used only by archived SetKD configurations. The
        # MissingGT route has no prototype teacher target.
        reference = reference_prototype if reference_prototype.numel() else teacher
        if reference.numel() and gt_prototype.numel():
            reference_gt = chamfer_l2_per_sample(reference.float(), gt_prototype).mean()
            student_reference = chamfer_l2_per_sample(prediction.float(), reference.float()).mean()
            reference_field = d2p_field(partial.float(), reference.float(), tau)
            result.update(
                {
                    "reference_proto_cdl2": float(reference_gt),
                    "student_reference_cdl2": float(student_reference),
                    "reference_d2p_mae_gt": float((reference_field - gt_field).abs().mean()),
                }
            )
        results.append(result)
    return mean_dict(results)


def benchmark_latency(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device, warmup: int = 30, runs: int = 100) -> float:
    model.eval()
    partial = batch["partial"][:1].to(device)
    views = batch["views"][:1].to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(partial, views)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(runs):
            model(partial, views)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / runs


def checkpoint_payload(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    # DINOv3 is a fixed public backbone loaded from its local checkpoint. Do
    # not duplicate its 1.2 GB weights in every experiment checkpoint.
    student_state = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("view_encoder.model.")
    }
    return {
        "epoch": epoch,
        "best_metric": best_metric,
        "model": student_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
    }


def build_model(config: dict[str, Any]) -> nn.Module:
    model_config = dict(config["model"])
    model_type = model_config.pop("type", "lite")
    # Export metadata belongs to manifests, not module constructors.
    model_config.pop("export_source", None)
    if model_type == "lite":
        return LiteVCPSetKD(**model_config)
    if model_type == "dinov3":
        return DINOv3SetKD(**model_config)
    if model_type == "litedino":
        return LiteDINOPrototypeStudent(**model_config)
    raise ValueError(f"Unsupported model.type: {model_type}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_type = str(config["model"].get("type", "lite")).lower()
    loss_mode = str(config["loss"].get("mode", "setkd")).lower()
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_dir = Path(args.output_root) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "run.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    dataset_kwargs = {
        "dataset_root": config["data"]["dataset_root"],
        "image_root": config["data"]["image_root"],
        "category_file": config["data"]["category_file"],
        "image_size": config["data"]["image_size"],
        "num_prototype_points": config["model"]["num_prototype_points"],
        "load_views": bool(config["data"].get("load_views", True)),
        "load_teacher": bool(config["data"].get("load_teacher", True)),
        "taxonomy_id": str(config["data"].get("taxonomy_id", "11")),
        "image_taxonomy_id": str(config["data"].get("image_taxonomy_id", config["data"].get("taxonomy_id", "11"))),
        "reference_root": config["data"].get("reference_root"),
        "load_gt_prototype": bool(config["data"].get("load_gt_prototype", True)),
        "train_variants": int(config["data"].get("train_variants", 8)),
        "test_variants": int(config["data"].get("test_variants", 1)),
    }
    train_set = LiteVCPDistillDataset(split="train", **dataset_kwargs)
    test_set = LiteVCPDistillDataset(split="test", **dataset_kwargs)
    loader_kwargs = {
        "num_workers": config["data"]["num_workers"],
        "pin_memory": device.type == "cuda",
        "persistent_workers": config["data"]["num_workers"] > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collate_distill_batch,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config["train"]["eval_batch_size"],
        shuffle=False,
        collate_fn=collate_distill_batch,
        **loader_kwargs,
    )

    model = build_model(config).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"], eta_min=config["train"]["min_lr"])
    use_amp = bool(config["train"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 0
    best_metric = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        unexpected = [name for name in incompatible.unexpected_keys if not name.startswith("view_encoder.model.")]
        missing = [name for name in incompatible.missing_keys if not name.startswith("view_encoder.model.")]
        if unexpected or missing:
            raise RuntimeError(f"Checkpoint/model mismatch; missing={missing}, unexpected={unexpected}")
        if not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_metric = float(checkpoint["best_metric"])
        print(f"Loaded checkpoint: {args.resume}")

    if args.eval:
        metrics = evaluate(model, test_loader, device, config["loss"]["tau"], use_amp)
        latency_ms = benchmark_latency(model, next(iter(test_loader)), device)
        metrics.update({
            "parameters": model.num_parameters,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "latency_ms_batch1": latency_ms,
        })
        print(json.dumps(metrics, indent=2, sort_keys=True))
        (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return

    log_path = output_dir / "train.jsonl"
    for epoch in range(start_epoch, config["train"]["epochs"]):
        model.train()
        epoch_terms: list[dict[str, float]] = []
        start_time = time.perf_counter()
        for batch in train_loader:
            partial = batch["partial"].to(device, non_blocking=True)
            views = batch["views"].to(device, non_blocking=True)
            teacher = batch["teacher"].to(device, non_blocking=True)
            reference_prototype = batch["reference_prototype"].to(device, non_blocking=True)
            reference_view_tokens = batch["reference_view_tokens"].to(device, non_blocking=True)
            gt_missing = batch["gt_missing"].to(device, non_blocking=True)
            gt_prototype = batch["gt_prototype"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if model_type == "litedino" and loss_mode == "feature_distill_missing_gt":
                    if not reference_view_tokens.numel():
                        raise RuntimeError("C-VGP feature alignment requires cached DINO view tokens")
                    prediction, view_tokens = model.forward_with_view_tokens(partial, views)
                    loss, terms = feature_distill_missing_gt_loss(
                        prediction,
                        view_tokens,
                        reference_view_tokens,
                        gt_missing,
                        feature_weight=float(config["loss"].get("feature_weight", 0.25)),
                    )
                elif loss_mode == "missing_gt":
                    prediction = model(partial, views)
                    loss, terms = direct_missing_gt_loss(prediction, gt_missing)
                elif model_type == "litedino":
                    # Retained only to make archived checkpoints inspectable.
                    if not reference_prototype.numel() or not reference_view_tokens.numel():
                        raise RuntimeError("Legacy compatibility loss requires an offline DINO reference cache")
                    prediction, view_tokens = model.forward_with_view_tokens(partial, views)
                    loss, terms = litedino_loss(
                        prediction,
                        view_tokens,
                        reference_prototype,
                        reference_view_tokens,
                        gt_prototype,
                        gt_missing,
                        partial,
                        **config["loss"],
                    )
                else:
                    prediction = model(partial, views)
                    loss, terms = setkd_loss(
                        prediction,
                        teacher,
                        gt_prototype,
                        gt_missing,
                        partial,
                        **config["loss"],
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            epoch_terms.append({key: float(value) for key, value in terms.items()})
        scheduler.step()

        record = mean_dict(epoch_terms)
        record.update({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_seconds": time.perf_counter() - start_time})
        if (epoch + 1) % config["train"]["eval_every"] == 0 or epoch == config["train"]["epochs"] - 1:
            metrics = evaluate(model, test_loader, device, config["loss"]["tau"], use_amp)
            record.update({f"val_{key}": value for key, value in metrics.items()})
            current_metric = metrics[
                "student_missing_cdl2"
                if loss_mode in {"missing_gt", "feature_distill_missing_gt"}
                else "student_proto_cdl2"
            ]
            if current_metric < best_metric:
                best_metric = current_metric
                torch.save(checkpoint_payload(model, optimizer, scheduler, epoch, best_metric, config), output_dir / "ckpt-best.pth")
        torch.save(checkpoint_payload(model, optimizer, scheduler, epoch, best_metric, config), output_dir / "ckpt-last.pth")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    best_checkpoint = output_dir / "ckpt-best.pth"
    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = [name for name in incompatible.unexpected_keys if not name.startswith("view_encoder.model.")]
    missing = [name for name in incompatible.missing_keys if not name.startswith("view_encoder.model.")]
    if unexpected or missing:
        raise RuntimeError(f"Checkpoint/model mismatch; missing={missing}, unexpected={unexpected}")
    metrics = evaluate(model, test_loader, device, config["loss"]["tau"], use_amp)
    metrics.update({
        "parameters": model.num_parameters,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "latency_ms_batch1": benchmark_latency(model, next(iter(test_loader)), device),
    })
    print("Best checkpoint", best_checkpoint)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
