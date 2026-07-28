"""Run a CPU-friendly training/evaluation smoke test on bundled synthetic data."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.generate_smoke_data import ensure_smoke_data  # noqa: E402
from litevcp.data import LiteVCPDistillDataset, collate_distill_batch  # noqa: E402
from litevcp.losses import d2p_field, direct_missing_gt_loss, prototype_cdmiss_per_sample  # noqa: E402
from litevcp.model import LiteDINOPrototypeStudent  # noqa: E402


def _dataset(root: Path, split: str) -> LiteVCPDistillDataset:
    return LiteVCPDistillDataset(
        dataset_root=root / "points",
        image_root=root / "scc",
        category_file=root / "Tooth_smoke.json",
        split=split,
        image_size=64,
        num_prototype_points=30,
        load_views=True,
        load_teacher=False,
        load_gt_prototype=False,
        train_variants=1,
        test_variants=1,
    )


def run_smoke_test(data_root: Path) -> None:
    torch.manual_seed(2027)
    np.random.seed(2027)
    torch.set_num_threads(1)
    fixture_root = ensure_smoke_data(data_root)

    train_set = _dataset(fixture_root, "train")
    test_set = _dataset(fixture_root, "test")
    train_loader = DataLoader(
        train_set,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_distill_batch,
    )
    train_batch = next(iter(train_loader))

    model = LiteDINOPrototypeStudent(num_prototype_points=30, width=8, teacher_feature_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    first_parameter = next(model.parameters())
    parameter_before = first_parameter.detach().clone()
    prediction = model(train_batch["partial"], train_batch["views"])
    loss, _ = direct_missing_gt_loss(prediction, train_batch["gt_missing"])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if prediction.shape != (2, 30, 3):
        raise RuntimeError(f"Unexpected training output shape: {tuple(prediction.shape)}")
    if not math.isfinite(float(loss.detach())):
        raise RuntimeError("Training loss is not finite")
    if torch.equal(parameter_before, first_parameter.detach()):
        raise RuntimeError("Optimizer step did not update the C-VGP parameters")

    test_batch = collate_distill_batch([test_set[0]])
    model.eval()
    with torch.no_grad():
        test_prediction = model(test_batch["partial"], test_batch["views"])
        cdmiss = prototype_cdmiss_per_sample(test_prediction, test_batch["gt_missing"], k_pred=10).mean()
        field = d2p_field(test_batch["partial"], test_prediction, tau=0.15)

    if test_prediction.shape != (1, 30, 3):
        raise RuntimeError(f"Unexpected test output shape: {tuple(test_prediction.shape)}")
    if field.shape != (1, 2048) or not torch.all((field >= 0.0) & (field <= 1.0)):
        raise RuntimeError("D2P field failed its shape/range check")
    if not math.isfinite(float(cdmiss)):
        raise RuntimeError("Synthetic CDMiss is not finite")

    print("VGP synthetic smoke test: PASS")
    print(f"  fixture: 2 train cases, 1 test case, 1 variant each")
    print(f"  train tensors: partial={tuple(train_batch['partial'].shape)}, views={tuple(train_batch['views'].shape)}")
    print(f"  C-VGP output: {tuple(prediction.shape)}, one-step loss={float(loss.detach()):.6f}")
    print(f"  test output: {tuple(test_prediction.shape)}, D2P field={tuple(field.shape)}")
    print("  note: synthetic values verify execution only; they are not benchmark results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "synthetic_t3ds_smoke",
        help="Bundled/generated synthetic fixture root.",
    )
    args = parser.parse_args()
    run_smoke_test(args.data_root)


if __name__ == "__main__":
    main()
