"""Set-based objectives for missing-region prototype prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def squared_min_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared distance from every source point to its closest target point."""
    return torch.cdist(source, target, p=2).square().amin(dim=-1)


def chamfer_l2_per_sample(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return squared_min_distance(source, target).mean(dim=1) + squared_min_distance(target, source).mean(dim=1)


def prototype_cdmiss_per_sample(
    prototype: torch.Tensor,
    gt_missing: torch.Tensor,
    k_pred: int = 10,
) -> torch.Tensor:
    """Measure missing-region coverage by the nearest prototype points.

    This matches the downstream ``CDMiss`` direction: every real missing point
    queries its closest ``k_pred`` predicted prototype points.  It is purposely
    one-sided because the prototype is sparse while ``gt_missing`` is dense.
    """
    distances = torch.cdist(gt_missing.float(), prototype.float(), p=2).square()
    k = min(max(int(k_pred), 1), prototype.shape[1])
    return distances.topk(k=k, largest=False, dim=-1).values.mean(dim=-1).mean(dim=-1)


def direct_missing_gt_loss(
    prediction: torch.Tensor,
    gt_missing: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise a sparse prototype directly with the full missing region.

    The new paper route deliberately does not create a separate FPS target,
    imitate an external prototype, or add a D2P-field target here.  The only
    geometry target is the dense ``missing_gt`` point set itself.
    """
    missing_set = chamfer_l2_per_sample(prediction.float(), gt_missing.float()).mean()
    return missing_set, {
        "total": missing_set.detach(),
        "missing_set": missing_set.detach(),
    }


def d2p_field(query_points: torch.Tensor, prototype: torch.Tensor, tau: float) -> torch.Tensor:
    squared_distance = squared_min_distance(query_points, prototype)
    return torch.exp(-squared_distance / (tau * tau))


def quality_weight(teacher: torch.Tensor, gt_missing: torch.Tensor, sigma: float) -> torch.Tensor:
    """Downweight teacher imitation when its predicted region is less reliable."""
    teacher_error = squared_min_distance(teacher, gt_missing).mean(dim=1)
    return torch.exp(-teacher_error / (sigma * sigma)).clamp_(min=0.20, max=1.0)


def setkd_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    gt_prototype: torch.Tensor,
    gt_missing: torch.Tensor,
    field_queries: torch.Tensor,
    tau: float = 0.05,
    quality_sigma: float = 0.10,
    teacher_weight: float = 0.50,
    field_gt_weight: float = 0.50,
    field_teacher_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Legacy prototype/D2P compatibility objective; unused by the paper route."""
    quality = quality_weight(teacher, gt_missing, quality_sigma)
    gt_set = chamfer_l2_per_sample(prediction, gt_prototype).mean()
    teacher_set = (quality * chamfer_l2_per_sample(prediction, teacher)).mean()

    pred_field = d2p_field(field_queries, prediction, tau)
    gt_field = d2p_field(field_queries, gt_missing, tau)
    teacher_field = d2p_field(field_queries, teacher, tau)
    field_gt = F.smooth_l1_loss(pred_field, gt_field)
    field_teacher = (F.smooth_l1_loss(pred_field, teacher_field, reduction="none").mean(dim=1) * quality).mean()

    loss = gt_set + teacher_weight * teacher_set + field_gt_weight * field_gt + field_teacher_weight * field_teacher
    terms = {
        "total": loss.detach(),
        "gt_set": gt_set.detach(),
        "teacher_set": teacher_set.detach(),
        "field_gt": field_gt.detach(),
        "field_teacher": field_teacher.detach(),
        "teacher_quality": quality.mean().detach(),
    }
    return loss, terms


def normalized_view_feature_loss(student_tokens: torch.Tensor, reference_tokens: torch.Tensor) -> torch.Tensor:
    """Align per-view visual structure without making token scale a target."""
    student = F.normalize(student_tokens.float(), dim=-1)
    reference = F.normalize(reference_tokens.float(), dim=-1)
    return F.mse_loss(student, reference)


def feature_distill_missing_gt_loss(
    prediction: torch.Tensor,
    student_view_tokens: torch.Tensor,
    reference_view_tokens: torch.Tensor,
    gt_missing: torch.Tensor,
    feature_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Use missing GT for geometry and DINO only for visual-token alignment."""
    missing_set = chamfer_l2_per_sample(prediction.float(), gt_missing.float()).mean()
    feature = normalized_view_feature_loss(student_view_tokens, reference_view_tokens)
    loss = missing_set + float(feature_weight) * feature
    return loss, {
        "total": loss.detach(),
        "missing_set": missing_set.detach(),
        "feature": feature.detach(),
    }


def litedino_loss(
    prediction: torch.Tensor,
    student_view_tokens: torch.Tensor,
    reference_prototype: torch.Tensor,
    reference_view_tokens: torch.Tensor,
    gt_prototype: torch.Tensor,
    gt_missing: torch.Tensor,
    field_queries: torch.Tensor,
    tau: float = 0.05,
    quality_sigma: float = 0.10,
    prototype_weight: float = 0.25,
    field_gt_weight: float = 0.50,
    field_reference_weight: float = 0.10,
    feature_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Legacy compatibility objective; not used by the paper's C-VGP route."""
    quality = quality_weight(reference_prototype, gt_missing, quality_sigma)
    gt_set = chamfer_l2_per_sample(prediction, gt_prototype).mean()
    reference_set = (quality * chamfer_l2_per_sample(prediction, reference_prototype)).mean()

    pred_field = d2p_field(field_queries, prediction, tau)
    gt_field = d2p_field(field_queries, gt_missing, tau)
    reference_field = d2p_field(field_queries, reference_prototype, tau)
    field_gt = F.smooth_l1_loss(pred_field, gt_field)
    field_reference = (
        F.smooth_l1_loss(pred_field, reference_field, reduction="none").mean(dim=1) * quality
    ).mean()
    feature = normalized_view_feature_loss(student_view_tokens, reference_view_tokens)
    loss = (
        gt_set
        + float(prototype_weight) * reference_set
        + float(field_gt_weight) * field_gt
        + float(field_reference_weight) * field_reference
        + float(feature_weight) * feature
    )
    terms = {
        "total": loss.detach(),
        "gt_set": gt_set.detach(),
        "reference_set": reference_set.detach(),
        "field_gt": field_gt.detach(),
        "field_reference": field_reference.detach(),
        "feature": feature.detach(),
        "reference_quality": quality.mean().detach(),
    }
    return loss, terms
