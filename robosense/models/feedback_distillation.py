from __future__ import annotations

import torch
from torch.nn import functional as F


def binary_soft_target_kd(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    temperature: float,
    quality_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact teacher-to-student Bernoulli KL at the registered temperature."""
    valid = teacher_probabilities.ge(0)
    if not valid.any():
        return student_logits.sum() * 0
    tau = max(float(temperature), 1e-6)
    teacher = teacher_probabilities[valid].float().to(student_logits.device).clamp(1e-6, 1 - 1e-6)
    softened_teacher = torch.sigmoid(torch.logit(teacher) / tau)
    teacher_distribution = torch.stack((1 - softened_teacher, softened_teacher), dim=-1)
    student_log_distribution = torch.log_softmax(
        torch.stack((torch.zeros_like(student_logits[valid]), student_logits[valid]), dim=-1) / tau,
        dim=-1,
    )
    per_sample = F.kl_div(student_log_distribution, teacher_distribution, reduction="none").sum(dim=-1)
    if quality_weights is None:
        return per_sample.mean()
    weights = quality_weights.float().to(per_sample.device)[valid].clamp_min(0)
    if weights.sum() <= 0:
        return student_logits.sum() * 0
    return (per_sample * weights).sum() / weights.sum()


def replay_detection_loss(
    student_logits: torch.Tensor,
    replay_labels: torch.Tensor,
    previous_probabilities: torch.Tensor | None,
    *,
    positive_weight: float,
    consistency_weight: float,
) -> torch.Tensor:
    valid = replay_labels.ge(0)
    if not valid.any():
        return student_logits.sum() * 0
    labels = replay_labels.float().to(student_logits.device)
    loss = F.binary_cross_entropy_with_logits(
        student_logits[valid], labels[valid],
        pos_weight=student_logits.new_tensor(float(positive_weight)),
    )
    if previous_probabilities is not None:
        previous = previous_probabilities.float().to(student_logits.device)
        protected = valid & previous.ge(0)
        if protected.any():
            loss = loss + float(consistency_weight) * F.mse_loss(
                torch.sigmoid(student_logits[protected]), previous[protected]
            )
    return loss

