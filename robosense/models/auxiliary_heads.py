"""Sensorless auxiliary heads over pooled multimodal language-model states."""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .feedback_distillation import binary_soft_target_kd, replay_detection_loss
from .sensor_encoder import _text_hidden_size, compute_anomaly_pool_positions


class MultimodalAuxiliaryHeads(nn.Module):
    """Detection and task heads that consume a pooled video/audio/text state."""

    def __init__(
        self,
        input_dim: int,
        task_embedding_dim: int,
        num_task_labels: int,
        num_action_labels: int,
        num_object_labels: int,
        anomaly_head: bool,
        task_aware_head: bool,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.task_embedding_dim = task_embedding_dim
        self.num_task_labels = num_task_labels
        self.num_action_labels = num_action_labels
        self.num_object_labels = num_object_labels
        self.anomaly_head = nn.Linear(input_dim, 1) if anomaly_head else None
        if task_aware_head:
            self.task_projection = nn.Sequential(nn.Linear(input_dim, task_embedding_dim), nn.LayerNorm(task_embedding_dim))
            self.task_classifier = nn.Linear(task_embedding_dim, num_task_labels)
            self.action_classifier = nn.Linear(task_embedding_dim, num_action_labels)
            self.object_classifier = nn.Linear(task_embedding_dim, num_object_labels)
            self.disagreement_head = nn.Linear(task_embedding_dim + 1, 1)
        else:
            self.task_projection = None
            self.task_classifier = None
            self.action_classifier = None
            self.object_classifier = None
            self.disagreement_head = None


def _masked_bce(logits: torch.Tensor, labels: torch.Tensor, positive_weight: float) -> Optional[torch.Tensor]:
    labels = labels.float().to(logits.device)
    valid = labels.ge(0)
    if not valid.any():
        return None
    return F.binary_cross_entropy_with_logits(
        logits[valid], labels[valid], pos_weight=logits.new_tensor(float(positive_weight))
    )


def _masked_ce(logits: torch.Tensor, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if labels is None:
        return None
    labels = labels.long().to(logits.device)
    valid = labels.ge(0)
    if not valid.any():
        return None
    return F.cross_entropy(logits[valid], labels[valid])


def attach_multimodal_auxiliary_heads(
    model: nn.Module,
    adapter_path: Optional[str] = None,
    anomaly_head: bool = True,
    export_fused_embeddings: bool = False,
    task_aware_head: bool = False,
    export_task_outputs: bool = False,
    num_task_labels: int = 0,
    num_action_labels: int = 0,
    num_object_labels: int = 0,
    task_embedding_dim: int = 256,
    task_loss_weight: float = 0.2,
    action_loss_weight: float = 0.1,
    object_loss_weight: float = 0.1,
    relation_distillation_loss_weight: float = 0.0,
    previous_edge_consistency_loss_weight: float = 0.0,
    disagreement_loss_weight: float = 0.0,
    anomaly_loss_weight: float = 1.0,
    anomaly_positive_weight: float = 1.0,
    distillation_loss_weight: float = 0.0,
    distillation_temperature: float = 1.0,
    language_model_loss_weight: float = 1.0,
    replay_loss_weight: float = 0.0,
    replay_consistency_weight: float = 0.5,
) -> nn.Module:
    """Patch a model so auxiliary losses use its native multimodal hidden states."""

    if hasattr(model, "reassemble_multimodal_auxiliary_heads"):
        return model
    heads = MultimodalAuxiliaryHeads(
        _text_hidden_size(model), task_embedding_dim, num_task_labels, num_action_labels,
        num_object_labels, anomaly_head, task_aware_head,
    )
    model.add_module("reassemble_multimodal_auxiliary_heads", heads)
    model.reassemble_export_fused_embeddings = export_fused_embeddings
    model.reassemble_export_task_outputs = export_task_outputs
    model.reassemble_task_loss_weight = task_loss_weight
    model.reassemble_action_loss_weight = action_loss_weight
    model.reassemble_object_loss_weight = object_loss_weight
    model.reassemble_relation_distillation_loss_weight = relation_distillation_loss_weight
    model.reassemble_previous_edge_consistency_loss_weight = previous_edge_consistency_loss_weight
    model.reassemble_disagreement_loss_weight = disagreement_loss_weight
    model.reassemble_anomaly_loss_weight = anomaly_loss_weight
    model.reassemble_anomaly_positive_weight = anomaly_positive_weight
    model.reassemble_distillation_loss_weight = distillation_loss_weight
    model.reassemble_distillation_temperature = distillation_temperature
    model.reassemble_language_model_loss_weight = language_model_loss_weight
    model.reassemble_replay_loss_weight = replay_loss_weight
    model.reassemble_replay_consistency_weight = replay_consistency_weight
    model.reassemble_relation_memory_student = None
    model.reassemble_relation_memory_teacher = None
    original_forward = model.forward

    def auxiliary_forward(
        this,
        input_ids=None,
        anomaly_labels=None,
        teacher_scores=None,
        task_labels=None,
        action_labels=None,
        object_labels=None,
        active_task_masks=None,
        previous_edge_scores=None,
        teacher_embeddings=None,
        teacher_quality_weights=None,
        disagreement_labels=None,
        replay_labels=None,
        anomaly_pool_positions=None,
        return_anomaly_logits=False,
        return_fused_embedding=False,
        **kwargs,
    ):
        heads = this.reassemble_multimodal_auxiliary_heads
        compute_auxiliary = heads.anomaly_head is not None and (
            anomaly_labels is not None or return_anomaly_logits or return_fused_embedding
        )
        if compute_auxiliary:
            kwargs["output_hidden_states"] = True
        outputs = original_forward(input_ids=input_ids, **kwargs)
        if not compute_auxiliary:
            return outputs

        attention_mask = kwargs.get("attention_mask")
        labels = kwargs.get("labels")
        if anomaly_pool_positions is None:
            if labels is not None:
                anomaly_pool_positions = compute_anomaly_pool_positions(labels, attention_mask)
            elif attention_mask is not None:
                anomaly_pool_positions = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
            else:
                anomaly_pool_positions = input_ids.new_full((input_ids.shape[0],), input_ids.shape[1] - 1)
        hidden = outputs.hidden_states[-1]
        positions = anomaly_pool_positions.to(hidden.device).long()
        fused = hidden[torch.arange(positions.shape[0], device=positions.device), positions]
        anomaly_logits = heads.anomaly_head(fused.float()).squeeze(-1)
        # Preserve Transformers' ModelOutput type. Generation accesses cache fields
        # through attributes (for example ``outputs.past_key_values``); converting
        # it to a plain dict breaks autoregressive prediction even though training
        # and non-generative evaluation continue to work.
        outputs["anomaly_logits"] = anomaly_logits
        if return_fused_embedding:
            outputs["fused_embedding"] = fused

        task_embedding = None
        if heads.task_projection is not None:
            task_embedding = F.normalize(heads.task_projection(fused.float()), dim=-1)
            task_logits = heads.task_classifier(task_embedding)
            if active_task_masks is not None:
                active_mask = active_task_masks.bool().to(task_logits.device).clone()
                if this.training and task_labels is not None:
                    targets = task_labels.long().to(task_logits.device)
                    valid = targets.ge(0) & targets.lt(active_mask.shape[-1])
                    if valid.any():
                        rows = torch.arange(active_mask.shape[0], device=active_mask.device)[valid]
                        active_mask[rows, targets[valid]] = True
                task_logits = task_logits.masked_fill(~active_mask, torch.finfo(task_logits.dtype).min)
            action_logits = heads.action_classifier(task_embedding)
            object_logits = heads.object_classifier(task_embedding)
            disagreement_logits = heads.disagreement_head(
                torch.cat([task_embedding, anomaly_logits.detach().unsqueeze(-1)], dim=-1)
            ).squeeze(-1)
            for key, value in {
                "task_embedding": task_embedding,
                "task_logits": task_logits,
                "action_logits": action_logits,
                "object_logits": object_logits,
                "disagreement_logits": disagreement_logits,
            }.items():
                outputs[key] = value

        if labels is None:
            return outputs
        if this.reassemble_language_model_loss_weight == 0:
            total_loss = anomaly_logits.float().sum() * 0.0
        else:
            total_loss = outputs["loss"] * this.reassemble_language_model_loss_weight
        if anomaly_labels is not None:
            anomaly_loss = _masked_bce(anomaly_logits, anomaly_labels, this.reassemble_anomaly_positive_weight)
            if anomaly_loss is not None:
                total_loss = total_loss + this.reassemble_anomaly_loss_weight * anomaly_loss
        if task_embedding is not None:
            for logits, targets, weight in [
                (task_logits, task_labels, this.reassemble_task_loss_weight),
                (action_logits, action_labels, this.reassemble_action_loss_weight),
                (object_logits, object_labels, this.reassemble_object_loss_weight),
            ]:
                auxiliary_loss = _masked_ce(logits, targets)
                if auxiliary_loss is not None:
                    total_loss = total_loss + weight * auxiliary_loss
            if this.training and disagreement_labels is not None and this.reassemble_disagreement_loss_weight > 0:
                disagreement_loss = _masked_bce(disagreement_logits, disagreement_labels, 1.0)
                if disagreement_loss is not None:
                    total_loss = total_loss + this.reassemble_disagreement_loss_weight * disagreement_loss
        if this.training and teacher_scores is not None and this.reassemble_distillation_loss_weight > 0:
            temperature = max(float(this.reassemble_distillation_temperature), 1e-6)
            kd_loss = binary_soft_target_kd(
                anomaly_logits, teacher_scores, temperature, teacher_quality_weights
            )
            total_loss = total_loss + this.reassemble_distillation_loss_weight * (temperature**2) * kd_loss
        if this.training and previous_edge_scores is not None and this.reassemble_previous_edge_consistency_loss_weight > 0:
            previous = previous_edge_scores.float().to(anomaly_logits.device)
            protected = previous.ge(0)
            if protected.any():
                total_loss = total_loss + this.reassemble_previous_edge_consistency_loss_weight * F.mse_loss(
                    torch.sigmoid(anomaly_logits[protected]), previous[protected]
                )
        if this.training and replay_labels is not None and this.reassemble_replay_loss_weight > 0:
            replay_loss = replay_detection_loss(
                anomaly_logits, replay_labels, previous_edge_scores,
                positive_weight=this.reassemble_anomaly_positive_weight,
                consistency_weight=this.reassemble_replay_consistency_weight,
            )
            total_loss = total_loss + this.reassemble_replay_loss_weight * replay_loss
        if this.training and teacher_embeddings is not None and this.reassemble_relation_distillation_loss_weight > 0:
            teacher_embeddings = teacher_embeddings.float().to(fused.device)
            valid = torch.isfinite(teacher_embeddings).all(dim=1) & teacher_embeddings.norm(dim=1).gt(0)
            if valid.sum() > 1:
                student_embed = F.normalize(fused[valid].float(), dim=-1)
                teacher_embed = F.normalize(teacher_embeddings[valid], dim=-1)
                relation_loss = F.mse_loss(student_embed @ student_embed.T, teacher_embed @ teacher_embed.T)
                total_loss = total_loss + this.reassemble_relation_distillation_loss_weight * relation_loss
        outputs["loss"] = total_loss
        return outputs

    model.forward = MethodType(auxiliary_forward, model)
    if adapter_path:
        load_multimodal_auxiliary_heads(model, adapter_path)
    return model


def save_multimodal_auxiliary_heads(model: nn.Module, output_dir: str) -> Path:
    module = model.reassemble_multimodal_auxiliary_heads
    path = Path(output_dir) / "auxiliary_heads.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "input_dim": module.input_dim,
        "task_embedding_dim": module.task_embedding_dim,
        "num_task_labels": module.num_task_labels,
        "num_action_labels": module.num_action_labels,
        "num_object_labels": module.num_object_labels,
        "state_dict": {key: value.detach().cpu() for key, value in module.state_dict().items()},
    }, path)
    return path


def load_multimodal_auxiliary_heads(model: nn.Module, path_or_dir: str) -> None:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "auxiliary_heads.pt"
    if not path.exists():
        raise FileNotFoundError(f"Multimodal auxiliary checkpoint not found: {path}")
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    model.reassemble_multimodal_auxiliary_heads.load_state_dict(payload["state_dict"])

