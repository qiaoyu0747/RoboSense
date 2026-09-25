"""Learned time-series token injection for REASSEMBLE sensor fusion."""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F

IGNORE_INDEX = -100
from .feedback_distillation import binary_soft_target_kd, replay_detection_loss


class SensorTokenEncoder(nn.Module):
    """Compress fixed-rate robot signals into a small set of LLM embeddings."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_tokens: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        temporal_stride: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.temporal_stride = temporal_stride
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.query_tokens = nn.Parameter(torch.randn(num_tokens, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward_features(self, sensor: torch.Tensor, sensor_mask: torch.Tensor) -> torch.Tensor:
        if self.temporal_stride > 1:
            sensor = sensor[:, :: self.temporal_stride]
            sensor_mask = sensor_mask[:, :: self.temporal_stride]
        hidden = self.encoder(self.input_proj(sensor), src_key_padding_mask=~sensor_mask.bool())
        queries = self.query_tokens.unsqueeze(0).expand(sensor.shape[0], -1, -1)
        tokens, _ = self.cross_attn(
            queries,
            hidden,
            hidden,
            key_padding_mask=~sensor_mask.bool(),
            need_weights=False,
        )
        return self.norm(tokens)

    def forward(self, sensor: torch.Tensor, sensor_mask: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.forward_features(sensor, sensor_mask))


def _text_hidden_size(model: nn.Module) -> int:
    config = model.config
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    elif hasattr(config, "text_config"):
        config = config.text_config
    return int(config.hidden_size)


def compute_anomaly_pool_positions(labels: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return the final prompt position before the first supervised answer token."""

    supervised = labels.ne(IGNORE_INDEX)
    has_supervision = supervised.any(dim=1)
    first_supervised = supervised.long().argmax(dim=1)
    positions = first_supervised.sub(1).clamp_min(0)
    if attention_mask is not None:
        fallback = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
    else:
        fallback = labels.new_full((labels.shape[0],), labels.shape[1] - 1)
    return torch.where(has_supervision, positions, fallback)


def _masked_ce(logits: torch.Tensor, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if labels is None:
        return None
    labels = labels.long().to(logits.device)
    valid = labels.ge(0)
    if not valid.any():
        return None
    return F.cross_entropy(logits[valid], labels[valid])


def attach_sensor_fusion(
    model: nn.Module,
    input_dim: int = 52,
    hidden_dim: int = 256,
    num_tokens: int = 8,
    adapter_path: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    freeze_sensor_encoder: bool = False,
    sensor_learning_rate: Optional[float] = None,
    anomaly_head: bool = False,
    export_fused_embeddings: bool = False,
    task_aware_head: bool = False,
    export_task_outputs: bool = False,
    num_task_labels: int = 68,
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
    sensor_loss_weight: float = 0.5,
    temporal_stride: int = 1,
    distillation_loss_weight: float = 0.0,
    distillation_temperature: float = 1.0,
    consistency_loss_weight: float = 0.0,
    consistency_sensor_dropout_probability: float = 0.0,
    language_model_loss_weight: float = 1.0,
    replay_loss_weight: float = 0.0,
    replay_consistency_weight: float = 0.5,
) -> nn.Module:
    """Attach a trainable sensor encoder and patch forward to inject its tokens."""

    if hasattr(model, "reassemble_sensor_fusion"):
        return model
    fusion = SensorTokenEncoder(input_dim, _text_hidden_size(model), hidden_dim, num_tokens, temporal_stride=temporal_stride)
    model.add_module("reassemble_sensor_fusion", fusion)
    if anomaly_head:
        model.add_module("reassemble_anomaly_head", nn.Linear(_text_hidden_size(model), 1))
        model.add_module("reassemble_sensor_classifier", nn.Linear(hidden_dim, 1))
    if task_aware_head:
        text_dim = _text_hidden_size(model)
        model.add_module("reassemble_task_projection", nn.Sequential(nn.Linear(text_dim, task_embedding_dim), nn.LayerNorm(task_embedding_dim)))
        model.add_module("reassemble_task_classifier", nn.Linear(task_embedding_dim, num_task_labels))
        model.add_module("reassemble_action_classifier", nn.Linear(task_embedding_dim, num_action_labels))
        model.add_module("reassemble_object_classifier", nn.Linear(task_embedding_dim, num_object_labels))
        model.add_module("reassemble_disagreement_head", nn.Linear(task_embedding_dim + 1, 1))
    model.reassemble_anomaly_loss_weight = anomaly_loss_weight
    model.reassemble_anomaly_positive_weight = anomaly_positive_weight
    model.reassemble_export_fused_embeddings = export_fused_embeddings
    model.reassemble_export_task_outputs = export_task_outputs
    model.reassemble_task_loss_weight = task_loss_weight
    model.reassemble_action_loss_weight = action_loss_weight
    model.reassemble_object_loss_weight = object_loss_weight
    model.reassemble_relation_distillation_loss_weight = relation_distillation_loss_weight
    model.reassemble_previous_edge_consistency_loss_weight = previous_edge_consistency_loss_weight
    model.reassemble_disagreement_loss_weight = disagreement_loss_weight
    model.reassemble_relation_memory_student = None
    model.reassemble_relation_memory_teacher = None
    model.reassemble_sensor_loss_weight = sensor_loss_weight
    model.reassemble_distillation_loss_weight = distillation_loss_weight
    model.reassemble_distillation_temperature = distillation_temperature
    model.reassemble_consistency_loss_weight = consistency_loss_weight
    model.reassemble_consistency_sensor_dropout_probability = consistency_sensor_dropout_probability
    model.reassemble_language_model_loss_weight = language_model_loss_weight
    model.reassemble_replay_loss_weight = replay_loss_weight
    model.reassemble_replay_consistency_weight = replay_consistency_weight
    model.reassemble_sensor_learning_rate = sensor_learning_rate
    original_forward = model.forward

    def sensor_forward(
        this,
        input_ids=None,
        sensor=None,
        sensor_mask=None,
        sensor_positions=None,
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
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        can_inject = (
            sensor is not None
            and sensor_mask is not None
            and sensor_positions is not None
            and input_ids is not None
            and sensor_positions.shape == input_ids.shape
        )
        if can_inject:
            counts = sensor_positions.sum(dim=1)
            if not torch.all(counts == this.reassemble_sensor_fusion.num_tokens):
                raise ValueError(
                    f"Expected {this.reassemble_sensor_fusion.num_tokens} sensor positions, got {counts.tolist()}."
                )
            if inputs_embeds is None:
                inputs_embeds = this.get_input_embeddings()(input_ids)
            sensor_features = this.reassemble_sensor_fusion.forward_features(sensor.float(), sensor_mask)
            sensor_tokens = this.reassemble_sensor_fusion.output_proj(sensor_features)
            inputs_embeds = inputs_embeds.clone()
            for batch_index in range(inputs_embeds.shape[0]):
                inputs_embeds[batch_index, sensor_positions[batch_index]] = sensor_tokens[batch_index].to(
                    inputs_embeds.dtype
                )
            compute_anomaly = (
                hasattr(this, "reassemble_anomaly_head")
                and anomaly_labels is not None
                and (kwargs.get("labels") is not None or return_anomaly_logits)
            )
            if compute_anomaly:
                kwargs["output_hidden_states"] = True
            outputs = original_forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
            if compute_anomaly:
                attention_mask = kwargs.get("attention_mask")
                if anomaly_pool_positions is None:
                    labels = kwargs.get("labels")
                    if labels is not None:
                        anomaly_pool_positions = compute_anomaly_pool_positions(labels, attention_mask)
                    elif attention_mask is not None:
                        anomaly_pool_positions = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
                    else:
                        anomaly_pool_positions = input_ids.new_full((input_ids.shape[0],), input_ids.shape[1] - 1)
                anomaly_pool_positions = anomaly_pool_positions.to(outputs.hidden_states[-1].device).long()
                fused = outputs.hidden_states[-1][
                    torch.arange(anomaly_pool_positions.shape[0], device=anomaly_pool_positions.device),
                    anomaly_pool_positions,
                ]
                anomaly_logits = this.reassemble_anomaly_head(fused.float()).squeeze(-1)
                sensor_logits = this.reassemble_sensor_classifier(sensor_features.mean(dim=1).float()).squeeze(-1)
                labels_float = anomaly_labels.float().to(anomaly_logits.device)
                # DDP reconstructs ModelOutput instances from their mapping keys. Custom
                # keys are not accepted by Qwen's typed output constructor, so return a
                # plain mapping whenever the anomaly head adds fields.
                outputs = dict(outputs)
                outputs["anomaly_logits"] = anomaly_logits
                if return_fused_embedding:
                    outputs["fused_embedding"] = fused
                outputs["sensor_anomaly_logits"] = sensor_logits
                task_embedding = None
                if hasattr(this, "reassemble_task_projection"):
                    task_embedding = F.normalize(this.reassemble_task_projection(fused.float()), dim=-1)
                    task_logits = this.reassemble_task_classifier(task_embedding)
                    if active_task_masks is not None:
                        mask = active_task_masks.bool().to(task_logits.device).clone()
                        # OOD feedback can introduce a labelled task that was not in
                        # the initial checkpoint's active-task set.  Keep the mask for
                        # all other classes, but never mask the supervised target: CE
                        # on a masked target otherwise becomes +inf in BF16.
                        if this.training and task_labels is not None:
                            targets = task_labels.long().to(task_logits.device)
                            valid = targets.ge(0) & targets.lt(mask.shape[-1])
                            if valid.any():
                                rows = torch.arange(mask.shape[0], device=mask.device)[valid]
                                mask[rows, targets[valid]] = True
                        task_logits = task_logits.masked_fill(~mask, torch.finfo(task_logits.dtype).min)
                    action_logits = this.reassemble_action_classifier(task_embedding)
                    object_logits = this.reassemble_object_classifier(task_embedding)
                    disagreement_logits = this.reassemble_disagreement_head(
                        torch.cat([task_embedding, anomaly_logits.detach().unsqueeze(-1)], dim=-1)
                    ).squeeze(-1)
                    outputs.update({
                        "task_embedding": task_embedding,
                        "task_logits": task_logits,
                        "action_logits": action_logits,
                        "object_logits": object_logits,
                        "disagreement_logits": disagreement_logits,
                    })
                if kwargs.get("labels") is not None:
                    if this.reassemble_language_model_loss_weight == 0:
                        # Multiplying a NaN/Inf language loss by zero is still NaN.
                        # The feedback objective intentionally disables generation,
                        # so construct a graph-connected exact zero instead.
                        outputs["loss"] = anomaly_logits.float().sum() * 0.0
                    else:
                        outputs["loss"] = outputs["loss"] * this.reassemble_language_model_loss_weight
                    valid_supervised = labels_float.ge(0)
                    if valid_supervised.any():
                        positive_weight = anomaly_logits.new_tensor(float(this.reassemble_anomaly_positive_weight))
                        anomaly_loss = F.binary_cross_entropy_with_logits(
                            anomaly_logits[valid_supervised], labels_float[valid_supervised], pos_weight=positive_weight
                        )
                        sensor_loss = F.binary_cross_entropy_with_logits(
                            sensor_logits[valid_supervised], labels_float[valid_supervised], pos_weight=positive_weight
                        )
                        outputs["loss"] = (
                            outputs["loss"]
                            + this.reassemble_anomaly_loss_weight * anomaly_loss
                            + this.reassemble_sensor_loss_weight * sensor_loss
                        )
                    if task_embedding is not None:
                        for logits, targets, weight in (
                            (task_logits, task_labels, this.reassemble_task_loss_weight),
                            (action_logits, action_labels, this.reassemble_action_loss_weight),
                            (object_logits, object_labels, this.reassemble_object_loss_weight),
                        ):
                            auxiliary_loss = _masked_ce(logits, targets)
                            if auxiliary_loss is not None:
                                outputs["loss"] = outputs["loss"] + weight * auxiliary_loss
                        if this.training and disagreement_labels is not None and this.reassemble_disagreement_loss_weight > 0:
                            outputs["loss"] = outputs["loss"] + this.reassemble_disagreement_loss_weight * F.binary_cross_entropy_with_logits(
                                disagreement_logits, disagreement_labels.float().to(disagreement_logits.device)
                            )
                    if this.training and teacher_scores is not None and this.reassemble_distillation_loss_weight > 0:
                        temperature = max(float(this.reassemble_distillation_temperature), 1e-6)
                        distillation_loss = binary_soft_target_kd(
                            anomaly_logits, teacher_scores, temperature, teacher_quality_weights
                        )
                        outputs["loss"] = (
                            outputs["loss"]
                            + this.reassemble_distillation_loss_weight * (temperature**2) * distillation_loss
                        )
                    if this.training and previous_edge_scores is not None and this.reassemble_previous_edge_consistency_loss_weight > 0:
                        previous = previous_edge_scores.float().to(anomaly_logits.device)
                        protected = previous >= 0
                        if protected.any():
                            outputs["loss"] = outputs["loss"] + this.reassemble_previous_edge_consistency_loss_weight * F.mse_loss(
                                torch.sigmoid(anomaly_logits[protected]), previous[protected]
                            )
                    if this.training and replay_labels is not None and this.reassemble_replay_loss_weight > 0:
                        replay_loss = replay_detection_loss(
                            anomaly_logits, replay_labels, previous_edge_scores,
                            positive_weight=this.reassemble_anomaly_positive_weight,
                            consistency_weight=this.reassemble_replay_consistency_weight,
                        )
                        outputs["loss"] = outputs["loss"] + this.reassemble_replay_loss_weight * replay_loss
                    if this.training and teacher_embeddings is not None and this.reassemble_relation_distillation_loss_weight > 0:
                        student_embed = F.normalize(fused.float(), dim=-1)
                        teacher_embed = F.normalize(teacher_embeddings.float().to(student_embed.device), dim=-1)
                        relation_loss = student_embed.sum() * 0
                        if student_embed.shape[0] > 1:
                            student_relation = student_embed @ student_embed.T
                            teacher_relation = teacher_embed @ teacher_embed.T
                            relation_loss = relation_loss + F.mse_loss(student_relation, teacher_relation)
                        if this.reassemble_relation_memory_student is not None:
                            memory_student = this.reassemble_relation_memory_student.to(student_embed.device)
                            memory_teacher = this.reassemble_relation_memory_teacher.to(student_embed.device)
                            relation_loss = relation_loss + F.mse_loss(
                                student_embed @ memory_student.T, teacher_embed @ memory_teacher.T
                            )
                        with torch.no_grad():
                            student_memory = student_embed.detach()
                            teacher_memory = teacher_embed.detach()
                            if this.reassemble_relation_memory_student is not None:
                                student_memory = torch.cat(
                                    [this.reassemble_relation_memory_student.to(student_memory.device), student_memory], dim=0
                                )
                                teacher_memory = torch.cat(
                                    [this.reassemble_relation_memory_teacher.to(teacher_memory.device), teacher_memory], dim=0
                                )
                            this.reassemble_relation_memory_student = student_memory[-32:]
                            this.reassemble_relation_memory_teacher = teacher_memory[-32:]
                        outputs["loss"] = outputs["loss"] + this.reassemble_relation_distillation_loss_weight * relation_loss
                    if (
                        this.reassemble_consistency_loss_weight > 0
                        and this.reassemble_consistency_sensor_dropout_probability > 0
                        and torch.rand((), device=anomaly_logits.device)
                        < this.reassemble_consistency_sensor_dropout_probability
                    ):
                        dropped_features = this.reassemble_sensor_fusion.forward_features(
                            torch.zeros_like(sensor).float(), torch.ones_like(sensor_mask).bool()
                        )
                        dropped_tokens = this.reassemble_sensor_fusion.output_proj(dropped_features)
                        dropped_embeds = inputs_embeds.clone()
                        for batch_index in range(dropped_embeds.shape[0]):
                            dropped_embeds[batch_index, sensor_positions[batch_index]] = dropped_tokens[batch_index].to(
                                dropped_embeds.dtype
                            )
                        consistency_kwargs = {key: value for key, value in kwargs.items() if key != "labels"}
                        consistency_kwargs["output_hidden_states"] = True
                        dropped_outputs = original_forward(
                            input_ids=input_ids, inputs_embeds=dropped_embeds, **consistency_kwargs
                        )
                        dropped_fused = dropped_outputs.hidden_states[-1][
                            torch.arange(anomaly_pool_positions.shape[0], device=anomaly_pool_positions.device),
                            anomaly_pool_positions,
                        ]
                        dropped_logits = this.reassemble_anomaly_head(dropped_fused.float()).squeeze(-1)
                        consistency_loss = F.mse_loss(torch.sigmoid(dropped_logits), torch.sigmoid(anomaly_logits).detach())
                        outputs["loss"] = outputs["loss"] + this.reassemble_consistency_loss_weight * consistency_loss
            return outputs
        if inputs_embeds is not None:
            return original_forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        return original_forward(input_ids=input_ids, **kwargs)

    model.forward = MethodType(sensor_forward, model)
    if pretrained_path:
        load_pretrained_sensor_encoder(model, pretrained_path)
    if adapter_path:
        load_sensor_fusion(model, adapter_path)
    if freeze_sensor_encoder:
        for parameter in model.reassemble_sensor_fusion.parameters():
            parameter.requires_grad_(False)
    return model


def save_sensor_fusion(model: nn.Module, output_dir: str) -> Path:
    module = getattr(model, "reassemble_sensor_fusion")
    path = Path(output_dir) / "sensor_fusion.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "input_dim": module.input_dim,
        "output_dim": module.output_dim,
        "hidden_dim": module.hidden_dim,
        "num_tokens": module.num_tokens,
        "temporal_stride": module.temporal_stride,
        "state_dict": {key: value.detach().cpu() for key, value in module.state_dict().items()},
    }
    if hasattr(model, "reassemble_anomaly_head"):
        payload["anomaly_head_state_dict"] = {
            key: value.detach().cpu() for key, value in model.reassemble_anomaly_head.state_dict().items()
        }
    if hasattr(model, "reassemble_sensor_classifier"):
        payload["sensor_classifier_state_dict"] = {
            key: value.detach().cpu() for key, value in model.reassemble_sensor_classifier.state_dict().items()
        }
    for name in [
        "reassemble_task_projection", "reassemble_task_classifier", "reassemble_action_classifier",
        "reassemble_object_classifier", "reassemble_disagreement_head",
    ]:
        if hasattr(model, name):
            payload[f"{name}_state_dict"] = {
                key: value.detach().cpu() for key, value in getattr(model, name).state_dict().items()
            }
    torch.save(payload, path)
    return path


def load_sensor_fusion(model: nn.Module, path_or_dir: str) -> None:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "sensor_fusion.pt"
    if not path.exists():
        raise FileNotFoundError(f"Sensor fusion checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    module = getattr(model, "reassemble_sensor_fusion")
    module.load_state_dict(payload["state_dict"])
    if "anomaly_head_state_dict" in payload and hasattr(model, "reassemble_anomaly_head"):
        model.reassemble_anomaly_head.load_state_dict(payload["anomaly_head_state_dict"])
    if "sensor_classifier_state_dict" in payload and hasattr(model, "reassemble_sensor_classifier"):
        model.reassemble_sensor_classifier.load_state_dict(payload["sensor_classifier_state_dict"])
    for name in [
        "reassemble_task_projection", "reassemble_task_classifier", "reassemble_action_classifier",
        "reassemble_object_classifier", "reassemble_disagreement_head",
    ]:
        key = f"{name}_state_dict"
        if key in payload and hasattr(model, name):
            getattr(model, name).load_state_dict(payload[key])


def load_pretrained_sensor_encoder(model: nn.Module, path: str) -> None:
    """Load shape-compatible standalone sensor-encoder weights."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload.get("sensor_encoder_state_dict", payload.get("state_dict", payload))
    module = getattr(model, "reassemble_sensor_fusion")
    target = module.state_dict()
    compatible = {key: value for key, value in source.items() if key in target and target[key].shape == value.shape}
    if not compatible:
        raise ValueError(f"No compatible sensor-encoder weights found in {path}.")
    module.load_state_dict(compatible, strict=False)

