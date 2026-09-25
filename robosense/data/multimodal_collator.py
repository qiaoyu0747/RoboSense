from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


FLOAT_FIELDS = (
    "anomaly", "teacher_score", "previous_edge_score", "teacher_embedding",
    "teacher_quality_weight", "disagreement_label", "replay_label",
)
LONG_FIELDS = ("task_label", "action_label", "object_label")
OUTPUT_NAMES = {
    "anomaly": "anomaly_labels",
    "teacher_score": "teacher_scores",
    "previous_edge_score": "previous_edge_scores",
    "teacher_embedding": "teacher_embeddings",
    "teacher_quality_weight": "teacher_quality_weights",
    "disagreement_label": "disagreement_labels",
    "replay_label": "replay_labels",
    "task_label": "task_labels",
    "action_label": "action_labels",
    "object_label": "object_labels",
}


def collate_multimodal_metadata(
    features: list[dict[str, Any]],
    base_collator: Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]],
    *,
    sensor_token_id: int | None = None,
    sensor_num_tokens: int = 8,
) -> dict[str, torch.Tensor]:
    """Existing RoboSense metadata collation, independent of a dataset backend."""

    rows = [dict(feature) for feature in features]
    metadata = {name: [row.pop(name, None) for row in rows] for name in (*FLOAT_FIELDS, *LONG_FIELDS)}
    active_masks = [row.pop("active_task_mask", None) for row in rows]
    sensor_paths = [row.pop("sensor", None) for row in rows]
    batch = base_collator(rows)

    for name in FLOAT_FIELDS:
        values = metadata[name]
        if all(value is not None for value in values):
            batch[OUTPUT_NAMES[name]] = torch.tensor(values, dtype=torch.float32)
    for name in LONG_FIELDS:
        values = metadata[name]
        if all(value is not None for value in values):
            batch[OUTPUT_NAMES[name]] = torch.tensor(values, dtype=torch.long)
    if all(value is not None for value in active_masks):
        batch["active_task_masks"] = torch.tensor(active_masks, dtype=torch.bool)

    has_sensor = any(path is not None for path in sensor_paths)
    if has_sensor:
        if not all(path is not None for path in sensor_paths) or sensor_token_id is None:
            raise ValueError("sensor fusion requires one sensor file per sample and sensor_token_id")
        tensors, masks = [], []
        for sensor_path in sensor_paths:
            with np.load(Path(sensor_path)) as payload:
                tensors.append(torch.from_numpy(payload["sensor"].copy()))
                masks.append(torch.from_numpy(payload["mask"].copy()))
        batch["sensor"] = torch.stack(tensors)
        batch["sensor_mask"] = torch.stack(masks).bool()
        batch["sensor_positions"] = batch["input_ids"].eq(sensor_token_id)
        counts = batch["sensor_positions"].sum(dim=1)
        if not torch.all(counts == sensor_num_tokens):
            raise ValueError(f"expected {sensor_num_tokens} sensor placeholders, got {counts.tolist()}")
    return batch
