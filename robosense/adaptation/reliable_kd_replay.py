from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any


def build_feedback_records(
    routed_rows: list[dict[str, Any]],
    *,
    cloud_threshold: float,
    minimum_confidence: float = 0.7,
) -> list[dict[str, Any]]:
    """Attach delayed labels and the registered reliable-teacher weight."""

    output = []
    for row in routed_rows:
        cloud_score = float(row["cloud_score"])
        label = int(row["anomaly"])
        cloud_prediction = int(cloud_score >= cloud_threshold)
        confidence = max(cloud_score, 1.0 - cloud_score)
        reliable = cloud_prediction == label and confidence >= minimum_confidence
        item = dict(row)
        item.update(
            teacher_score=cloud_score,
            teacher_quality_weight=1.0 if reliable else 0.0,
            replay_label=-1.0,
            previous_edge_score=-1.0,
        )
        output.append(item)
    return output


def _rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def build_replay_records(
    feedback: list[dict[str, Any]],
    seen_train: list[dict[str, Any]],
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select one unique, class/task-stratified historical record per feedback record."""

    pools: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seen_train:
        key = (int(row["anomaly"]), str(row.get("task_text", "")).strip().lower())
        pools[key].append(row)
    for rows in pools.values():
        rows.sort(key=lambda row: _rank(seed, str(row["sample_id"])))

    all_rows = sorted(seen_train, key=lambda row: _rank(seed, str(row["sample_id"])))
    used: set[str] = set()
    replay = []
    for source in feedback:
        key = (int(source["anomaly"]), str(source.get("task_text", "")).strip().lower())
        candidates = pools.get(key, []) + all_rows
        selected = next((row for row in candidates if str(row["sample_id"]) not in used), None)
        if selected is None:
            raise ValueError("not enough unique seen_train records for replay")
        used.add(str(selected["sample_id"]))
        item = dict(selected)
        item.update(
            anomaly=-1.0,
            teacher_score=-1.0,
            teacher_quality_weight=0.0,
            replay_label=float(selected["anomaly"]),
            previous_edge_score=float(selected["edge_score"]),
        )
        replay.append(item)
    return replay


REGISTERED_ADAPTATION = {
    "lora_rank": 32,
    "epochs": 3,
    "learning_rate": 2e-5,
    "effective_batch_size": 8,
    "supervised_weight": 1.0,
    "kd_weight": 0.25,
    "replay_weight": 0.5,
    "temperature": 2.0,
    "teacher_confidence": 0.7,
    "trainable": ["thinker_lora", "anomaly_head", "task_heads"],
    "frozen": ["cloud_teacher", "vision_tower", "audio_tower", "sensor_encoder", "base_weights"],
}
