from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


TRAIN_SPLITS = {"seen_train", "adaptation_stream", "edge_distill_base"}
SELECTION_SPLITS = {"seen_val", "ood_gate"}
FINAL_SPLITS = {"id_test_full", "id_test_matched", "ood_test", "realworld_ood_test"}
ALL_SPLITS = TRAIN_SPLITS | SELECTION_SPLITS | FINAL_SPLITS


@dataclass(frozen=True)
class SampleRecord:
    dataset: str
    sample_id: str
    recording_id: str
    split: str
    anomaly: int
    task_text: str
    video_paths: tuple[str, ...] = ()
    audio_path: str | None = None
    sensor_path: str | None = None
    sensor_summary: str | None = None
    failure_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.sample_id:
            errors.append("empty sample_id")
        if not self.recording_id:
            errors.append(f"{self.sample_id}: empty recording_id")
        if self.split not in ALL_SPLITS:
            errors.append(f"{self.sample_id}: unsupported split {self.split}")
        if self.anomaly not in (0, 1):
            errors.append(f"{self.sample_id}: anomaly must be 0 or 1")
        if not self.task_text.strip():
            errors.append(f"{self.sample_id}: empty task text")
        return errors

    @property
    def allow_training(self) -> bool:
        return self.split in TRAIN_SPLITS

    @property
    def allow_selection(self) -> bool:
        return self.split in SELECTION_SPLITS

    @property
    def final_test(self) -> bool:
        return self.split in FINAL_SPLITS

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["video_paths"] = list(self.video_paths)
        row.update(
            allow_training=self.allow_training,
            allow_selection=self.allow_selection,
            final_test=self.final_test,
        )
        return row


def _resolve(path: str | None, root: Path) -> str | None:
    if not path:
        return None
    value = Path(os.path.expandvars(path)).expanduser()
    return str(value if value.is_absolute() else root / value)


def load_manifest(path: str | Path, data_root: str | Path | None = None) -> list[SampleRecord]:
    manifest = Path(path)
    root = Path(data_root or os.environ.get("ROBOSENSE_DATA_ROOT", manifest.parent))
    records: list[SampleRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        record = SampleRecord(
            dataset=str(row["dataset"]), sample_id=str(row["sample_id"]),
            recording_id=str(row["recording_id"]), split=str(row["split"]),
            anomaly=int(row["anomaly"]), task_text=str(row["task_text"]),
            video_paths=tuple(_resolve(value, root) for value in row.get("video_paths", [])),
            audio_path=_resolve(row.get("audio_path"), root),
            sensor_path=_resolve(row.get("sensor_path"), root),
            sensor_summary=row.get("sensor_summary"), failure_type=row.get("failure_type"),
            metadata=dict(row.get("metadata", {})),
        )
        errors = record.validate()
        if errors:
            raise ValueError(f"line {line_number}: {'; '.join(errors)}")
        if record.sample_id in seen:
            raise ValueError(f"line {line_number}: duplicate sample_id {record.sample_id}")
        seen.add(record.sample_id)
        records.append(record)
    return records


def assert_recording_isolation(records: Iterable[SampleRecord]) -> None:
    memberships: dict[str, set[str]] = {}
    for record in records:
        memberships.setdefault(record.recording_id, set()).add(record.split)
    for recording_id, splits in memberships.items():
        development = splits & (TRAIN_SPLITS | SELECTION_SPLITS)
        final = splits & FINAL_SPLITS
        if development and final:
            raise ValueError(f"recording leakage for {recording_id}: {sorted(splits)}")
