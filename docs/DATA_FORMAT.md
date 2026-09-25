# Data format

RoboSense uses one JSON object per execution. All assets derived from the same recording must retain one `recording_id` and one split.

```json
{
  "dataset": "reassemble",
  "sample_id": "example-0001",
  "recording_id": "recording-001",
  "split": "seen_train",
  "anomaly": 0,
  "task_text": "Insert Ethernet",
  "video_paths": ["reassemble/videos/example-0001.mp4"],
  "audio_path": "reassemble/audio/example-0001.wav",
  "sensor_path": "reassemble/sensors/example-0001.npz",
  "sensor_summary": "Robot sensor summary: ...",
  "metadata": {"action": "insert", "object": "ethernet"}
}
```

Supported splits are `seen_train`, `seen_val`, `adaptation_stream`, `ood_gate`, `edge_distill_base`, `id_test_full`, `id_test_matched`, `ood_test`, and `realworld_ood_test`.

Sensor NPZ files contain:

- `sensor`: normalized `T × C` float tensor;
- `mask`: length-`T` validity mask.

REASSEMBLE uses `C=52`; ImperfectPour uses `C=44`. The prompt must contain exactly eight `<|fim_pad|>` placeholders when sensor-token fusion is enabled.

Router prediction records additionally contain `edge_score`, `cloud_score`, endpoint thresholds, a 256-dimensional `fused_embedding`, and the sample's task and recording identifiers. Final-test rows are never used when fitting representations, classifiers, calibration or thresholds.
