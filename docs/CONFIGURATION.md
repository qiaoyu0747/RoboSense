# Configuration

The four YAML files are sanitized from the completed seed-42 training configurations. They retain LoRA rank 32, three epochs, learning rate `5e-5`, BF16, anomaly-AUPRC checkpoint selection, and the registered auxiliary loss weights.

Required environment variables:

- `ROBOSENSE_DATASET_DIR`: LLaMA-Factory datasets and `dataset_info.json`.
- `ROBOSENSE_OUTPUT_ROOT`: checkpoints and generated artifacts.
- `ROBOSENSE_EDGE_MODEL`: local path or Hugging Face identifier for Qwen2.5-Omni-3B.
- `ROBOSENSE_CLOUD_MODEL`: local path or Hugging Face identifier for Qwen2.5-Omni-7B.
- `ROBOSENSE_SENSOR_PRETRAINED`: REASSEMBLE standalone sensor-encoder checkpoint used to initialize the learned sensor tokens.

The public dataset aliases are intentionally generic. Map them to locally prepared ShareGPT records:

| Dataset | Train alias | Validation alias |
|---|---|---|
| REASSEMBLE | `reassemble_seen_train` | `reassemble_seen_val` |
| RoboFAC | `robofac_seen_train` | `robofac_seen_val` |
| FAILURE | `failure_seen_train` | `failure_seen_val` |
| ImperfectPour | `imperfectpour_seen_train` | `imperfectpour_seen_val` |

For cloud training, `train_cloud.sh` changes only the base model and output location. Modality preprocessing, labels, optimization and checkpoint-selection policy remain matched to the edge configuration.

For reliable KD with replay, continue from the selected Edge-3B-FT adapter and set:

```yaml
learning_rate: 2.0e-5
num_train_epochs: 3.0
language_model_loss_weight: 0.0
distillation_loss_weight: 0.25
distillation_temperature: 2.0
replay_loss_weight: 0.5
replay_consistency_weight: 0.5
```

Cloud-7B, vision/audio towers, the sensor encoder and base weights remain frozen. Only the Thinker LoRA and registered anomaly/task heads are updated.
