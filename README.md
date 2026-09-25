# RoboSense

Minimal research code for multimodal edge–cloud robot failure detection.

RoboSense fine-tunes a Qwen2.5-Omni-3B edge model and a Qwen2.5-Omni-7B cloud model, predicts the expected correction benefit of cloud verification, and periodically adapts the edge model using reliable cloud knowledge distillation and historical replay.

This repository contains method code and sanitized configurations only. Datasets, model weights, checkpoints, predictions, telemetry, and baseline implementations are not redistributed.

## Method

1. **Multimodal detection.** Edge-3B and Cloud-7B produce anomaly probabilities and 256-dimensional fused representations.
2. **Sensor fusion.** Robot time series are encoded into eight learned tokens. REASSEMBLE additionally includes a structured sensor summary in the prompt.
3. **Net-benefit routing.** Two calibrated classifiers estimate whether cloud inference will correct an edge error or harm a correct edge decision. The routing score is `P(fix) - P(harm)`.
4. **Cloud-to-edge adaptation.** Correct Cloud-7B predictions with confidence at least 0.7 provide KD targets. Edge-3B updates LoRA and task heads using supervised loss, KD, and seen-training replay.

Routing uses only edge probability, confidence, entropy, task-conditioned OOD distance, and a 16-dimensional PCA projection of the fused edge representation. Latency, energy, bandwidth, and payload size are evaluation outcomes, not router inputs.

## Dataset interfaces

| Dataset | Inputs |
|---|---|
| REASSEMBLE | Video, audio, sensor summary, eight learned sensor tokens, task text |
| RoboFAC | Synchronized RGB views and task text |
| FAILURE | Video, audio and task text |
| ImperfectPour | Video, eight learned proprioception tokens and task text |

## Installation

The completed experiments used Transformers 4.55.0 and LLaMA-Factory commit `7e24047c97d34c99c21faf11991959a6a3cb9134`.

```bash
python -m pip install -r requirements.txt
git clone https://github.com/hiyouga/LLaMA-Factory.git third_party/LLaMA-Factory
git -C third_party/LLaMA-Factory checkout 7e24047c97d34c99c21faf11991959a6a3cb9134
git -C third_party/LLaMA-Factory apply ../../patches/llamafactory-robosense.patch
python -m pip install -e third_party/LLaMA-Factory
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

The patch is extracted from the training implementation used in the completed experiments. It adds multimodal anomaly/task heads, sensor-token injection, KD/replay metadata collation, auxiliary checkpoint handling, and anomaly-AUPRC checkpoint selection.

## Configuration

Set external paths rather than editing source files:

```bash
export ROBOSENSE_DATA_ROOT=/path/to/datasets
export ROBOSENSE_DATASET_DIR=/path/to/llamafactory/data
export ROBOSENSE_OUTPUT_ROOT=/path/to/outputs
export ROBOSENSE_EDGE_MODEL=Qwen/Qwen2.5-Omni-3B
export ROBOSENSE_CLOUD_MODEL=Qwen/Qwen2.5-Omni-7B
export ROBOSENSE_SENSOR_PRETRAINED=/path/to/reassemble_sensor_encoder.pt
```

Register the dataset names referenced by `configs/*.yaml` in LLaMA-Factory's `dataset_info.json`. See [data format](docs/DATA_FORMAT.md) and [configuration](docs/CONFIGURATION.md).

## Commands

```bash
python -m robosense.cli audit --manifest /path/to/manifest.jsonl
bash scripts/train_edge.sh reassemble
bash scripts/train_cloud.sh reassemble
bash scripts/fit_router.sh predictions.jsonl outputs/reassemble/router.joblib
bash scripts/adapt_edge.sh predictions.jsonl outputs/reassemble/router.joblib \
  outputs/reassemble/adaptation 0.50
bash scripts/evaluate.sh final_predictions.jsonl outputs/reassemble/metrics.json
```

The generated feedback and replay records are consumed by the patched LLaMA-Factory training path with `language_model_loss_weight: 0`, `distillation_loss_weight: 0.25`, `distillation_temperature: 2`, and `replay_loss_weight: 0.5`.

## Repository contents

- `robosense/models`: exact sensor encoder, multimodal heads, KD and replay losses extracted from the completed implementation.
- `robosense/routing`: net-benefit feature construction, cross-fitted correction/harm classifiers and exact-budget routing.
- `robosense/adaptation`: reliable teacher filtering and deterministic historical replay.
- `robosense/evaluation`: detection metrics, threshold selection and grouped bootstrap utilities.
- `patches`: integration with the pinned LLaMA-Factory revision.
- `configs`: sanitized versions of the four dataset training configurations.

## Third-party software

RoboSense builds on [Qwen2.5-Omni](https://huggingface.co/collections/Qwen/qwen25-omni-67de7e5e3ba8e47b585c5eb8). Their respective licenses and model terms continue to apply. Dataset licenses are not changed by this code release.

## Citation

The paper citation will be added after publication. For the software release, use [citation.bib](citation.bib).

## License

RoboSense-specific code is released under the MIT License. Third-party code and models remain under their original terms.
