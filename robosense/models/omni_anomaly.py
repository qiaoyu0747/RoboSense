"""Public entry points for the exact anomaly modules used in RoboSense."""

from .auxiliary_heads import (
    attach_multimodal_auxiliary_heads,
    load_multimodal_auxiliary_heads,
    save_multimodal_auxiliary_heads,
)
from .sensor_encoder import attach_sensor_fusion, load_sensor_fusion, save_sensor_fusion

__all__ = [
    "attach_sensor_fusion", "load_sensor_fusion", "save_sensor_fusion",
    "attach_multimodal_auxiliary_heads", "load_multimodal_auxiliary_heads",
    "save_multimodal_auxiliary_heads",
]
