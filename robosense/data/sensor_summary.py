from __future__ import annotations

import math

import numpy as np
import pandas as pd



SENSOR_FIELDS = {
    "duration": "duration_sec",
    "force_x_absmax": "robot_state__measured_force__d0__absmax",
    "force_y_absmax": "robot_state__measured_force__d1__absmax",
    "force_z_absmax": "robot_state__measured_force__d2__absmax",
    "force_x_range": "robot_state__measured_force__d0__range",
    "force_y_range": "robot_state__measured_force__d1__range",
    "force_z_range": "robot_state__measured_force__d2__range",
    "torque_x_absmax": "robot_state__measured_torque__d0__absmax",
    "torque_y_absmax": "robot_state__measured_torque__d1__absmax",
    "torque_z_absmax": "robot_state__measured_torque__d2__absmax",
    "gripper_left_initial": "robot_state__gripper_positions__d0__first",
    "gripper_right_initial": "robot_state__gripper_positions__d1__first",
    "gripper_left_final": "robot_state__gripper_positions__d0__last",
    "gripper_right_final": "robot_state__gripper_positions__d1__last",
    "gripper_left_delta": "robot_state__gripper_positions__d0__delta",
    "gripper_right_delta": "robot_state__gripper_positions__d1__delta",
    "gripper_left_range": "robot_state__gripper_positions__d0__range",
    "gripper_right_range": "robot_state__gripper_positions__d1__range",
    "pose_x_delta": "robot_state__pose__d0__delta",
    "pose_y_delta": "robot_state__pose__d1__delta",
    "pose_z_delta": "robot_state__pose__d2__delta",
    "velocity_x_absmax": "robot_state__velocity__d0__absmax",
    "velocity_y_absmax": "robot_state__velocity__d1__absmax",
    "velocity_z_absmax": "robot_state__velocity__d2__absmax",
}


def fmt(value: float, digits: int = 4) -> str:
    if value is None or pd.isna(value) or not math.isfinite(float(value)):
        return "missing"
    return f"{float(value):.{digits}f}"


def label_relative(z: float) -> str:
    if pd.isna(z):
        return "not available"
    az = abs(float(z))
    if az >= 3.0:
        return "extremely unusual"
    if az >= 2.0:
        return "unusual"
    if az >= 1.0:
        return "somewhat different"
    return "typical"


def build_reference_stats(features: pd.DataFrame) -> pd.DataFrame:
    train_normal = features[(features["split"] == "train") & (features["anomaly"] == 0)].copy()
    rows = []
    for (action, obj), group in train_normal.groupby(["action", "object"]):
        row = {"action": action, "object": obj}
        for name, col in SENSOR_FIELDS.items():
            if col not in group:
                continue
            values = group[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) == 0:
                row[f"{name}__mean"] = np.nan
                row[f"{name}__std"] = np.nan
            else:
                row[f"{name}__mean"] = float(values.mean())
                row[f"{name}__std"] = float(values.std(ddof=0) or 1e-6)
        rows.append(row)
    return pd.DataFrame(rows)


def feature_value(row, col: str) -> float:
    return float(row[col]) if col in row and not pd.isna(row[col]) else np.nan


def make_sensor_summary(row, ref_row=None) -> str:
    force = [
        feature_value(row, SENSOR_FIELDS["force_x_absmax"]),
        feature_value(row, SENSOR_FIELDS["force_y_absmax"]),
        feature_value(row, SENSOR_FIELDS["force_z_absmax"]),
    ]
    torque = [
        feature_value(row, SENSOR_FIELDS["torque_x_absmax"]),
        feature_value(row, SENSOR_FIELDS["torque_y_absmax"]),
        feature_value(row, SENSOR_FIELDS["torque_z_absmax"]),
    ]
    grip_initial = [
        feature_value(row, SENSOR_FIELDS["gripper_left_initial"]),
        feature_value(row, SENSOR_FIELDS["gripper_right_initial"]),
    ]
    grip_final = [
        feature_value(row, SENSOR_FIELDS["gripper_left_final"]),
        feature_value(row, SENSOR_FIELDS["gripper_right_final"]),
    ]
    grip_delta = [
        feature_value(row, SENSOR_FIELDS["gripper_left_delta"]),
        feature_value(row, SENSOR_FIELDS["gripper_right_delta"]),
    ]
    pose_delta = [
        feature_value(row, SENSOR_FIELDS["pose_x_delta"]),
        feature_value(row, SENSOR_FIELDS["pose_y_delta"]),
        feature_value(row, SENSOR_FIELDS["pose_z_delta"]),
    ]
    velocity_peak = [
        feature_value(row, SENSOR_FIELDS["velocity_x_absmax"]),
        feature_value(row, SENSOR_FIELDS["velocity_y_absmax"]),
        feature_value(row, SENSOR_FIELDS["velocity_z_absmax"]),
    ]

    lines = [
        "Robot sensor summary:",
        f"- duration_sec: {fmt(feature_value(row, SENSOR_FIELDS['duration']))}",
        f"- measured_force_absmax_xyz: [{fmt(force[0])}, {fmt(force[1])}, {fmt(force[2])}]",
        f"- measured_torque_absmax_xyz: [{fmt(torque[0])}, {fmt(torque[1])}, {fmt(torque[2])}]",
        f"- gripper_initial_lr: [{fmt(grip_initial[0])}, {fmt(grip_initial[1])}]",
        f"- gripper_final_lr: [{fmt(grip_final[0])}, {fmt(grip_final[1])}]",
        f"- gripper_delta_lr: [{fmt(grip_delta[0])}, {fmt(grip_delta[1])}]",
        f"- end_effector_position_delta_xyz: [{fmt(pose_delta[0])}, {fmt(pose_delta[1])}, {fmt(pose_delta[2])}]",
        f"- velocity_absmax_xyz: [{fmt(velocity_peak[0])}, {fmt(velocity_peak[1])}, {fmt(velocity_peak[2])}]",
    ]

    if ref_row is not None and len(ref_row) == 1:
        ref = ref_row.iloc[0]
        comparisons = []
        for name in [
            "duration",
            "force_z_absmax",
            "torque_z_absmax",
            "gripper_left_final",
            "gripper_right_final",
            "pose_z_delta",
        ]:
            col = SENSOR_FIELDS[name]
            value = feature_value(row, col)
            mean = ref.get(f"{name}__mean", np.nan)
            std = ref.get(f"{name}__std", np.nan)
            z = (value - mean) / std if not pd.isna(value) and not pd.isna(mean) and std else np.nan
            comparisons.append(f"{name}: {label_relative(z)} (z={fmt(z, 2)})")
        lines.append("- compared_to_successful_same_task: " + "; ".join(comparisons))

    return "\n".join(lines)


def answer_for(row, structured_output: bool = False) -> str:
    label = "anomalous" if int(row["anomaly"]) == 1 else "normal"
    reason = "The action failed." if int(row["anomaly"]) == 1 else "The action succeeded."

