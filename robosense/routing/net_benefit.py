from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def _decision_center(score: float, threshold: float) -> float:
    if score <= threshold:
        return 0.5 * score / threshold
    return 0.5 + 0.5 * (score - threshold) / (1.0 - threshold)


def exact_mask(scores: np.ndarray, budget: float, sample_ids: list[str]) -> np.ndarray:
    count = math.floor(float(budget) * len(scores))
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), sample_ids[index]))
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:count]] = True
    return mask


def correction_targets(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([row["anomaly"] for row in rows], dtype=bool)
    edge = np.asarray([row["edge_score"] >= row["edge_threshold"] for row in rows])
    cloud = np.asarray([row["cloud_score"] >= row["cloud_threshold"] for row in rows])
    return ((edge != labels) & (cloud == labels)).astype(int), ((edge == labels) & (cloud != labels)).astype(int)


@dataclass
class _BinaryModel:
    kind: str
    probability: float | None = None
    scaler_mean: list[float] | None = None
    scaler_scale: list[float] | None = None
    coef: list[float] | None = None
    intercept: float | None = None

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, c: float) -> "_BinaryModel":
        if np.unique(y).size < 2:
            return cls("beta_constant", probability=float((y.sum() + 0.5) / (len(y) + 1.0)))
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=42)
        model.fit(scaler.transform(x), y)
        return cls("logistic", scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                   coef=model.coef_[0].tolist(), intercept=float(model.intercept_[0]))

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "beta_constant":
            return np.full(len(x), self.probability)
        z = (x - np.asarray(self.scaler_mean)) / np.asarray(self.scaler_scale)
        logits = z @ np.asarray(self.coef) + float(self.intercept)
        return 1 / (1 + np.exp(-np.clip(logits, -50, 50)))


@dataclass
class _Calibration:
    kind: str
    coef: float | None = None
    intercept: float | None = None
    x: list[float] | None = None
    y: list[float] | None = None

    @classmethod
    def fit(cls, probability: np.ndarray, labels: np.ndarray, method: str) -> "_Calibration":
        if np.unique(labels).size < 2:
            return cls("constant", intercept=float((labels.sum() + 0.5) / (len(labels) + 1.0)))
        if method == "platt":
            model = LogisticRegression(C=1e6, max_iter=2000).fit(probability[:, None], labels)
            return cls("platt", coef=float(model.coef_[0, 0]), intercept=float(model.intercept_[0]))
        model = IsotonicRegression(out_of_bounds="clip").fit(probability, labels)
        return cls("isotonic", x=model.X_thresholds_.tolist(), y=model.y_thresholds_.tolist())

    def predict(self, probability: np.ndarray) -> np.ndarray:
        if self.kind == "constant":
            return np.full(len(probability), self.intercept)
        if self.kind == "platt":
            logits = probability * float(self.coef) + float(self.intercept)
            return 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        return np.interp(probability, self.x, self.y)


class NetBenefitRouter:
    """The correction-minus-harm router used by the completed V4/V5 studies."""

    feature_names = ("edge_probability", "confidence", "entropy", "ood_distance") + tuple(
        f"embedding_pca_{index}" for index in range(16)
    )

    def __init__(self) -> None:
        self.pca: PCA | None = None
        self.prototypes: dict[str, np.ndarray] = {}
        self.global_prototype: np.ndarray | None = None
        self.fix_model: _BinaryModel | None = None
        self.harm_model: _BinaryModel | None = None
        self.fix_calibration: _Calibration | None = None
        self.harm_calibration: _Calibration | None = None
        self.threshold = 0.0

    def fit_representation(self, seen_train: list[dict[str, Any]]) -> None:
        embeddings = _unit(np.asarray([row["fused_embedding"] for row in seen_train]))
        self.pca = PCA(n_components=16, svd_solver="full").fit(embeddings)
        grouped: dict[str, list[np.ndarray]] = {}
        for row, embedding in zip(seen_train, embeddings):
            grouped.setdefault(str(row.get("task_text", "unknown")).strip().lower(), []).append(embedding)
        self.prototypes = {key: _unit(np.mean(value, axis=0)[None])[0] for key, value in grouped.items()}
        self.global_prototype = _unit(np.mean(embeddings, axis=0)[None])[0]

    def features(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if self.pca is None or self.global_prototype is None:
            raise RuntimeError("fit_representation must be called first")
        result = []
        for row in rows:
            embedding = _unit(np.asarray(row["fused_embedding"])[None])[0]
            prototype = self.prototypes.get(str(row.get("task_text", "unknown")).strip().lower(), self.global_prototype)
            probability = _decision_center(float(row["edge_score"]), float(row["edge_threshold"]))
            entropy = -(probability * math.log(max(probability, 1e-12)) +
                        (1 - probability) * math.log(max(1 - probability, 1e-12)))
            result.append(np.r_[probability, max(probability, 1 - probability), entropy,
                                1 - embedding @ prototype, self.pca.transform(embedding[None])[0]])
        return np.vstack(result)

    def fit(self, development: list[dict[str, Any]], gate: list[dict[str, Any]], *, budget: float = 0.25) -> None:
        x, x_gate = self.features(development), self.features(gate)
        groups = np.asarray([row["recording_id"] for row in development])
        fix, harm = correction_targets(development)
        gate_fix, gate_harm = correction_targets(gate)
        candidates = []
        for c in (0.01, 0.1, 1.0, 10.0):
            folds = min(5, len(np.unique(groups)))
            splitter = GroupKFold(folds)
            oof_fix, oof_harm = np.zeros(len(x)), np.zeros(len(x))
            for train, test in splitter.split(x, fix, groups):
                oof_fix[test] = _BinaryModel.fit(x[train], fix[train], c).predict(x[test])
                oof_harm[test] = _BinaryModel.fit(x[train], harm[train], c).predict(x[test])
            for method in ("platt", "isotonic"):
                fix_cal, harm_cal = _Calibration.fit(oof_fix, fix, method), _Calibration.fit(oof_harm, harm, method)
                fix_model, harm_model = _BinaryModel.fit(x, fix, c), _BinaryModel.fit(x, harm, c)
                benefit = fix_cal.predict(fix_model.predict(x_gate)) - harm_cal.predict(harm_model.predict(x_gate))
                mask = exact_mask(benefit, budget, [row["sample_id"] for row in gate])
                net = int((gate_fix[mask] - gate_harm[mask]).sum())
                brier = float(np.mean((fix_cal.predict(fix_model.predict(x_gate)) - gate_fix) ** 2) +
                              np.mean((harm_cal.predict(harm_model.predict(x_gate)) - gate_harm) ** 2)) / 2
                candidates.append((net, -brier, -c, method == "platt", fix_model, harm_model, fix_cal, harm_cal))
        selected = max(candidates, key=lambda value: value[:4])
        self.fix_model, self.harm_model, self.fix_calibration, self.harm_calibration = selected[4:]
        benefit = self.score(gate)
        utilities = gate_fix - gate_harm
        choices = [(int(utilities[benefit > threshold].sum()), -int((benefit > threshold).sum()), float(threshold))
                   for threshold in np.r_[np.inf, np.unique(benefit), -np.inf]]
        self.threshold = max(choices)[2]

    def score(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if any(value is None for value in (self.fix_model, self.harm_model, self.fix_calibration, self.harm_calibration)):
            raise RuntimeError("router has not been fit")
        x = self.features(rows)
        return self.fix_calibration.predict(self.fix_model.predict(x)) - self.harm_calibration.predict(self.harm_model.predict(x))

    def route(self, rows: list[dict[str, Any]], budget: float = 0.25, positive_only: bool = True) -> np.ndarray:
        benefit = self.score(rows)
        if positive_only:
            eligible = benefit > self.threshold
            ranked = np.where(eligible, benefit, -np.inf)
            return exact_mask(ranked, budget, [row["sample_id"] for row in rows]) & eligible
        return exact_mask(benefit, budget, [row["sample_id"] for row in rows])
