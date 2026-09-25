from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        upper = scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1]
        mask = (scores >= edges[index]) & upper
        if mask.any():
            result += mask.mean() * abs(float(labels[mask].mean()) - float(scores[mask].mean()))
    return float(result)


def binary_metrics(labels: Iterable[int], scores: Iterable[float], threshold: float) -> dict[str, Any]:
    y = np.asarray(list(labels), dtype=np.int64)
    probability = np.asarray(list(scores), dtype=np.float64)
    if y.size == 0 or probability.shape != y.shape:
        raise ValueError("labels and scores must be equal-sized nonempty vectors")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("scores must be finite probabilities")
    prediction = (probability >= threshold).astype(np.int64)
    classes = np.unique(y)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    both_classes = len(classes) == 2
    return {
        "n": int(y.size),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "auprc": float(average_precision_score(y, probability)) if both_classes else None,
        "auroc": float(roc_auc_score(y, probability)) if both_classes else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)) if both_classes else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "ece": expected_calibration_error(y, probability),
        "brier": float(np.mean((probability - y) ** 2)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def select_threshold(labels: Iterable[int], scores: Iterable[float]) -> dict[str, Any]:
    y = np.asarray(list(labels), dtype=np.int64)
    probability = np.asarray(list(scores), dtype=np.float64)
    candidates = np.unique(np.r_[0.0, probability, 1.0])
    ranked: list[tuple[float, float, float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = binary_metrics(y, probability, float(threshold))
        balanced = metrics["balanced_accuracy"]
        ranked.append((metrics["f1"], -1.0 if balanced is None else balanced,
                       -float(threshold), metrics))
    selected = max(ranked, key=lambda item: item[:3])[3]
    return {"selection_rule": "max_f1_then_balanced_accuracy_then_lower_threshold",
            "candidate_count": len(candidates), "selected": selected}


def grouped_bootstrap(labels: Iterable[int], scores: Iterable[float], groups: Iterable[str],
                      threshold: float, iterations: int = 10_000, seed: int = 42,
                      confidence: float = 0.95) -> dict[str, Any]:
    y = np.asarray(list(labels), dtype=np.int64)
    probability = np.asarray(list(scores), dtype=np.float64)
    group_values = np.asarray(list(groups), dtype=object)
    if not (len(y) == len(probability) == len(group_values)):
        raise ValueError("bootstrap inputs must have equal lengths")
    unique = np.asarray(sorted({str(value) for value in group_values}), dtype=object)
    group_lookup = {value: index for index, value in enumerate(unique)}
    group_index = np.asarray(
        [group_lookup[str(value)] for value in group_values], dtype=np.int64
    )
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = defaultdict(list)

    # Resampling groups with replacement is equivalent to assigning each group a
    # multinomial multiplicity. Operating on those weights avoids reconstructing
    # sample vectors and repeatedly entering sklearn 10,000 times.
    prediction = probability >= threshold
    descending = np.argsort(-probability, kind="mergesort")
    sorted_scores = probability[descending]
    tie_starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    chunk_size = 256
    probabilities = np.full(len(unique), 1.0 / len(unique), dtype=np.float64)
    for start in range(0, iterations, chunk_size):
        count = min(chunk_size, iterations - start)
        group_weights = rng.multinomial(len(unique), probabilities, size=count)
        weights = group_weights[:, group_index].astype(np.float64, copy=False)
        positives = weights * y
        negatives = weights * (1 - y)
        tp = positives[:, prediction].sum(axis=1)
        fp = negatives[:, prediction].sum(axis=1)
        fn = positives[:, ~prediction].sum(axis=1)
        tn = negatives[:, ~prediction].sum(axis=1)
        total_positive = tp + fn
        total_negative = tn + fp

        precision = np.divide(
            tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0
        )
        recall = np.divide(
            tp, total_positive, out=np.zeros_like(tp), where=total_positive > 0
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(tp),
            where=(precision + recall) > 0,
        )
        specificity = np.divide(
            tn, total_negative, out=np.zeros_like(tn), where=total_negative > 0
        )
        balanced = (recall + specificity) / 2
        valid = (total_positive > 0) & (total_negative > 0)

        sorted_positive = positives[:, descending]
        sorted_negative = negatives[:, descending]
        positive_by_score = np.add.reduceat(sorted_positive, tie_starts, axis=1)
        negative_by_score = np.add.reduceat(sorted_negative, tie_starts, axis=1)
        cumulative_positive = np.cumsum(positive_by_score, axis=1)
        cumulative_total = np.cumsum(
            positive_by_score + negative_by_score, axis=1
        )
        precision_at_score = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        average_precision = np.divide(
            np.sum(positive_by_score * precision_at_score, axis=1),
            total_positive,
            out=np.zeros_like(total_positive),
            where=total_positive > 0,
        )

        cumulative_negative = np.cumsum(negative_by_score, axis=1)
        lower_negative = total_negative[:, None] - cumulative_negative
        concordant = np.sum(
            positive_by_score * (lower_negative + 0.5 * negative_by_score),
            axis=1,
        )
        auroc = np.divide(
            concordant,
            total_positive * total_negative,
            out=np.zeros_like(total_positive),
            where=valid,
        )

        collected["f1"].extend(f1.tolist())
        collected["precision"].extend(precision.tolist())
        collected["recall"].extend(recall.tolist())
        collected["auprc"].extend(average_precision[valid].tolist())
        collected["auroc"].extend(auroc[valid].tolist())
        collected["balanced_accuracy"].extend(balanced[valid].tolist())
    alpha = (1.0 - confidence) / 2.0
    return {
        "iterations": iterations,
        "group_count": len(unique),
        "confidence": confidence,
        "metrics": {
            key: {
                "mean": float(np.mean(values)),
                "lower": float(np.quantile(values, alpha)),
                "upper": float(np.quantile(values, 1.0 - alpha)),
                "valid_iterations": len(values),
            }
            for key, values in collected.items()
        },
    }


def grouped_paired_bootstrap(
    labels: Iterable[int],
    baseline_scores: Iterable[float],
    candidate_scores: Iterable[float],
    groups: Iterable[str],
    *,
    threshold: float = 0.5,
    candidate_threshold: float | None = None,
    iterations: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired grouped confidence intervals for candidate-minus-baseline metrics."""
    y = np.asarray(list(labels), dtype=np.int64)
    baseline = np.asarray(list(baseline_scores), dtype=np.float64)
    candidate = np.asarray(list(candidate_scores), dtype=np.float64)
    group_values = np.asarray(list(groups), dtype=object)
    if not (
        len(y) == len(baseline) == len(candidate) == len(group_values)
        and len(y) > 0
    ):
        raise ValueError("paired bootstrap inputs must be equal-sized nonempty vectors")
    unique = np.asarray(sorted({str(value) for value in group_values}), dtype=object)
    group_lookup = {value: index for index, value in enumerate(unique)}
    group_index = np.asarray(
        [group_lookup[str(value)] for value in group_values], dtype=np.int64
    )
    probability = np.full(len(unique), 1.0 / len(unique), dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = defaultdict(list)

    candidate_threshold = threshold if candidate_threshold is None else candidate_threshold

    def metrics_for(
        scores: np.ndarray, weights: np.ndarray, decision_threshold: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        positive = weights * y
        negative = weights * (1 - y)
        predicted = scores >= decision_threshold
        tp = positive[:, predicted].sum(axis=1)
        fp = negative[:, predicted].sum(axis=1)
        fn = positive[:, ~predicted].sum(axis=1)
        total_positive = tp + fn
        total_negative = negative.sum(axis=1)
        precision = np.divide(
            tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0
        )
        recall = np.divide(
            tp, total_positive, out=np.zeros_like(tp), where=total_positive > 0
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(tp),
            where=(precision + recall) > 0,
        )
        specificity = np.divide(
            negative[:, ~predicted].sum(axis=1),
            total_negative,
            out=np.zeros_like(total_negative),
            where=total_negative > 0,
        )
        balanced_accuracy = (recall + specificity) / 2
        order = np.argsort(-scores, kind="mergesort")
        sorted_scores = scores[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
        by_score_positive = np.add.reduceat(positive[:, order], starts, axis=1)
        by_score_total = np.add.reduceat(weights[:, order], starts, axis=1)
        cumulative_positive = np.cumsum(by_score_positive, axis=1)
        cumulative_total = np.cumsum(by_score_total, axis=1)
        precision_at_score = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        auprc = np.divide(
            np.sum(by_score_positive * precision_at_score, axis=1),
            total_positive,
            out=np.zeros_like(total_positive),
            where=total_positive > 0,
        )
        valid = (total_positive > 0) & (total_negative > 0)
        return auprc, f1, balanced_accuracy, valid

    for start in range(0, iterations, 256):
        count = min(256, iterations - start)
        multiplicity = rng.multinomial(len(unique), probability, size=count)
        weights = multiplicity[:, group_index].astype(np.float64, copy=False)
        base_auprc, base_f1, base_balanced, base_valid = metrics_for(
            baseline, weights, threshold
        )
        cand_auprc, cand_f1, cand_balanced, cand_valid = metrics_for(
            candidate, weights, candidate_threshold
        )
        valid = base_valid & cand_valid
        deltas["auprc"].extend((cand_auprc[valid] - base_auprc[valid]).tolist())
        deltas["f1"].extend((cand_f1 - base_f1).tolist())
        deltas["balanced_accuracy"].extend(
            (cand_balanced[valid] - base_balanced[valid]).tolist()
        )

    alpha = (1.0 - confidence) / 2.0
    metric_results = {}
    for metric, values in deltas.items():
        vector = np.asarray(values, dtype=np.float64)
        lower_tail = (np.count_nonzero(vector <= 0) + 1) / (len(vector) + 1)
        upper_tail = (np.count_nonzero(vector >= 0) + 1) / (len(vector) + 1)
        metric_results[metric] = {
            "mean_delta": float(vector.mean()),
            "lower": float(np.quantile(vector, alpha)),
            "upper": float(np.quantile(vector, 1.0 - alpha)),
            "two_sided_p": float(min(1.0, 2 * min(lower_tail, upper_tail))),
            "valid_iterations": len(vector),
        }
    return {
        "iterations": iterations,
        "group_count": len(unique),
        "confidence": confidence,
        "metrics": metric_results,
    }

