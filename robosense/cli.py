from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import joblib
import yaml

from .adaptation.reliable_kd_replay import build_feedback_records, build_replay_records
from .data.manifest import assert_recording_isolation, load_manifest
from .evaluation.detection_metrics import binary_metrics, select_threshold
from .routing.net_benefit import NetBenefitRouter


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def audit_manifest(args: argparse.Namespace) -> None:
    records = load_manifest(args.manifest, args.data_root)
    assert_recording_isolation(records)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.split] = counts.get(record.split, 0) + 1
    print(json.dumps({"passed": True, "records": len(records), "splits": counts}, indent=2))


def train(args: argparse.Namespace) -> None:
    config_text = os.path.expandvars(Path(args.config).read_text())
    if "${" in config_text:
        raise SystemExit("configuration contains unresolved environment variables")
    config = yaml.safe_load(config_text)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        resolved = handle.name
    try:
        subprocess.run([args.llamafactory_cli, "train", resolved], check=True)
    finally:
        Path(resolved).unlink(missing_ok=True)


def fit_router(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.predictions)
    train_rows = [row for row in rows if row["split"] == "seen_train"]
    development = [row for row in rows if row["split"] in {"seen_val", "adaptation_stream"}]
    gate = [row for row in rows if row["split"] == "ood_gate"]
    if not train_rows or not development or not gate:
        raise ValueError("router fitting requires seen_train, development, and ood_gate rows")
    router = NetBenefitRouter()
    router.fit_representation(train_rows)
    router.fit(development, gate, budget=args.budget)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(router, args.output)


def adapt(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.predictions)
    router: NetBenefitRouter = joblib.load(args.router)
    adaptation = sorted([row for row in rows if row["split"] == "adaptation_stream"], key=lambda row: row["sample_id"])
    train_rows = [row for row in rows if row["split"] == "seen_train"]
    mask = router.route(adaptation, budget=args.budget, positive_only=args.positive_only)
    routed = [row for row, selected in zip(adaptation, mask) if selected]
    feedback = build_feedback_records(routed, cloud_threshold=args.cloud_threshold)
    replay = build_replay_records(feedback, train_rows)
    _write_jsonl(Path(args.output_dir) / "feedback.jsonl", feedback)
    _write_jsonl(Path(args.output_dir) / "replay.jsonl", replay)
    _write_json(Path(args.output_dir) / "feedback_audit.json", {
        "adaptation_records": len(adaptation), "routed": len(routed),
        "reliable_teacher_records": sum(row["teacher_quality_weight"] > 0 for row in feedback),
        "replay_records": len(replay), "budget": args.budget,
    })


def evaluate(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.predictions)
    selection = [row for row in rows if row["split"] == "seen_val"]
    threshold = args.threshold
    if threshold is None:
        threshold = select_threshold([row["anomaly"] for row in selection], [row["score"] for row in selection])["selected"]["threshold"]
    output = {"threshold": threshold, "scopes": {}}
    for split in ("id_test_full", "id_test_matched", "ood_test", "realworld_ood_test"):
        scope = [row for row in rows if row["split"] == split]
        if scope:
            output["scopes"][split] = binary_metrics(
                [row["anomaly"] for row in scope], [row["score"] for row in scope], threshold
            )
    _write_json(args.output, output)


def main() -> None:
    parser = argparse.ArgumentParser(prog="robosense")
    commands = parser.add_subparsers(dest="command", required=True)
    audit_cmd = commands.add_parser("audit")
    audit_cmd.add_argument("--manifest", required=True)
    audit_cmd.add_argument("--data-root")
    audit_cmd.set_defaults(handler=audit_manifest)

    train_cmd = commands.add_parser("train")
    train_cmd.add_argument("--config", required=True)
    train_cmd.add_argument("--role", choices=("edge", "cloud"), required=True)
    train_cmd.add_argument("--llamafactory-cli", default="llamafactory-cli")
    train_cmd.set_defaults(handler=train)

    route_cmd = commands.add_parser("fit-router")
    route_cmd.add_argument("--predictions", required=True)
    route_cmd.add_argument("--output", required=True)
    route_cmd.add_argument("--budget", type=float, default=0.25)
    route_cmd.set_defaults(handler=fit_router)

    adapt_cmd = commands.add_parser("adapt")
    adapt_cmd.add_argument("--predictions", required=True)
    adapt_cmd.add_argument("--router", required=True)
    adapt_cmd.add_argument("--output-dir", required=True)
    adapt_cmd.add_argument("--budget", type=float, default=0.25)
    adapt_cmd.add_argument("--cloud-threshold", type=float, required=True)
    adapt_cmd.add_argument("--positive-only", action="store_true")
    adapt_cmd.set_defaults(handler=adapt)

    evaluate_cmd = commands.add_parser("evaluate")
    evaluate_cmd.add_argument("--predictions", required=True)
    evaluate_cmd.add_argument("--output", required=True)
    evaluate_cmd.add_argument("--threshold", type=float)
    evaluate_cmd.set_defaults(handler=evaluate)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
