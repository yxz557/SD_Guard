from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitResult:
    group_to_split: dict[str, str]
    score: float
    trial: int


def _feature_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["suite_name"]), str(row["attack_type"]), int(row["label"]))


def _candidate_assignment(
    groups: list[str],
    group_sizes: dict[str, int],
    ratios: tuple[float, float, float],
    rng: random.Random,
) -> dict[str, str]:
    order = list(groups)
    rng.shuffle(order)
    total = sum(group_sizes.values())
    targets = {
        "train": total * ratios[0],
        "validation": total * ratios[1],
        "test": total * ratios[2],
    }
    assignment: dict[str, str] = {}
    split_index = 0
    running = 0
    for group in order:
        split_name = SPLITS[split_index]
        size = group_sizes[group]
        if split_index < len(SPLITS) - 1:
            before_error = abs(running - targets[split_name])
            after_error = abs(running + size - targets[split_name])
            if running > 0 and before_error < after_error:
                split_index += 1
                split_name = SPLITS[split_index]
                running = 0
        assignment[group] = split_name
        running += size
    return assignment


def _score_assignment(
    rows: list[dict[str, Any]],
    assignment: dict[str, str],
    ratios: tuple[float, float, float],
) -> float:
    ratio_by_split = dict(zip(SPLITS, ratios, strict=True))
    global_cells = Counter(_feature_key(row) for row in rows)
    global_labels = Counter(int(row["label"]) for row in rows)
    split_cells: dict[str, Counter] = {name: Counter() for name in SPLITS}
    split_labels: dict[str, Counter] = {name: Counter() for name in SPLITS}
    split_totals = Counter()

    for row in rows:
        split_name = assignment[str(row["group_id"])]
        split_cells[split_name][_feature_key(row)] += 1
        split_labels[split_name][int(row["label"])] += 1
        split_totals[split_name] += 1

    score = 0.0
    total = len(rows)
    for split_name in SPLITS:
        ratio = ratio_by_split[split_name]
        expected_total = total * ratio
        score += 4.0 * (split_totals[split_name] - expected_total) ** 2 / (expected_total + 1.0)
        for label, count in global_labels.items():
            expected = count * ratio
            actual = split_labels[split_name][label]
            score += 3.0 * (actual - expected) ** 2 / (expected + 1.0)
        for cell, count in global_cells.items():
            expected = count * ratio
            actual = split_cells[split_name][cell]
            score += (actual - expected) ** 2 / (expected + 1.0)
    return score


def find_group_stratified_split(
    rows: Iterable[dict[str, Any]],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 20261106,
    trials: int = 10_000,
) -> SplitResult:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot split an empty dataset")
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("Expected three positive train/validation/test ratios")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1, got {ratios}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    if len(grouped) < 3:
        raise ValueError("At least three groups are required")

    group_sizes = {group: len(items) for group, items in grouped.items()}
    groups = sorted(grouped)
    best: SplitResult | None = None
    for trial in range(trials):
        assignment = _candidate_assignment(
            groups,
            group_sizes,
            ratios,
            random.Random(seed + trial),
        )
        score = _score_assignment(rows, assignment, ratios)
        if best is None or score < best.score:
            best = SplitResult(assignment, score, trial)
    assert best is not None
    return best


def summarize_split(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name in SPLITS:
        selected = [row for row in rows if row["split"] == split_name]
        by_label = Counter(int(row["label"]) for row in selected)
        by_suite = Counter(str(row["suite_name"]) for row in selected)
        by_attack = Counter(str(row["attack_type"]) for row in selected)
        by_suite_attack_label = Counter(
            f"{row['suite_name']}::{row['attack_type']}::{row['label']}" for row in selected
        )
        summary[split_name] = {
            "samples": len(selected),
            "groups": len({row["group_id"] for row in selected}),
            "labels": dict(sorted(by_label.items())),
            "suites": dict(sorted(by_suite.items())),
            "attacks": dict(sorted(by_attack.items())),
            "suite_attack_label": dict(sorted(by_suite_attack_label.items())),
        }
    return summary

