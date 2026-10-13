"""Standalone, standard-library-only CPU implementation of LA Current5-v1.

Adapted from LocateAnything ``game_finetune/core/common.py`` (parser, IoU,
Kuhn matching) and ``game_finetune/evaluate/eval_current5.py`` (aggregation).
The supplied snapshots are named ``la-common.py`` and ``eval_current5.py``;
their *raw-file* SHA256 values are recorded in SOURCE_SHA256 below. In
particular, the supplied evaluator differs from the historical evaluator SHA
in protocol.json; do not substitute that manifest hash for the snapshot hash.
Modifications: remove inference/GPU/filesystem dependencies, accept already
parsed records, add separate quality diagnostics, and require complete cell
coverage before publishing overall scores. GroundAnything's root LICENSE is
Apache-2.0. The supplied LA snapshots have no license/copyright header or
accompanying LICENSE; their upstream ownership/license needs verification
before redistribution. This note does not relicense the adapted LA logic.

Records contain gt_groups/pred_groups (label -> lists of pixel-space xyxy
boxes), pred_status, latency_seconds and query_labels. LA normalized-token
answers can be converted with parse_la_answer; GA parsing is intentionally
outside this module. Empty/invalid no-box predictions still count as true for
LA's historical negative none_accuracy, with quality reported separately.
"""
from __future__ import annotations

import math
import re
import statistics as st
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

SOURCE_SHA256 = {
    "eval_current5.py": "e243a86db729578400b096bfdfd1b9fdd7d3dcc232c6856ac9b0d72931029a58",
    "la-common.py": "3280c3602d278fa91a8f21f1e68b903c40a611491ec57c97f1dca8bea0982c80",
}
THRESHOLDS = [round(0.50 + 0.05 * index, 2) for index in range(10)]
FORMS = ("category", "description", "multiclass", "multi_instance", "negative")
PRIMARY_METRIC = {form: "mean_f1_0.50_0.95" for form in FORMS}
PRIMARY_METRIC["negative"] = "none_accuracy"
BOXES_STATUS = "boxes"
NONE_STATUS = "none"
EMPTY_STATUS = "empty"

# Exact LA token grammar: unsigned integers, no extra whitespace, lowercase none.
_OUTPUT_TOKEN_RE = re.compile(
    r"<ref>(?P<label>.*?)</ref>"
    r"|<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>"
    r"|<box>none</box>",
    re.DOTALL,
)


def normalized(value: str) -> str:
    return " ".join((value or "").split()).strip().casefold()


def _parse_output_units(
    answer: str,
    image_width: int,
    image_height: int,
    categories: Iterable[str] | None = None,
) -> tuple[list[dict], bool]:
    """Adapted from LA common._parse_output_units, preserving its rules."""
    canonical = {category.casefold(): category for category in categories or []}
    current_label = "unknown"
    boxes = []
    saw_none = False
    for match in _OUTPUT_TOKEN_RE.finditer(answer):
        label = match.group("label")
        if label is not None:
            cleaned = " ".join(label.split()).strip()
            current_label = canonical.get(cleaned.casefold(), cleaned or "unknown")
            continue
        if match.group("x1") is None:
            saw_none = True
            continue
        coords = [
            max(0, min(1000, int(match.group(name))))
            for name in ("x1", "y1", "x2", "y2")
        ]
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({
            "label": current_label,
            "normalized_box": coords,
            "box": [x1 / 1000 * image_width, y1 / 1000 * image_height,
                    x2 / 1000 * image_width, y2 / 1000 * image_height],
        })
    return boxes, saw_none


def parse_model_output_with_status(
    answer: str,
    image_width: int,
    image_height: int,
    categories: Iterable[str] | None = None,
) -> tuple[list[dict], str]:
    """LA parser: valid boxes win over none; malformed/empty output is empty."""
    boxes, saw_none = _parse_output_units(answer, image_width, image_height, categories)
    if boxes:
        return boxes, BOXES_STATUS
    return boxes, NONE_STATUS if saw_none else EMPTY_STATUS


def _normalized_groups(groups: Mapping) -> dict[str, list]:
    grouped = defaultdict(list)
    for label, boxes in groups.items():
        # Keep empty keys: the evaluator counts len(groups) for categories.
        grouped[normalized(str(label or "unknown"))].extend(boxes or [])
    return dict(grouped)


def parse_la_answer(answer: str, width: int, height: int) -> tuple[dict, str]:
    """GT/LA answer parser with evaluator label normalization, pixel xyxy output.

    Active refs persist across boxes and none markers, missing/empty refs use
    unknown, coordinates are clamped to [0,1000], and degenerate boxes are
    discarded. No extra format repair, lowercasing tokens, or deduplication.
    """
    boxes, status = parse_model_output_with_status(answer, width, height, None)
    grouped = defaultdict(list)
    for box in boxes:
        grouped[normalized(str(box.get("label") or "unknown"))].append(box["box"])
    return dict(grouped), status


def box_iou(left: list[float], right: list[float]) -> float:
    """LA common.box_iou: nonpositive union has IoU zero."""
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def maximum_matches(predictions: list, ground_truth: list, threshold: float) -> int:
    """Maximum bipartite matching (LA's IoU-ordered Kuhn augmenting paths)."""
    adjacency = [
        [index for index, target in enumerate(ground_truth) if box_iou(box, target) >= threshold]
        for box in predictions
    ]
    matched_prediction = [-1] * len(ground_truth)

    def augment(prediction_index: int, visited: set[int]) -> bool:
        candidates = sorted(
            adjacency[prediction_index],
            key=lambda target_index: box_iou(predictions[prediction_index], ground_truth[target_index]),
            reverse=True,
        )
        for target_index in candidates:
            if target_index in visited:
                continue
            visited.add(target_index)
            previous = matched_prediction[target_index]
            if previous == -1 or augment(previous, visited):
                matched_prediction[target_index] = prediction_index
                return True
        return False

    matches = 0
    for prediction_index in range(len(predictions)):
        if augment(prediction_index, set()):
            matches += 1
    return matches


def _empty_counts() -> dict:
    return {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in THRESHOLDS}


def _count_metrics(row: dict) -> dict:
    precision = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
    recall = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**row, "precision": precision, "recall": recall, "f1": f1}


def iou_metrics(ground_truth: dict, predictions: dict) -> dict:
    """Label-sensitive TP/FP/FN and metrics at every Current5 threshold."""
    counts = _empty_counts()
    for label in sorted(set(ground_truth) | set(predictions)):
        gt = ground_truth.get(label) or []
        pred = predictions.get(label) or []
        for threshold in THRESHOLDS:
            matches = maximum_matches(pred, gt, threshold)
            counts[threshold]["tp"] += matches
            counts[threshold]["fp"] += len(pred) - matches
            counts[threshold]["fn"] += len(gt) - matches
    return {f"{threshold:.2f}": _count_metrics(counts[threshold]) for threshold in THRESHOLDS}


def add_counts(target: dict, source: dict) -> None:
    for threshold in THRESHOLDS:
        row = source[f"{threshold:.2f}"]
        for key in ("tp", "fp", "fn"):
            target[threshold][key] += int(row[key])


def finalize_counts(counts: dict) -> dict:
    """Aggregate counts first (micro), then average F1 across thresholds."""
    rows = {f"{threshold:.2f}": _count_metrics(counts[threshold]) for threshold in THRESHOLDS}
    return {"thresholds": rows, "mean_f1_0.50_0.95": st.mean(row["f1"] for row in rows.values())}


def _box_count_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    return str(count) if count < 4 else "4+"


def _empty_bucket_counts() -> dict:
    return {
        bucket: {"requests": 0, "gt_boxes": 0, "pred_boxes": 0, "tp_0.50": 0,
                 "fn_0.50": 0, "count_abs_error": 0.0, "all_correct_0.50": 0}
        for bucket in ("0", "1", "2", "3", "4+")
    }


def _finalize_box_count_buckets(buckets: dict) -> dict:
    finalized = {}
    for bucket, row in buckets.items():
        gt_boxes = int(row["gt_boxes"])
        tp = int(row["tp_0.50"])
        pred = int(row["pred_boxes"])
        fn = int(row["fn_0.50"])
        metrics = _count_metrics({"tp": tp, "fp": pred - tp, "fn": fn})
        finalized[bucket] = {
            **row,
            "requests": int(row["requests"]),
            "gt_boxes": gt_boxes,
            "pred_boxes": pred,
            "tp_0.50": tp,
            "fn_0.50": fn,
            "recall_0.50": metrics["recall"],
            "f1_0.50": metrics["f1"],
            "count_mae": row["count_abs_error"] / row["requests"] if row["requests"] else 0.0,
            "all_correct_0.50": int(row["all_correct_0.50"]) / row["requests"] if row["requests"] else 0.0,
        }
        finalized[bucket].pop("count_abs_error", None)
    return finalized


def _latency_summary(latencies: list) -> dict:
    return {
        "mean": st.mean(latencies) if latencies else 0.0,
        # LA uses round/index selection, not an interpolated percentile.
        "p95": sorted(latencies)[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))]
        if latencies else 0.0,
    }


def score_records(records: Iterable[Mapping], form: str) -> dict:
    """Score one form/game cell without importing a worker or any GPU library.

    Required: gt_groups and pred_groups, containing already-valid pixel boxes.
    Labels are normalized like evaluator.gt_groups/pred_groups. Optional status
    defaults to boxes/empty, latency to zero, query_labels to an empty list.
    Optional retry metadata is retained for category/multiclass aggregation.
    The function does not filter invalid boxes supplied directly in groups:
    validating/parsing those is the caller's responsibility, as in LA.
    """
    if form not in FORMS:
        raise ValueError(f"Unknown Current5 form: {form}")
    records = list(records)
    category = form in ("category", "multiclass")
    negative = form == "negative"
    counts = _empty_counts()
    statuses = Counter()
    latencies = []
    requested_categories = gt_categories = pred_categories = covered_categories = 0
    retry_requests = retry_categories = 0
    category_retry = False
    gt_box_count = pred_box_count = all_correct = false_positive_requests = 0
    explicit_none = empty = no_box_requests = valid_no_box_requests = 0
    valid_parse_requests = gt_nonempty_requests = inconsistent_status_requests = 0
    count_errors, ious = [], []
    buckets = _empty_bucket_counts()
    for record in records:
        gt = _normalized_groups(record["gt_groups"])
        pred = _normalized_groups(record["pred_groups"])
        gt_boxes = [box for boxes in gt.values() for box in boxes]
        pred_boxes = [box for boxes in pred.values() for box in boxes]
        status = str(record.get("pred_status", BOXES_STATUS if pred_boxes else EMPTY_STATUS))
        latencies.append(float(record.get("latency_seconds", 0.0)))
        statuses[status] += 1
        gt_box_count += len(gt_boxes)
        pred_box_count += len(pred_boxes)
        gt_nonempty_requests += int(bool(gt_boxes))
        explicit_none += int(status == NONE_STATUS)
        empty += int(status == EMPTY_STATUS)
        no_box_requests += int(not pred_boxes)
        valid_no_box_requests += int(not pred_boxes and status == NONE_STATUS)
        consistent = (status == BOXES_STATUS and bool(pred_boxes)) or (
            status in (NONE_STATUS, EMPTY_STATUS) and not pred_boxes
        )
        inconsistent_status_requests += int(not consistent)
        valid_parse_requests += int(consistent and status in (BOXES_STATUS, NONE_STATUS))
        if category:
            row_metrics = iou_metrics(gt, pred)
            requested_categories += len(record.get("query_labels") or [])
            gt_categories += len(gt)
            pred_categories += len(pred)
            covered_categories += sum(1 for label in gt if label in pred and pred[label])
            retry = record.get("retry") or {}
            retries = int(retry.get("retry_count", 0) or 0)
            retry_requests += int(bool(retries))
            retry_categories += retries
            category_retry = category_retry or bool(record.get("category_retry", False)) or bool(retries)
        else:
            row_metrics = iou_metrics({"query": gt_boxes}, {"query": pred_boxes})
            matches = int(row_metrics["0.50"]["tp"])
            correct = matches == len(gt_boxes) == len(pred_boxes)
            all_correct += int(correct)
            count_error = abs(len(pred_boxes) - len(gt_boxes))
            count_errors.append(count_error)
            bucket = buckets[_box_count_bucket(len(gt_boxes))]
            bucket["requests"] += 1
            bucket["gt_boxes"] += len(gt_boxes)
            bucket["pred_boxes"] += len(pred_boxes)
            bucket["tp_0.50"] += matches
            bucket["fn_0.50"] += int(row_metrics["0.50"]["fn"])
            bucket["count_abs_error"] += count_error
            bucket["all_correct_0.50"] += int(correct)
            if gt_boxes:
                # Historical mean_best_iou is ONE maximum per GT-nonempty request.
                ious.append(max((box_iou(g, p) for g in gt_boxes for p in pred_boxes), default=0.0))
            elif pred_boxes:
                false_positive_requests += 1
        add_counts(counts, row_metrics)
    n = len(records)
    summary = finalize_counts(counts)
    summary.update({
        "form": form,
        "requests": n,
        "pred_boxes": pred_box_count,
        "boxes_per_request": pred_box_count / n if n else 0.0,
        "latency_seconds": _latency_summary(latencies),
        "parse_status_counts": dict(statuses),
        # These diagnostics never replace LA's historical primary metrics.
        "quality": {
            "valid_parse_rate": valid_parse_requests / n if n else 0.0,
            "empty_or_invalid_prediction_rate": (n - valid_parse_requests) / n if n else 0.0,
            "no_box_requests": no_box_requests,
            "explicit_none_no_box_requests": valid_no_box_requests,
            "explicit_none_no_box_rate": valid_no_box_requests / n if n else 0.0,
            "gt_nonempty_requests": gt_nonempty_requests,
            "inconsistent_parse_status_requests": inconsistent_status_requests,
        },
    })
    if category:
        summary.update({
            "requested_categories": requested_categories,
            "gt_categories": gt_categories,
            "pred_categories": pred_categories,
            "category_coverage": covered_categories / gt_categories if gt_categories else 0.0,
            "category_retry": category_retry,
            "retry_requests": retry_requests,
            "retry_categories": retry_categories,
        })
    else:
        finalized_buckets = _finalize_box_count_buckets(buckets)
        summary.update({
            "gt_boxes": gt_box_count,
            "request_all_correct_0.50": all_correct / n if n else 0.0,
            "count_mae": st.mean(count_errors) if count_errors else 0.0,
            "mean_best_iou": st.mean(ious) if ious else 0.0,
            "false_positive_requests": false_positive_requests,
            "false_positive_rate": false_positive_requests / n if n else 0.0,
            "hallucinated_boxes_per_request": pred_box_count / n if n and negative else 0.0,
            # Preserve LA's exact definition, including GT-nonempty edge cases.
            "none_accuracy": (n - false_positive_requests) / n if n and negative else 0.0,
            "explicit_none_rate": explicit_none / n if n else 0.0,
            "empty_prediction_rate": empty / n if n else 0.0,
            "box_count_buckets": finalized_buckets,
            "two_box_recall_0.50": finalized_buckets["2"]["recall_0.50"],
            "three_plus_box_recall_0.50": sum(finalized_buckets[b]["tp_0.50"] for b in ("3", "4+"))
            / max(1, sum(finalized_buckets[b]["gt_boxes"] for b in ("3", "4+"))),
        })
    return summary


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summary_cells(cells: Mapping, protocol: Mapping) -> dict:
    """LA form/game macro and training-weighted summary from nested cell metrics.

    cells is form -> game -> metrics. protocol is the LA manifest with games
    -> game -> forms -> form -> sampled_rows, and train_weights. Only manifest
    games are considered. Form means are unweighted means of game primaries,
    NOT pooled box/request micro-F1. A full aggregate requires all five forms,
    exact expected game identities, and each cell's requests == sampled_rows.
    Missing, short, excessive or nonfinite cells leave overall values None;
    partial per-form means remain visible with explicit coverage diagnostics.
    """
    if protocol.get("protocol", "current5-v1") != "current5-v1":
        raise ValueError("Expected current5-v1 protocol")
    games = protocol["games"]
    expected = {
        form: {
            game: int(info.get("forms", {}).get(form, {}).get("sampled_rows", 0))
            for game, info in games.items()
            if int(info.get("forms", {}).get(form, {}).get("sampled_rows", 0)) > 0
        }
        for form in FORMS
    }
    form_scores, coverage = {}, {}
    complete_forms = []
    for form in FORMS:
        key = PRIMARY_METRIC[form]
        values = {}
        completed = set()
        missing, incomplete = [], []
        for game, expected_requests in expected[form].items():
            metrics = cells.get(form, {}).get(game)
            if not isinstance(metrics, Mapping):
                missing.append(game)
                continue
            value = metrics.get(key)
            requests = metrics.get("requests", 0)
            valid = _finite_number(value) and _finite_number(requests) and requests > 0
            if valid:
                values[game] = float(value)
            if valid and requests == expected_requests and metrics.get("form", form) == form:
                completed.add(game)
            else:
                incomplete.append(game)
        if values:
            form_scores[form] = {
                "metric": key, "games": len(values), "mean": st.mean(values.values()),
                "minimum": min(values.values()), "per_game": values,
            }
        # No sampled cells for a form cannot constitute a full Current5 score.
        if expected[form] and completed == set(expected[form]):
            complete_forms.append(form)
        coverage[form] = {
            "games": len(values), "expected_games": len(expected[form]),
            "complete_games": len(completed), "missing_games": missing,
            "incomplete_games": incomplete,
        }
    full = len(complete_forms) == len(FORMS)
    equal_macro = weighted = worst_game = None
    if full:
        equal_macro = st.mean(form_scores[form]["mean"] for form in FORMS)
        weights = {form: float(protocol["train_weights"][form]) for form in FORMS}
        if any(not math.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("train_weights must be finite, nonnegative and have positive total")
        weighted = sum(form_scores[form]["mean"] * weights[form] for form in FORMS) / sum(weights.values())
        game_values = []
        for game in games:
            values = [form_scores[form]["per_game"][game] for form in FORMS if game in expected[form]]
            if values:
                game_values.append((game, st.mean(values)))
        if game_values:
            game, score = min(game_values, key=lambda item: item[1])
            worst_game = {"game": game, "score": score}
    return {
        "protocol": "current5-v1", "forms": form_scores, "coverage": coverage,
        "complete": full, "complete_forms": complete_forms,
        "current5_equal_view_macro": equal_macro,
        "current5_training_weighted": weighted,
        "current5_worst_game": worst_game,
    }
