"""Dependency-free metrics shared by LocateAnything and GroundAnything.

The evaluator consumes a small JSON-compatible record contract.  Coordinate
space is always explicit: ``pixel``, ``ga_999``, ``la_1000`` or ``normalized``.
All matching is category-aware and one-to-one.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence


class CoordinateError(ValueError):
    """A box is malformed or outside the declared image coordinate space."""


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "unknown"

    @property
    def values(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class Sample:
    """One prediction and its ground truth in the evaluator input contract."""

    sample_id: str
    width: int
    height: int
    ground_truth: Mapping[str, Any]
    prediction: Mapping[str, Any]
    category: str = "unknown"
    split: str = "all"
    game: str = "unknown"


THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CoordinateError(f"coordinate is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise CoordinateError(f"coordinate is not finite: {value!r}")
    return number


def normalize_box(
    coordinates: Sequence[Any],
    width: int,
    height: int,
    coordinate_space: str,
    *,
    label: str = "unknown",
) -> Box:
    """Convert one explicit-coordinate-space xyxy box into pixels.

    ``normalized`` means values in ``[0, 1]``.  The two model grids use their
    protocol maxima, so LA ``1000`` and GA ``999`` both map their endpoints to
    the image endpoints.  No coordinate space is inferred from the values.
    """

    if width <= 0 or height <= 0:
        raise CoordinateError("image width and height must be positive")
    if isinstance(coordinates, (str, bytes)):
        raise CoordinateError("a box must contain four coordinates")
    try:
        coordinate_count = len(coordinates)
    except TypeError as exc:
        raise CoordinateError("a box must contain four coordinates") from exc
    if coordinate_count != 4:
        raise CoordinateError("a box must contain four coordinates")
    values = tuple(_finite(value) for value in coordinates)
    if coordinate_space == "pixel":
        pixels = values
    elif coordinate_space == "ga_999":
        pixels = (values[0] / 999 * width, values[1] / 999 * height,
                  values[2] / 999 * width, values[3] / 999 * height)
    elif coordinate_space == "la_1000":
        pixels = (values[0] / 1000 * width, values[1] / 1000 * height,
                  values[2] / 1000 * width, values[3] / 1000 * height)
    elif coordinate_space == "normalized":
        pixels = (values[0] * width, values[1] * height,
                  values[2] * width, values[3] * height)
    else:
        raise CoordinateError(f"unsupported coordinate_space: {coordinate_space!r}")
    x1, y1, x2, y2 = pixels
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise CoordinateError(f"box is outside image bounds: {coordinates!r}")
    return Box(x1, y1, x2, y2, str(label))


def _entries(value: Any, default_label: str) -> list[tuple[str, Sequence[Any]]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CoordinateError("boxes/predictions must be a list")
    entries: list[tuple[str, Sequence[Any]]] = []
    for item in value:
        if isinstance(item, Mapping):
            coordinates = item.get("coordinates", item.get("box"))
            label = str(item.get("label", item.get("category", default_label)))
        else:
            coordinates, label = item, default_label
        entries.append((label, coordinates))
    return entries


def normalize_prediction(
    prediction: Mapping[str, Any],
    width: int,
    height: int,
    *,
    default_coordinate_space: str | None = None,
) -> tuple[list[Box], int, str | None]:
    """Normalize predictions, returning ``(valid_boxes, invalid_count, error)``."""

    space = prediction.get("coordinate_space", default_coordinate_space)
    raw_entries = prediction.get("predictions", prediction.get("boxes", []))
    if not raw_entries and space is None:
        return [], 0, None
    if not isinstance(space, str):
        return [], 0, "prediction.coordinate_space is required"
    boxes: list[Box] = []
    invalid = 0
    try:
        entries = _entries(raw_entries, "unknown")
    except CoordinateError as exc:
        return [], 1, str(exc)
    for label, coordinates in entries:
        try:
            boxes.append(normalize_box(coordinates, width, height, space, label=label))
        except CoordinateError:
            invalid += 1
    return boxes, invalid, None


def _sample_from_mapping(record: Mapping[str, Any]) -> Sample:
    return Sample(
        sample_id=str(record.get("id", record.get("sample_id", ""))),
        width=int(record["width"]),
        height=int(record["height"]),
        ground_truth=record["ground_truth"],
        prediction=record["prediction"],
        category=str(record.get("category", "unknown")),
        split=str(record.get("split", "all")),
        game=str(record.get("game", "unknown")),
    )


def _ground_truth_boxes(sample: Sample) -> tuple[list[Box], int]:
    ground_truth = sample.ground_truth
    space = ground_truth.get("coordinate_space")
    if not isinstance(space, str):
        raise CoordinateError("ground_truth.coordinate_space is required")
    default_label = str(ground_truth.get("category", sample.category))
    boxes: list[Box] = []
    invalid = 0
    for label, coordinates in _entries(ground_truth.get("boxes", []), default_label):
        try:
            boxes.append(normalize_box(coordinates, sample.width, sample.height, space, label=label))
        except CoordinateError:
            invalid += 1
    return boxes, invalid


def box_iou(left: Box | Sequence[float], right: Box | Sequence[float]) -> float:
    a = left.values if isinstance(left, Box) else tuple(float(v) for v in left)
    b = right.values if isinstance(right, Box) else tuple(float(v) for v in right)
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    union += max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def one_to_one_matches(ground_truth: Sequence[Box], predictions: Sequence[Box], threshold: float) -> list[tuple[int, int, float]]:
    """Return deterministic category-aware maximum-cardinality matches.

    This is the same Kuhn-style bipartite matching family used by LA.  Edges
    below the IoU threshold are removed; candidates are visited in descending
    IoU order, so a prediction cannot match more than one ground-truth box.
    """

    adjacency: list[list[tuple[int, float]]] = []
    for pred in predictions:
        candidates = [
            (index, box_iou(pred, target))
            for index, target in enumerate(ground_truth)
            if pred.label == target.label and box_iou(pred, target) >= threshold
        ]
        adjacency.append(sorted(candidates, key=lambda item: (-item[1], item[0])))
    matched: dict[int, tuple[int, float]] = {}

    def augment(pred_index: int, visited: set[int]) -> bool:
        for gt_index, iou in adjacency[pred_index]:
            if gt_index in visited:
                continue
            visited.add(gt_index)
            previous = matched.get(gt_index)
            if previous is None or augment(previous[0], visited):
                matched[gt_index] = (pred_index, iou)
                return True
        return False

    for pred_index in range(len(predictions)):
        augment(pred_index, set())
    return sorted((gt_index, pred_index, iou) for gt_index, (pred_index, iou) in matched.items())


def _score_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_sample(record: Sample | Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one sample and retain enough detail for later aggregation."""

    sample = record if isinstance(record, Sample) else _sample_from_mapping(record)
    gt_boxes, gt_invalid = _ground_truth_boxes(sample)
    status = str(sample.prediction.get("status", "success"))
    pred_boxes, pred_invalid, parse_error = normalize_prediction(sample.prediction, sample.width, sample.height)
    if status != "success":
        parse_error = sample.prediction.get("error") or f"prediction status: {status}"
        pred_boxes, pred_invalid = [], 0
    gt_none = not gt_boxes
    pred_none = not pred_boxes and not parse_error
    none_outcome = "none_correct" if gt_none and pred_none else None
    if gt_none and not pred_none:
        none_outcome = "none_false_positive"
    elif not gt_none and pred_none:
        none_outcome = "none_false_negative"
    result: dict[str, Any] = {
        "id": sample.sample_id,
        "split": sample.split,
        "game": sample.game,
        "category": sample.category,
        "parse_error": bool(parse_error),
        "illegal_coordinates": pred_invalid > 0 or gt_invalid > 0,
        "invalid_prediction_boxes": pred_invalid,
        "gt_none": gt_none,
        "pred_none": pred_none,
        "none_outcome": none_outcome,
        "latency_ms": sample.prediction.get("latency_ms"),
        "thresholds": {},
        "matched_ious": {},
    }
    for threshold in THRESHOLDS:
        matches = one_to_one_matches(gt_boxes, pred_boxes, threshold)
        tp = len(matches)
        result["thresholds"][str(threshold)] = _score_counts(tp, len(pred_boxes) - tp, len(gt_boxes) - tp)
        result["matched_ious"][str(threshold)] = [iou for _, _, iou in matches]
    return result


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def aggregate_results(results: Iterable[Mapping[str, Any]], *, _include_groups: bool = True) -> dict[str, Any]:
    """Aggregate sample results, per-class groups, and latency percentiles."""

    rows = list(results)
    total = len(rows)
    thresholds: dict[str, dict[str, Any]] = {}
    for threshold in THRESHOLDS:
        key = str(threshold)
        tp = sum(int(row["thresholds"][key]["tp"]) for row in rows)
        fp = sum(int(row["thresholds"][key]["fp"]) for row in rows)
        fn = sum(int(row["thresholds"][key]["fn"]) for row in rows)
        scored = _score_counts(tp, fp, fn)
        ious = [iou for row in rows for iou in row["matched_ious"][key]]
        scored["mIoU"] = sum(ious) / len(ious) if ious else 0.0
        thresholds[key] = scored
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    none_counts = {name: sum(row.get("none_outcome") == name for row in rows)
                   for name in ("none_correct", "none_false_positive", "none_false_negative")}
    none_gt_total = sum(bool(row.get("gt_none")) for row in rows)
    groupings: dict[str, dict[str, Any]] = {}
    if _include_groups:
        for field in ("split", "game"):
            groupings[field] = {}
            for value in sorted({str(row.get(field, "unknown")) for row in rows}):
                groupings[field][value] = aggregate_results(
                    (row for row in rows if str(row.get(field, "unknown")) == value),
                    _include_groups=False,
                )
    per_class: dict[str, Any] = {}
    if _include_groups:
        for group in sorted({str(row.get("category", "unknown")) for row in rows}):
            per_class[group] = aggregate_results(
                (row for row in rows if str(row.get("category", "unknown")) == group),
                _include_groups=False,
            )
    output: dict[str, Any] = {
        "samples": total,
        "valid_rate": sum(not row["parse_error"] and not row["illegal_coordinates"] for row in rows) / total if total else 0.0,
        "parse_error_rate": sum(bool(row["parse_error"]) for row in rows) / total if total else 0.0,
        "illegal_coordinate_rate": sum(bool(row["illegal_coordinates"]) for row in rows) / total if total else 0.0,
        "thresholds": thresholds,
        "f1_at_0.50": thresholds["0.5"]["f1"],
        "f1_at_0.75": thresholds["0.75"]["f1"],
        "f1_at_0.90": thresholds["0.9"]["f1"],
        "f1_at_0.50:0.95": sum(item["f1"] for item in thresholds.values()) / len(thresholds) if thresholds else 0.0,
        "false_positive": thresholds["0.5"]["fp"],
        "false_negative": thresholds["0.5"]["fn"],
        "mIoU": thresholds["0.5"]["mIoU"],
        "iou_at_0.75": thresholds["0.75"]["mIoU"],
        "iou_at_0.90": thresholds["0.9"]["mIoU"],
        "none": {**none_counts, "gt_total": none_gt_total,
                 "accuracy": none_counts["none_correct"] / none_gt_total if none_gt_total else 0.0},
        "latency_ms": {"count": len(latencies), "p50": _percentile(latencies, 0.50), "p95": _percentile(latencies, 0.95),
                       "throughput_samples_per_second": (1000 * len(latencies) / sum(latencies)) if sum(latencies) > 0 else None},
        "per_class": per_class,
    }
    if _include_groups:
        output["groups"] = groupings
    return output
