"""Shared evaluation utilities for game UI grounding."""

from .metrics import (
    Box,
    CoordinateError,
    Sample,
    aggregate_results,
    box_iou,
    evaluate_sample,
    normalize_box,
    normalize_prediction,
    one_to_one_matches,
)

__all__ = [
    "Box",
    "CoordinateError",
    "Sample",
    "aggregate_results",
    "box_iou",
    "evaluate_sample",
    "normalize_box",
    "normalize_prediction",
    "one_to_one_matches",
]
