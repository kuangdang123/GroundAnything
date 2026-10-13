"""Artificial CPU fixtures for the standalone Current5-v1 contract."""
import ast
import copy
import itertools
import math
import unittest
from pathlib import Path

from eval.current5_metrics import (
    FORMS,
    SOURCE_SHA256,
    THRESHOLDS,
    box_iou,
    finalize_counts,
    iou_metrics,
    maximum_matches,
    parse_la_answer,
    parse_model_output_with_status,
    score_records,
    summary_cells,
)


BOX = [0, 0, 10, 10]
FAR = [20, 20, 30, 30]


def record(gt=None, pred=None, status=None, latency=0.0, labels=None):
    pred = {} if pred is None else pred
    return {
        "gt_groups": {} if gt is None else gt,
        "pred_groups": pred,
        "pred_status": status if status is not None else ("boxes" if any(pred.values()) else "empty"),
        "latency_seconds": latency,
        "query_labels": labels or [],
    }


def protocol_fixture():
    return {
        "protocol": "current5-v1",
        "games": {
            game: {"forms": {form: {"sampled_rows": 2} for form in FORMS}}
            for game in ("alpha", "beta")
        },
        "train_weights": {form: i + 1 for i, form in enumerate(FORMS)},
    }


def cells_fixture():
    return {
        form: {
            "alpha": {"form": form, "requests": 2,
                      "mean_f1_0.50_0.95": i / 10, "none_accuracy": i / 10},
            "beta": {"form": form, "requests": 2,
                     "mean_f1_0.50_0.95": (i + 2) / 10, "none_accuracy": (i + 2) / 10},
        }
        for i, form in enumerate(FORMS)
    }


class LAParserTests(unittest.TestCase):
    def test_refs_persist_and_normalize_without_deduplication(self):
        answer = (
            "prefix <box><0><0><1000><1000></box>"
            "<ref>  Big\n PLAYER </ref><box><10><20><30><40></box>"
            "<box>none</box><box><10><20><30><40></box>"
            "<ref> </ref><box><0><0><1><1></box>"
        )
        groups, status = parse_la_answer(answer, 200, 100)
        self.assertEqual(status, "boxes")
        self.assertEqual(groups["unknown"], [[0, 0, 200, 100], [0, 0, 0.2, 0.1]])
        self.assertEqual(groups["big player"], [[2, 2, 6, 4], [2, 2, 6, 4]])

    def test_clamp_before_degenerate_filter(self):
        groups, status = parse_la_answer(
            "<ref>A</ref><box><0><0><1200><2000></box>"
            "<box><1000><0><2000><1000></box>"
            "<box><1><1><1><5></box><box><4><5><1><2></box>", 200, 100
        )
        self.assertEqual((groups, status), ({"a": [[0, 0, 200, 100]]}, "boxes"))

    def test_literal_grammar_and_statuses(self):
        invalid = ["", "garbage", "<ref>a</ref>", "<box>NONE</box>",
                   "<box><-1><0><9><9></box>", "<box><0.0><0><9><9></box>",
                   "<box> <0><0><9><9></box>", "<box><0><0><0><9></box>"]
        for answer in invalid:
            with self.subTest(answer=answer):
                self.assertEqual(parse_la_answer(answer, 10, 10), ({}, "empty"))
        self.assertEqual(parse_la_answer("text <box>none</box>", 10, 10), ({}, "none"))
        self.assertEqual(parse_la_answer("<box><0><0><0><9></box><box>none</box>", 10, 10), ({}, "none"))
        self.assertEqual(parse_la_answer("<box>none</box><box><0><0><9><9></box>", 10, 10)[1], "boxes")

    def test_common_canonical_labels_and_full_box_metadata(self):
        boxes, status = parse_model_output_with_status(
            "<ref> player </ref><box><100><200><300><400></box>", 200, 100, ["PLAYER"]
        )
        self.assertEqual(status, "boxes")
        self.assertEqual(boxes, [{"label": "PLAYER", "normalized_box": [100, 200, 300, 400],
                                  "box": [20, 20, 60, 40]}])


class MatchingTests(unittest.TestCase):
    def test_iou_degenerate_disjoint_and_intersection(self):
        self.assertEqual(box_iou(BOX, BOX), 1)
        self.assertEqual(box_iou(BOX, FAR), 0)
        self.assertEqual(box_iou([1, 1, 1, 1], [1, 1, 1, 1]), 0)
        self.assertEqual(box_iou([5, 5, 0, 0], BOX), 0)
        self.assertAlmostEqual(box_iou(BOX, [5, 0, 15, 10]), 1 / 3)

    def test_maximum_matching_reassigns_non_greedily(self):
        # First prediction prefers GT0 but can also match GT1. The second can
        # ONLY match GT0, so an IoU-greedy one-pass matcher would return one.
        gt = [[0, 0, 10, 10], [4, 0, 14, 10]]
        pred = [[1, 0, 11, 10], [0, 0, 10, 10]]
        self.assertGreater(box_iou(pred[0], gt[0]), box_iou(pred[0], gt[1]))
        self.assertGreaterEqual(box_iou(pred[0], gt[1]), 0.5)
        self.assertLess(box_iou(pred[1], gt[1]), 0.5)
        self.assertEqual(maximum_matches(pred, gt, 0.5), 2)
        self.assertEqual(maximum_matches(pred[::-1], gt[::-1], 0.5), 2)

    def test_one_box_cannot_match_multiple_and_threshold_is_inclusive(self):
        self.assertEqual(maximum_matches([BOX], [BOX, BOX], 0.5), 1)
        self.assertEqual(maximum_matches([BOX, BOX], [BOX], 0.5), 1)
        self.assertEqual(maximum_matches([BOX], [[0, 0, 20, 10]], 0.5), 1)
        self.assertEqual(maximum_matches([BOX], [[0, 0, 20, 10]], 0.55), 0)
        self.assertEqual(maximum_matches([], [BOX], 0.5), 0)
        self.assertEqual(maximum_matches([BOX], [], 0.5), 0)

    def test_matching_matches_exhaustive_small_graphs(self):
        candidates = [BOX, [1, 0, 11, 10], [4, 0, 14, 10], FAR]
        for pred in itertools.combinations_with_replacement(candidates, 3):
            for gt in itertools.combinations_with_replacement(candidates, 3):
                best = 0
                for assignment in itertools.product(range(-1, len(gt)), repeat=len(pred)):
                    used = [index for index in assignment if index >= 0]
                    if len(used) != len(set(used)):
                        continue
                    if all(j == -1 or box_iou(pred[i], gt[j]) >= 0.5 for i, j in enumerate(assignment)):
                        best = max(best, len(used))
                self.assertEqual(maximum_matches(list(pred), list(gt), 0.5), best)

    def test_iou_metrics_and_finalize_counts(self):
        rows = iou_metrics({"a": [BOX]}, {"b": [BOX]})
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows["0.50"], {"tp": 0, "fp": 1, "fn": 1,
                                        "precision": 0.0, "recall": 0.0, "f1": 0.0})
        counts = {t: {"tp": 2, "fp": 1, "fn": 2} for t in THRESHOLDS}
        summary = finalize_counts(counts)
        self.assertAlmostEqual(summary["mean_f1_0.50_0.95"], 4 / 7)
        self.assertEqual(counts[0.5], {"tp": 2, "fp": 1, "fn": 2})


class RecordScoringTests(unittest.TestCase):
    def test_label_sensitive_category_but_description_ignores_label(self):
        row = record({"gt": [BOX]}, {"wrong": [BOX]}, labels=["gt"])
        for form in ("category", "multiclass"):
            score = score_records([row], form)
            self.assertEqual(score["mean_f1_0.50_0.95"], 0)
            self.assertEqual(score["category_coverage"], 0)
            self.assertEqual(score["requested_categories"], 1)
        for form in ("description", "multi_instance"):
            score = score_records([row], form)
            self.assertEqual(score["mean_f1_0.50_0.95"], 1)
            self.assertEqual(score["request_all_correct_0.50"], 1)

    def test_category_normalization_coverage_and_retry_metadata(self):
        row = record({" Big  PLAYER ": [BOX]}, {"big\nplayer": [FAR]}, labels=["Big Player", "item"])
        row["retry"] = {"retry_count": 2}
        score = score_records([row], "category")
        self.assertEqual(score["category_coverage"], 1)  # presence, not localization
        self.assertEqual(score["mean_f1_0.50_0.95"], 0)
        self.assertEqual(score["gt_categories"], 1)
        self.assertEqual(score["pred_categories"], 1)
        self.assertEqual(score["requested_categories"], 2)
        self.assertEqual(score["retry_requests"], 1)
        self.assertEqual(score["retry_categories"], 2)
        self.assertTrue(score["category_retry"])

    def test_micro_f1_is_not_mean_request_f1(self):
        rows = [record({"a": [BOX]}, {"a": [BOX]}), record({"a": [BOX] * 9}, {})]
        for form in ("category", "multiclass", "description", "multi_instance"):
            score = score_records(iter(rows), form)
            self.assertEqual(score["thresholds"]["0.50"]["tp"], 1)
            self.assertEqual(score["thresholds"]["0.50"]["fn"], 9)
            self.assertAlmostEqual(score["mean_f1_0.50_0.95"], 2 / 11)
            self.assertNotAlmostEqual(score["mean_f1_0.50_0.95"], 0.5)

    def test_mean_across_iou_thresholds(self):
        score = score_records([record({"a": [[0, 0, 20, 10]]}, {"a": [BOX]})], "category")
        self.assertEqual(score["thresholds"]["0.50"]["f1"], 1)
        self.assertEqual(score["thresholds"]["0.55"]["f1"], 0)
        self.assertEqual(score["mean_f1_0.50_0.95"], 0.1)

    def test_negative_empty_invalid_count_as_true_with_separate_quality(self):
        rows = [record(status="none"), record(status="empty"), record(status="invalid"),
                record(pred={"a": [BOX, FAR]})]
        score = score_records(rows, "negative")
        self.assertEqual(score["none_accuracy"], 0.75)
        self.assertEqual(score["false_positive_requests"], 1)
        self.assertEqual(score["false_positive_rate"], 0.25)
        self.assertEqual(score["hallucinated_boxes_per_request"], 0.5)
        self.assertEqual(score["explicit_none_rate"], 0.25)
        self.assertEqual(score["empty_prediction_rate"], 0.25)
        self.assertEqual(score["request_all_correct_0.50"], 0.75)
        self.assertEqual(score["quality"]["valid_parse_rate"], 0.5)
        self.assertEqual(score["quality"]["empty_or_invalid_prediction_rate"], 0.5)
        self.assertEqual(score["quality"]["explicit_none_no_box_rate"], 0.25)
        self.assertEqual(score["parse_status_counts"], {"none": 1, "empty": 1, "invalid": 1, "boxes": 1})

    def test_negative_preserves_gt_nonempty_edge_case(self):
        score = score_records([record({"a": [BOX]}, {"a": [FAR]})], "negative")
        self.assertEqual(score["none_accuracy"], 1)  # exact historical LA rule
        self.assertEqual(score["request_all_correct_0.50"], 0)
        self.assertEqual(score["hallucinated_boxes_per_request"], 1)
        self.assertEqual(score["quality"]["gt_nonempty_requests"], 1)

    def test_buckets_all_correct_mae_and_best_iou(self):
        rows = [record(status="none"), record({"a": [BOX]}, {"z": [FAR]}),
                record({"a": [BOX, FAR]}, {"a": [BOX]}),
                record({"a": [BOX] * 3}, {"a": [BOX] * 3}),
                record({"a": [BOX] * 4}, {"a": [BOX] * 2})]
        score = score_records(rows, "multi_instance")
        buckets = score["box_count_buckets"]
        self.assertEqual(list(buckets), ["0", "1", "2", "3", "4+"])
        self.assertEqual(score["request_all_correct_0.50"], 2 / 5)
        self.assertEqual(score["count_mae"], 3 / 5)
        self.assertEqual(score["mean_best_iou"], 3 / 4)  # one max per nonempty GT row
        self.assertEqual(score["two_box_recall_0.50"], 0.5)
        self.assertEqual(score["three_plus_box_recall_0.50"], 5 / 7)
        self.assertEqual(buckets["0"]["all_correct_0.50"], 1)
        self.assertEqual(buckets["0"]["f1_0.50"], 0)
        self.assertAlmostEqual(buckets["2"]["f1_0.50"], 2 / 3)
        self.assertEqual(buckets["4+"]["count_mae"], 2)
        self.assertNotIn("count_abs_error", buckets["4+"])

    def test_latency_uses_la_round_index_percentile(self):
        # Five samples: round(0.95 * 4) == 4, i.e. maximum, not interpolation.
        rows = [record(latency=t) for t in [1, 5, 2, 4, 3]]
        self.assertEqual(score_records(rows, "description")["latency_seconds"], {"mean": 3, "p95": 5})

    def test_empty_cells_and_unknown_forms(self):
        for form in FORMS:
            score = score_records([], form)
            self.assertEqual(score["requests"], 0)
            self.assertEqual(score["mean_f1_0.50_0.95"], 0)
            self.assertEqual(score["latency_seconds"], {"mean": 0.0, "p95": 0.0})
            self.assertEqual(score["quality"]["valid_parse_rate"], 0)
        with self.assertRaises(ValueError):
            score_records([], "single")

    def test_does_not_mutate_record(self):
        row = record({" A ": [BOX]}, {"a": [BOX]}, latency=2)
        original = copy.deepcopy(row)
        score_records([row], "description")
        self.assertEqual(row, original)


class CellSummaryTests(unittest.TestCase):
    def test_game_macro_form_macro_training_weighted_and_worst_game(self):
        summary = summary_cells(cells_fixture(), protocol_fixture())
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["complete_forms"], list(FORMS))
        self.assertAlmostEqual(summary["forms"]["category"]["mean"], 0.1)
        self.assertAlmostEqual(summary["current5_equal_view_macro"], 0.3)
        self.assertAlmostEqual(summary["current5_training_weighted"], 5.5 / 15)
        self.assertEqual(summary["current5_worst_game"]["game"], "alpha")
        self.assertAlmostEqual(summary["current5_worst_game"]["score"], 0.2)

    def test_negative_primary_is_none_accuracy_not_box_f1(self):
        cells = cells_fixture()
        for cell in cells["negative"].values():
            cell["mean_f1_0.50_0.95"] = 0.0
            cell["none_accuracy"] = 1.0
        summary = summary_cells(cells, protocol_fixture())
        self.assertEqual(summary["forms"]["negative"]["metric"], "none_accuracy")
        self.assertEqual(summary["forms"]["negative"]["mean"], 1.0)
        self.assertAlmostEqual(summary["current5_equal_view_macro"], 0.4)
        self.assertAlmostEqual(summary["current5_training_weighted"], 8 / 15)

    def test_form_mean_unweighted_by_game_request_count(self):
        protocol = protocol_fixture()
        protocol["games"]["beta"]["forms"]["category"]["sampled_rows"] = 100
        cells = cells_fixture()
        cells["category"]["beta"]["requests"] = 100
        summary = summary_cells(cells, protocol)
        self.assertTrue(summary["complete"])
        self.assertAlmostEqual(summary["forms"]["category"]["mean"], 0.1)

    def test_partial_or_invalid_coverage_never_publishes_full_summary(self):
        for bad in (None, 0, 1, 3, "2", math.nan):
            cells = cells_fixture()
            if bad is None:
                del cells["category"]["alpha"]
            else:
                cells["category"]["alpha"]["requests"] = bad
            with self.subTest(requests=bad):
                summary = summary_cells(cells, protocol_fixture())
                self.assertFalse(summary["complete"])
                self.assertIsNone(summary["current5_equal_view_macro"])
                self.assertIsNone(summary["current5_training_weighted"])
                self.assertIsNone(summary["current5_worst_game"])
                self.assertEqual(summary["coverage"]["category"]["complete_games"], 1)

    def test_nonfinite_metric_or_wrong_form_blocks_complete(self):
        for bad in (None, "0.2", math.nan, math.inf, True):
            cells = cells_fixture()
            cells["category"]["alpha"]["mean_f1_0.50_0.95"] = bad
            self.assertFalse(summary_cells(cells, protocol_fixture())["complete"])
        cells = cells_fixture()
        cells["category"]["alpha"]["form"] = "negative"
        self.assertFalse(summary_cells(cells, protocol_fixture())["complete"])

    def test_wrong_game_cannot_replace_missing_game(self):
        cells = cells_fixture()
        cells["category"]["unexpected"] = cells["category"].pop("alpha")
        summary = summary_cells(cells, protocol_fixture())
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["coverage"]["category"]["missing_games"], ["alpha"])

    def test_form_with_no_expected_games_is_not_complete(self):
        protocol = protocol_fixture()
        for info in protocol["games"].values():
            info["forms"]["negative"]["sampled_rows"] = 0
        self.assertFalse(summary_cells(cells_fixture(), protocol)["complete"])

    def test_game_missing_one_form_excluded_only_from_that_form(self):
        protocol = protocol_fixture()
        del protocol["games"]["alpha"]["forms"]["negative"]
        cells = cells_fixture()
        del cells["negative"]["alpha"]
        summary = summary_cells(cells, protocol)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["coverage"]["negative"]["expected_games"], 1)
        self.assertAlmostEqual(summary["forms"]["negative"]["mean"], 0.6)
        self.assertAlmostEqual(summary["current5_worst_game"]["score"], 0.15)

    def test_invalid_protocol_and_weights(self):
        protocol = protocol_fixture()
        protocol["protocol"] = "other"
        with self.assertRaises(ValueError):
            summary_cells(cells_fixture(), protocol)
        for bad in (0, -1, math.inf, math.nan):
            protocol = protocol_fixture()
            protocol["train_weights"] = {form: bad for form in FORMS}
            with self.assertRaises(ValueError):
                summary_cells(cells_fixture(), protocol)


class IndependenceTests(unittest.TestCase):
    def test_only_standard_library_imports_and_documented_source_hashes(self):
        path = Path(__file__).resolve().parents[1] / "eval" / "current5_metrics.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                roots.add(node.module.split(".")[0])
        self.assertLessEqual(roots, {"__future__", "math", "re", "statistics", "collections"})
        self.assertEqual(set(SOURCE_SHA256), {"eval_current5.py", "la-common.py"})
        for digest in SOURCE_SHA256.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
