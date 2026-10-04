import unittest

from eval.game_ui.metrics import aggregate_results, box_iou, evaluate_sample, normalize_box


_MISSING = object()


def sample(prediction, *, gt_boxes=_MISSING, gt_space="pixel", category="button", latency=10):
    boxes = [[10, 10, 30, 30]] if gt_boxes is _MISSING else gt_boxes
    return {
        "id": "s1",
        "width": 100,
        "height": 100,
        "category": category,
        "ground_truth": {"coordinate_space": gt_space, "boxes": boxes},
        "prediction": {"status": "success", "coordinate_space": "pixel", "predictions": prediction, "latency_ms": latency},
    }


class GameUIMetricsTests(unittest.TestCase):
    def test_ga_grid_maps_to_pixel(self):
        box = normalize_box([0, 0, 999, 999], 1920, 1080, "ga_999")
        self.assertEqual(box.values, (0.0, 0.0, 1920.0, 1080.0))

    def test_perfect_box_and_thresholds(self):
        result = evaluate_sample(sample([{"label": "button", "coordinates": [10, 10, 30, 30]}]))
        aggregate = aggregate_results([result])
        self.assertEqual(aggregate["thresholds"]["0.5"]["tp"], 1)
        self.assertEqual(aggregate["f1_at_0.50"], 1.0)

    def test_none_and_parse_error_are_distinct(self):
        negative = evaluate_sample(sample([], gt_boxes=[]))
        self.assertEqual(negative["none_outcome"], "none_correct")
        failed = evaluate_sample({**sample([]), "prediction": {"status": "invalid_response"}})
        self.assertTrue(failed["parse_error"])
        self.assertEqual(failed["none_outcome"], None)

    def test_one_prediction_cannot_match_two_ground_truth_boxes(self):
        result = evaluate_sample(sample([{"label": "button", "coordinates": [10, 10, 30, 30]}], gt_boxes=[[10, 10, 30, 30], [10, 10, 30, 30]]))
        self.assertEqual(result["thresholds"]["0.5"]["tp"], 1)
        self.assertEqual(result["thresholds"]["0.5"]["fn"], 1)

    def test_invalid_coordinates_count_as_illegal(self):
        result = evaluate_sample(sample([{"label": "button", "coordinates": [0, 0, 120, 20]}]))
        self.assertTrue(result["illegal_coordinates"])
        self.assertEqual(result["invalid_prediction_boxes"], 1)

    def test_per_class_and_latency(self):
        first = evaluate_sample(sample([{"label": "button", "coordinates": [10, 10, 30, 30]}], category="button", latency=5))
        second = evaluate_sample(sample([{"label": "icon", "coordinates": [10, 10, 30, 30]}], category="icon", latency=15))
        aggregate = aggregate_results([first, second])
        self.assertEqual(set(aggregate["per_class"]), {"button", "icon"})
        self.assertEqual(aggregate["latency_ms"]["p50"], 10.0)
        self.assertEqual(aggregate["false_positive"], 0)
        self.assertEqual(aggregate["false_negative"], 0)


if __name__ == "__main__":
    unittest.main()
