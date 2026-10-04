import json
import tempfile
import unittest
import sys
from pathlib import Path

from train.vlm.train_vlm_lora import _swift_args, build_command


class VLMLauncherTest(unittest.TestCase):
    def recipe(self):
        return {
            "model": {"path": "weights/base_model", "output_dir": "outputs/test_vlm"},
            "datasets": [{"path": "data/train.gam.jsonl"}],
            "training": {
                "tuner_type": "lora",
                "lora_rank": 64,
                "lora_alpha": 128,
                "lora_target_modules": ["q_proj", "v_proj"],
                "freeze_llm": True,
                "freeze_vit": True,
                "freeze_aligner": False,
                "max_steps": 3,
            },
        }

    def test_builds_explicit_lora_command(self):
        command = build_command(self.recipe(), nproc_per_node=2)
        self.assertEqual(command[:3], [sys.executable, "-m", "swift"])
        self.assertIn("--lora_rank", command)
        self.assertIn("64", command)
        self.assertIn("--target_modules", command)
        self.assertIn("--nproc_per_node", command)

    def test_full_tuning_does_not_emit_lora_flags(self):
        recipe = self.recipe()
        recipe["training"] = {"tuner_type": "full", "freeze_llm": False, "freeze_vit": False}
        args = _swift_args(recipe, nproc_per_node=1, resume=None)
        self.assertIn("--tuner_type", args)
        self.assertNotIn("--lora_rank", args)


if __name__ == "__main__":
    unittest.main()
