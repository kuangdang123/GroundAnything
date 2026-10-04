"""Launch GroundAnything VLM SFT through ms-swift.

The repository owns the recipe and preflight contract; ms-swift owns the
model-specific SFT loop.  Keeping this adapter small makes the launcher
auditable and avoids maintaining a second multimodal collator in GAM.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
_PATH_FIELDS = {"path", "manifest", "output_dir", "resume_from_checkpoint"}


def _project_path(value: str, *, must_exist: bool = False) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or value.startswith("~"):
        raise ValueError(f"resource path must be project-relative: {value}")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"resource path escapes project: {value}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"required VLM training input does not exist: {value}")
    return resolved


def load_recipe(path: str) -> dict[str, Any]:
    recipe_path = _project_path(path, must_exist=True)
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise ValueError("VLM native_config must be a mapping")
    if recipe.get("mode") == "DLM":
        raise ValueError("VLM launcher cannot consume a DLM recipe")
    model = recipe.get("model")
    if not isinstance(model, dict) or model.get("family") != "groundinganything_vlm":
        raise ValueError("VLM recipe model.family must be groundinganything_vlm")
    model_path = model.get("path")
    if not isinstance(model_path, str):
        raise ValueError("VLM recipe requires model.path")
    _project_path(model_path, must_exist=True)
    datasets = recipe.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("VLM recipe requires a non-empty datasets list")
    for entry in datasets:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("each VLM dataset requires a project-relative path")
        _project_path(entry["path"], must_exist=True)
    training = recipe.get("training")
    if not isinstance(training, dict):
        raise ValueError("VLM recipe requires a training mapping")
    tuner = str(training.get("tuner_type", "lora")).lower()
    if tuner not in {"lora", "full"}:
        raise ValueError("training.tuner_type must be lora or full")
    if tuner == "lora":
        rank = training.get("lora_rank", 64)
        if type(rank) is not int or rank < 1:
            raise ValueError("training.lora_rank must be a positive integer")
        targets = training.get("lora_target_modules")
        if not isinstance(targets, list) or not targets or not all(isinstance(x, str) and x for x in targets):
            raise ValueError("LoRA requires a non-empty lora_target_modules list")
    return recipe


def _swift_args(recipe: dict[str, Any], *, resume: str | None, nproc_per_node: int) -> list[str]:
    model = recipe["model"]
    data = recipe["datasets"]
    training = recipe["training"]
    args = ["sft", "--model", str(_project_path(model["path"]))]
    args += ["--dataset", *[str(_project_path(item["path"])) for item in data]]
    args += ["--output_dir", str(_project_path(model.get("output_dir", "outputs/vlm_lora")))]
    args += ["--tuner_type", str(training.get("tuner_type", "lora"))]
    if training.get("tuner_type", "lora") == "lora":
        args += ["--lora_rank", str(training.get("lora_rank", 64))]
        args += ["--lora_alpha", str(training.get("lora_alpha", 128))]
        args += ["--lora_dropout", str(training.get("lora_dropout", 0.05))]
        args += ["--target_modules", *[str(x) for x in training["lora_target_modules"]]]
    for key in (
        "freeze_vit", "freeze_aligner", "freeze_llm", "learning_rate", "weight_decay",
        "warmup_ratio", "max_length", "max_steps", "num_train_epochs",
        "per_device_train_batch_size", "gradient_accumulation_steps", "gradient_checkpointing",
        "dataset_num_proc", "dataloader_num_workers", "save_steps", "logging_steps",
        "save_strategy", "report_to", "torch_dtype", "attn_impl", "packing",
    ):
        if key in training:
            args += ["--" + key, str(training[key]).lower() if isinstance(training[key], bool) else str(training[key])]
    if resume:
        args += ["--resume_from_checkpoint", str(_project_path(resume, must_exist=True))]
    if nproc_per_node > 1:
        args += ["--nproc_per_node", str(nproc_per_node)]
    return args


def build_command(recipe: dict[str, Any], *, resume: str | None = None, nproc_per_node: int = 1) -> list[str]:
    """Build a Swift command without importing GPU/model code."""
    return [sys.executable, "-m", "swift", *_swift_args(recipe, resume=resume, nproc_per_node=nproc_per_node)]


def _write_audit(recipe: dict[str, Any], command: list[str]) -> None:
    output = _project_path(recipe["model"].get("output_dir", "outputs/vlm_lora"))
    output.mkdir(parents=True, exist_ok=True)
    training = recipe["training"]
    audit = {
        "model_family": recipe["model"]["family"],
        "base_model": recipe["model"]["path"],
        "tuner_type": training.get("tuner_type", "lora"),
        "freeze_vit": bool(training.get("freeze_vit", True)),
        "freeze_aligner": bool(training.get("freeze_aligner", False)),
        "freeze_llm": bool(training.get("freeze_llm", True)),
        "lora_rank": training.get("lora_rank") if training.get("tuner_type", "lora") == "lora" else None,
        "lora_target_modules": training.get("lora_target_modules", []) if training.get("tuner_type", "lora") == "lora" else [],
        "command": command,
    }
    (output / "trainable_parameter_policy.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="project-relative VLM native config")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.nproc_per_node < 1 or args.nnodes < 1 or not 0 <= args.node_rank < args.nnodes:
        parser.error("invalid distributed topology")
    recipe = load_recipe(args.config)
    command = build_command(recipe, resume=args.resume_from_checkpoint, nproc_per_node=args.nproc_per_node)
    print(json.dumps({"command": command, "shell_command": shlex.join(command)}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    if importlib.util.find_spec("swift") is None:
        raise SystemExit("ms-swift is required for vlm-train; install a tested swift release before execution")
    _write_audit(recipe, command)
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
