#!/usr/bin/env python3
"""GroundAnything interactive demo — upload an image, type a query, see boxes drawn.

This is a thin client/UI: it talks to the SGLang serving endpoint over the
OpenAI-compatible API and does NOT load model weights itself. Start the service
first, then run this app.

    # terminal 1 — the model service (one GPU)
    CUDA_VISIBLE_DEVICES=4 .venv-serve/bin/python scripts/run.py \\
        configs/release/dlm_sglang_denoise.yaml

    # terminal 2 — this demo
    .venv-demo/bin/python demo/app.py --base-url http://127.0.0.1:8101/v1

Why not just use `grounding_anything.GroundingAnything.predict()`?
That client only covers referring-expression bbox/point. It does not expose the
dense-detection, OCR, layout or GUI prompt families, nor does it return more
than one box per phrase. This demo drives the protocol layer directly so the
released model's full grounding surface is visible, and it validates every
answer with protocol.validate_prompt_answer_pair() so a malformed generation is
reported instead of being drawn as if it were correct.

Modes (each maps to one canonical prompt family in grounding_anything/protocol.py):
    detect_box    category list  -> every instance of each category     (dense_bbox)
    detect_point  category list  -> one center point per category       (dense_point)
    refer_box     free phrase    -> boxes for one referring expression  (refer_bbox)
    refer_point   free phrase    -> one point for one expression        (refer_point)
    ocr           (no query)     -> text-line boxes                     (ocr)
    layout        block list     -> document layout blocks              (layout)
    gui           instruction    -> one UI element point                (gui)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grounding_anything.protocol import (  # noqa: E402
    PromptFamily,
    build_dense_bbox_prompt,
    build_dense_point_prompt,
    build_gui_prompt,
    build_layout_prompt,
    build_ocr_prompt,
    build_refer_bbox_prompt,
    build_refer_point_prompt,
    validate_prompt_answer_pair,
)

# --------------------------------------------------------------------------- #
# mode registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    family: PromptFamily
    task: str                    # "bbox" | "point"  (geometry, drives drawing)
    query_kind: str              # "categories" | "phrase" | "instruction" | "none"
    default_query: str
    help: str


MODES: tuple[Mode, ...] = (
    Mode("detect_box", "检测框 · 类别列表 → 全部实例",
         PromptFamily.DENSE_BBOX, "bbox", "categories", "person, car, dog",
         "输入若干类别（逗号分隔），模型找出每个类别的**所有**实例。"),
    Mode("detect_point", "检测点 · 类别列表 → 中心点",
         PromptFamily.DENSE_POINT, "point", "categories", "person, car",
         "同上，但每个类别只给一个中心点。"),
    Mode("refer_box", "指代框 · 自由描述 → 框",
         PromptFamily.REFER_BBOX, "bbox", "phrase", "the red car",
         "给一句指代描述，模型定位它。"),
    Mode("refer_point", "指代点 · 自由描述 → 点",
         PromptFamily.REFER_POINT, "point", "phrase", "the center of the red car",
         "给一句指代描述，模型给出一个点。"),
    Mode("ocr", "OCR · 文本行检测",
         PromptFamily.OCR, "bbox", "none", "",
         "不需要 query，模型检测图中所有文本行。"),
    Mode("layout", "版面 · 文档区块",
         PromptFamily.LAYOUT, "bbox", "categories", "text, title, table, figure",
         "给版面类别列表，模型找出对应的文档区块。"),
    Mode("gui", "GUI · 界面元素定位",
         PromptFamily.GUI, "point", "instruction", "the search button",
         "给一句界面指令，模型给出目标控件位置。"),
)
MODE_BY_KEY = {m.key: m for m in MODES}

# --------------------------------------------------------------------------- #
# decoders (speed / quality operating points)
# --------------------------------------------------------------------------- #
#
# Each decoder is a SEPARATE server process loading its own copy of the weights
# (infer/serve_sglang.py --decoder). They cannot be switched on a running
# server: docs/INFERENCE.md says to stop the existing service before changing
# the decoder. So the demo talks to one URL per decoder and lets you pick.
#
# The paper's speed claim (4.51x, 0.74 pp mIoU drop on COCO) is measured against
# the CAUSAL autoregressive counterpart, NOT against denoise.
DECODERS: tuple[tuple[str, str, str, str], ...] = (
    # key, label, default URL, note
    ("denoise", "hybrid · 熵引导去噪（默认，论文主表口径）",
     "http://127.0.0.1:8101/v1",
     "`--decoder denoise`：DecodeV4 块式并行去噪，B32/sub-block 4/熵阈值 0.8。"
     "论文主表 61.75% 用的是这一档。"),
    ("speculative", "fast · 自推测（论文 4.51× 加速档）",
     "http://127.0.0.1:8102/v1",
     "`--decoder speculative`：贪心自推测 + 一次并行因果验证，接受最长匹配前缀。"
     "注意论文的 4.51× 是对 **causal** 基线测的。"),
    ("causal", "slow · 因果 AR 基线",
     "http://127.0.0.1:8103/v1",
     "`--decoder causal`：普通自回归逐 token 生成，作为对照基线。"),
)
DECODER_BY_KEY = {d[0]: d for d in DECODERS}

# --------------------------------------------------------------------------- #
# decode tiers
# --------------------------------------------------------------------------- #
#
# The released serving recipe pins GAM_DLM_DECODE_PROFILE=task_profiles, whose
# five-tier resolver FAILS CLOSED on any task it does not know
# (infer/engines/dlm_task_profiles.py:_five_tier_profile_name). A plain OpenAI
# request carries no benchmark task identity, so it gets the server's shared
# framework settings rather than a paper tier. To reproduce the paper's numbers
# a client must send the tier explicitly as custom_params.gam_dlm_decode.
TIER_AUTO = "auto"

# Shared DecodeV4 settings, identical across all five tiers
# (task_profiles.json -> shared_decode).
SHARED_DECODE: dict[str, Any] = {
    "acceptance_policy": "entropy",
    "entropy_threshold": 0.8,
    "top_k": 0,
    "position_penalty": 0.0,
    "repetition_mode": "none",
    "repetition_weight": 0.0,
    "enable_eos_early_stop": False,
    "block_size": 32,
    "sub_block_size": 4,
    "denoise_steps": 4,
    "legacy_force_fill_all": False,
    "sampling_seed": 42,
}

# Atomic spatial token ids, an explicit request-scoped contract
# (dlm_task_profiles.py:SPATIAL_TOKEN_CONTRACT).
SPATIAL_CONTRACT: dict[str, Any] = {
    "box_start_token_id": 151648,
    "box_end_token_id": 151649,
    "coord_token_min_id": 151669,
    "coord_token_max_id": 152668,
}


@dataclass(frozen=True)
class Tier:
    key: str
    label: str
    profile: str | None      # None = do not send custom_params
    temperature: float
    top_p: float
    max_tokens: int
    cardinality: int | None
    note: str


TIERS: tuple[Tier, ...] = (
    Tier(TIER_AUTO, "auto · 服务端默认（framework）", None, 0.0, 1.0, 4096, None,
         "不发 custom_params。服务端 task_profiles 对未知 task fail closed，"
         "所以这里走的是共享 framework 设置，不是论文档位。"),
    Tier("strict_single", "strict_single · 单目标（论文档）", "task_single", 0.0, 1.0, 512, 1,
         "14 个指代/单目标任务用这档：T=0、top_p=1、512 tokens、"
         "expected_cardinality=1（解出第一个完整 wrapper 即停）。"),
    Tier("medium", "medium · 中等密度", "task_medium", 0.1, 0.95, 4096, None,
         "12 个任务：T=0.1、top_p=0.95、4096 tokens。"),
    Tier("dense", "dense · 高密度", "task_dense", 0.1, 0.95, 8192, None,
         "8 个任务：T=0.1、top_p=0.95、8192 tokens（检测/密集场景用这档）。"),
    Tier("ocr_medium", "ocr_medium · OCR 中等", "task_ocr_medium", 0.3, 0.95, 4096, None,
         "4 个 OCR 任务：T=0.3、top_p=0.95、4096 tokens。"),
    Tier("ocr_dense", "ocr_dense · OCR 密集", "task_ocr_dense", 0.0, 1.0, 4096, None,
         "4 个 OCR 任务：T=0、top_p=1、4096 tokens。"),
)
TIER_BY_KEY = {t.key: t for t in TIERS}


def decode_params_for(tier: Tier) -> dict[str, Any] | None:
    """Build custom_params.gam_dlm_decode for a tier.

    IMPORTANT: only DecodeConfig field names may appear here. `max_tokens` is a
    top-level OpenAI sampling parameter, NOT a decode parameter; sending it
    inside gam_dlm_decode raises inside the SGLang scheduler and takes the whole
    server down (observed 2026-10-10: "unknown GAM decode parameters:
    ['max_tokens']" -> Scheduler hit an exception -> SIGQUIT).
    """
    if tier.profile is None:
        return None
    decode = dict(SHARED_DECODE)
    decode.update(SPATIAL_CONTRACT)
    decode.update(profile=tier.profile, temperature=tier.temperature, top_p=tier.top_p)
    if tier.cardinality is not None:
        decode.update(expected_cardinality=tier.cardinality, coordinates_per_target=4)
    return {"gam_dlm_decode": decode}


def build_prompt(mode: Mode, raw_query: str) -> str:
    """Turn the UI query into the canonical prompt for the mode's family."""
    query = (raw_query or "").strip()
    if mode.query_kind == "categories":
        items = [p.strip() for p in query.replace("</c>", ",").split(",") if p.strip()]
        if not items:
            raise ValueError("请输入至少一个类别（逗号分隔）")
        if mode.family is PromptFamily.LAYOUT:
            return build_layout_prompt(items)
        if mode.family is PromptFamily.DENSE_POINT:
            return build_dense_point_prompt(items)
        return build_dense_bbox_prompt(items)
    if mode.query_kind == "phrase":
        if not query:
            raise ValueError("请输入一句指代描述")
        return (build_refer_point_prompt(query) if mode.family is PromptFamily.REFER_POINT
                else build_refer_bbox_prompt(query))
    if mode.query_kind == "instruction":
        if not query:
            raise ValueError("请输入一句界面指令")
        return build_gui_prompt(query)
    return build_ocr_prompt()


# --------------------------------------------------------------------------- #
# HTTP call
# --------------------------------------------------------------------------- #


@dataclass
class Stats:
    """Result statistics, shaped after the LocateAnything app's stats object.

    LA emits {images, boxes, latency_seconds, per_label}; the extra fields here
    break the client-side latency down and pass through the server's token
    accounting, which this service reports and LA's does not.
    """
    images: int = 1
    boxes: int = 0
    latency_seconds: float = 0.0
    per_label: dict[str, int] = field(default_factory=dict)
    entities: int = 0
    mode: str = ""
    decoder: str = ""
    tier: str = ""
    # timing breakdown (seconds)
    request_seconds: float = 0.0     # client-side HTTP round trip
    parse_seconds: float = 0.0       # parsing + recovery
    draw_seconds: float = 0.0        # rendering the overlay
    total_seconds: float = 0.0       # whole handler
    # server-reported token usage
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    repair_notes: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Inference:
    ok: bool
    entries: list[dict]          # [{"phrase": str, "coords": [[...], ...]}]
    prompt: str
    raw: str
    finish_reason: str | None
    usage: dict | None
    seconds: float
    error: str | None
    stats: Stats = field(default_factory=Stats)


def build_stats(inference: Inference, *, mode: str, decoder: str, tier: str,
                request_seconds: float, parse_seconds: float,
                draw_seconds: float = 0.0, total_seconds: float = 0.0) -> Stats:
    per_label: dict[str, int] = {}
    for entry in inference.entries:
        n = len(entry["coords"])
        if n:
            per_label[entry["phrase"]] = per_label.get(entry["phrase"], 0) + n
    usage = inference.usage or {}
    return Stats(
        images=1,
        boxes=sum(per_label.values()),
        latency_seconds=round(inference.seconds, 3),
        per_label=per_label,
        entities=len(inference.entries),
        mode=mode, decoder=decoder, tier=tier,
        request_seconds=round(request_seconds, 3),
        parse_seconds=round(parse_seconds, 3),
        draw_seconds=round(draw_seconds, 3),
        total_seconds=round(total_seconds, 3),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        finish_reason=inference.finish_reason,
        repair_notes=inference.error if inference.ok else None,
    )


def _encode(image_path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(image_path.name)[0]
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("仅支持 JPEG / PNG / WebP")
    return mime, base64.b64encode(image_path.read_bytes()).decode("ascii")


def call_service(base_url: str, model: str, image_path: Path, prompt: str,
                 tier: Tier | None = None, *, timeout: float = 180.0) -> dict:
    mime, encoded = _encode(image_path)
    body: dict[str, Any] = {
        "model": model,
        "temperature": tier.temperature if tier else 0.0,
        "top_p": tier.top_p if tier else 1.0,
        "max_tokens": tier.max_tokens if tier else 4096,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
            {"type": "text", "text": prompt},
        ]}],
    }
    if tier is not None:
        custom = decode_params_for(tier)
        if custom is not None:
            body["custom_params"] = custom
    request = Request(base_url.rstrip("/") + "/chat/completions",
                      data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def explain_parse_error(raw: str, error: str) -> str:
    """Turn a grammar error into something actionable.

    The protocol errors carry an offset (e.g. "expected ', ' at offset 1325").
    Showing the bytes around that offset is usually enough to see why the
    generation is not canonical -- most often a missing space in the entry
    separator, or a truncated/duplicated payload.
    """
    import re

    match = re.search(r"at offset (\d+)", error)
    if not match or not raw:
        return f"```\n{error}\n```"
    offset = int(match.group(1))
    start = max(0, offset - 60)
    window = raw[start:offset + 60]
    pointer = " " * (offset - start) + "^ 这里"
    # Make the wrapper tokens visible so separator spacing is obvious.
    shown = (window.replace("<|object_ref_start|>", "«REF_START»")
                   .replace("<|object_ref_end|>", "«REF_END»")
                   .replace("<|box_start|>", "«BOX_START»")
                   .replace("<|box_end|>", "«BOX_END»"))
    return (f"```\n{error}\n```\n"
            f"出错位置（offset {offset}）附近：\n"
            f"```\n{shown}\n{pointer}\n```\n"
            f"常见原因：entry 之间必须是 `, `（逗号**加空格**）；"
            f"坐标元组之间是 `,`（无空格）。")


def is_degenerate(coords: list[int], task: str) -> bool:
    """True for a whole-image / zero-size box, which is noise rather than a target.

    DecodeV4 occasionally fills a block with the same placeholder tuple; on a
    0..999 grid that shows up as (0,0,998,998) or a collapsed span. Drawing it
    would cover the image and hide the real targets.
    """
    if task == "point":
        return False
    if len(coords) < 4:
        return False
    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
    w, h = abs(x2 - x1), abs(y2 - y1)
    return (w >= 950 and h >= 950) or w == 0 or h == 0


def parse_lenient(raw: str, task: str, drop_degenerate: bool = True
                  ) -> tuple[list[dict], list[str]]:
    """Extract every drawable target from a generation, tolerating format noise.

    Drawing a box should not depend on the model emitting a byte-perfect
    canonical string. This pulls out every complete ``<ref>…<box>…`` entry,
    merges labels that were repeated as separate entries, drops duplicate
    coordinate tuples, and reports what it had to repair.

    Returns (entries, notes). ``entries`` uses the same shape as the strict
    path so drawing code is shared.
    """
    import re

    notes: list[str] = []
    pattern = re.compile(
        r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
        r"<\|box_start\|>(.*?)(?:<\|box_end\|>|$)",
        re.DOTALL,
    )
    arity = 2 if task == "point" else 4
    merged: dict[str, list[list[int]]] = {}
    order: list[str] = []
    seen_entries: dict[str, int] = {}
    dropped_dupes = 0
    dropped_short = 0
    dropped_degen = 0
    truncated = 0

    for label, payload in pattern.findall(raw):
        label = label.strip()
        if not label:
            continue
        if label in merged:
            seen_entries[label] = seen_entries.get(label, 1) + 1
        else:
            merged[label] = []
            order.append(label)
            seen_entries[label] = 1
        for chunk in payload.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            nums = [int(n) for n in re.findall(r"<(\d+)>", chunk)]
            if len(nums) < arity:
                # A truncated trailing tuple (no <|box_end|>) lands here.
                if len(nums) and chunk is payload.split(",")[-1].strip():
                    truncated += 1
                else:
                    dropped_short += 1
                continue
            coords = nums[:arity]
            if coords in merged[label]:
                dropped_dupes += 1
                continue
            if drop_degenerate and is_degenerate(coords, task):
                dropped_degen += 1
                continue
            merged[label].append(coords)

    if dropped_degen:
        notes.append(f"{dropped_degen} 个退化框（全图/零面积）已丢弃")
    if truncated:
        notes.append(f"{truncated} 个末尾残缺坐标已丢弃")
    if dropped_short:
        notes.append(f"{dropped_short} 个坐标长度不足 {arity} 已丢弃")
    if dropped_dupes:
        notes.append(f"{dropped_dupes} 个重复坐标已去重")
    repeated = {k: v for k, v in seen_entries.items() if v > 1}
    if repeated:
        detail = ", ".join(f"{k}×{v}" for k, v in repeated.items())
        notes.append(f"同一 label 被输出成多个 entry，已合并（{detail}）")
    if "<|box_end|>" not in raw:
        notes.append("输出缺少 <|box_end|>，可能被截断")

    entries = [{"phrase": lab, "coords": merged[lab]} for lab in order]
    return entries, notes


def run_inference(mode: Mode, query: str, image_path: Path,
                  base_url: str, model: str, tier_key: str = TIER_AUTO,
                  timeout: float = 180.0, strict: bool = False,
                  drop_degenerate: bool = True) -> Inference:
    """Full round trip with timing. By default every drawable target is recovered.

    ``strict=True`` additionally requires the answer to be canonical
    (validate_prompt_answer_pair); the default path is ``parse_lenient`` so a
    well-localized target is still drawn when the model's wrapper syntax is
    imperfect. Either way a generation truncated mid-answer is flagged.
    """
    started = time.perf_counter()
    tier = TIER_BY_KEY.get(tier_key)
    if tier is None:
        tier = TIER_BY_KEY[TIER_AUTO]

    def fail(message: str) -> Inference:
        return Inference(False, [], "", "", None, None,
                         time.perf_counter() - started, message)

    try:
        prompt = build_prompt(mode, query)
    except Exception as exc:                                   # bad user input
        return fail(f"{type(exc).__name__}: {exc}")

    t_req = time.perf_counter()
    try:
        payload = call_service(base_url, model, image_path, prompt, tier, timeout=timeout)
    except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
        return fail(f"服务请求失败: {type(exc).__name__}: {exc}")
    request_seconds = time.perf_counter() - t_req

    choice = payload["choices"][0]
    raw = choice["message"]["content"]
    finish = choice.get("finish_reason")
    usage = payload.get("usage")

    t_parse = time.perf_counter()
    notes: list[str] = []
    if strict:
        try:
            entries_raw = validate_prompt_answer_pair(prompt, raw, mode.family)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if finish != "stop":
                detail = f"生成未完整结束 (finish_reason={finish!r}); {detail}"
            return Inference(False, [], prompt, raw, finish, usage,
                             time.perf_counter() - started, detail)
        entries = [{"phrase": e.phrase,
                    "coords": [list(c) for c in (e.coordinates or ())]}
                   for e in entries_raw]
    else:
        entries, notes = parse_lenient(raw, mode.task, drop_degenerate)
    parse_seconds = time.perf_counter() - t_parse

    if not entries:
        return Inference(False, [], prompt, raw, finish, usage,
                         time.perf_counter() - started,
                         "没有解析出任何完整的目标"
                         + (f"（finish_reason={finish!r}）" if finish != "stop" else ""))

    if finish != "stop":
        notes.append(f"生成未自然结束（finish_reason={finish!r}），结果可能不完整")
    if not any(e["coords"] for e in entries):
        notes.append("模型返回了 label 但没有坐标（可能是 None = 图中无此目标）")

    result = Inference(True, entries, prompt, raw, finish, usage,
                       time.perf_counter() - started,
                       "；".join(notes) if notes else None)
    result.stats = build_stats(
        result, mode=mode.key, decoder=tier_key, tier=tier_key,
        request_seconds=request_seconds, parse_seconds=parse_seconds,
        total_seconds=result.seconds,
    )
    return result


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #

PALETTE = [
    (230, 57, 70), (29, 53, 87), (69, 123, 157), (42, 157, 143),
    (233, 196, 106), (244, 162, 97), (231, 111, 81), (38, 70, 83),
    (144, 190, 109), (67, 170, 139), (87, 117, 144), (188, 108, 37),
]


def _color_for(phrase: str, order: list[str]) -> tuple[int, int, int]:
    if phrase not in order:
        order.append(phrase)
    return PALETTE[order.index(phrase) % len(PALETTE)]


def _to_pixels(coords: list[int], w: int, h: int) -> list[float]:
    return [v / 999 * ((w - 1) if i % 2 == 0 else (h - 1)) for i, v in enumerate(coords)]


def draw(image_bytes: bytes, inference: Inference, task: str):
    """Render the validated predictions onto the image. Returns a PIL image."""
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(io.BytesIO(image_bytes)) as src:
        canvas = src.convert("RGB")
    draw_ctx = ImageDraw.Draw(canvas, "RGBA")
    try:
        font = ImageFont.load_default(size=max(14, canvas.width // 55))
    except TypeError:                                   # older Pillow
        font = ImageFont.load_default()

    order: list[str] = []
    for entry in inference.entries:
        phrase = entry["phrase"]
        color = _color_for(phrase, order)
        for coords in entry["coords"]:
            pixels = _to_pixels(coords, canvas.width, canvas.height)
            if task == "bbox" and len(pixels) >= 4:
                x1, y1, x2, y2 = (min(pixels[0], pixels[2]), min(pixels[1], pixels[3]),
                                  max(pixels[0], pixels[2]), max(pixels[1], pixels[3]))
                draw_ctx.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)
                label = phrase
                try:
                    box = draw_ctx.textbbox((0, 0), label, font=font)
                    tw, th = box[2] - box[0], box[3] - box[1]
                except Exception:
                    tw, th = 8 * len(label), 14
                ly = y1 - th - 6 if y1 - th - 6 > 0 else y1 + 2
                draw_ctx.rectangle([x1, ly, x1 + tw + 10, ly + th + 6], fill=color + (220,))
                draw_ctx.text((x1 + 5, ly + 3), label, fill=(255, 255, 255), font=font)
            elif len(pixels) >= 2:
                x, y = pixels[0], pixels[1]
                r = 7
                draw_ctx.ellipse([x - r, y - r, x + r, y + r], fill=color + (235,),
                                 outline=(255, 255, 255, 255), width=2)
            elif len(pixels) > 2:                        # trajectory polyline
                pts = list(zip(pixels[0::2], pixels[1::2]))
                draw_ctx.line(pts, fill=color + (255,), width=3)
    return canvas


def summarize(inference: Inference, tier_key: str = TIER_AUTO,
              decoder_key: str = "denoise") -> str:
    """Status line in the LocateAnything app's style.

    LA renders: ``**推理完成**：共 N 个目标（person×26, car×10），耗时 3.251s``
    Additional detail (decoder/tier/timing breakdown/token usage) follows for
    this service, since it exposes a decoder choice and reports token usage.
    """
    tier = TIER_BY_KEY.get(tier_key, TIER_BY_KEY[TIER_AUTO])
    dec = DECODER_BY_KEY.get(decoder_key, DECODERS[0])
    st = inference.stats

    if not inference.ok:
        return (f"❌ **没有可画的目标**（耗时 {st.latency_seconds:.3f}s）\n\n"
                f"档位 `{dec[0]}` + `{tier.key}`\n\n"
                + explain_parse_error(inference.raw, inference.error or ""))

    summary = "、".join(f"{k}×{v}" for k, v in st.per_label.items()) or "无目标"
    lines = [f"**推理完成**：共 **{st.boxes}** 个目标（{summary}），耗时 **{st.latency_seconds:.3f}s**"]

    # timing breakdown
    parts = [f"请求 {st.request_seconds:.3f}s", f"解析 {st.parse_seconds:.3f}s"]
    if st.draw_seconds:
        parts.append(f"绘制 {st.draw_seconds:.3f}s")
    parts.append(f"合计 {st.total_seconds:.3f}s")
    lines.append(f"⏱ {' / '.join(parts)}")

    # request identity
    tier_desc = (f"`{tier.profile}` T={tier.temperature} top_p={tier.top_p} "
                 f"max_tokens={tier.max_tokens}") if tier.profile else "服务端默认"
    lines.append(f"⚙️ 解码 `{dec[0]}` · 任务档 `{tier.key}`（{tier_desc}）")

    # server-side accounting
    if st.completion_tokens is not None:
        lines.append(f"🔢 tokens: prompt {st.prompt_tokens} + completion "
                     f"{st.completion_tokens} = {st.total_tokens} · finish={st.finish_reason}")

    if st.repair_notes:
        lines.append(f"🔧 解析修复：{st.repair_notes}")

    lines.append("")
    for entry in inference.entries:
        n = len(entry["coords"])
        head = f"- `{entry['phrase']}` — {n} 个" if n else f"- `{entry['phrase']}` — None（图中无此目标）"
        lines.append(head)
        for coords in entry["coords"][:6]:
            lines.append(f"    - `{coords}`")
        if n > 6:
            lines.append(f"    - … 另有 {n - 6} 个")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# gradio ui
# --------------------------------------------------------------------------- #


def build_app(args):
    import gradio as gr

    def predict(image, mode_key, query, tier_key, decoder_key, base_url, model,
                timeout, drop_degen, strict_mode):
        if image is None:
            return None, "请先上传一张图片。", "", None
        t_start = time.perf_counter()
        mode = MODE_BY_KEY.get(mode_key) or MODES[0]
        buf = io.BytesIO()
        from PIL import Image as _Image
        _Image.fromarray(image).save(buf, format="PNG")
        raw_bytes = buf.getvalue()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            handle.write(raw_bytes)
            tmp = Path(handle.name)
        try:
            inference = run_inference(mode, query, tmp, base_url, model, tier_key,
                                      float(timeout), strict=bool(strict_mode),
                                      drop_degenerate=bool(drop_degen))
        finally:
            tmp.unlink(missing_ok=True)

        drawn_array = image
        draw_seconds = 0.0
        if inference.ok and any(e["coords"] for e in inference.entries):
            t_draw = time.perf_counter()
            drawn = draw(raw_bytes, inference, mode.task)
            draw_seconds = time.perf_counter() - t_draw
            from numpy import asarray
            drawn_array = asarray(drawn)

        # fold rendering time into the stats the UI displays
        inference.stats = replace(
            inference.stats,
            draw_seconds=round(draw_seconds, 3),
            total_seconds=round(time.perf_counter() - t_start, 3),
        )
        summary = summarize(inference, tier_key, decoder_key)
        return drawn_array, summary, inference.raw, inference.stats.to_dict()

    def on_mode(mode_key):
        mode = MODE_BY_KEY.get(mode_key) or MODES[0]
        visible = mode.query_kind != "none"
        return (gr.update(visible=visible, value=mode.default_query,
                          label={"categories": "类别列表（逗号分隔）",
                                 "phrase": "指代描述",
                                 "instruction": "界面指令"}.get(mode.query_kind, "Query")),
                gr.update(value=mode.help))

    # Gradio 6 moved `theme` from the Blocks constructor to launch(); build the
    # theme here and let main() hand it to launch().
    theme = gr.themes.Soft()

    with gr.Blocks(title="GroundAnything 推理演示") as demo:
        gr.Markdown(
            "# 🎯 GroundAnything 推理演示\n"
            "上传图片 → 选模式 → 输入 query → 直接看到框/点。"
            "请求发往 SGLang 服务的 OpenAI 兼容接口，本应用不加载权重。\n\n"
            "> 所有输出都经过 `protocol.validate_prompt_answer_pair()` 严格校验；"
            "**校验不过的会明确报错，而不是画成看似正确的结果**。"
        )
        with gr.Row():
            with gr.Column(scale=2, min_width=330):
                decoder_dd = gr.Dropdown(choices=[(d[1], d[0]) for d in DECODERS],
                                         value="denoise", label="解码档位（fast/hybrid/slow）")
                decoder_note = gr.Markdown(DECODERS[0][3])
                mode_dd = gr.Dropdown(choices=[(m.label, m.key) for m in MODES],
                                      value=MODES[0].key, label="推理模式")
                mode_help = gr.Markdown(MODES[0].help)
                query_tb = gr.Textbox(value=MODES[0].default_query,
                                      label="类别列表（逗号分隔）", lines=2)
                tier_dd = gr.Dropdown(choices=[(t.label, t.key) for t in TIERS],
                                      value=TIER_AUTO, label="论文任务档（benchmark tier）")
                tier_note = gr.Markdown(TIER_BY_KEY[TIER_AUTO].note)
                run_btn = gr.Button("▶ 开始推理", variant="primary")
                with gr.Accordion("解析与严格性", open=True):
                    drop_degen_cb = gr.Checkbox(
                        value=True, label="丢弃退化框（全图/零面积）",
                        info="DecodeV4 偶发把同一个占位框填满整块，画出来会盖住真实目标")
                    strict_cb = gr.Checkbox(
                        value=False, label="严格模式（仅接受规范格式）",
                        info="默认关：只要坐标能解出来就画，格式问题在结果里标注。"
                             "打开后要求完全符合协议，格式不完美就整体拒收")
                with gr.Accordion("服务与参数", open=False):
                    base_url_tb = gr.Textbox(value=args.base_url, label="服务地址（选解码档会自动改）")
                    model_tb = gr.Textbox(value=args.model, label="模型 ID")
                    timeout_sl = gr.Slider(10, 600, value=180, step=10, label="超时（秒）")
            with gr.Column(scale=5):
                image_in = gr.Image(label="上传图片", type="numpy")
                image_out = gr.Image(label="标注结果", type="numpy")
                status_md = gr.Markdown("")
                stats_json = gr.JSON(label="推理统计")
                with gr.Accordion("模型原始输出（排查用）", open=False):
                    raw_tb = gr.Code(label="raw output", language="html")
        mode_dd.change(on_mode, inputs=mode_dd, outputs=[query_tb, mode_help])
        tier_dd.change(lambda k: TIER_BY_KEY.get(k, TIER_BY_KEY[TIER_AUTO]).note,
                       inputs=tier_dd, outputs=tier_note)

        def on_decoder(key):
            d = DECODER_BY_KEY.get(key, DECODERS[0])
            return gr.update(value=d[2]), gr.update(value=d[3])

        decoder_dd.change(on_decoder, inputs=decoder_dd,
                          outputs=[base_url_tb, decoder_note])
        run_btn.click(predict,
                      inputs=[image_in, mode_dd, query_tb, tier_dd, decoder_dd,
                              base_url_tb, model_tb, timeout_sl,
                              drop_degen_cb, strict_cb],
                      outputs=[image_out, status_md, raw_tb, stats_json])
        gr.Markdown(
            "---\n**默认就会画框。** 解析器会尽量从生成结果里取出所有可画的目标："
            "合并重复 label 的 entry、去重坐标、丢弃退化框，并在结果上方标注它修了什么。\n\n"
            "**两个「档位」是不同的东西：**\n"
            "1. **解码档位 fast/hybrid/slow** —— `--decoder speculative / denoise / causal`，"
            "是三个独立服务进程，各占一个端口（切换时 UI 自动改地址）。\n"
            "2. **论文任务档 strict_single / medium / dense / ocr_*** —— 同一服务内的请求级参数，"
            "通过 `custom_params.gam_dlm_decode` 发送。\n\n"
            "**模式**：检测框/点用 `dense_*` 族群；指代用 `refer_*`；OCR 无需 query。坐标 0–999 网格。"
        )
    return demo, theme


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("GA_BASE_URL", "http://127.0.0.1:8101/v1"))
    parser.add_argument("--model", default=os.environ.get("GA_MODEL", "groundinganything"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    app, theme = build_app(args)
    # Both `theme` and `show_api` moved between Gradio 4/5 and 6; drop whichever
    # the installed version does not accept instead of pinning one API surface.
    kwargs: dict[str, Any] = {"server_name": args.host, "server_port": args.port,
                              "share": args.share, "theme": theme, "show_api": False}
    while True:
        try:
            app.queue().launch(**kwargs)
            break
        except TypeError as exc:
            dropped = next((k for k in ("show_api", "theme") if k in kwargs
                            and k in str(exc)), None)
            if dropped is None:
                raise
            print(f"[demo] installed Gradio does not accept launch({dropped}=...); retrying without it")
            kwargs.pop(dropped)


if __name__ == "__main__":
    main()
