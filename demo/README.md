# GroundAnything 推理演示（Gradio）

上传图片 → 选模式 → 输入 query → **直接看到框/点**。

仓库原本只有 `examples/predict.py` 命令行版，没有交互界面；本目录补上。

## 架构：客户端/服务分离

```
┌────────────────────┐        HTTP (OpenAI 兼容)        ┌──────────────────┐
│  demo/app.py        │  ──────────────────────────────> │  SGLang 服务      │
│  Gradio 6 (无模型)   │   /v1/chat/completions          │  :8101           │
│  :7860              │ <────────────────────────────── │  GPU 4           │
└────────────────────┘                                  └──────────────────┘
```

**本应用不加载模型权重**，只发 HTTP 请求。这跟参考实现
(`LocateAnything/game_finetune/app/gradio_app.py`) 不同——那个 app 用 transformers
在进程内直载模型；GroundAnything 发布的是 SGLang 服务，所以 UI 保持轻量客户端即可。

## 启动

```bash
# 终端 1：模型服务（占一张 GPU）
cd /workspace/project-nas-1000073/liangjiaqi/GroundAnything
CUDA_VISIBLE_DEVICES=4 .venv-serve/bin/python -u -m infer.serve_sglang \
  --model weights/dlm_bundle --decoder denoise \
  --output outputs/sglang/denoise --port 8101
# 就绪标志：日志出现 "The server is fired up and ready to roll!"
# 首次加载 9.69 GB 权重约 6 分钟（从 NAS 读）

# 终端 2：界面
PYTHONPATH=$PWD/demo:$PWD /workspace/cpfs-data/ga-demo-venv/bin/python demo/app.py \
  --base-url http://127.0.0.1:8101/v1 --model groundinganything --port 7860
```

浏览器打开 `http://<devbox>:7860`。从 Windows 本地用 SSH 隧道：

```powershell
ssh -L 7860:127.0.0.1:7860 notebook
# 然后本地浏览器开 http://127.0.0.1:7860
```

## 七种模式

几何类型（`bbox` / `point`）与语义族群（prompt family）在协议层是正交的，
本应用把它们组合成 7 个模式，覆盖已发布模型的全部 grounding 面：

| 模式 | prompt family | 几何 | query |
|---|---|---|---|
| 检测框 | `dense_bbox` | box | 类别列表（逗号分隔），输出**每个类别的全部实例** |
| 检测点 | `dense_point` | point | 类别列表 |
| 指代框 | `refer_bbox` | box | 一句指代描述 |
| 指代点 | `refer_point` | point | 一句指代描述 |
| OCR | `ocr` | box | 无需 query，检测所有文本行 |
| 版面 | `layout` | box | 区块类别列表 |
| GUI | `gui` | point | 一句界面指令 |

> 内置客户端 `grounding_anything.GroundingAnything.predict()` 只覆盖
> `refer_bbox`/`refer_point`，且每 phrase 只回一个框。本应用直接驱动
> `grounding_anything/protocol.py` 的协议层，所以能把 dense / OCR / layout / GUI 都展示出来。

## 关键设计：宁可报错，不画错

所有输出都过 `protocol.validate_prompt_answer_pair(prompt, answer, family)`——
与评测链路同一个门。校验不通过时：

- **不画框**，而是把错误和**模型原始输出**一起显示（折叠面板）
- 明确区分"服务请求失败" / "生成未完整结束 (finish_reason)" / "协议校验失败"

这不是洁癖。实测中模型确实会产出**不合规**的输出（见下），如果宽松解析就会画出看似正确、
实则错误的框——对 grounding 任务来说这是最危险的失败模式。

## 输出格式与计时（对齐 LocateAnything app）

结果区对齐 LA（`LocateAnything/game_finetune/app/gradio_app.py`）的约定：
LA 产出 `{"images", "boxes", "latency_seconds", "per_label"}` 并渲染成
`**推理完成**：共 N 个目标（person×26, car×10），耗时 3.251s`，本 demo 沿用同一形状。

**`推理统计` JSON 面板**（LA 用 `gr.JSON`，这里也一样）：

```json
{
  "images": 1,
  "boxes": 36,
  "latency_seconds": 3.251,
  "per_label": { "person": 26, "car": 10 },
  "entities": 2,
  "mode": "detect_box",
  "decoder": "denoise",
  "tier": "auto",
  "request_seconds": 3.18,
  "parse_seconds": 0.021,
  "draw_seconds": 0.412,
  "total_seconds": 3.402,
  "prompt_tokens": 1043,
  "completion_tokens": 337,
  "total_tokens": 1380,
  "finish_reason": "stop"
}
```

比 LA 多出的字段都是为了定位瓶颈：LA 只给一个总 `latency`，
这里把**请求 / 解析 / 绘制**拆开，并透传服务端的 token 计量（LA 的服务不上报 usage）。

**状态行**（LA 风格 + 计时明细）：

```
**推理完成**：共 36 个目标（person×26、car×10），耗时 3.251s
⏱ 请求 3.180s / 解析 0.021s / 绘制 0.412s / 合计 3.402s
⚙️ 解码 denoise · 任务档 auto（服务端默认）
🔢 tokens: prompt 1043 + completion 337 = 1380 · finish=stop
```

计时用 `time.perf_counter()`（单调时钟，不受系统时间调整影响）。
`latency_seconds` 只算模型往返，`total_seconds` 含解析与绘制——两者分开才能看出
慢在服务端还是本地。

## ⚠️ 两个「档位」是不同的东西，别混淆

| | 解码档位（decoder） | 论文任务档（benchmark tier） |
|---|---|---|
| UI 名称 | `解码档位（fast/hybrid/slow）` | `论文任务档（benchmark tier）` |
| 取值 | `speculative` / `denoise` / `causal` | `strict_single` / `medium` / `dense` / `ocr_medium` / `ocr_dense` |
| 粒度 | **进程级**——`--decoder` 启动参数 | **请求级**——`custom_params.gam_dlm_decode` |
| 切换代价 | 要**另起一个服务**（各自加载一份权重、各自端口） | 同一个服务内即时生效 |
| 改的是什么 | 生成算法（自推测 / 块式去噪 / 因果 AR） | 采样温度、top_p、token 预算、单目标基数 |

`docs/INFERENCE.md` 明确要求**改变 decoder 前必须先停掉现有服务**，所以三档在 demo 里是
**三个并列的服务**，切换档位时 UI 自动改服务地址：

| 档位 | decoder | 默认端口 | 论文定位 |
|---|---|---|---|
| **hybrid** | `denoise` | 8101 | 熵引导 DecodeV4，论文主表 61.75% 用这档 |
| **fast** | `speculative` | 8102 | 自推测，论文报 4.51× 加速（**基线是 causal**） |
| **slow** | `causal` | 8103 | 因果 AR 基线 |

### 三档实测对比（teaser.png，3 次取中位数，同一图同一 query）

| 模式 | denoise (hybrid) | speculative (fast) | causal (slow) |
|---|---|---|---|
| `detect_box person` | 5.37 s / 31 目标 | **5.19 s / 78** | **13.64 s / 74** |
| `refer_point person` | 2.88 s / 30 点 | **1.03 s / 1** | **0.79 s / 1** |

**关于论文的 4.51×**：那个数字是对 **causal** 基线测的，不是对 denoise。
实测 speculative 比 causal 快约 **2.6×**（5.19 vs 13.64），方向一致但没到 4.51×——
论文的 operating point 与这里不同（我的服务跑 BF16、无 CUDA Graph、单请求），
**不能直接拿这里的数字去对标论文**。

**三档的输出内容本身也不同**（78 / 31 / 74 个目标），说明这不只是速度取舍，
而是**不同的模型行为**——这点比论文的叙述更强：换 decoder 会改变预测结果，不只是变快变慢。

### 同卡并存需要覆盖显存预留

`infer/serve_sglang.py` 原本把 `--mem-fraction-static` 硬编码为 `0.65`（80 GB 卡上约 52 GB），
**两个服务无法共存于同卡**。已加向后兼容的环境变量覆盖（默认值不变）：

```bash
GAM_SGLANG_MEM_FRACTION=0.25 CUDA_VISIBLE_DEVICES=4 .venv-serve/bin/python -u -m infer.serve_sglang \
  --model weights/dlm_bundle --decoder causal --output outputs/sglang/causal --port 8103
```

本机实测：`denoise` 与 `causal` 共用 GPU 4（各占 39.6 GB / 42.6 GB 区间内），KV cache 30650 tokens。

## 论文任务档（benchmark tier）——请求级，即时生效

服务端发布配置固定 `GAM_DLM_DECODE_PROFILE=task_profiles`，而该模式对任何未注册的 task
**fail closed**（`infer/engines/dlm_task_profiles.py:_five_tier_profile_name` 要求恰好匹配一档）。
普通 OpenAI 请求不带 benchmark task 身份，**因此不会自动套用论文明细**——
`infer/decode/config.py:3-6` 明确写了：请求级 `custom_params.gam_dlm_decode` 优先级高于服务端 YAML。

所以 UI 提供档位下拉框，按 `infer/decode/configs/task_profiles.json` 原值发送：

| 档位 key | profile 名 | T | top_p | max_tokens | expected_cardinality | 论文覆盖 |
|---|---|---|---|---|---|---|
| `auto` | *（不发）* | 0.0 | 1.0 | 4096 | — | 服务端 framework 默认，**非论文档位** |
| `strict_single` | `task_single` | 0.0 | 1.0 | **512** | **1** | 14 个指代/单目标任务 |
| `medium` | `task_medium` | 0.1 | 0.95 | 4096 | — | 12 个任务 |
| `dense` | `task_dense` | 0.1 | 0.95 | **8192** | — | 8 个检测/密集任务 |
| `ocr_medium` | `task_ocr_medium` | 0.3 | 0.95 | 4096 | — | 4 个 OCR 任务 |
| `ocr_dense` | `task_ocr_dense` | 0.0 | 1.0 | 4096 | — | 4 个 OCR 任务 |

五档共享的 DecodeV4 设置（`shared_decode`）与原子空间 token 契约也一并发送：
`acceptance_policy=entropy`、`entropy_threshold=0.8`、`top_k=0`、`block_size=32`、
`sub_block_size=4`、`denoise_steps=4`、`sampling_seed=42`，
以及 `box_start/end_token_id=151648/151649`、`coord_token_min/max_id=151669/152668`。

**实测档位确实生效**（同一图同一 query）：`auto` 与 `strict_single`（T=0）两次结果完全一致
（160 tokens / 31 目标）；`dense`（T=0.1）两次**不同**（160/31 与 225/44）——采样参数真的改变了生成。

### ⚠️ 一个能打挂服务的坑（务必遵守）

**`custom_params.gam_dlm_decode` 里只能放 `DecodeConfig` 的字段名**（共 33 个，见
`infer/decode/config.py`）。`max_tokens` 是**顶层** OpenAI 采样参数，**不是** decode 参数。

我最初把整份档位字典塞进去，结果：

```
ValueError: unknown GAM decode parameters: ['max_tokens']
Scheduler hit an exception: ...
SIGQUIT received. It usually means one child failed.
```

**调度器抛异常 → 整个服务（含已加载的 23 GB 权重）退出**，不只是这一个请求失败。
在 `decode_params_for()` 里已加注释说明。

## 实测记录（2026-10-10，A800-80GB，GPU 4）

| 图 | 模式 | query | 结果 | 耗时 |
|---|---|---|---|---|
| `docs/assets/teaser.png` (3559×3246) | 检测框 | person | ✅ **31 个目标**，5/5 抽样像素验证命中 | 3.6 s |
| `docs/assets/teaser.png` | 指代点 | the person | ✅ 30 个点 | 2.2 s |
| `docs/assets/demo-poster.jpg` (1920×1080) | 检测点 | person | ✅ 9 个目标，5/5 命中 | 0.8 s |
| `docs/assets/demo-poster.jpg` | OCR | — | ✅ 检出 "Ground" / "Anything" 等文字框 | 0.7 s |

### 两类模型输出不合规（协议校验正确拦下，保留原始输出可复现）

**① entry 分隔符少了空格** —— 也就是常看到的 `GrammarError: expected ', ' ... found ','`：

```
<|object_ref_start|>dog<|object_ref_end|><|box_start|><299><335><384><774>,<673><418><755><744><|box_end|>,
                                                                                                          ^ 这里
```

`<|box_end|>` 之后跟的是 `,`，而协议要求 `, `（逗号**加空格**）。
坐标元组之间才是 `,`（无空格）——两者层级不同，容易混淆。
UI 的错误面板会自动截取出错 offset 附近的片段并标出插入符（`explain_parse_error()`）。

**② 同一 label 输出成多个 entry**：

```
...<|box_end|>, <|object_ref_start|>dog<|object_ref_end|><|box_start|>...
→ GrammarError: duplicate phrase entry: 'dog'
```

协议要求一个 label **只出现一次**，它的**所有实例放在同一个 payload 里**，
用 `,`（无空格）分隔坐标元组。模型有时会为同一个 label 开第二个 entry。

**③ 退化重复**：`refer_bbox` 上出现 `<0><0><998><998>` 重复 12 次或同一坐标元组重复数十次，
被 `CoordinateError: duplicate coordinate tuples are not canonical` 拒绝。

**④ refer 族更容易失控**：`refer_point "the center of the dog"` 曾生成 3841 tokens 才停，
而同图 `detect_point` 只要 0.8 s。

> 速度上能直观看到并行解码优势：`detect_box` 在 3559×3246 大图上 **3.6 秒**出 31 个框。

## 运行环境

- demo venv：`/workspace/cpfs-data/ga-demo-venv`（gradio 6.30.0 + pillow + numpy，约 30 包）
  —— 与模型栈隔离，**不需要** torch
- 模型环境：仓库内 `.venv-serve`（见根目录 `CONTEXT.md` 的部署节）

## Gradio 版本兼容

本应用同时兼容 Gradio 4/5/6。6.0 把 `theme` 与 `show_api` 从 `Blocks()` 挪到了 `launch()`，
代码用 `try/except TypeError` 逐个丢弃不支持的参数并重试，所以升级/降级 gradio 都不会启动失败。

## 默认就会画框：宽松解析（重要设计变更）

**画框不该依赖模型输出字节级完美。** 早期版本用 `validate_prompt_answer_pair()` 做硬门禁，
结果是模型明明定位到了人（`<302><336><383><772>`），却因为格式瑕疵**一个框都不画**。
现在默认走 `parse_lenient()`：

| 处理 | 说明 |
|---|---|
| 提取 | 用正则抓出所有完整的 `<ref>…<box>…` entry，**容忍缺失的 `<\|box_end\|>`** |
| 合并 | 同一 label 被输出成多个 entry → 合并成一个 |
| 去重 | 重复的坐标元组 → 只留一个 |
| 丢弃退化框 | 全图框 `(0,0,998,998)`、零面积框 → 丢弃（UI 可关） |
| 标注 | 结果上方列出**修了什么**（如"16 个退化框已丢弃；同一 label 合并 person×2"） |
| 截断提示 | `finish_reason != stop` 会明确警告结果可能不完整 |

UI 的「解析与严格性」面板提供两个开关：

- **丢弃退化框**（默认开）：DecodeV4 偶发把同一占位框填满整块，画出来会盖住真实目标
- **严格模式**（默认关）：打开后要求完全符合协议，格式不完美就整体拒收

严格校验**仍然保留**用于评测对齐——`scripts/evaluate.py` 走的是那套门禁，
论文数字必须在规范格式下比较。但**交互演示没有必要那么严**。

### 实测：同一失败案例的修复前后

`demo-poster.jpg` + `detect_box person`（模型输出：16 个重复全图框 + 两个 `person` entry）：

```
修复前： ❌ CoordinateError: duplicate coordinate tuples are not canonical   → 不画
修复后： ✅ [[302, 336, 383, 772]]                                          → 画出真实的人
        notes: 16 个退化框（全图/零面积）已丢弃；同一 label 被输出成多个 entry，已合并（person×2）
```

关闭「丢弃退化框」则保留全部：`[[0,0,998,998], [302,336,383,772]]`。

> ⚠️ 宽松解析只修复**表示形式**，不修复**定位错误**。如果模型框错了位置，宽松解析照样会画出来。
> 它降低的是"格式噪声导致的假失败"，不是"错误结果"。因此 notes 必须保留可见。

## 已知边界

- **没有验证数值正确性**：像素抽样证明"坐标 → 画布"这一环正确，但不证明坐标本身框得准。
  要评估定位质量必须走评测链路（`scripts/evaluate.py`），本 demo 不是评测工具。
- **退化生成是模型侧问题**，不是 prompt 或档位问题：实测同一图上六个档位输出完全一致，
  prompt 也是正确的 `Locate all the instances that match the following categories: person.`。
  只在少数图上观察到，**未系统量化占比**。
- 服务侧 `--max-running-requests 1`（发布配置），并发点击会排队。
- `--share` 未开启；跨机访问请用 SSH 隧道。
