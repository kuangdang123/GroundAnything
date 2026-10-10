# Claim → 代码映射：GroundAnything（arXiv 2609.39600）

> 配对文件：`docs/papers/2026-yu-groundanything.md`（单篇精读笔记）。
> 仓库根：`D:\Desktop\GroundAnything`（git remote `https://github.com/kuangdang123/GroundAnything.git`）。
> 所有 `path:line` 均来自本次 `read` / `grep` 工具的**原始输出行号**，未凭记忆填写。
> 本机为 Windows、无 CUDA、无 GPU，**未执行任何前向/训练/评测**，故"已实现"仅指**静态代码核对通过**，
> 不包含运行验证。标注 `[静态核对]` 的条目只读了代码，未跑过。
>
> 状态口径：
> - **IMPLEMENTED** = 能定位到实现该机制的主体代码，语义与论文一致
> - **PARTIAL** = 有实现但与论文描述存在可指出的差异（差异单独列出）
> - **NOT FOUND** = 仓库内无对应实现（已用针对性 grep 检索，检索式列在条目内）

---

## 0. 仓库结构速览（用于定位）

```
GroundAnything/
├── grounding_anything/    # 轻量 HTTP 客户端 + protocol.py（prompt/解析/坐标协议，725 行）
├── infer/
│   ├── decode/            # DecodeV4 请求契约 + 可靠性/采样/结构约束（sglang 路径）
│   ├── dlm/               # 原生 PyTorch DLM 参考服务 + hierarchy/causal/speculative 解码 + prefix cache
│   └── engines/sglang_gam/# 默认服务用的自定义 SGLang 适配层（含 gam_speculative.py / gam_hierarchy.py）
├── models/
│   ├── dlm/hybrid.py      # ★ DLM 训练包装器（Qwen3.5 路线）：分块、互补腐蚀、联合损失、flex mask（2152 行）
│   ├── dlm/vlm.py         # DLM 训练包装器（纯 Qwen3 路线，316 行）
│   └── vlm/               # GroundAnything-VLM 模型定义（与 GroundAnything-Model/ 同源副本）
├── train/
│   ├── dlm/train_dlm.py   # ★ AR→Diffusion 转换训练入口（--mode DLM，1742 行）
│   ├── tokenizer/         # 词表扩展 / 校验 / 权重核对
│   └── rl/                # Causal-JustGRPO、奖励、rollout、拓扑适配
├── eval/                  # 30-benchmark 评测套件（任务、prompt、指标）
├── tools/                 # JSONL 构建 / LocateAnything 标注转换（★ 不是模型转换）
└── configs/               # release / train / eval / rl 配方
```

Python 文件共 **229 个**（不含 `.git`、`__pycache__`、`third_party`；`tests/` 下有 **6 个**测试文件）。
最大单文件为 `models/dlm/hybrid.py`（2152 行），其次 `models/dlm/vlm.py`（316 行）+
`models/vlm/modeling_groundinganything.py`（1528 行）。
`GroundAnything-Model/` 是 HF 模型包（含 9.2 GB `model.safetensors`），与 `models/vlm/` 是同一份代码的两处拷贝。

---

## 1. 核心机制映射

### M1 — Blockwise 去噪（块划分 + 块内双向/块间因果）

| 项 | 结论 |
|:---|:---|
| 状态 | **IMPLEMENTED** `[静态核对]` |

| 论文声明 | 代码位置 | 函数/类 |
|:---|:---|:---|
| 响应 Y 切成长度 B=32 的连续块，块可跨几何 tuple 边界、遇 response-span 边界停止、末块可短（§3.4.2 / §A.7） | `models/dlm/hybrid.py:70` | `compute_response_blocks(labels, block_size)` |
| 训练固定 B=32；B=8 仅作为隔离实验 | `models/dlm/hybrid.py:184-195`；`train/dlm/train_dlm.py:1286-1291`；`train/dlm/train_dlm.py:1442` | `GAMQwen35DLM.__init__` 校验 `block_size in (8,32)`；`allow_nondefault_block_size` 仅 B8 实验放开 |
| 推理侧 B=32、子块 4、从左到右 | `infer/decode/config.py:29-31`（`block_size=32, sub_block_size=4, denoise_steps=4`）；`infer/decode/sglang_algorithm.py:269-272`（`for rel_start in range(denoise_start, total_len, config.sub_block_size)`） | `DecodeConfig` / `GAMDecodeV2Block`（类定义在 `infer/decode/sglang_algorithm.py:37`） |
| 参考 SGLang 服务强制 B=32 | `infer/engines/sglang_gam/__init__.py:55-60`；`infer/engines/sglang_gam/gam_speculative.py:71-72`；`infer/engines/sglang_gam/compat.py:109-112` | 启动期 hard check |
| 一个物理块 = 1 anchor + 31 mask | `infer/engines/sglang_gam/gam_hierarchy.py:96-101`（注释 `[AR] + [MASK] * 31 and returns 31 tokens`）；`infer/engines/sglang_gam/__init__.py:126-136`（`padding = block_size - 1`） | `GAMHierarchyBlock` |
| 块完成后一次因果 pass 重建权威 KV（D+1 NFE，"不验证、不拒绝"） | `infer/engines/sglang_gam/gam_hierarchy.py`（causal commit 路径）；`infer/engines/sglang_gam/compat.py:78-82`（注释明确区分 bidirectional denoise forward 与 eager causal verifier） | — |
| 块间因果 + 块内双向的**训练**注意力 | `models/dlm/hybrid.py:751`（`_block_mask`，其内 `:769-791` 的 `mask_mod`：`n2n = ~kv_clean & q_turn.eq(kv_turn)`；`n2c = kv_clean & q_turn.gt(kv_turn)`）；`models/dlm/hybrid.py:803`（`_packed_block_mask`，其内 `:817-829` 的 `noisy_mask_mod`） | `GAMQwen35DLM._block_mask` / `_packed_block_mask` |
| clean 流严格因果、不能读 noisy 视图 | `models/dlm/hybrid.py:1324-1330`（`create_causal_mask`）；`models/dlm/hybrid.py:641-647`（注释："The clean stream is independent of the noisy stream (C2C only)"） | — |
| 训练用 `flex_attention` + BlockMask（128×128 tile 分类） | `models/dlm/hybrid.py:40-67`（`_flex_attention_kernel` + `torch.compile`）；`models/dlm/hybrid.py:852-970`（`_packed_block_mask` 的 tile 分类） | — |

**与论文的差异**：无。

### M2 — 互补腐蚀 + 联合 AR/MDM 目标（式 2/6/7）

| 项 | 结论 |
|:---|:---|
| 状态 | **IMPLEMENTED**（与论文逐项吻合，已用发布训练配置交叉验证）`[静态核对]` |

| 论文声明 | 代码位置 | 函数 |
|:---|:---|:---|
| 每块采 `t_b ~ U(0,1)`，块内每位置 `m_i^(1) ~ Bernoulli(t_b)` | `models/dlm/hybrid.py:585-598` | `GAMQwen35DLM._sample_noise` |
| `m_i^(2) = 1 − m_i^(1)`（互补） | `models/dlm/hybrid.py:649-653`（`mask = item["selected"] if view == 0 else response & ~item["selected"]`） | `_build_streams` |
| 两个视图都 mask EOS | `models/dlm/hybrid.py:653`（`mask \|= response & item["ids"].eq(self.im_end_token_id)`）；`models/dlm/hybrid.py:597` | 同上 |
| 视觉 embedding 只出现在 clean 流 | `models/dlm/hybrid.py:645-646`（`clean_streams = [item["embeds"] ...]`，只对 clean 保留视觉 embedding）；`models/dlm/hybrid.py:656`（noisy 流用 `embed_tokens(noisy_ids)[item["text"]]`，`text` 掩掉视觉 token） | 同上 |
| `L = 0.5 L_MDM + 0.5 L_AR` | `models/dlm/hybrid.py:201-202`（默认 0.5/0.5）；`models/dlm/hybrid.py:234-243`（只允许 (0.5,0.5) 或 (1.0,0.0)）；`models/dlm/hybrid.py:2151`（`loss = self.mdm_loss_weight * mdm_loss + self.causal_loss_weight * causal_loss`） | `GAMQwen35DLM` |
| MDM 均值含被 mask 的 EOS，无 inverse-noise 加权 | `models/dlm/hybrid.py:2143`（`mdm_loss = self._selective_causal_ce(noisy_hidden, streams["noisy_labels"])`）；无 weighting 参数 | — |
| 保留 next-token shift | `models/dlm/hybrid.py`（`token_shift`）；`train/dlm/train_dlm.py:1440-1442` | — |

**✅ 已解决的疑点**：论文说块内每位置以概率 `t_b ~ U(0,1)` 被 mask；代码写成
`probabilities = minimum_noise_level + (1 - minimum_noise_level) * t`
（`models/dlm/hybrid.py:590-591`），形式上是**带下界的插值**。核对：
`minimum_noise_level` 默认值为 **0.0**（`models/dlm/hybrid.py:180`，校验区间 `[0,1)` 见 `:190-191`），
且**所有发布的训练配置都显式设为 0.0**：
`configs/train/general.yaml:47`、`specialist.yaml:47`、`sft1.yaml:52`、`sft2.yaml:52`、
`dlm_h800_smoke.yaml:45`、`dlm_h800_resume.yaml:45`（均为 `minimum_noise_level: 0.0`）。
训练入口从 YAML 读取该值（`train/dlm/train_dlm.py:1441`），RL 侧两条路径显式传 0.0
（`train/rl/current/trainer.py:148`、`train/rl/branches/trace/trainer.py:189`）。
→ 下界为 0 时 `probabilities == t`，**与论文 `m_i ~ Bernoulli(t_b)` 完全一致**，不构成差异。
仅当有人把 `minimum_noise_level` 调大于 0 时才会偏离论文。

**✅ 联合损失权重已核对**：`configs/train/*.yaml` 与 `configs/release/dlm_train_h800_smoke.yaml`
中 `mdm_loss_weight: 0.5` / `causal_loss_weight: 0.5`（如 `configs/train/general.yaml:49-50`），
与论文式 2 的 `0.5 L_MDM + 0.5 L_AR` 一致。

**⚠️ 差异（实现动机，非语义）**：clean 流只做一次 forward 供两个 noisy 视图共享
（`models/dlm/hybrid.py:641-645` 注释解释这是省 25% 语言 token 工作量），与论文"`[w^(1); w^(2); x]`
共享一个 clean 多模态流"的语义一致，只是实现更省算力。

### M3 — 注意力可见性（图 4）

| 项 | 结论 |
|:---|:---|
| 状态 | **IMPLEMENTED** `[静态核对]` |

`models/dlm/hybrid.py:785-787`：
```
n2n = ~kv_clean & q_turn.eq(kv_turn)     # 同一块（turn）内 → 双向
n2c = kv_clean & q_turn.gt(kv_turn)      # 严格在前的 clean 块
allowed = q_valid & kv_valid & (n2n | n2c)
```
`models/dlm/hybrid.py:817-829` 是 varlen packed 等价版本。clean 流用 `create_causal_mask`
（`models/dlm/hybrid.py:1324-1330`、`models/dlm/vlm.py:299-304`）且**不包含 noisy KV**，因此
clean token 天然读不到 noisy 视图——与论文"Clean tokens use causal attention and cannot read
noisy views"一致。图 4 的四种可见性（N→N 双向、N→C 前缀、C→C 因果、C→N 屏蔽）**全部可对上**。

### M4 — Entropy-guided decoding（式 24 + 空集兜底）

| 项 | 结论 |
|:---|:---|
| 状态 | **PARTIAL**（熵的取值域与论文/模型卡不一致）`[静态核对]` |

| 论文声明 | 代码位置 | 函数 |
|:---|:---|:---|
| `H_j = −Σ_{v∈V_gen} p_j(v) log p_j(v)`，**在可生成词表上、排除 mask token** | `infer/decode/reliability.py:41-46` | `raw_reliability` |
| 在**未修改**的 token 分布上算（T=1、无采样过滤） | `infer/decode/reliability.py:26-37`（docstring："Compute calibrated statistics at T=1 before any sampling filter"） | 同上 |
| 提交所有 `H_j ≤ τ` | `infer/decode/reliability.py:158-169`（`accepted = adjusted <= float(config.entropy_threshold)`） | `normal_acceptance` |
| 若为空，只提交 `argmin H_j` | `infer/decode/reliability.py:172-188`（`scores.argmin()`） | `most_reliable_position` |
| 提交后不重 mask、不修正 | `infer/decode/reliability.py` 无 resample 路径；`infer/decode/config.py:66-67`（注释：新策略"always drain one position at a time and never fill all residual masks in one operation"） | — |
| 发布配方：τ=0.8, B=32, sub-block 4 | `infer/decode/configs/task_profiles.json` → `shared_decode = {acceptance_policy:"entropy", entropy_threshold:0.8, block_size:32, sub_block_size:4, denoise_steps:4, sampling_seed:42}`；`infer/decode/task_profiles.py:165-209`（`task_single/medium/dense/ocr_medium/ocr_dense` 全部 τ=0.8） | `decoder_profile` |
| 任务分层路由 | `infer/decode/configs/task_profiles.json` → `tiers`（strict_single 14 / medium 12 / dense 8 / ocr_medium 4 / ocr_dense 4，共 42） | — |

**⚠️ 差异 1（重要）**：论文 §B.1 与模型卡 `GroundAnything-Model/README.md:277` 都写明熵是在
**可生成词表 `V_gen`、排除 mask token** 上算的。代码 `infer/decode/reliability.py:37-46` 直接对
传入的 `raw_logits` 全词表做 `logsumexp` 与 `-(p·log p).sum(dim=-1)`，**没有任何词表裁剪**；
调用方 `infer/decode/sglang_algorithm.py:315-337` 传的是 `full_logits` 的切片（`shifted_local_logits`），
也没有把 `|<MASK>|` (ID 152670) 或其他特殊 token 置 `-inf`。
（唯一的词表裁剪出现在 `infer/dlm/hierarchy.py:146-152`，那是**坐标语法约束**路径 —— 把
`row_logits[:coord_min]` 与 `raw_logits[coord_max+1:]` 置 `-inf`，不是熵统计路径。）
→ **结论**：本仓库的熵是在全词表上算的，与论文/模型卡的定义不同；这会系统性改变 H_j 的数值尺度，
进而改变同一个 τ 的行为。属于复现时必须先对齐的头号差异。

**⚠️ 差异 2（需注意）**：`infer/decode/reliability.py:129-155` 的 `adjusted_entropy` 支持在熵上叠加
**位置惩罚**（`entropy_position`）与**重复风险权重**（`entropy_position_repetition`）。这两个策略在论文中
**没有描述**，属于仓库扩展。发布配方里 `position_penalty=0.0`、`repetition_mode="none"`，
所以默认路径与论文一致。

**⚠️ 差异 3（论文未报告）**：发布 profile 的 medium / dense 层用 `temperature=0.1`、`ocr_medium` 用
`temperature=0.3`（`infer/decode/configs/task_profiles.json` → `tiers`；`infer/decode/task_profiles.py:174-209`）。
论文 §4.1 与 §B.1 只说主表用 entropy-guided decoding，**没有报告任何温度采样设置**，也没有说明
"采样与可靠性评估分离"是否意味着对部分任务用了 T>0。若复现表 1–4，必须按发布 profile 的
per-tier 温度才能复现，而论文正文不足以推出这一点。

**参照实现**：`infer/dlm/selection.py:131`（`entropy_acceptance`）是原生 PyTorch 参考路线里的对应实现。

### M5 — Self-speculative decoding（线性 + 二次）

| 项 | 结论 |
|:---|:---|
| 线性 | **IMPLEMENTED** `[静态核对]` |
| **二次** | **NOT FOUND**（仓库未实现） |

| 论文声明 | 代码位置 | 函数/类 |
|:---|:---|:---|
| 双向起草 + 因果验证，共享权重 | `infer/engines/sglang_gam/gam_speculative.py:1-7`（模块 docstring）；`:32-33`（`class GAMSpeculativeBlock(DllmAlgorithm)`，docstring "One bidirectional draft plus causal longest-prefix verification"） | `GAMSpeculativeBlock` |
| 双向 mask vs 下三角因果 mask | `infer/engines/sglang_gam/gam_speculative.py:57-64`（`self._causal_mask = torch.tril(torch.ones(B, B))`；`self._bidir_mask = torch.ones(B*B)`；两者各有 packed 版本） | — |
| **只接受最长匹配前缀**，首个 mismatch 终止 | `infer/engines/sglang_gam/execution_ops.py:45` | `longest_prefix_and_correction(ids, predictions)` |
| 被拒后缀从 cache 移除 | `infer/engines/sglang_gam/gam_speculative.py:418-425`（注释："Rejected suffix states are removed from the cache" 语义对应） | — |
| 每轮 2 NFE、2B query token（线性） | `infer/engines/sglang_gam/gam_speculative.py:279-320`（Phase 1 全 B32 双向 draft → Phase 2 因果 verify，两次 forward） | — |
| 服务入口 | `infer/decoding.py:32`（`settings['algorithm_config'] = {'block_size': 32, 'token_shift': 1, 'debug': False}`）；`README.md:130-134`（`python3 run.py serve --decoder speculative`） | `decoder_settings` |
| 纯 PyTorch 参考路线 | `infer/dlm/hierarchy.py:494`（`class HierarchyBlockDecoder`）；`infer/dlm/hierarchy.py:684-688`（`Qwen35PrefixDraftRunner(model, self.config.block_size) if self.config.use_prefix_cache`）；`infer/dlm/prefix_cache.py:441`（`class Qwen35PrefixDraftRunner`） | — |
| 双向 draft 原语 | `models/dlm/hybrid.py:1507`（`draft_block_logits`）、`models/dlm/hybrid.py:1601-1628`（`draft_block`，docstring "Predict every position after `block_start` in one DLM NFE"） | `GAMQwen35DLM` |
| 因果 verifier / AR 基线 | `infer/dlm/causal.py:13` | `causal_generate_cached`（docstring "Cached causal baseline for wall-clock comparisons with DLM decoding"） |

**NOT FOUND：二次自推测（quadratic self-speculation，§B.2 / 图 6b）**
- 检索式：`quadratic|fused_verify|row_leader|B\+1|B \+ 1`（全仓库，含 `.py` / `.md` / `.json`）
- 命中仅 3 处，**均非实现**：`eval/OCR/utils.py:1111`（OCR 匹配注释里的无关用词）、
  `GroundAnything-Model/README.md:285`（引用论文图 6 的 caption）、
  `GroundAnything-Model/tokenizer.json:88454`（BPE 词表里的普通词 `Ġquadratic`）。
- 交叉证据：`README.md:309` 明写 "The documented `--decoder speculative` route uses
  **linear drafting and greedy verification**."；`infer/dlm/config.py` 的 `mode` 取值里也只有
  `speculative` 一个（无 quadratic 变体）。
- → 论文 §4.3.1 关于二次 schedule 的结论（**TPF 3.17、TPS 49.9**，图 8）**无法用本仓库复现**。

### M6 — 坐标 token 词表与编解码

| 项 | 结论 |
|:---|:---|
| 状态 | **IMPLEMENTED**（与论文逐项吻合，已用发布 checkpoint 实测交叉验证） |

| 论文声明 | 代码位置 / 实测 | 备注 |
|:---|:---|:---|
| 1000 个坐标 token `<0>`–`<999>`，ID **151669–152668** | `train/tokenizer/extend_tokens.py:36-39`（`BASE_TOKENIZER_LENGTH=151669`、`TARGET_TOKENIZER_LENGTH=152670`、`TARGET_EMBEDDING_ROWS=152670`）；`infer/engines/dlm_task_profiles.py:335-336`（`coord_token_min_id: 151669`、`coord_token_max_id: 152668`） | **实测**：`GroundAnything-Model/added_tokens.json` 中 `<数字>` 形式 token 恰 **1000** 个，ID 范围 **151669–152668** ✔ |
| `</c>` = **152669** | 同上；`grounding_anything/protocol.py:37`（`SEP_TOKEN = "</c>"`） | **实测**：`</c>` → 152669 ✔ |
| `|<MASK>|` = **152670**，原子 token | `models/dlm/hybrid.py:25`（`MASK_TOKEN = "|<MASK>|"`）；`models/dlm/hybrid.py:99-148`（`initialize_mask_token`，末尾校验 `check != [mask_id]` 则抛错）；`infer/dlm/build_public_model_bundle.py:427`（"包含原子 `|<MASK>|`（ID 152670），固定 block size 为 32"） | **实测**：`|<MASK>|` → 152670 ✔ |
| mask 行初始化 = 4 个特殊 token 行的 FP32 均值 | `models/dlm/hybrid.py:128-143`（`seed_tokens = ["<|im_start|>","<|im_end|>","<|vision_start|>","<|vision_end|>"]`；`embeddings[mask_id].copy_(embeddings[seed_ids].float().mean(dim=0)...)`；input/output 两个矩阵分别写） | 与 §A.2 完全一致 |
| 式 4：`q(v) = floor((999v + 500)/1000)` | `grounding_anything/protocol.py:804-812`（`return (value * GRID_MAX + 500) // 1000`，`GRID_MAX = 999`） | 整数精确、round-half-up，**不使用 Python 银行家舍入**（docstring 显式说明） |
| 0–1000 → 0–999 重映射（LocateAnything 兼容） | 同上 `remap_1000_coordinate`；`grounding_anything/protocol.py:815-839`（`remap_1000_coordinates`，含映射后塌缩校验） | 与 `tools/convert_locany_to_gam.py:40-43` 的 `min(999, round(value*999/1000))` 等价 |
| 坐标 token → 像素（box 用 width/height，point 用 width−1/height−1） | `eval/utils/prompt_mode.py:596-621`（`gam_grid_to_abs`、`gam_grid_point_to_pixel`）；`eval/utils/prompt_mode.py:640-674`（`pixel_boxes_to_gam_grid`，含窄框不塌缩的修正） | 注意 box 用 `*width`、point 用 `*(width-1)`，**两者语义不同**，是刻意的 |
| 坐标 token 解析 | `grounding_anything/protocol.py:412-425`（`_parse_coordinate_token`）；`:475`（`parse_answer`）；`:377`（`serialize_answer`） | — |
| 结构 token（object_ref / box / vision / im）约定 | `grounding_anything/protocol.py:33-42`；表 7 逐项对应 | — |

### M7 — Prompt 模板（表 6）

| 项 | 结论 |
|:---|:---|
| 状态 | **IMPLEMENTED**（逐字节一致） |

| 论文表 6 | 代码位置 |
|:---|:---|
| `Locate all the instances that match the following categories: [CATEGORIES].` | `grounding_anything/protocol.py:44-46`（`DENSE_BBOX_PROMPT_PREFIX`） |
| `Locate the target referred to by the following description: [EXPRESSION].` | `grounding_anything/protocol.py:47-49`（`REFER_BBOX_PROMPT_PREFIX`） |
| `Point to: [CATEGORIES].` | `grounding_anything/protocol.py:50`（`DENSE_POINT_PROMPT_PREFIX`） |
| `Point to the target referred to by the following description: [EXPRESSION].` | `grounding_anything/protocol.py:51-53`（`REFER_POINT_PROMPT_PREFIX`） |
| `OCR task detect all the text in box format.` | `grounding_anything/protocol.py:64`（`OCR_PROMPT`） |
| `Detect all document layout elements that match the following categories: [CATEGORIES].` | `grounding_anything/protocol.py:68-70`（`LAYOUT_PROMPT_PREFIX`） |
| `Point to the UI element to click for the following instruction: [INSTRUCTION].` | `grounding_anything/protocol.py:71-73`（`GUI_PROMPT_PREFIX`） |
| `Given reference boxes <|box_start|>[BOXES]<|box_end|> indicating one or more objects, find all similar objects in the image and output their bounding boxes.` | `grounding_anything/protocol.py:58-62`（`VISUAL_PROMPT_PREFIX` + `VISUAL_PROMPT_SUFFIX`） |

构建/解析函数成对存在：`build_dense_bbox_prompt`/`parse_dense_bbox_prompt`（`:562`/`:569`）、
`build_refer_bbox_prompt`/`parse_refer_bbox_prompt`（`:624`/`:630`）、
`build_dense_point_prompt`/`parse_dense_point_prompt`（`:636`/`:643`）、
`build_refer_point_prompt`/`parse_refer_point_prompt`（`:684`/`:690`）、
`build_ocr_prompt`/`parse_ocr_prompt`（`:740`/`:757`）、
`build_layout_prompt`/`parse_layout_prompt`（`:765`/`:772`）、
`build_gui_prompt`/`parse_gui_prompt`（`:792`/`:798`）、
`build_visual_prompt`/`parse_visual_prompt`（`:696`/`:714`）。
另有 `validate_prompt_answer_pair`（`:846`）做 prompt/答案配对校验。

### M8 — AR-to-Diffusion 转换

| 项 | 结论 |
|:---|:---|
| 状态 | **PARTIAL**（无独立转换脚本；以训练入口形式实现，且 RL 拓扑与论文不一致）`[静态核对]` |

| 论文声明 | 代码位置 | 备注 |
|:---|:---|:---|
| 从 coordinate alignment 后的 AR checkpoint 初始化 | `train/dlm/train_dlm.py:1395-1398`（`AutoProcessor.from_pretrained(model_path)` → `model_loader.from_pretrained(...)`）；`train/dlm/train_dlm.py:361`（`load_initial_dlm_checkpoint(model, checkpoint)`）；`README.md:411`（"To fine-tune the released DLM, use the complete `weights/dlm_bundle` as the initial DLM checkpoint and set the native `model.path` to the matching autoregressive base model"） | 没有单独的 `convert_*.py`；转换 = "用 AR 权重初始化 + 新加 mask token + 跑 0.5/0.5 联合损失训练" |
| 新增 mask token 并 mean-init 两个（untied）词表矩阵 | `models/dlm/hybrid.py:99-148`；调用点 `train/dlm/train_dlm.py:1432`（`mask_id = initialize_mask_token(processor.tokenizer, model)`）与 `:1438`（`mask_token_id=mask_id`） | ✔ |
| 两条 DLM 包装路线 | `models/dlm/hybrid.py:163`（`class GAMQwen35DLM`，"Direct Conversion wrapper around a Qwen3.5 VLM"）；`models/dlm/vlm.py:1`（"GroundAnything/Kimi-K3-ViT + plain-Qwen3 Direct-Conversion DLM route"，`train/dlm/train_dlm.py:39` 导入 `GAMQwen3DLM`） | Qwen3.5 路线与 Qwen3 路线并存 |
| 转换训练入口 | `train/dlm/train_dlm.py:1087`（`main`，`--mode DLM`）；`README.md:386`（"Keep the training mode as `DLM`"） | — |
| 训练阶段配方（Stage 1–4 / 转换 / RL） | `configs/train/general.yaml`、`specialist.yaml`、`sft1.yaml`、`sft2.yaml`；`configs/release/*_train.yaml`；`configs/training_stages.yaml`（753 B） | — |
| 参数白名单（block_size / mdm_loss_weight / causal_loss_weight 等） | `train/release_parameters.py:4` | 含 `block_size`、`causal_loss_weight`、`mdm_loss_weight`、`fixed_block_size_experiment`、`block_annealing` 等 |
| 权重打包为可推理 bundle | `infer/dlm/build_public_model_bundle.py:202-254`（mask row 与 additional token 写入）、`:360-379`（原子性校验）、`:427` | — |

**⚠️ 差异（重要）**：论文表 13 说 GRPO 用 **256 GPU**；仓库里发布的 RL 拓扑是
**24 / 56 / 64 H800**：
- `train/rl/current/trainer.py:1`：`"""24-H800 Causal-JustGRPO trainer for GAM DLM RL V2."""`
- `train/rl/distributed/causal.py:1`：`"""Causal-JustGRPO V2 adapter for 7x8 H800."""`
- `train/rl/distributed/multiroute_56.py:1`：`"""Isolated RLV2 control: same 17.7K data/rewards as RLV3 on 56 H800."""`
- `train/rl/branches/rebalanced_64/trainer.py:2`：`"""Causal-JustGRPO RLV2 on eight H800 nodes (64 ranks)."""`
- `train/rl/distributed/multiroute_64.py:1`：`"""RLV2 multi-route control adapter for eight-node (64-GPU) H800."""`
- 入口：`configs/release/rl_train.yaml`；`train/rl/branches/dare/main_causal_grpo.py:6-7`
  已显式禁用（`raise SystemExit("This RL branch is disabled. Use configs/release/rl_train.yaml.")`）

→ 论文的 256-GPU GRPO 配方**没有以可直接运行的 launch YAML 形式发布**；发布的是 24/56/64 卡的控制与
适配实验路径。这意味着 RL 阶段的**逐位复现**需要自行重写拓扑。

**⚠️ 易混淆点**：`tools/convert_locany_to_gam.py` **不是** AR→Diffusion 转换器。
它是**数据**转换器（LocateAnything ShareGPT 标注 → GAM JSONL），见 `:1-6` 的 docstring 与
`train/data/row_contract.py` 的 `validate_uniform_row` 调用。检索"convert"时勿误认。

### M9 — GRPO 目标与优势归一化（式 3/8/9/11）

| 项 | 结论 |
|:---|:---|
| 状态 | **IMPLEMENTED** `[静态核对]` |

| 论文声明 | 代码位置 | 函数 |
|:---|:---|:---|
| ε = 0.2 | `train/rl/config.py:34`（`clip_epsilon: float = 0.2`）；`train/rl/current/config.py:25`（`clip_epsilon: float = 0.2`，并在 `:56` 的校验里固定为 0.2）；`train/rl/branches/trace/trainer.py:690`（`p99_limit = min(float(config.clip_epsilon), 0.20)`） | — |
| clipped 目标 `min(rA, clip(r,1−ε,1+ε)A)` | `train/rl/loss.py:18-35` | `tracerl_token_loss` |
| 组内标准化用**总体标准差**（式 8 的 `1/G Σ`）+ ε=1e-6 | `train/rl/loss.py:8-15`（`std = values.std(unbiased=False)`；`(values - values.mean()) / (std + eps)`，`eps=1.0e-6`） | `group_normalized_advantage` |
| 先分量标准化再加权、再组内标准化（式 9） | `train/rl/reward_adapter.py:66-79` | `gdpo_total_rewards`（注释："Match active-column GDPO scaling before final group normalization"） |
| 每次 rollout 一次更新、无 KL | `train/rl/current/config.py`；`train/rl/current/trainer.py:448-484`（单次 gather→advantage→update 路径）；`train/rl/current/trainer.py:1-5` | `train_causal_justgrpo` |
| 冻结 ViT，更新 projector + language | `train/rl/current/trainer.py:139-147`（`initialize_mask_token` 后构造 DLM wrapper）；`train/release_parameters.py` 白名单 | — |
| block/sub-block 一致性 | `train/rl/config.py:17-20`（`block_size: int = 32`、`sub_block_size: int = 4`）；`train/rl/config.py:51-54`（校验里固定 32/4）；`train/rl/trajectory.py:24-67`（`validate(block_size=32)`，校验 diffusion commit step 不跨物理 B32 块） | `GAMTraceTrajectory` |
| 响应数 / 温度 | `train/rl/config.py:15`（`num_generations: int = 8`）、`:24`（`temperature: float = 0.5`）、`:74-75`（只允许 {0.3, 0.5, 0.7}）；`configs/rl/default.yaml:9`（`temperature: 0.7`）；`train/rl/branches/trace/trainer.py:778`（`target = 0.7 if zero_rate > 0.50 else ...`） | `TraceRLConfig` |
| 一致的固定契约（防止配置漂移） | `train/rl/config.py:46-84`（`validate()` 强制 `num_generations=8`、`block_size=32`、`causal_anchor_tokens=1`、`diffusion_tokens_per_block=31`、`sub_block_size=4`、`token_shift=1`、`top_p=1.0`、`top_k=0`、`position_penalty=0.0`；`:72-73` 强制 `entropy_threshold == 0.8`） | 与论文 §3.4.3/§A.8 的 G=8、T=0.7、τ=0.8、ε=0.2 **一致** |
| RLV2（VLM）变体：冻结 ViT+projector、KL 0.02、batch 级标准化 | `train/rl/branches/dare/materialize_causal_hf_checkpoint.py`、`train/rl/branches/rebalanced/trainer.py:119-143`、`train/rl/distributed/*`（`_released_validate` 保留原校验后加拓扑校验） | — |

### M10 — 任务奖励（式 12–14 框、式 15–21 OCR、式 22–23 pointing）

| 项 | 结论 |
|:---|:---|
| 状态 | **IMPLEMENTED**，但**框奖励有多版本**（论文只描述了其中一个）`[静态核对]` |

**框奖励（式 14：`R_strict = clip[0,1](0.10 R_fmt + 0.10 R_count + 0.50 F̄ + 0.25 R_IoU + 0.05 R_sort − 0.07 P_big − 0.03 P_dup)`）**

| 论文项 | 代码 | 值 |
|:---|:---|:---|
| `F̄` 平均 IoU 0.50 / 0.75 / 0.95 处的定位 F1 | `train/rl/rewards/grounding_bbox_reward.py:45` | `STRICT_BASELINE_IOU_THRESHOLDS = (0.5, 0.75, 0.95)` ✔ |
| 权重 0.10 / 0.10 / 0.50 / 0.25 / 0.05 | `train/rl/rewards/grounding_bbox_reward.py:50-56`（`ORIGINAL_REWARD_WEIGHTS = {"format":0.10, "count":0.10, "f1":0.50, "matched_iou":0.25, "sort":0.05}`） | **逐项吻合** ✔ |
| 罚项 0.07（oversized）/ 0.03（duplicate） | `train/rl/rewards/grounding_bbox_reward.py:147-148`（`ORIGINAL_BIGBOX_PENALTY_WEIGHT = 0.07`；`ORIGINAL_DUPLICATE_PENALTY_WEIGHT = 0.03`） | **吻合** ✔ |
| 打分函数 | `train/rl/rewards/grounding_bbox_reward.py:945`（`score_completion_original`）、`:966`（`score_completion_original_strict_iou`，docstring "Task319 baseline with F1 thresholds changed to 0.5, 0.75 and 0.95"）、`:1408`（`score_completion` 分发） | — |
| GT-wise 匹配（一个预测可支撑多个 GT）→ `R_set` | `train/rl/rewards/grounding_bbox_reward.py` 的 `_match_pairs`（`:486`）、`precision_recall_fbeta_at_threshold`（`:512`）；集合完备性公式的实现在同一文件 | 论文 §A.10.1 式 12–13 |
| 发布配方用哪一版 | `train/rl/reward_adapter.py:30-32,44-55`（`JointGAMGroundingRexOmniReward` + `JointGAMGroundingStrictIoUReward`，权重 **0.7 / 0.3**） | 与论文 "(0.7, 0.3)" **吻合** ✔ |

> **重要提示**：`train/rl/rewards/grounding_bbox_reward.py` 里同时存在
> `RECALL_V2`（`:59`）、`BEST_V3`（`:72`）、`PARETO_V4`–`V8`（`:86/99/113/125/137`）、
> `DCGRPO`（`:1256`）、`JointPRNash`（`:1515`）、`ScaleGatedNash`（`:1550`）、
> `ParetoConsensusJointPR`（`:1606`）等 **10+ 个奖励变体**，各有独立权重与罚项系数，
> 并注册为 `grounding_bbox_reward_*` 的 Swift ORM（`:1756-1806`）。
> **论文只描述了 ORIGINAL 一版**（对应 `score_completion_original*`）。
> `docs/investigation/repo-survey.md` 之外的读者容易误以为论文用的是最新变体 —— 不是。

**OCR 奖励**（论文 §A.10.2 式 15–21 只给了一套；仓库有多套）

| 论文项 | 代码 |
|:---|:---|
| Hungarian 一对一匹配、NFKC + case fold、Levenshtein `E_ij`、soft affinity `sqrt(I·E)` | `train/rl/rewards/ocr_dual_reference_reward.py:169`（`maximum_assignment`）、`:65-75`（`normalize_text`/`edit_similarity`）、`:262`（`affinity = sqrt(overlap)*similarity if overlap>=0.10 and similarity>=0.20`） |
| `0.45 H̄ + 0.35 F(A_soft) + 0.10 H_0.50 + 0.10 C` | `train/rl/rewards/ocr_dual_reference_reward.py:268-277`（`0.45*hard_f1_miou + 0.35*soft_f1 + ...`，并输出 `hard_f1_miou` 字段名） |
| 阈值集 `{0.50, 0.55, …, 0.95}` | `train/rl/rewards/ocr_dual_reference_reward.py:31`（`IOU_THRESHOLDS = tuple(round(0.50 + 0.05*i, 2) for i in range(10))`）；`train/rl/rewards/ocr_multiview_reward.py:43`；`eval/OCR/utils.py:74` |
| word/line 双粒度一致性、`(α,β)` 组合 | `train/rl/rewards/ocr_reference_group_reward.py:75-183`（`:162` `semantic = 0.35*dual + 0.55*group_score`；`:164` `dual = 0.75*high + 0.25*low`）；`train/rl/rewards/ocr_gam_native_reward.py:164-186`；`train/rl/rewards/ocr_multiview_reward.py:250-413` |
| 互补参考一致性 `Wtext = 0.82(0.70 max + 0.30 min) + 0.18 Wc` | `train/rl/rewards/ocr_multiview_reward.py:454-462`（`primary_score = 0.70*high + 0.30*low`；`teacher_score = 0.82*primary_score + 0.18*hybrid_score`；`semantic = 0.95*teacher_score + 0.05*ppocr_geometry`）；`train/rl/rewards/ocr_gam_native_reward.py:220-223` |
| 格式门控的**二次**作用（表 14 的 1.00/0.60/0.30/0.00 与 1.00/0.45/0.35/0.15/0.00） | `train/rl/rewards/ocr_reference_group_reward.py:179`（`semantic + 0.10*format_reward*format_coverage`）；`train/rl/rewards/ocr_gam_native_reward.py:183`（`base = (semantic + 0.10*parsed.format_reward*coverage) * format_gate`）；格式分在解析函数里给（`:149` `format_reward = 0.35`、`:161` `format_reward = 0.3`） |
| 罚项 0.05 dup / 0.10 over / 0.05 invalid / 0.08 large | `train/rl/rewards/ocr_reference_group_reward.py:183`（`-0.05*dup -0.10*overgeneration -0.05*invalid`）、`:179` 附近；`train/rl/rewards/ocr_gam_native_reward.py:186`；`ocr_dual_reference_reward.py:406` |
| 发布配方用哪一套 | `train/rl/reward_adapter.py:32,56-62`（`JointGAMOCRMixedReward`，权重 **1.0**，与论文"OCR 各给一个标量奖励"吻合）；路由在 `train/rl/rewards/ocr_mixed_reference_reward.py:34`（`detect_reward_mode`，按样本 solution schema 选 `multi_teacher_views_v1` 或 `merged_single_teacher`） |

> 仓库里 OCR 奖励同样有 **6 个以上变体**（`ocr_dual_reference` / `ocr_reference_group` /
> `ocr_multiview` / `ocr_mixed_reference` / `ocr_gam_native` / `ocr_multiteacher`）。
> 论文的式 15–21 与 `ocr_dual_reference_reward.py` + `ocr_multiview_reward.py` 的加权系数最接近，
> 但**没有一套代码能逐项对上论文全部 7 条式子**（例如论文的 `(α,β)=(0.90,0.35)` / `(0.75,0.50)`
> 在代码里以 `0.35*dual + 0.55*group_score` 的等价改写形式出现，系数并非字面相同）。
> → 标记为 **PARTIAL**：公式族一致、个别系数无法一一对上。

**Pointing 奖励（式 22–23）**

| 论文项 | 代码 | 值 |
|:---|:---|:---|
| `s(d) = exp[−(1/2)(d/75)²]`，`d ≤ 200`，否则 0 | `train/rl/shared/reward.py:28-29`（`POINT_SIGMA = 75.0`、`POINT_MAX_DISTANCE = 200.0`）；`:99-103`（`math.exp(-0.5 * (distance / POINT_SIGMA) ** 2)`，`distance > 200` 返回 0.0） | **逐项吻合** ✔ |
| 一对一匹配（按 label） | `train/rl/shared/reward.py:106-133`（`_assignment`，优先 `scipy.optimize.linear_sum_assignment`，无 scipy 时贪心回退） | — |
| `F2 = 5PR/(4P+R)`，分母为 0 时取 0 | `train/rl/shared/reward.py:151-155`（`5.0*precision*recall/(4.0*precision + recall) if precision+recall else 0.0`） | **吻合** ✔ |
| count 一致 = 小/大 | `train/rl/shared/reward.py:156-160`（`min/max`） | **吻合** ✔ |
| format：1 / 0.5（可恢复）/ 0 | `train/rl/shared/reward.py:73-80`（`useful and not errors → 1.0`；`useful → 0.5`；else `0.0`） | **吻合** ✔ |
| 权重 `0.55 F2 + 0.25 R_count + 0.10 F1,label + 0.10 R_fmt` | `train/rl/shared/reward.py:177-182` | **逐项吻合** ✔ |
| 路由 | `train/rl/shared/reward.py:187-208`（`MultiRouteGAMRewardAdapter`，按 `rlv3_route` 分发） | — |

> **注意**：pointing 奖励只在 **RLV3** 路径（`train/rl/shared/reward.py`）里；
> **RLV2** 的 `train/rl/reward_adapter.py:44-63` **只有** `grounding_bbox_gam` 与 `ocr_bbox_text_gam`
> 两个 task_type，遇到其他 task 直接 `raise ValueError`。而 `train/rl/shared/reward.py:1-6` 的 docstring
> 自己说明："Only the five point-output routes need a new scorer because the released joint recipe
> has no point reward." → 论文的 pointing 奖励对应 RLV3 路径。

### M11 — F1mIoU 指标与评测指标实现

| 项 | 结论 |
|:---|:---|
| 状态 | **IMPLEMENTED** `[静态核对]` |

| 论文声明 | 代码位置 | 函数 |
|:---|:---|:---|
| **F1mIoU = 阈值聚合 F1，不是平均匹配 IoU** | `eval/utils/detection_utils.py:66`（`IOU_THRESHOLDS = tuple(round(0.50 + 0.05*i, 2) for i in range(10))`，即 0.50:0.05:0.95 共 10 档）；`eval/utils/detection_utils.py:886-901`（`agg_f1_mean_iou`：逐阈值算 `2·meanP·meanR/(meanP+meanR)` 后取 10 档均值） | `agg_f1_mean_iou` |
| 中间载荷（每样本每阈值的 R/P） | `eval/utils/detection_utils.py:711-728`（`_metric_payload`，`"F1mIoU": {"recalls": ..., "precisions": ...}`） | — |
| IoU 0.50 / 0.95 单点 | `eval/utils/detection_utils.py:721-722`（`f1_iou_50`、`f1_iou_95`） | — |
| F1 由**平均 P / 平均 R**算，而非逐样本 F1 再平均 | `eval/utils/detection_utils.py:851-873`（中文注释明确："F1 用'平均 P / 平均 R'计算，而非逐样本 F1 再平均"） | `agg_f1` |
| 匹配规则（逐 label、贪心取最大 IoU、不重复使用预测） | `eval/utils/detection_utils.py:662-689` | `_score_sample` |
| 类别感知变体 | `eval/utils/detection_utils.py:696-697`（`_gam_category_aware` 分支） | `_score_category_aware_iou_sweep` |
| **F1@Point / point-in-mask** | `eval/metrics/detection_metrics.py:31`（`is_point_in_mask(point, mask)`）、`:42`（`is_point_in_box`）、`:187`（`calculate_pointing_metrics(gt_masks_dict, pred_points_dict)`）、`:245`（掩码命中判定） | 与 §4.2 "SAM-derived masks determine point correctness" 对应 |
| **OCR 的 loose-match F1mIoU + 独立 parse error** | `eval/OCR/utils.py:74-79`（`IOU_THRESHOLDS`、`loose_match_F1mIoU`）、`:1195`（`_sample_loose_miou`）、`:1341`（`"loose_match_F1mIoU": sum(loose_metrics.values())`）、`:1377-1473`（HierText / ICDAR2015 / TotalText / SROIE 各自的 loose/strict 聚合入口） | — |
| **GUI action accuracy / exact accuracy** | `eval/GUI/screenspot_pro/screenspot_utils.py:1122`（`eval_sample_positive_gt`）、`:1172-1224`（`action_acc` + `parse_error_rate` 载荷）、`:1243-1406`（`action_acc` 聚合器，`:1282` `num_correct_action / num_total`）；`eval/GUI/osworld_g/osworldg_utils.py:220-254`（`exact_acc` + `parse_error_rate`） | — |
| ScreenSpot-V2 | `eval/GUI/screenspot_v2/screenspot_v2_utils.py`（105 行） | — |
| 数据集/任务注册 | `eval/task_config.py`（166 行，逐任务 CPU/像素/生成参数表）；`configs/eval/tasks.json`（**42** 条 recipe） | — |
| 评测执行 | `eval/eval_runner.py:810`；`scripts/evaluate.py`（266 行，唯一严格前端） | — |

> **注意**：`eval/metrics/detection_metrics.py`（1446 行）另有一套 `calculate_keypoint_*` /
> `calculate_oks` / `calculate_iou` 等通用指标，属于上游 lmms-eval 风格代码，**不是**论文 F1mIoU 的实现。
> 论文 F1mIoU 的实现以 `eval/utils/detection_utils.py` 为准。

### M12 — 30-benchmark 任务清单与套件范围

| 项 | 结论 |
|:---|:---|
| 状态 | **IMPLEMENTED**（任务清单可复现）`[静态核对]` |

- 论文"30 benchmarks / 11 类感知能力"（§4.1）
- 仓库：`eval/README.md:5`（"the canonical guide for the **30-benchmark suite** used in the
  GroundAnything paper"）；`eval/README.md:60-123` 给出**逐条 30 个 task ID 列表**并 `assert len(tasks) == 30`
  ；`eval/README.md:62`（"The registry contains **42 task recipes**. These **30 entries** match this
  paper's suite"）；`eval/README.md:75` 列出被排除的 12 条。
- 注册表实测：`configs/eval/tasks.json` 为 dict，**42** 个 key ✔（与 README 一致）
- 任务分组（`eval/README.md:64-73`）：Detection 4 / Referring boxes 5 / Object pointing 6 /
  Spatial pointing 4 / GUI 3 / OCR 4 / Document layout 2 / Visual prompting 2 = **30** ✔
- 一键生成 30 任务全量配置：`eval/README.md:82-143`（`full_suite_denoise.yaml` / `full_suite_speculative.yaml`
  / `full_suite_vlm.yaml`）
- 发布模板默认 `limit: 8` 只是 smoke test；全量需 `limit: null`（`eval/README.md:149`、`README.md:367`）
- 评测模式只有 7 种：`eval/utils/prompt_mode.py:27-35`（`VLM, GAM, DLM, RLV2, REXOMNI, LOCATEANYTHING, GROUNDINGDINO`）

### M13 — 训练算力/配方（表 11/12/13）

| 项 | 结论 |
|:---|:---|
| 状态 | **PARTIAL** |

| 论文 | 仓库 |
|:---|:---|
| Stage 1/2/3/4：128 GPU | `configs/release/general_train.yaml`、`specialist_train.yaml`、`sft1_train.yaml`、`sft2_train.yaml`；`README.md:388`（"The supplied stage templates target **8 nodes × 8 devices**" = 64 ranks，可调）；`configs/train/*.yaml` 内为 native 配方 |
| AR→Diffusion Phase I/II：256 GPU | 未在 `configs/` 中找到 256 卡 launch YAML；`README.md:388` 只提供 8×8 模板 |
| GRPO：256 GPU（表 13） | **冲突**：仓库发布的是 24 / 56 / 64 H800 拓扑（见 M8 差异段） |
| 基础预训练 > 700B tokens | 训练数据与 token 计数**不在仓库**；`README.md:400`（"a generic JSONL-to-training-cache converter is not included"） |
| 数据引擎（§A.3） | **NOT FOUND**（见 M14） |

### M14 — 数据引擎（§3.3 / §A.3）

| 项 | 结论 |
|:---|:---|
| 状态 | **NOT FOUND**（未开源） |

- 检索式：`teacher|field_fusion|fusion|coverage|candidate_generation|data_engine`（全仓库 `*.py`）
- 命中 130 处，**全部是其他含义**：`train/rl/rewards/ocr_*` 的"multi-teacher **参考视图**"
  （奖励用的参考标注，不是标注生成流水线）、`train/rl/*/trainer.py` 的 **teacher-forcing** 前向、
  各种 "checkpoint/report **coverage**" 完整性校验。
- 仓库只有数据的**消费端契约**：`train/data/row_contract.py`（70 行，`validate_uniform_row`）、
  `train/data/special_token_format.py`（2 行）、`train/data/__init__.py`（61 行）、
  `tools/validate_gam_jsonl.py`（33 行）、`tools/build_gam_cache.py`（136 行）、
  `tools/convert_locany_to_gam.py`（214 行）
- `README.md:400`：训练需要"prepared caches and manifests"，且 **"a generic JSONL-to-training-cache
  converter is not included"**
- → 论文的字段级教师融合、`a(f) ∈ {pass,fail,pending}` 校验状态机、专家迭代循环**均未发布**。

### M15 — 统计协议（10 runs = 5 seeds × 2）

| 项 | 结论 |
|:---|:---|
| 状态 | **NOT FOUND**（未实现） |

- 论文 §4.1：自家模型 10 runs（5 seeds × 2 runs）；本地 baseline 3 runs（3 seeds × 1 run）
- 检索式：`five seeds|10 runs|ten runs|seed.*repeat|num_runs|repeats`（全仓库 `*.py`）
  → 命中的全是 `repeat_interleave`（GQA KV 广播）与 RL 数据重采样，**与多种子评测无关**
- 反向证据（最直接）：`eval/README.md:198`
  > "The script reports **per-task results** and does not compute a paper-wide aggregate score.
  > One invocation is **one complete run of the selected suite**; it does not reproduce the paper's
  > ten-run statistical averaging by itself."
- → ① **30-benchmark 聚合分（72.42 / 61.75 / 71.35 / 53.32）没有计算代码**；
  ② **多种子重复与方差统计没有脚本**。复现论文主数字需要自行实现聚合与重复协议。

### M16 — 执行优化（SGLang / CUDA Graph / FP8）

| 项 | 结论 |
|:---|:---|
| SGLang 集成 | **IMPLEMENTED**（默认服务路径） |
| CUDA Graph | **IMPLEMENTED 但默认禁用** |
| 选择性 FP8 | **PARTIAL**（有 kernel 注册与兼容补丁，但无公开开关、无数值对齐证据） |

| 论文（§B.3 / 图 15） | 代码 |
|:---|:---|
| SGLang eager = 默认 | `infer/serve_sglang.py:20-38`（环境注入，含 `GAM_SGLANG_DISABLE_CUDA_GRAPH="1"`）；`README.md:324-326`（表格：SGLang eager **Enabled**；CUDA Graph **Disabled**；Selective FP8 **Disabled**） |
| CUDA Graph replay | `infer/engines/sglang_gam/compat.py:115-129,319-320`（`TritonAttnBackend.init_forward_metadata_capture_cuda_graph` / `..._replay_cuda_graph` 的 monkey-patch）；`infer/engines/sglang_gam/triton_verify_graph.py:34-67`（draft/verify 双后端图捕获）；`infer/engines/sglang_gam/gam_speculative.py:355`（注释："A CUDA graph can safely accelerate the bidirectional draft even ..."） |
| CUDA Graph 支持的形状 | `README.md:331`（"The supported capture case is B32 with one request and no tensor, pipeline, or data parallel expansion"） |
| 无 `--cuda-graph` 开关 | `README.md:333`（"The release launcher does not expose a `--cuda-graph` switch."）；`GroundAnything-Model/README.md:310` 同 |
| 选择性 FP8 作用于语言线性层 | `infer/engines/sglang_gam/compat.py:542-558`（注册 FP8 helper 签名）、`:686-713`（`fp8_utils.triton_scaled_mm` 的 scale 兼容补丁，`_GAM_FP8_SCALE_PATCHED`） |
| FP8 **不是**发布权重格式 | `infer/dlm/build_public_model_bundle.py:443`（"权重精度保持原始 BF16/FP32 mixed checkpoint，不包含 INT8/FP8 量化权重"） |
| **未建立 FP8 保精度** | 论文 §D.2 p.35 自己声明 "it does not establish that FP8 preserves accuracy"；仓库侧也没有 FP8 on/off 的匹配质量对照脚本 |

---

## 2. "仓库未实现 / 未开源"清单（复现阻塞项）

| # | 论文声明 | 状态 | 阻塞程度 | 检索依据 |
|:--|:---|:---|:---|:---|
| N1 | **MTP 对照**（causal MTP + block-MTP，§4.3.2 / §C） | **未开源**（论文自述为 "in-house causal drafter"） | **高**：Finding 2.2 无法验证 | 全仓库检索 `mtp\|MTP\|multi_token\|multi-token` → 仅命中 Qwen3.5 的 "cached multi-token fast path"（`models/qwen35_runtime.py:256`，是 KV cache 多 token 扩展，**不是 MTP 头**）与 LocateAnything 的评测/奖励兼容代码；**无任何 MTP 模块或 drafter** |
| N2 | **二次自推测**（§B.2 / 图 6b，TPF 3.17 / TPS 49.9） | **未实现** | **中**：图 8 的二次曲线无法复现 | 检索 `quadratic\|fused_verify\|row_leader` → 3 处均为文档/无关命中（见 M5） |
| N3 | **数据引擎**（§3.3 / §A.3，字段级融合 + 校验状态机 + 专家迭代） | **未开源** | **高**：训练数据无法重建 | 见 M14 |
| N4 | **30-benchmark 聚合分**（72.42 / 61.75 / 71.35 / 53.32） | **未实现** | **高**：主 claim 的核心数字无法复算 | `eval/README.md:198` 明确不计算 paper-wide aggregate；`eval/eval_runner.py` 无聚合逻辑；全仓库检索 `overall\|aggregate\|Average` 无匹配聚合实现 |
| N5 | **10 runs = 5 seeds × 2 的统计协议** | **未实现** | **中**：无法复现方差/显著性 | 见 M15 |
| N6 | **256-GPU GRPO 配方**（表 13） | **未发布对应 launch YAML** | **中**：RL 阶段只能按 24/56/64 卡路径近似 | 见 M8 差异段 |
| N7 | **训练数据与 token 计数**（> 700B tokens 的构成） | **未发布** | **高** | `README.md:400`；仓库无数据清单 |
| N8 | **通用 JSONL → 训练 cache 转换器** | **未提供** | **高** | `README.md:400` 原文 |
| N9 | **外部 baseline 的本地适配器**（GPT-6 Astra / Qwen3.7-Max / Kimi-K3 / SenseNova-Vision / SEED1.5-VL / Molmo / UI-R1 等） | **未实现** | **中**：表 1–4 的多数行只能照抄 | `eval/utils/prompt_mode.py:27-35` 只有 7 个模式；`REXOMNI` / `LOCATEANYTHING` / `GROUNDINGDINO` 有适配器，`astra` / `qwen3.7` 全仓库**零命中**（`kimi-k3` 仅出现在视觉编码器与 `eval/GUI/screenspot_pro/screenspot_utils.py:96` 的模型尺寸表里） |
| N10 | **Stage 1–3 预训练数据与配方细节** | **部分发布**（有 YAML 骨架，无数据） | **高** | `configs/train/general.yaml`、`specialist.yaml` 存在；数据缺失 |
| N11 | **去污染 / 训练-评测划分脚本** | **未实现、论文亦未报告** | **高**：复现对比的**首要风险** | 全仓库检索 `split\|contamination\|decontaminat\|leak` 无相关实现；论文全文亦无此声明 |
| N12 | **FP8 精度对照**（论文 §D.2 自认未建立） | **未提供** | **低-中**：图 15 的 3.51×/4.06× 无法分解出 FP8 贡献 | 见 M16 |

---

## 3. 论文数字 ↔ 仓库可验证性对照

| 论文数字 | 能否用本仓库复现 | 需要的额外条件 |
|:---|:---|:---|
| 词表布局（1000 坐标 token、ID 151669–152668、`</c>`=152669、`|<MASK>|`=152670） | ✅ **可**（已用 `added_tokens.json` 实测吻合） | 无 |
| 架构规格（表 8） | ✅ **可**（`config.json` 的 `text_config`/`vision_config` 逐项吻合） | 无 |
| 参数计数（表 9，4,843,669,504） | ✅ **可，且已 bit-exact 验证**（Lead 2026-10-09：只读 safetensors header 逐张量求和 = 4,843,669,504，与表 9 **完全相同，偏差 0.0%**；拆分 vision+projector 428,482,048 / language+lm_head 4,415,187,456） | 无（只需读 header，不必加载 9.2 GB 权重） |
| Prompt 模板（表 6） | ✅ **可**（逐字节一致） | 无 |
| 式 4 坐标量化 | ✅ **可**（`protocol.py:812` 精确实现） | 无 |
| 熵引导解码算法（式 24） | ⚠️ **有偏差**：熵的全词表 vs `V_gen` | 需先对齐熵的取值域 |
| τ=0.8 / B=32 / sub-block 4 | ✅ **可**（发布 profile 明确） | 无 |
| 4.51× / 0.74 pp | ❌ **不可**（无 H800；且依赖 49.6 TPS 的 AR 参考点） | H800 级 GPU + GroundAnything-VLM 权重 + COCO 评测集 |
| 2.56× / 2.98 pp（τ=0.8） | ❌ **不可** | 同上 |
| 二次自推测 TPF 3.17 / TPS 49.9 | ❌ **不可**（未实现） | 需自行实现 §B.2 二次 schedule |
| 30 benchmark 逐任务数值（表 1–4） | ⚠️ **理论可**（任务清单、prompt、metric 齐备） | 30 个数据集的本地下载（`configs/datasets.yaml` 只给相对路径）+ GPU 服务 |
| 30-benchmark 聚合分（72.42 / 61.75 / 53.32 / 71.35） | ❌ **不可**（无聚合代码、无 53.32/71.35 的表） | 需自行定义聚合方式；53.32/71.35 无逐任务分解 |
| DLM vs MTP（图 11–14） | ❌ **不可**（MTP 未开源） | 需自行实现两种 MTP drafter |
| 训练配方（表 11/12/13） | ⚠️ **部分可**（YAML 骨架在，数据与拓扑缺） | 训练数据 + 256 卡拓扑 |
| 图 15 的 3.45–4.06× | ⚠️ **理论可**（CUDA Graph/FP8 代码在，默认关闭） | GPU + 打开开关 + 自行测；且无 FP8 精度对照 |

---

## 4. 关键差异汇总（论文 ↔ 代码）

| # | 论文/模型卡说法 | 代码实际 | 影响 |
|:--|:---|:---|:---|
| D1 | 熵在**可生成词表 `V_gen`、排除 mask token** 上算（§B.1 式 24；`GroundAnything-Model/README.md:277`） | `infer/decode/reliability.py:37-46` 对**全词表** logits 算熵，无裁剪；调用方 `infer/decode/sglang_algorithm.py:315-337` 传全词表切片 | **高**：同一个 τ 的语义不同，解码行为不可直接对齐 |
| D2 | 块内每位置以 `Bernoulli(t_b)`、`t_b~U(0,1)` 腐蚀（§A.7） | ~~`models/dlm/hybrid.py:590-591` 有噪声下界~~ **已核实不是差异**：`minimum_noise_level` 默认 0.0（`models/dlm/hybrid.py:180`）且所有发布训练配置显式设为 0.0（`configs/train/general.yaml:47` 等 6 个 YAML），此时 `probabilities == t`，与论文一致 | **无**（仅当有人调大该值时才偏离） |
| D3 | 主表用 entropy-guided decoding，未提温度 | 发布 profile 的 medium/dense 用 `T=0.1`、ocr_medium 用 `T=0.3`（`infer/decode/configs/task_profiles.json` → `tiers`） | **中**：论文正文不足以复现表 1–4 |
| D4 | 主表只用 entropy-guided（无 position penalty / repetition） | 代码支持 `entropy_position` / `entropy_position_repetition`（`infer/decode/reliability.py:129-155`），发布配方置 0/none | **低**：默认路径一致，但存在论文未描述的扩展策略 |
| D5 | GRPO 用 256 GPU（表 13） | 发布拓扑为 24 / 56 / 64 H800（`train/rl/current/trainer.py:1` 等） | **中**：RL 阶段不可逐位复现 |
| D6 | OCR 奖励是 §A.10.2 式 15–21 的一套公式 | 仓库有 6+ 个 OCR 奖励变体、10+ 个框奖励变体，系数与论文式子**不能逐项一一对上** | **中**：需先确定论文对应哪个变体 |
| D7 | MTP 是 "in-house causal drafter"（§C.1） | 仓库**完全不存在** MTP 代码 | **高**：Finding 2.2 不可验证 |
| D8 | 30 benchmark 有统一综合分 | `eval/README.md:198`：只报逐任务，不算 paper-wide aggregate | **高**：主 claim 数字不可复算 |

---

## 5. 给复现者的最小行动清单

1. **先对齐 D1**：把 `infer/decode/reliability.py` 的熵裁剪到 `V_gen`（排除 `|<MASK>|`=152670 及
   非生成性特殊 token），重测同一 τ 下的 COCO 行为，确认与论文的 τ=0.8 是否仍是同一运行点。
2. **核对 D2（已完成，此处留作复核记录）**：`minimum_noise_level` 在 `models/dlm/hybrid.py:180`
   默认为 0.0，且 `configs/train/{general,specialist,sft1,sft2,dlm_h800_smoke,dlm_h800_resume}.yaml`
   全部显式设为 0.0 —— 与论文 `Bernoulli(t_b)` 一致。复现时若改这个值就会偏离论文。
3. **以 `configs/eval/tasks.json` + `eval/README.md:89-120` 为基准**搭建 30 任务全量配置，
   自行实现聚合与多种子协议（N4/N5 无现成代码）。
4. **接受两个复现缺口**：MTP 对照（N1）与二次自推测（N2）必须自行实现；
   数据引擎（N3）无法重建，只能用公开数据近似。
5. **注意无 GPU 的前置条件**：所有服务路径（`infer/serve_sglang.py`、`infer/serve_vllm.py`）与
   训练路径（`configs/release/*_train.yaml`）都要求 CUDA/PPU 环境；`README.md:92` 列出
   测试过的加速器为 NVIDIA B300 / B200 / H200 / H800 与 PPU。本机 Windows 无 GPU，
   **只能做 CPU 级的契约/单元测试**（如 `tests/` 下的 6 个测试文件：
   `test_batch_predict.py`、`test_game_ui_metrics.py`、`test_gam_cache_builder.py`、
   `test_locany_to_gam.py`、`test_vllm_deployment.py`、`test_vlm_launcher.py`）。

---

## 6. 引用来源与边界说明

- 本文件的全部 `path:line` 来自本次会话中 `read` / `grep` / `pwsh`（`Get-ChildItem`、Python `json`）的原始输出。
- **未**修改任何源码、README、config、测试；**未**触碰 `docs/investigation/` 下队友文件
  （`repo-survey.md` 仅被 grep 只读检索过一次；其中提到的 `configs/task_profiles.json` 路径
  **经本笔记核实为不存在**，实际文件是 `infer/decode/configs/task_profiles.json`。
  该条属于"引自队友侦察、未二次核实"的反例，已在本笔记中修正）。
- 未执行任何模型前向、训练或评测脚本；所有状态判定都是静态代码核对的结果，运行时行为未验证。
- 论文侧的数字与出处见配套文件 `docs/papers/2026-yu-groundanything.md` 的 §7 claim–证据对照表。

### 补充：权重级验证（Lead，2026-10-09 追加）

上表第 485 行的"参数计数"一项原本标为 ⚠️ 可近似。Lead 随后用**只读 safetensors header**
（只读文件头 8 字节长度 + JSON header，**不解码任何张量数据**，不占显存）完成了逐张量精确核对：

| 核对项 | 结果 |
|:---|:---|
| 张量总数 / dtype | 569 个，**全部 BF16**，全部 `base_model.*` 前缀 |
| 精确参数总和 | **4,843,669,504**，与表 9 **完全相同（偏差 0.0%）** |
| 拆分 | vision+projector 428,482,048；language+lm_head 4,415,187,456 |
| 表 8 十二项规格 | **12/12 通过**（ViT 27 层/1024/4096/patch14；LLM 36 层/2560/9728/32 Q/8 KV/head 128；max_pos 262144；vocab 152671） |
| 视觉塔是否真的在权重里 | ✅ 165 个 `visual.*` 张量，含 27 个 block（`blocks.0`–`blocks.26`）与 projector `fc1 [4096,4096]`、`fc2 [2560,4096]` —— 与论文"2×2 聚合 + 两层 MLP（4096→4096→2560）"一致 |
| embed_tokens vs lm_head | 形状均为 `[152671, 2560]`（**形状层面无法判定是否 tie，需查 modeling 代码**） |
| config 的 `vocab_size` 不一致 | `config.json` 报 **152670**，而 `embed_tokens` 实为 **152671** 行 —— 差 1 正是后加的 `|<MASK>|`(id 152670)。**读取时须以张量为准**，`serve_sglang.py:88` 的 `|<MASK>| == vocab-1` 断言即依赖此关系 |

结论：**发布权重与论文声明的架构是同一份东西**，不是"同名不同物"。
边界：这仍是静态校验——**没有做前向、没有验证数值正确性，也没有验证权重与论文性能数字的因果**。
