# GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed

> 精读笔记（paper-reading 技能模板）。一手材料：本地 PDF（61 页，含 22 页正文 + 附录 A–E），
> 路径 `GroundAnything Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed.pdf`。
> PDF 文本层抽取**成功**（PyMuPDF 1.28.2，61 页 / 210,436 字符），本笔记所有数字均来自文本层；
> 图（Figure）内的数值来自文本层，**未与图像像素二次核对**，凡此类数值均标注 `[图内数值]`。
> 表内数值来自表体文本层，与正文交叉核对一致的已标注。

---

## 元数据

| 项 | 值 |
|:---|:---|
| 标题 | GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed |
| 作者 | Qize Yu\*, Lianrui Fan\*, Bowen Ping\*, Xini Ding\*, Zetian Song\*, Junbo Niu, Kaixuan Wang, Tianxing Chen, Yue Chen, Minghua He, Yuran Wang, Jie Huang, Haojun Zhang, Min Chen, Hao Li, Wenxuan Song, Ruihai Wu, Xianming Liu, Shilong Liu, Shuchang Zhou, Ping Luo†, Shiyu Huang†（\* 同等贡献，† 通讯，Qize Yu 为 project lead） |
| 单位 | XPeng Inc.；Peking University；The University of Hong Kong；UC Berkeley；Princeton；NUS；HKUST (GZ)（p.1） |
| venue | arXiv 预印本，未报告会议/期刊接收（PDF 文本层无 venue 字段） |
| 年份 | 2026（arXiv:2609.39600v1 [cs.CV]，30 Sep 2026，p.1） |
| 链接 | https://arxiv.org/abs/2609.39600 |
| 代码 | https://github.com/groundingpi/GroundAnything（本地 git remote 为 https://github.com/kuangdang123/GroundAnything.git） |
| 模型 | https://huggingface.co/GroundingPI/GroundAnything（DLM）、https://huggingface.co/GroundingPI/GroundAnything-VLM（AR） |
| 项目页 | https://groundingpi.github.io/groundanything/ |
| 页数 / 结构 | 61 页；正文 §1–§5（p.4–17）、参考文献（p.18–21）、附录 A–E（p.22–61）；共 33 表、33 图 |

## TL;DR（一句话）

把视觉定位重新表述为"结构化视觉证据抽取"，用**块内双向 + 块间因果**的 masked diffusion
（B=32）直接由 AR checkpoint 转换而来，在 30 个 grounding benchmark 上让 4B 的扩散解码
（61.75%）首次超过同规模 AR 前作，并在 COCO 上以 **4.51× 加速 / 仅掉 0.74 pp F1mIoU** 的自推测模式
证明"并行解码 + 精确坐标"可以兼得。

---

## 1. 问题定义与显式假设

**问题**：AR grounding 模型把空间预测序列化，带来 (a) 顺序延迟、(b) 对输出 token 强加因果顺序
（Abstract；§1 p.4）。密集场景与复杂版面下代价最大。

**显式假设**（论文自述，非我的推断）：
1. 「我们可能以不同顺序注意到物体或读到不同标牌，却对'有什么、在哪里'得到相同判断」→ grounding
   变量由图像与 query **联合约束**，不存在内在的从左到右生成顺序（§1 p.4、§3.1 p.5）。
2. 双向 masked diffusion 天然适配该结构：假设可并行涌现并用共享视觉证据 + 已恢复结构联合精化（§1）。
3. 语义顺序（含轨迹时间顺序）仍需保留 —— 这是双向注意力的**边界**，论文在 §3.1 p.6 显式声明：
   "Semantic ordering, including trajectory time order, is preserved."
4. 「Diffusion 是 Transformer 骨干内的一种生成策略」，其价值在合适负载（并行填充、结构化输出），
   而非普遍优于 AR（§4.5 Takeaway 2，p.16）。

## 2. 核心 claim（3 条）

### Claim 1 —— 定位能力：同一 4B 骨干的两个变体同时刷新"同规模"最佳
> "Across 30 grounding benchmarks, our autoregressive variant, GroundAnything-VLM, establishes a new
> overall state of the art among similarly sized models at 72.42%, remaining competitive with
> GPT-6 Astra (71.35%). With entropy-guided decoding, GroundAnything also surpasses the prior state of
> the art at this scale, averaging 61.75% versus 53.32% for the fast MTP-based LocateAnything model."
> —— Abstract，p.2；§4.2 p.9；图 16

### Claim 2 —— 速度：自推测模式在 COCO 上以极小质量损失换取 4.51× 加速
> "an optional self-speculative mode achieves a 4.51× speedup over the AR counterpart with a 0.74
> percentage-point drop in COCO F1mIoU" —— Abstract，p.2；图 1，p.1；§4.3.1 Finding 2.1，p.12；§D.1

### Claim 3 —— 设计：块内双向扩散比因果 MTP 更适合"联合约束的空间预测"草稿
> "Bidirectional diffusion with a limited number of refinement steps improves draft-generation
> efficiency over the tested causal MTP baseline." —— Finding 2.2，§4.3.2，p.15

注：Claim 3 的措辞在正文里比 §1 的贡献点更保守（§1 只说"systematic acceleration studies"），
且 §C.4 显式限定 "neither establishes superiority over every MTP design"。

## 3. 方法组件（按数据流）

### 3.1 输入/输出协议（§A.1，表 6 / 表 7；与 `grounding_anything/protocol.py` 逐字节一致，见第二节）

- 条目模板：`<|object_ref_start|>[IDENTIFIER]<|object_ref_end|><|box_start|>[PAYLOAD]<|box_end|>`
- box = 4 个原子坐标 token，顺序 `(x1,y1,x2,y2)`；point = 2 个，顺序 `(x,y)`
- 同一 identifier 的多个 tuple 共用一对 wrapper，tuple 之间用**无空格逗号**分隔；不同条目用「逗号+空格」分隔
- 被查询但缺失的类别输出普通文本 `None`（"not a padding element"，§A.1 p.23）
- OCR 转写写在 identifier 字段，不另开文本通道
- **坐标规范化（式 4）**：`q(v) = floor((999v + 500) / 1000)`，输入为 [0,1000] 网格时
- 输入 xywh → xyxy；多边形 → 轴对齐外接框；非有限/越界/反向/退化框直接 validation 失败
- box 按 x1 稳定排序；**无序点按 x 排序；语义顺序永不被空间排序替换**（§A.1 p.23）

### 3.2 架构（§3.2 p.6、§A.2 p.23–24，表 8 / 表 9 / 图 2）

| 部件 | 配置（表 8） | 本地 checkpoint 实测 |
|:---|:---|:---|
| 视觉编码器 | MoonViT-V2 (Kimi K3)：27 层；width 1024；FFN 4096；12 heads；QKV 1536；patch 14 | `vision_config`：27 层/1024/4096/12 heads/patch 14，`image_size=448` ✔ |
| Projector | LayerNorm(1024) → 2×2 聚合 → bias-free linear 4096→4096→2560 + GELU → RMSNorm(2560)，ε=1e-5 | 未在 config.json 暴露（在 modeling 代码内） |
| 语言骨干 | Qwen3-4B：36 层；width 2560；FFN 9728；32 Q heads / 8 KV heads；head_dim 128 | `text_config`：36 层（36×`full_attention`）/2560/9728/32/8/128 ✔ |
| 数值 | SiLU；attn dropout 0；RMSNorm ε=1e-6；RoPE base 5,000,000 | `rms_norm_eps=1e-6` ✔；`rope_theta` 不在顶层 config 中（未报告/未暴露） |
| 位置容量 | 262,144 | `max_position_embeddings=262144` ✔ |
| 视觉 token | 动态处理，最多 1024 个投影视觉 token（= 2×2 聚合前的 4096 patch）；通道 mean/std = (0.5,0.5,0.5) | 未在 config 暴露 |

- 词表（§A.2 p.24）：base 词表 **151,669** 条目 → +1000 坐标 token + `</c>` = **152,670**；
  坐标 ID **151669–152668**，`</c>` = **152669**；GroundAnything 再加 `|<MASK>|` = **152670**，
  总 **152,671**。EOS=151645，padding=151643，image/vision_start/vision_end = 151655/151652/151653。
  → **本地 `GroundAnything-Model/added_tokens.json` 实测**：坐标 token 恰 1000 个，ID 151669–152668；
  `</c>`=152669；`|<MASK>|`=152670；`<|im_end|>`=151645；`<|endoftext|>`=151643；
  `<|vision_start|>`=151652、`<|vision_end|>`=151653、`<|image_pad|>`=151655 —— **全部逐项吻合**。
- 输入 embedding 与输出 head **不共享（untied）**；因果与扩散两种模式共享**全部**模型权重
- mask 行初始化：用 `<|im_start|>`、`<|im_end|>`、`<|vision_start|>`、`<|vision_end|>` 四行在
  **FP32** 下的均值，分别写入两个（untied）词表矩阵，再 cast 回模型 dtype；**新增 5,120 参数**
  （= 2 × 2560）
- 参数量（表 9）：vision 401,214,464；projector 27,267,584；decoder(excl. vocab) 3,633,511,936；
  input embedding 390,835,200（VLM）/ 390,837,760（DLM）；output head 同；**总计 4,843,664,384（VLM）
  / 4,843,669,504（DLM）**。表注：**"4B 指的是语言骨干"**，完整多模态模型约 **4.844B**。

### 3.3 数据引擎（§3.3 p.6，图 3；§A.3 p.24）—— **未开源，不可复现**

流程：候选生成 → 字段级融合 → 任务校验 → 版本化监督 → 专家迭代。
- **字段级融合**：互补教师提供语义/定位/分割/文本/界面/版面证据；类别发现先于类别条件定位；
  预测映射回原图后，仅当 query scope 与标注粒度一致才对齐
- **校验与覆盖**：每个依赖有状态 `a(f) ∈ {pass, fail, pending}`；查询级覆盖 `c(I,Q)` 同状态空间
- **接受条件（式 5）**：`A(I,Q,Y) = 1[∀f∈F, a(f)=pass] · 1[h(Q)=0 ∨ c(I,Q)=pass]`
- **空目标声明**：必须包含 query-scoped 缺失事实 `f_abs`，防止空集上"真空真"通过校验
- 字段级融合/校验/专家迭代的具体实现、教师型号、迭代轮数 —— **未报告**

### 3.4 训练管线（§A.4 p.25，表 10）

| 阶段 | GroundAnything-VLM | GroundAnything |
|:---|:---|:---|
| Base VLM Training (Pretrain 1) | Stage 1–3；因果交叉熵 | Stage 1–3；因果交叉熵 |
| Coordinate Alignment (Pretrain 2 / SFT) | Stage 4；assistant-only 空间交叉熵 | Stage 4；同左 |
| AR-to-Diffusion conversion | — | `0.5 L_MDM + 0.5 L_AR`；权重共享 |
| Reinforcement learning | Causal GRPO | Causal GRPO；AR/扩散共享权重 |

- **Stage 1**：冻结 ViT + LLM，仅训 projector（图像-描述对，assistant-only CE）
- **Stage 2**：全部解冻，纯文本 + 通用图文联合训练；`L_mix = L_text + L_vlm`，两域**各自全局 token 均值**后相加
- **Stage 3**：通用 VQA/指令跟随/图像描述/视频；context 与 packing 均为 32K；**无独立纯文本配额**，
  空间监督集中在 Pretrain 2。**基础预训练累计 exposure > 700B tokens**（§A.5 p.25）
- **Stage 4**：从 Stage 3 初始化，更新 ViT + projector + language，8 个空间任务组
  （detection、GUI、referring grounding、referring pointing、OCR、document layout、
  dense pointing & counting、visual prompting）；context/packing 8K

### 3.5 AR→Diffusion 转换（§3.4.2 p.7、§A.7 p.26–27，图 4）

- 保留预训练骨干与 grounding 词表，**新增 1 个 mask token**；`x = [V, P, Y]`，把 Y 切成
  **B = 32** 的连续块；**块可以跨越几何 tuple 边界**，遇 response-span 边界停止，最后一块可短，
  **无需语义 padding**（§A.7 p.26）
- **互补腐蚀**：每块采 `t_b ~ U(0,1)`；块内每个 supervised 非 EOS 位置抽
  `m_i^(1) ~ Bernoulli(t_b)`，并令 `m_i^(2) = 1 − m_i^(1)`。视图 a 把目标替换为 `|<MASK>|`。
  **两个视图都把 EOS mask 掉**。图像、prompt、header、非监督 turn、空 thinking 前缀**不腐蚀**
- 联合输入 `[w^(1); w^(2); x]`：**视觉 embedding 只出现在 clean 流 x 里**；每个 noisy 视图保留一份
  未腐蚀的 prompt 拷贝
- **注意力可见性（图 4）**：noisy 响应块内 → 双向（自读）；可读 clean 图像/prompt 上下文与
  **严格在前的** clean 响应块；**不能读**当前/未来的 clean 目标，也不能读另一个 noisy 视图。
  clean token 用 token 级因果注意力，**不能读任何 noisy 视图**
- **联合目标（式 2、6、7）**：`L = 0.5 L_MDM + 0.5 L_AR`；
  `L_MDM = −[Σ_{a=1..2} Σ_{i∈M^(a)} log p_θ(y_i | x, w^(a); A_a)] / (|M^(1)|+|M^(2)|)`；
  `L_AR = −(1/|T|) Σ_{i∈T} log p_θ(y_i | V, P, y_<i)`。
  MDM 均值**包含被 mask 的 EOS 目标**，**无 inverse-noise 加权**；两者都保留 next-token shift
  （hidden 位置 `i−1` 预测 target `i`）
- 两子阶段（表 12）：**Phase I** 冻结 ViT，训 projector + language；**Phase II** 全部更新；
  优化器/调度器在阶段间**重启**

### 3.6 强化后训练：Causal GRPO（§3.4.3 p.7、§A.8 p.27，图 5）

- 每个 prompt 采 **G = 8** 个响应，温度 **T = 0.7**；更新前用无梯度 teacher-forcing 记录 detached
  旧策略 log-prob，第二遍算当前策略梯度；两者都用 **FP32 log-softmax**，且都是**因果策略的精确似然**
  （"not likelihoods of a diffusion denoising trajectory"）
- **组内标准化（式 8，population std）**：`Z_G(u_i) = (u_i − ū) / sqrt( (1/G) Σ (u_j − ū)² + 1e-6 )`
- **优势（式 9）**：`A_i = Z_G( Σ_{c∈C_task} w_c · Z_G(R_{c,i}) )` —— 先分量标准化再加权，**再组内标准化**
- 框奖励权重 **(0.7, 0.3)**；OCR 与 pointing 各给一个标量（单位权重）；常数组的优势为 0
- **策略目标（式 3/11）**：`r_{i,t}` = 当前/旧 因果 token 概率比（在 rollout 温度下），
  `L = −(1/G) Σ_i (1/|T_i|) Σ_{t∈T_i} min(r A, clip(r,1−ε,1+ε) A)`，**ε = 0.2**；
  **每次 rollout 一次更新**，**无 reference-policy KL**；冻结 ViT，更新 projector + language
- GroundAnything-VLM 变体：冻结 ViT **和** projector，保留冻结监督参考，**KL 系数 0.02**，
  用样本标准差（ε=1e-8），加权求和后对**整个响应 batch** 标准化（§A.9，表 13）

### 3.7 并行推理（§3.5 p.8、§B p.30–31）

- **Block 状态与 token 对齐（§B.1）**：图文 prefill 建立因果 prefix cache + 第一个 anchor token；
  一个物理块 = 1 个已知 anchor + **31 个 mask 位置**；**子块从左到右**完成，但每次去噪 forward
  仍**评估整个物理块**；已完成 token 对后续 mask 位置可见；prefix cache 全程复用
- **熵引导解码（式 24）**：`p_j = softmax(z_j)`，`H_j = −Σ_{v∈V_gen} p_j(v) log p_j(v)`
  （**在可生成词表上，排除 mask token**）。对当前子块仍 mask 的位置集合 U，
  **一次性提交所有** `S = {j ∈ U : H_j ≤ τ}`；若 S 为空，只提交 `argmin_{j∈U} H_j` 以保证进度。
  **提交的 token 既不重新 mask 也不修正**。采样与可靠性评估是**两个独立操作**
- **因果 cache 构造（§B.1 p.30）**：双向 hidden state 依赖右侧位置，不能直接当因果历史；
  块完成后用**一次因果 forward** 重算该块的权威 KV 并预测下一个 anchor。
  **该 pass 不验证、不拒绝、不替换**已完成的块。因此一个需要 D 次去噪 pass 的块共用 **D + 1 次 NFE**
  （不含 initial prefill）；pass 数取决于提交决策，**不是固定的 per-response 去噪步数**
- **其他提交策略（§B.1 p.31）**：Dynamic Decoding 用置信度阈值 + 强制最优位置兜底；
  Static Decoding 按规定的 per-pass 配额提交最高置信位置。**这些置信度不应与上面的未修改 logit 熵混同**
- **自推测（§B.2 p.31，图 6）**：用共享权重做双向起草 + 因果验证；
  **只接受与因果预测一致的最长连续 draft 前缀**，第一个不一致处终止接受，后面的偶然匹配全部丢弃，
  被拒后缀从 cache 中移除
  - **线性**：一次双向 pass 出候选块 + 一次因果 pass 并行验证。接受最长匹配前缀，并在首个 mismatch
    处提交 verifier correction。**每轮 2 NFE、2B query token**。seed 只计一次，不计入 draft acceptance
  - **二次**：融合 query 含 **B 组、每组 B+1** 个 token，第 i 组在重叠位置 `[t+i, …, t+i+B]` 上排成
    `[d_i, M, …, M]`；行首做因果验证，每行的 mask 位置构成双向提议组；所有 query 读 cached prefix。
    把行预测 `a_i` 与 `d_{i+1}` 比较，接受计数从已知首 token `d_0` 起算为 1；首个 mismatch 处选该行的
    提议作为下一 draft（**复用计算但不验证该提议**）。初始化后**每轮 1 NFE、B(B+1) query token**
  - 论文显式声明："linear/quadratic 讲的是 query-token 计数，**不是** Transformer FLOPs 或固定的接受 token 数"
- **执行优化（§B.3）**：Native PyTorch (Eager) → SGLang (Eager) → CUDA Graph replay → 选择性 FP8；
  FP8 只作用于**合规的语言模型线性层**，其余保持 BF16

---

## 4. 数据与评测

### 4.1 Benchmark 套件（§4.1 p.9、§4.2 p.9、§E p.36–37）

- **30 个 benchmark、11 类感知能力、44 个 baseline**（§4.1），对比 44/45 见红旗 R1
- 检测：COCO、LVIS、Dense200、VisDrone
- 指代：RefCOCO、RefCOCO+、RefCOCOg
- 目标 pointing：COCO、LVIS、Dense200、VisDrone、RefCOCOg val/test
- 空间/GUI：RefSpatial、RoboSpatial、ScreenSpot-V2、ScreenSpot-Pro、OSWorld-G
- OCR：HierText、ICDAR2015、TotalText、SROIE；版面：DocLayNet、M6Doc
- 视觉提示：FSC147、Dense200
- **训练集/评测集划分与去污染：未报告**（见红旗 R8）

### 4.2 指标定义（§4.1 p.9、§E p.37）

- **F1mIoU**：box grounding 与 OCR 使用。定义是 **"threshold-aggregated F1 score, not mean matched-box
  IoU"** —— 即对 IoU ∈ {0.50, 0.55, …, 0.95} 逐阈值算 F1 再取均值。box 任务同时报 IoU 0.50 / 0.95 / 0.50:0.95 聚合
- **F1@Point**：目标 pointing 用，point-in-mask F1，平衡漏检与误检（§4.2 p.10，遵循 Jiang et al. 2026 的 SAM 掩码判定）
- **任务特定 accuracy**：空间 pointing 用 point-in-mask accuracy；ScreenSpot 用 action accuracy；
  OSWorld-G 用 exact accuracy
- **OCR 额外要求** loose-match 协议下的转写一致，并**单独报告 parse-error rate**
- 聚合约定：**RefCOCO avg = RefCOCO/RefCOCOg/RefCOCO+ 的非加权均值**；**RefSpatial avg 平均
  Location 与 Placement 两个 split**
- 符号：`–`/`N/A`/`UNK` 分别表示未报告 / 不支持或未解决 / 零样本状态未说明；
  `*` 表示外部报告行；`†` 表示 prompt/协议对齐不确定（分数只作描述性）

### 4.3 评测协议与统计（§4.1 p.9）

- 自家模型：**所有数值结论基于 10 次运行 = 5 个随机种子 × 每种子 2 次**
- 本地评测的 baseline：**3 次运行 = 3 个随机种子 × 每种子 1 次**（理由：算力成本高）
- GPT-6 Astra 使用 thinking effort **High**
- **主表（表 1–4）全部用熵引导扩散解码，不用自推测** —— 理由：自推测混合了因果 AR 验证/纠正，
  "is therefore not pure DLM decoding"（§4.1 p.9）。这是论文做得很好的一处公平性声明
- **方差 / std / 置信区间：全文未报告**（见红旗 R2）

### 4.4 主表关键数值（精确抄录）

**表 1（Detection & referring, F1mIoU，p.11）**

| 模型 | COCO | LVIS | Dense200 | VisDrone | RefCOCOg val | RefCOCOg test | RefCOCO avg |
|:---|---:|---:|---:|---:|---:|---:|---:|
| GroundAnything-VLM | 63.70 | 56.63 | 75.55 | 40.74 | 84.37 | 83.56 | 83.43 |
| **GroundAnything** | 60.72 | 52.59 | 70.04 | 33.76 | **91.61** | **91.10** | 85.33 |
| Rex-Omni | 56.28 | 46.74 | 53.29 | 27.19 | 73.90 | 74.76 | 69.29 |
| LocateAnything Hybrid | 59.12 | 49.56 | 50.07 | 28.57 | 76.43 | 77.67 | 78.06 |
| SenseNova-Vision | 57.49 | 56.12 | 68.13 | 42.35 | 78.69 | 79.48 | 77.55 |
| GPT-6 Astra | 62.75 | 54.97 | 65.04 | 37.12 | 74.98 | 78.91 | 77.81 |
| Qwen3.7-Max (>1T) | 62.79 | 53.28 | 31.20 | 41.59 | 80.54 | 81.71 | 80.20 |
| GroundingDINO | 60.56 | 52.61 | 24.92 | 34.47 | 49.77 | 50.43 | 45.15 |

**表 2（Robot/spatial/GUI，accuracy，p.12）**

| 模型 | RefSpatial avg | RefSpatial Unseen | RoboSpatial Context | ScreenSpot-Pro | ScreenSpot-V2 | OSWorld-G |
|:---|---:|---:|---:|---:|---:|---:|
| GroundAnything-VLM | 69.00 | 62.34 | 72.13 | 65.34 | 94.89 | 74.65 |
| **GroundAnything** | 63.00 | 62.34 | 69.67 | **75.96** | **95.60** | **81.21** |
| GPT-6 Astra | 85.93 | 81.93 | 65.69 | 93.17 | 97.88 | 86.70 |
| Qwen3.7-Max | 68.75 | 57.14 | 69.67 | 55.06 | 81.13 | 49.29 |

**表 3（OCR / layout / visual prompting，F1mIoU，p.13）**

| 模型 | HierText | ICDAR2015 | TotalText | SROIE | DocLayNet | M6Doc | FSC147 | Dense200(VP) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| GroundAnything-VLM | 41.33 | 42.50 | 49.03 | 71.15 | 85.78 | 74.76 | 60.29 | 75.26 |
| **GroundAnything** | 33.31 | 41.81 | 43.08 | 43.64 | 68.38 | 56.69 | 52.43 | 63.52 |
| Rex-Omni | 34.46 | 45.65 | 52.35 | 48.35 | 68.06 | 54.95 | 57.15 | 55.50 |
| GPT-6 Astra | 39.58 | 48.87 | 53.55 | 53.57 | 77.54 | 60.59 | 61.04 | 68.94 |

**表 4（Object pointing，F1@Point，p.13）**

| 模型 | RefCOCOg val | RefCOCOg test | COCO | LVIS | Dense200 | VisDrone |
|:---|---:|---:|---:|---:|---:|---:|
| GroundAnything-VLM | 90.44 | 91.03 | 84.92 | 79.81 | 84.16 | 68.03 |
| **GroundAnything** | 85.75 | 85.73 | 71.32 | 64.57 | 77.14 | 57.94 |
| Rex-Omni | 84.96 | 85.32 | 79.74 | 70.04 | 76.66 | 51.97 |
| GPT-6 Astra | 87.80 | 84.90 | 82.17 | 77.14 | 86.57 | 65.62 |

**表 5（输出 token 效率，p.16）**

| 模型 | COCO boxes/img | COCO tokens/img | COCO tokens/box | Dense200 boxes/img | Dense200 tokens/img | Dense200 tokens/box |
|:---|---:|---:|---:|---:|---:|---:|
| SEED1.5-VL（外部） | 4.2 | 631.0 | 148.8 | 73.1 | 5446.3 | 74.5 |
| GroundAnything-VLM | 6.1 | 46.4 | **7.6** | 87.6 | 446.8 | **5.1** |
| GroundAnything | 6.1 | 46.4 | **7.6** | 87.6 | 446.8 | **5.1** |

→ §D.4：相对 SEED1.5-VL 约 **19.6×** 与 **14.6×** 更少 token/box；论文明确说明这是**序列化长度比，
不是同硬件时延比**。

### 4.5 速度评测关键数值

**解码策略扫描（B = 32，sub-block size 4，COCO F1mIoU）**

| 配置 | TPF | TPS | COCO F1mIoU | 出处 |
|:---|---:|---:|---:|:---|
| GroundAnything-VLM（AR 参考） | — | **49.6** | **63.70** | §D.1 p.34；图 8/9/10 虚线 |
| 线性自推测（B=32） | **2.18** | **223.8** | **62.96** | 图 8 [图内数值]、图 10 [图内数值]、§D.1 |
| 线性自推测（B=16） | 2.21 | 226.6 | 60.71 | 图 10 [图内数值] |
| 线性自推测（B=8 / B=4） | 2.01 / 1.41 | 215.9 / 157.7 | 60.39 / 57.75 | 图 10 [图内数值] |
| 二次自推测（B=32） | **3.17** | **49.9** | — | 图 8 [图内数值]、§D.1 |
| 二次自推测（B=16） | 3.17 | 88.0 | — | 图 8 [图内数值] |
| 熵引导（τ=0.8, B=32） | 1.24 | **127.0** | **60.72** | 图 9/10 [图内数值]、§4.3.1、§D.1 |
| 熵引导（B=16） | 1.23 | 136.6 | **61.58**（该扫描最佳） | 图 10 [图内数值]、§D.1 |
| 熵引导（B=8 / B=4） | 1.21 / 1.18 | 131.7 / 134.8 | 60.93 / 53.91 | 图 10 [图内数值] |

→ 两个主加速比：**223.8 / 49.6 = 4.51×**（质量差 **63.70 − 62.96 = 0.74 pp**）、
**127.0 / 49.6 = 2.56×**（质量差 **63.70 − 60.72 = 2.98 pp**）（§D.1 p.34）。

**熵阈值扫描（B=32，sub-block 4，图 9 [图内数值]）**

| τ | 0.2 | 0.4 | 0.6 | **0.8** | 1.0 | 1.2 |
|:---|---:|---:|---:|---:|---:|---:|
| TPF | 1.02 | 1.03 | 1.11 | 1.24 | 1.37 | 1.60 |
| TPS | 107.3 | 113.0 | 118.2 | 127.0 | 134.3 | 148.7 |
| COCO F1mIoU | 59.97 | 59.92 | 59.82 | **60.72** | 54.06 | 56.14 |

→ τ 在 0.8 之后质量**断崖**（60.72 → 54.06 @ τ=1.0），τ=1.2 部分回升到 56.14；论文自己说这条曲线
"supports a speed–accuracy trade-off rather than a universal monotonic accuracy law"（§D.1）

**DLM vs MTP（§4.3.2 p.14–15，§C p.31–33）**

| 量 | 数值 | 出处 |
|:---|:---|:---|
| K=16，one-step DLM 接受 draft token / 轮 | **3.48**（接受率 **21.76%**） | §C.2 p.32；图 11 [图内数值] |
| K=16，causal MTP 接受 draft token / 轮 | **2.69**（接受率 **16.81%**） | 同上 |
| 每接受 draft token 全轮成本 | MTP **7.71 ms** → DLM 1 step **5.38 ms**（**−30.2%**） | §C.2 p.32；图 12 [图内数值] |
| DLM 2 个 refinement step | **4.70 ms**（最低） | §4.3.2、§C.3 p.33 |
| DLM 4 / 8 个 step | 6.84 ms / 11.83 ms | §C.3 p.33；图 13 [图内数值] |
| 每 committed 输出 token 成本（2 step） | 仅降 **3.8%** | §C.3 p.33 |
| K=32，DLM 接受前缀 | 降到 **3.30** token | §C.2 p.32 |
| MTP 最低成本 | **6.58 ms @ K=8** | §C.2 p.32 |
| K=4 | MTP **7.15 ms** 优于 DLM **9.31 ms** | §C.2 p.32 |
| 固定 deadline 400 ms committed tokens | DLM **97.1** vs MTP **84.6** | §4.3.2 p.14；§C.3；图 13(c) |
| 固定 deadline 50 ms | MTP **10.6** vs DLM **8.6** | 同上 |

**草稿诊断（图 14 [图内数值]，§C.4 p.33）**：K=8，两个 DLM 用 2 个 refinement step

| 指标 | COCO | Dense200 | DocLayNet | GUI |
|:---|---:|---:|---:|---:|
| format error：causal MTP → DLM(bidir) | 5.01 → **1.80** | 4.13 → **1.64** | 3.82 → **1.48** | 0.55 → **0.49** |
| valid-box miss：causal MTP → DLM(bidir) | 12.8 → **9.5** | 24.9 → **14.8** | 19.0 → **11.5** | 8.0 → **7.5** |

→ 正文精确值：Dense200 **24.89% → 14.79%**（降 **10.10 pp**），DocLayNet 降 **7.52 pp**，
GUI 仅降 **0.46 pp**；重训的 **causal DLM 对照为 21.74%**（§4.3.2 p.15、§C.4 p.33）

**基础设施渐进优化（图 15 / §D.2 p.34）**

| 解码模式 | 最终 ms/token | 相对各自 Native PyTorch(Eager) 的累计加速 |
|:---|---:|---:|
| Dynamic Decoding | 6.59 | **3.48×** |
| Static Decoding (4) | 6.76 | **3.45×** |
| Entropy-Guided Decoding | 7.15 | **3.51×** |
| Self-Speculative Decoding | **3.73** | **4.06×** |

→ 论文明确警告：这些比值是**同一解码模式内**的实现对比，**不能**与 4.51× 的算法加速相乘（§D.2 p.34–35）。
且"it does not establish that FP8 preserves accuracy"（§D.2 p.35）。

### 4.6 消融（§4.4 p.15–16、§D.3 p.35、图 16）

| 消融项 | GroundAnything-VLM | GroundAnything |
|:---|---:|---:|
| 坐标表示：量化坐标 → 文本坐标 | 73.03 → 71.60（**−1.43 pp**） | 61.66 → 55.98（**−5.68 pp**） |
| 视觉编码器：MoonViT-V2 → MoonViT (Kimi-VL) | 72.81（−0.22） | 61.03（−0.63） |
| 视觉编码器：MoonViT-V2 → Qwen3-ViT (Qwen3-VL-4B) | 71.12（−1.91） | 60.12（−1.54） |
| 训练阶段：去掉 "Stage III" | 71.74（**−1.29 pp**） | 45.04（**−16.62 pp**） |
| 完整（消融协议下） | 73.03 | 61.66 |
| 最终 30-benchmark 汇总 | **72.42** | **61.75** |

- §4.4：文本坐标的 **0.25× 速度**注释只针对该表示对比，不等同于 SEED1.5-VL 的外部 token 计数比
- §D.3 明确承认：**完整消融分（73.03/61.66）与最终汇总（72.42/61.75）不同**，只做组内比较，
  不替换任何一组分数，也不假设聚合方式一致 —— 这是一处值得称赞的诚实声明
- §D.3 明确承认：Stage III 消融"changes a complete phase and does not isolate individual reward
  components or optimization choices within it"
- §D.3.1（式 31–32）给出多级视觉注入对坐标读出影响的**精确一阶敏感度恒等式**，
  但结论只是：现有消融支持编码器选择，**要归因于"没有 DeepStack"需要专门的注入实验**

### 4.7 与 baseline 的公平性

**做得好的部分**：
- 主表明确排除自推测，保证 DLM 与 AR 的解码语义可比（§4.1 p.9）
- 外部行用 `*` 标注、协议不确定用 `†` 标注，并逐条说明原因（§4.2 p.10、§E p.37）
- `N/A` 被明确定义为"接口或可靠 prompt/parser 组合不可用"，**不**等于模型缺乏该能力（§E p.37）
- DLM–MTP 对照（§C.1）显式匹配视觉预处理、projector、坐标词表、序列化、AR 骨干初始化、
  有序样本、监督 token exposure、优化器更新数、有效 batch、candidate-length 采样

**不足**：
- §C.1 自己承认："These controls do not imply equal trainable parameter counts or training FLOPs"
- 主要 MTP baseline 是**自研的 in-house causal drafter**，且**明确不是 LocateAnything 的复现**
  （"it is distinct from LocateAnything… is not a reproduction of LocateAnything"，§C.1 p.31）
- 自研模型 10 runs vs 本地 baseline 3 runs —— 评测投入不对称（§4.1）
- 表 1–4 的对手没有做逐任务 prompt/超参搜索的报告（见红旗 R1）

## 5. 算力

| 阶段 | GPUs | per-device / global batch | 序列长度 | 可训练模块 | 峰值 LR（language / vision / projector） | 出处 |
|:---|---:|:---|:---|:---|:---|:---|
| Stage 1 | **128** | 6 / 768 | 8K / 8K | Projector | frozen / frozen / 5e-5 | 表 11 p.26 |
| Stage 2 | **128** | 2 / 256 | 8K / 8K | All | 1e-5 / 1e-6 / 1e-6 | 表 11 |
| Stage 3 | **128** | 2 / 256 | **32K / 32K** | All | 1e-5 / 1e-6 / 1e-6 | 表 11 |
| Stage 4（Pretrain 2 / SFT） | **128** | 2 / 256 | 8K / 8K | All | 1e-5 / 1e-6 / 1e-6 | 表 11 |
| AR→Diffusion Phase I | **256** | 32 / 8192 | clean 4096 | Projector + language | 2e-5 / frozen / 2e-5 | 表 12 p.27 |
| AR→Diffusion Phase II | **256** | 32 / 8192 | clean 4096 | All | 2e-5 / 2e-6 / 2e-5 | 表 12 |
| GRPO（GroundAnything） | **256** | 1 / 1（prompt/response 32/256，8 响应/prompt） | — | Projector + language | 2e-6 | 表 13 p.28 |
| GRPO（GroundAnything-VLM） | **256** | 同上 | — | Language | 5e-7 | 表 13 |

- 四个预训练/对齐阶段统一：BF16、ZeRO-1、AdamW (β1,β2)=(0.9,0.95)、FlashAttention、activation checkpointing；
  LR 调度 Cosine，warmup 3%，每阶段 1 epoch（= 配置预算）
- GroundAnything GRPO：AdamW (0.9, 0.999)、ε=1e-8、weight decay 0、grad clip 1.0、BF16/ZeRO-1、FP32 优化器状态、
  seed **20260902**；VLM GRPO：weight decay 0.01、BF16/**ZeRO-2**、seed **20260812**
- 转换阶段：BF16、ZeRO-1、1 步梯度累积、AdamW (0.9,0.95)、eps 1e-8、wd 0.1、clip 1.0、Cosine + 3% warmup；
  Phase I seed 32/32、Phase II seed 42/42
- 基础预训练累计 exposure **> 700B tokens**
- **推理基准硬件：NVIDIA H800**（表 p.15 "NVIDIA H800 inference"）；本地仓库 README 说训练
  profile 面向 "H800-class CUDA GPUs"
- **未报告**：训练 GPU **型号**、总 GPU-hours、墙钟时长、能耗、是否使用 FP8/BF16 混精训练的完整细节

## 6. 术语表（本论文中的确切含义）

| 术语 | 本论文中的定义 | 出处 |
|:---|:---|:---|
| **DLM** | Diffusion Language Model，本论文指 masked/blockwise 扩散语言模型；解码时块内双向、块间因果 | §3.1 |
| **blockwise denoising** | 把响应 Y 切成 B=32 的连续块，块内并行去噪、块间因果；块可跨几何 tuple 边界 | §3.4.2、§A.7 |
| **B（物理块大小）** | 解码扫描中的物理块大小，**包含已知 anchor**；发布配方 B=32，即 1 anchor + 31 mask | §B.1、§D.1 p.34 |
| **K（candidate length）** | DLM–MTP 对照中的**新提议 token 数**，**不含** anchor / padding / verifier correction 或 bonus。**B 与 K 不可混同** | §C.2 p.32、§D.1 p.34 |
| **entropy-guided decoding** | 对仍 mask 的位置用**未修改** logit 分布算熵 `H_j = −Σ_{v∈V_gen} p_j(v) log p_j(v)`（可生成词表，排除 mask token），一次性提交所有 `H_j ≤ τ`；若为空则只提交 `argmin H_j`；提交后不重 mask 不修正 | §3.5 p.8、式 24、§B.1 p.30 |
| **Dynamic / Static Decoding** | 用**置信度**阈值 + 强制兜底 / 固定 per-pass 配额；**其置信度不等于**上面的未修改 logit 熵 | §3.5、§B.1 p.31 |
| **self-speculative decoding（线性）** | 双向起草 1 块 + 因果并行验证，接受最长匹配前缀，首个 mismatch 处提交 verifier correction；**每轮 2 NFE、2B query token** | §3.5、§B.2 p.31 |
| **self-speculative decoding（二次）** | 融合 query = B 组 × (B+1) token，行首因果验证、行内双向提议；接受计数从 d_0 起算；首个 mismatch 选该行提议为下一 draft（复用但不验证）；**初始化后每轮 1 NFE、B(B+1) query token** | §B.2 p.31 |
| **NFE** | Number of Function Evaluations，模型前向次数。一个需要 D 次去噪 pass 的块用 **D+1** 次 NFE（+1 为因果 cache 构造 pass），不含 prefill | §B.1 p.30 |
| **TPF** | committed output tokens **per model forward**；必须计入因果 cache 构造/验证 pass | §D.1 p.34 |
| **TPS** | output tokens per second | §D.1 |
| **MTP** | Multi-Token Prediction，多 token 预测的并行框解码路线（LocateAnything 系列）。本论文的主 MTP baseline 是**自研因果 recurrent future-token 模块**（horizon 间权重共享），另有 block-MTP 对照 | §2.1、§C.1 p.31 |
| **坐标 token** | `<0>`–`<999>` 共 **1000** 个原子量化坐标 token，ID 151669–152668；输入按式 4 量化 | §3.2、§A.1 式 4、表 7 |
| **F1mIoU** | **阈值聚合 F1**（IoU 0.50:0.05:0.95 逐阈值算 F1 再平均），**不是**匹配框的平均 IoU | §E p.37 |
| **F1@Point** | point-in-mask F1，平衡漏检与误检；point 正确性按 Jiang et al. (2026) 用 SAM 派生掩码判定 | §4.2 p.10 |
| **R_set / R_strict** | 框奖励的两项：集合完备性（GT-wise 匹配的软 F1，**允许一个预测支撑多个 GT**，因此是完备性代理而非有界一对一 F1）；严格定位（含 format/count/F̄/IoU/sort 加权项与 oversized/duplicate 罚项） | §A.10.1 式 12–14 |
| **Z_G** | 组内标准化算子；式 8 用**总体标准差**（population std），ε=1e-6 | §A.8 式 8 |
| **Stage 1–4** | Stage 1–3 属于 Pretrain 1；Stage 4 = Pretrain 2 = SFT。**注意与图 16 中名为 "Stage III" 的消融标签不是同一编号体系** | §A.4 p.25、§D.3 p.35 |
| **GAM** | 本地仓库的评测模式名，指 GAM spatial-token 协议（0–999 坐标）；论文正文未使用该缩写作为方法名 | 本地 README `GroundAnything-Model/README.md:166` |

## 7. Claim 与证据对照表

| # | Claim | 支撑位置 | 证据强度 | 边界或反例 |
|:--|:---|:---|:---|:---|
| C1a | GroundAnything-VLM 在 30 benchmark 上达到同规模最佳 72.42% | 摘要 p.2；§4.2 p.9；图 16 p.16；逐任务证据＝表 1–4 | **中**。逐任务数值齐备且有 10 runs，但 ①**没有任何一张表给出 72.42 这个聚合值本身**（只有图 16 的两个端点和正文一句话）；②**没有 std / 置信区间**；③"similarly sized"的分组标准是"语言骨干配置"（§E p.37），非总参数 | 语料不同划分、`†` 标注的协议不确定行仍计入 |
| C1b | GroundAnything-VLM 与 GPT-6 Astra (71.35%) 竞争力相当 | 摘要；§4.2 p.9 | **弱**。71.35 只在正文与图 16 出现，**无表**；Astra 的 `thinking effort High` 是单一设置（§4.1） | Astra 在 RefSpatial（85.93 vs 69.00）与 GUI（93.17 vs 65.34）**大幅领先**，"competitiveness"仅成立于总体均值 |
| C1c | GroundAnything（纯扩散）平均 61.75%，超过前作 53.32%（LocateAnything） | 摘要；§1 p.4；§4.2 p.9；图 16 | **中偏弱**。61.75 与 53.32 均**无独立表格**；表 1–4 里 LocateAnything Fast 的逐任务是 COCO 53.06 / LVIS 42.90 / Dense200 20.70 / VisDrone 9.82 / RefCOCOg 75.30–76.50 / DocLayNet 49.37 / M6Doc 47.03 等，**无法从正文表格复算出 53.32** | 表 3 显示 DLM 在 OCR/layout 上明显落后 AR（HierText 33.31 vs 41.33；SROIE 43.64 vs 71.15），论文自述"OCR and layout exhibit larger diffusion–AR gaps"（§4.2 p.10） |
| C1d | GroundAnything 在有定位精度要求时大幅领先（RefCOCOg 91.61/91.10） | 表 1 p.11；§4.2 p.10 | **中**。数值明确、超越所有列出对手（含 GPT-6 Astra 74.98/78.91），且 §E.1 给出阈值分解 | 论文未解释"扩散反而在 RefCOCOg 上超过自家 AR 5–8 pp"的机制；只在 §4.5 给一般性论述 |
| C2a | 自推测 4.51× 加速，COCO F1mIoU 仅掉 0.74 pp | 摘要；图 1；Finding 2.1 p.12；§D.1 p.34（223.8/49.6=4.51×；63.70−62.96=0.74） | **强**。分子分母与差值都可从 §D.1 的明确数值复算 | ①是 COCO 单数据集，**不可外推到 30-benchmark 均值**（§D.1 明示）；②**不是纯 DLM 解码**（混了因果 AR 验证），论文自己因此把它排除出主表；③§D.1 明示 0.74 pp 的差距**并不**意味 verifier 接受了不匹配 token |
| C2b | 熵引导纯扩散 2.56× 加速，掉 2.98 pp | §4.3.1 p.11；§D.1 p.34（127.0/49.6；63.70−60.72） | **强**。可复算 | 同样只在 COCO；τ=0.8 是在 COCO 扫描上选的，而 COCO 也是 benchmark 之一 |
| C2c | 基础设施渐进优化带来 3.45–4.06× 加速 | 表 p.15；图 15；§D.2 p.34 | **中**。四个模式的 ms/token 与倍数都明确 | ①是**同一模式内**的实现对比，不是跨方法；②**未验证 FP8 保精度**（§D.2 明示）；③图 15 只报时延，**没有每个基础设施阶段的匹配质量消融**（§B.3 也承认量化会改变 logits 与解码决策） |
| C3 | 块内双向扩散的草稿效率优于所测因果 MTP | Finding 2.2 p.15；图 11–14；§C.2–C.4 | **中**。K=16 时 3.48 vs 2.69 token/轮、5.38 vs 7.71 ms 都是明确数值，且有 attention 对照（重训 causal DLM 21.74% vs 双向 14.79%） | ①对照的 MTP 是**自研基线，不是 LocateAnything 复现**（§C.1）；②**block-MTP（双向）同样优于 causal MTP**，所以"双向性本身不是扩散独有的优势"（§C.4）；③K=4 时 MTP 反而更优（7.15 vs 9.31 ms）；④固定 deadline 50 ms 时 MTP 领先（10.6 vs 8.6 tokens）；⑤§C.4 明确"neither establishes superiority over every MTP design" |
| C4 | 紧凑序列化带来 19.6× / 14.6× 的 token 数下降 | 表 5 p.16；§D.4 p.36 | **中**。表格数值明确 | 论文自己声明："The ratios reflect serialization length, **not** matched-hardware latency"，检测实例数与模型执行差异使其**不能**读作加速 |

**无证据项**：无。所有 claim 都能指到表号/图号/节号；但 **C1a 的 72.42、C1b 的 71.35、C1c 的 61.75/53.32
没有可复算的聚合表**，只存在于正文与图中，属"有出处、无表格"。

## 8. 红旗清单（逐条核对）

| # | 红旗 | 判定 | 证据 |
|:--|:---|:---|:---|
| R1 | **对手没有消融 / 超参未对齐** | ⚠️ **部分命中** | 对手只被评测、没被消融；外部行靠 `*`/`†` 标注，`†` 明确说是"uncertain prompt/protocol alignment，分数只作描述性"（§4.2 p.10、§E p.37）。GPT-6 Astra 只测 `thinking effort High` 一个设置（§4.1）。本地评测的 baseline 只跑 3 runs。另外**图 7 caption 写 "Across 45 Baselines"，§4.1 写 "44 baselines"，差 1，未解释**。 |
| R2 | **单 seed / 无方差** | ⚠️ **命中（半）** | 自家模型 5 seeds × 2 runs = 10 runs，本地 baseline 3 seeds（§4.1），**不是单 seed**；但**全文没有任何 std / 标准差 / 置信区间 / error bar**（全文检索 `standard deviation`/`std`/`±`/`variance`/`confidence interval` 只命中通道归一化与 GRPO 的 population std）。10 次运行报点估计而不报离散度，无法判断差距是否显著。 |
| R3 | **在测试集上调超参 / 用测试集选 checkpoint** | ⚠️ **部分命中** | 值得肯定：DLM–MTP 实验**显式**在 validation prefix 上选 candidate length 与 denoising depth，然后固定用于 test，且"validation 与 test prefix 按图像分开"（§C.2 p.32、§C.3 p.33）；格式诊断窗口"在查看提议之前从公共参考序列化中选出"（§C.4 p.33）。可疑处：τ=0.8 与 B=32 的**运行点是在 COCO 扫描上选的**（§4.3.1、§D.1），而 COCO 同时是 benchmark 表 1 的一行；论文自己承认"The B = 32 operating point used for the headline comparison is not the maximum-quality choice for every decoding strategy"，且"the COCO sweep alone does not establish a universal optimum"（§D.1 p.34）。 |
| R4 | **算力不对称却宣称方法更优** | ⚠️ **部分命中** | 训练侧完全不对称：128 GPU × 3 阶段 + 256 GPU 转换 + 256 GPU RL + >700B token 预训练 + 专有数据引擎；对手是已发布 checkpoint 的推理评测（GPT-6 Astra 等为外部 API 结果），对手训练算力**未报告**。但这属于"资源不同"而非"同一实验内不对称"，论文也未据此宣称效率优势。**真正需要注意的**是 §C.1 自己承认的："These controls do not imply equal trainable parameter counts or training FLOPs"。 |
| R5 | **指标定义与前作不同却直接比数字** | ✅ **基本规避** | F1mIoU 明确"Following the grounding evaluation convention of Jiang et al. (2026)"（§E p.37），外部行大量直接取自该文（表 1/3/4 表注）；论文另外给出了 F1mIoU 的精确定义（阈值聚合 F1，非平均 IoU）。风险点是不同 parser/prompt 混用，已用 `*`/`†` 标注。 |
| R6 | **消融只做"全去掉"却宣称每个组件都有效** | ✅ **规避** | 消融粒度相当细：坐标表示（2 档）、视觉编码器（3 档）、Stage III 开关、块大小（4/8/16/32）、熵阈值（0.2–1.2 共 6 档）、refinement 深度（1/2/4/8）、candidate length（4/8/16/32）、线性 vs 二次、DLM vs causal DLM vs block-MTP vs causal MTP。§D.3 主动声明 Stage III 消融"changes a complete phase and does not isolate individual reward components"；§D.3.1 主动声明现有消融**不能**把优势归因于"没有 DeepStack"。 |
| R7 | **真实限制只写在附录** | ⚠️ **命中** | 论文**没有独立的 Limitations 章节**，只有 §4.5 Discussion 的两条 Takeaway（p.16，各 2–3 句）。真实限制几乎全在附录：VisDrone 极小目标 F1@IoU0.95 仅 3.42/2.24（§E.2 p.37）；RefSpatial Location/Placement 59.00/67.00 低于 Astra 的 86.00/85.86（§E.5 p.43）；OCR/layout 的扩散–AR 差距（§4.2 p.10 + 表 3）；**去掉 Stage III 使 DLM 掉 16.62 pp**（§D.3 p.35）；FP8 不保证精度（§D.2 p.35）；DLM 在 RefCOCO F1@IoU0.95 为 22.39，低于 LocateAnything Hybrid 的 27.61（§E.1 p.37）。 |
| R8 | **数据集划分不一致或泄漏** | ❌ **未报告 → 高风险** | 论文说训练数据"combines public datasets with annotations produced by our data engine"（§3.3 p.6），而评测的 COCO/LVIS/Dense200/RefCOCO 系列/HierText/DocLayNet 等本身也是公开数据集。**全文没有任何一句话说明评测集与训练数据的去污染/划分隔离**（检索 `split`/`contamination`/`decontaminat`/`leak` 无结果）。仅 DLM–MTP 的 prefix 实验声明了 validation/test 按图像分开（§C.2）。复现与对比时的**首要风险**。 |
| R9 | **其余观察** | — | ① 主表使用熵引导解码的**单一运行点**（τ=0.8, B=32）——论文承认该点在 COCO 上**不是**最优（B=16 时 61.58 > 60.72）；② **无 venue**，为 arXiv 预印本；③ 表 1 的 RefCOCO avg 对 DLM 只报 85.33，但 DLM 在 RefCOCOg val/test 上（91.61/91.10）明显高于 VLM（84.37/83.56），说明 avg 混合了 RefCOCO/RefCOCO+ 上 DLM 落后的部分 —— 论文未分解；④ 论文诚实声明用生成式 AI 辅助代码、润色、**部分图表生成**，以及数据标注流水线（AI Use Statement，p.17）。 |

## 9. 可复现性：缺哪些关键信息

**已给足**（超出同类论文的平均水平）
- 完整架构规格（表 8）与逐模块参数计数（表 9）
- 词表布局与每个特殊 token 的 ID（§A.2），且经本地 checkpoint 实测吻合
- 完整训练配方（表 11/12/13）：GPU 数、per-device/global batch、序列长度、峰值 LR、调度器、
  优化器超参、warmup、seed、权重衰减、梯度裁剪、并行策略
- 联合损失的精确定义（式 2/6/7），含 mask 均值口径、"含被 mask 的 EOS"、"无 inverse-noise 加权"
- 熵引导解码的精确算法（式 24 + 空集兜底 + 提交后不可变）
- 自推测两种 schedule 的 NFE 与 query-token 计数
- GRPO 的完整目标（式 3/8/9/10/11）、ε、rollout 温度、组大小、有无 KL
- 三个任务的奖励公式（式 12–14 框、式 15–21 + 表 14 OCR、式 22–23 pointing）
- 评测指标定义与报告约定（§4.1、§E）
- F1mIoU 的精确定义（阈值聚合 F1）

**缺失（复现的主要风险）**
1. **训练数据规模与构成**：只有">700B tokens 累计预训练 exposure"；各数据集/各任务组的数据量、
   数据引擎产出的标注量 —— **未报告**
2. **30 benchmark 的逐项聚合方式**：72.42 / 61.75 / 53.32 / 71.35 **没有表格**，无法从正文复算
3. **方差**：10 runs 的结构给了，但 std/CI **全部未报告**
4. **训练硬件型号与总时长**：只有 GPU 数量，无型号（表 11/12/13 未写）、无 GPU-hours、无墙钟时间
5. **训练/评测集去污染声明**：完全没有（红旗 R8）
6. **数据引擎实现**：教师模型、融合规则、迭代轮数 —— **未开源、未报告**
7. **MTP baseline 实现**：自研，**未开源**；明确不是 LocateAnything 复现
8. **二次自推测实现**：论文有完整算法描述，但**本地仓库未实现**（见 `claim-code-map.md`）
9. **baseline 的推理超参**：除 Astra 的 thinking effort 外，对手的 prompt/采样设置未逐项报告
10. **超参搜索预算**：τ、B、K、refinement 深度的扫描范围给了，但扫描所用的验证集规模未报告

## 10. 我的疑问（待验证点）

1. **RefCOCOg 上的反常优势**：DLM（91.61/91.10）反超同骨干 AR（84.37/83.56）约 7 pp，而其他任务上
   扩散普遍落后。"指代目标数量少、输出短"是否使并行解码的误差更小？还是 RefCOCOg 的 GT 与
   数据引擎标注的重叠方式不同？论文没有解释。
2. **72.42 与 73.03 的差值**：图 16 消融基线 73.03 vs 最终 72.42（−0.61 pp，AR）、61.66 vs 61.75
   （+0.09 pp，DLM）。§D.3 说两者聚合可能不同。若聚合方式确实不同，那 30-benchmark 均值到底怎么算？
3. **44 vs 45 baselines**：图 7 caption 与 §4.1 不一致，是否有一个 baseline 未计入某处？
4. **τ=0.8 的稳健性**：τ∈[0.6, 0.8] 区间质量平稳（59.82 → 60.72），τ=1.0 掉到 54.06，τ=1.2 回升到
   56.14。这种非单调性提示熵尺度与词表大小强相关；换 checkpoint 或换任务是否需要重标定 τ？
   （本地代码注释也承认 "Thresholds used for final profiles must be selected from GAM telemetry
   quantiles"，`infer/decode/task_profiles.py:45-46`）
5. **D+1 NFE 的记账**：块完成后有一次"不验证、不拒绝"的因果 pass。如果它预测的下一 anchor
   与双向 pass 的结果冲突，会怎样？§B.1 说 completed tokens 不可变，但 anchor 是下一块的首 token ——
   这个 token 的来源是否影响可复现性？
6. **FP8 精度**：论文明说"does not establish that FP8 preserves accuracy"，那么 3.51×（熵引导）
   与 4.06×（自推测）中 FP8 的贡献占比是多少？没有 FP8-off 的匹配质量对照。
7. **本地无 GPU 的含义**：本仓库只做代码级核对；所有"论文数字 vs 本地可跑性"的结论都基于
   "本机 Windows、无 CUDA、无 GPU"这一事实 —— 因此本笔记**不包含任何本地实测数值**。

## 11. 与我的课题的关系

| 维度 | 判断 |
|:---|:---|
| 可直接用 | ① 0–999 量化坐标 + 结构化 wrapper 的 I/O 协议（`grounding_anything/protocol.py` 与论文 §A.1 逐字节一致）；② 熵引导提交策略（`infer/decode/reliability.py`）；③ 三个任务的奖励公式（`train/rl/rewards/*`、`train/rl/shared/reward.py`）；④ F1mIoU 的阈值聚合定义；⑤ 30-benchmark 任务清单（`eval/README.md:89-120`） |
| 需改造 | ① "块内双向 + 块间因果"的 flex-attention mask 构造（绑定 Qwen3.5/`flex_attention`）；② 互补腐蚀的两个 noisy view 需要一次 clean forward 共享（`models/dlm/hybrid.py:641-645` 的注释说明这是为省 25% 语言 token 工作量的优化）；③ D+1 NFE 的因果 cache 重建路径 |
| 仅对照 | ① MTP 对照（自研未开源）；② 数据引擎（未开源）；③ 训练 stage 1–3 的数据与配方；④ 外部 baseline 行（无本地适配器） |
| 最大复现障碍 | 30-benchmark 聚合分不可复算 + 无方差 + 无去污染声明 + 训练数据未公开 |

---

### 附：本笔记的核查方法

- PDF：`pymupdf` 1.28.2 `page.get_text("text")`，61 页全文抽取到临时文件（未入库），逐页带 `<<<PAGE n>>>` 标记。
  表内数字按**表体文本流**顺序读取；纯图形元素（柱状图/散点）的数值来自文本层残余的数字标签，
  已在正文标注为 `[图内数值]`，**未做 OCR/像素级二次核对**。
- 仓库实测：`GroundAnything-Model/config.json`、`added_tokens.json` 用 Python `json` 直接解析核对；
  `configs/eval/tasks.json` 计数核对；代码引用均为 `read`/`grep` 工具的**原始行号**。
- 本机为 Windows、无 CUDA、无 GPU，**未执行任何模型前向、未运行训练或评测脚本**，因此本文不含任何本地实测指标。
- 未修改任何源码、README、config 或测试；未触碰 `docs/investigation/` 下队友文件。
