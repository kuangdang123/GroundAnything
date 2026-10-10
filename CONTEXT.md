# GroundAnything 上下文

> 本文件是给下一个会话的事实清单。所有带 ✅ 的条目在 2026-10-09 由命令实测核对过；带 ⬜ 的尚未核实。

## 这是什么

4B 参数视觉 grounding 基础模型（DLM / blockwise 并行去噪），论文《GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed》，arXiv 2609.39600，2026，一作 Qize Yu。

同一份 vision-language backbone 出两个 checkpoint：

| 检查点 | 解码方式 | 论文定位 |
| --- | --- | --- |
| `GroundingPI/GroundAnything` | 熵引导并行去噪 / 可选 self-speculative | 本仓库主角，论文报 61.75%（对比 LocateAnything 53.32%） |
| `GroundingPI/GroundAnything-VLM` | 自回归 | 论文报 30 benchmark 总平均 72.42% |

当前阶段：**已发布代码 + 权重，未在开发机上跑通**（缺 `weights/` 与 `data/`，见下）。

## 运行位置

- 本地：`D:\Desktop\GroundAnything`（Windows，**无 GPU**，Python 3.12.10，`C:\Users\Huya\AppData\Local\Programs\Python\Python312\python.exe`）
- GitHub：`https://github.com/kuangdang123/GroundAnything.git`（origin）。✅ origin/main = `5534c00`，**落后**本地 HEAD `c5f7840` 四个提交，即本地是最新的。
- 开发机：`ssh notebook`（`notebook-ssh.huya.info:60791`，ProxyJump `liangjiaqi@fort.huya.com:32200`，`~/.ssh/id_rsa`）
  - 项目根：`/workspace/project-nas-1000073/liangjiaqi/GroundAnything`（项目 NAS 1000073 挂载，实例内即 Overlay 的 `/workspace` 不是同一层：NAS 有 9.4 PB 空闲，Overlay 只剩 2.2 GB，**大文件必须写 NAS 路径**）
  - 兄弟项目：`/workspace/project-nas-1000073/liangjiaqi/GroundingPI`（9.2 GB）——**不是 GroundAnything 的另一个版本**，而是同一套脚手架下的另一个 4B 模型/论文（HF `GroundingPI/GroundingPI`，arXiv 2609.39601），带自己的 `GroundPI-Model`（9.2 GB / 2 分片）。`/workspace/nas-data/LocateAnything-evalbase` 同理是 LocateAnything 的**代码**副本（3.1 MB），不是评测数据。
  - 环境：`/opt/venv/bin/python` = Python 3.12.3，torch 2.7.1+cu126，flash_attn 2.8.3.post1；`HF_HOME=/models/huggingface`；pip 走 PyPI（4.4 s）或 aliyun 镜像（1.4 s）都通
  - ❌ **prompt 环境不可直接跑本项目**：`/opt/venv` 里 **transformers 4.57.1，且没有 sglang、没有 vllm**（全机 `find / -maxdepth 6` 也无目录），conda 不可用，**项目 venv 与 `weights/` 都不存在**。仓库依赖锁要求 train = transformers 5.7.0 / torch 2.10.0+cu129，serve = 5.5.4 / 2.9.1 —— **均不满足**，只有 eval profile 部分满足。所以第一步必然是 `python3 run.py setup serve`（会自建 `.venv-*`）。
  - 加速卡：8 × A800-SXM4-80GB，**共享且动态**。两次采样相差 15 分钟：10:0x 是 0 卡占 8 GB、3/4/5/6 各占 33.5 GB；10:2x 变成只有 0 卡 8065 MiB + 1 卡 54,587 MiB，**2–7 卡全部空闲**。占用者 `process_name=[Not Found]`（跨 PID namespace，属主不可见）。**放量前必须重跑 `nvidia-smi` 复核，不要相信任何历史快照。**
  - 磁盘：`/workspace/project-nas-1000073` 与 `/workspace/nas-data` 是**同一 10P 卷的两个导出**，可用 9.60 PiB；而 overlay `/` 与 `/tmp` 只剩 **2.16 GiB** —— 大文件必须落 NAS，临时文件也别写 `/tmp`（remote-scout 的校验输出就留在那儿了，见其报告 §10）

## 关键结论：代码不缺，缺的是权重与评测数据

✅ 2026-10-09 逐文件 sha256 比对：本地仓库（排除 `.git/` 与 `GroundAnything-Model/`）与 devbox 同路径仓库 **397 个文件全部逐字节一致，0 处内容差异**。

✅ devbox 上 `git status` 显示 341 个文件被修改，是**假象**：`git diff --numstat` 合计 64,748 增 / 64,748 删，且决定性判据 `git diff --ignore-cr-at-eol --stat | wc -l` = **0**（numstat 全是 57/57、486/486 式等量增删，`od -c` 见 `\r\n`，`core.autocrlf/eol` 未设置）。**因此不要为了"同步"去覆盖任何一侧。**

✅ 两端 HEAD 同为 `c5f7840a4d8a288f96e99654ccb1e836565a8e3b`；但**远端 main 比 GitHub `origin/main`（`5534c00`）领先 4 个未推送提交**——这 4 个提交是本地/远端独有、GitHub 上没有的，别误删。

✅ 本地 `GroundAnything-Model/` 与 HF `GroundingPI/GroundAnything` 仓库 **35/35 文件同名同大小，且 `checksums.sha256` 里 27 个小文件全部哈希通过**。

## ✅ 已修复：`git add -A` 撞 9.69 GB 权重的口子（2026-10-09）

**问题（实测）**：`git check-ignore -v GroundAnything-Model` 原本返回 NOT IGNORED（`.gitignore` 没有这条规则），`git status` 只把它显示为 `?? GroundAnything-Model/`。一次 `git add -A` 就会把 9.69 GB safetensors + 45 MB demo.mp4 + 20 MB 论文 PDF 写进 git 历史。

**修法（已执行，用户批准）**：往 `.gitignore:18-22` 追加

```gitignore
# Locally downloaded release artifacts: model checkpoints (multi-GB) and the
# paper PDF. Kept out of git; both are retrieved from their upstream sources.
/GroundAnything-Model/
*.pdf
```

**修复后实测**：

- `git check-ignore -v` → `.gitignore:20:/GroundAnything-Model/` 与 `.gitignore:21:*.pdf` 均命中
- `git status --short` → `GroundAnything-Model/` 与论文 PDF **不再出现**，只剩 ` M .gitignore` 与三个新文档目录
- `git ls-files --deleted` → 空；`git ls-files GroundAnything-Model` = 0、`git ls-files "*.pdf"` = 0，**没有误伤任何已跟踪文件**（仓库本来就没有跟踪的 pdf 或该目录文件）
- 这是本次会话唯一被修改的已跟踪文件

**同步提醒**：`weights/`、`vendor/`、`data/`、`outputs/`、`GroundAnything-Model/` 现在都已被忽略（`.gitignore:2,4,5,6,20`）——**这些目录 git 带不走，跨机同步必须单独搬**。

## 数据与权重（本地）

- 权重目录：`GroundAnything-Model/`（未跟踪，来自 NAS，**非** README 约定的 `weights/dlm_bundle`）
  - ✅ `model.safetensors` = 9,687,419,736 字节，sha256 = `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`，**与上游 checksums.sha256 期望值一致**
  - ✅ 它是**可直接上 SGLang 的 DLM bundle**，不需要 `prepare-model`：safetensors header 569 张量全部 `base_model.*` 前缀，含 `infer/serve_sglang.py:87` 要求的 `base_model.model.language_model.embed_tokens.weight`；`embed_tokens=[152671,2560]`，`|<MASK>|`=152670=vocab−1，满足 `serve_sglang.py:88` 入场断言；`chat_template.jinja` 就位
  - ⚠️ 只能当 **DLM** 用。当 VLM 用会被 `infer/serve_sglang_vlm.py:44-47` 拒绝：它要求 `model_type=="groundinganything_vlm"` 且 `architectures==["GroundAnythingVLMForConditionalGeneration"]`，而实际 `config.json` 是 `"groundinganything"`/`GroundAnythingForConditionalGeneration`；且该脚本 `:51` 还要 `model.safetensors.index.json`（本目录是单文件 safetensors，没有）
- ✅ `GroundAnything-Model/checksums.sha256` 与 HF 上游**逐字节一致**（可作为校验基准）
- ✅ `GroundAnything-Model/assets/` 9 个文件齐全（含 fig2-architecture.png、fig7-grounding-performance.png，这两个不在 checksums.sha256 里）
- ✅ 论文 PDF 就在仓库根：`GroundAnything Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed.pdf`（20,281,962 字节，与 devbox 同文件）

### 权重结构已逐张量验证（2026-10-09，只读 safetensors header）

方法：只读文件头 8 字节长度 + JSON header，**不解码张量数据、不占显存**。

| 核对项 | 结果 |
| --- | --- |
| 张量总数 / dtype | 569 个，**全部 BF16**，全部 `base_model.*` 前缀 |
| **精确参数总和** | **4,843,669,504** —— 与论文表 9 **完全相同，偏差 0.0%** |
| 拆分 | vision+projector 428,482,048；language+lm_head 4,415,187,456 |
| 论文表 8 十二项规格 | **12/12 通过**（ViT 27 层/1024/4096/patch14；LLM 36 层/2560/9728/32 Q heads/8 KV heads/head_dim 128；max_pos 262144；vocab 152671） |
| 视觉塔确实在权重里 | ✅ 165 个 `visual.*` 张量，27 个 block（`blocks.0`–`blocks.26`）+ projector `fc1 [4096,4096]`、`fc2 [2560,4096]`，与论文"2×2 聚合 + 两层 MLP 4096→4096→2560"一致 |
| embed_tokens vs lm_head | 形状都是 `[152671, 2560]`；**是否 weight tying 光看形状判定不了，需查 modeling 代码** |

⚠️ **陷阱**：`config.json` 的 `text_config.vocab_size` 报 **152670**，而 `embed_tokens` 实际是 **152671** 行。差 1 就是后加的 `|<MASK>|`（id 152670）。**读取时以张量为准**；`infer/serve_sglang.py:88` 的 `|<MASK>| == vocab-1` 断言正依赖这个关系。

边界：这是静态结构校验 —— **没做前向、没验证数值正确性、也没验证权重与论文性能数字的因果关系**。

**未下载（截至 2026-10-09 未执行，等确认）**：

| 缺什么 | 上游 | 体积 |
| --- | --- | --- |
| GroundAnything-VLM 自回归检查点 | `GroundingPI/GroundAnything-VLM`（model） | 9.09 GB（`model-00001/00002-of-00002.safetensors` + index） |
| 30-benchmark 评测数据 | `Skywalker0410/Grounding-EvalData`（**dataset**，需 `--repo-type dataset`） | 仓库 ~6.06 GB 文件（最大 coco.tar.gz 3.88 GB、doclaynet.tar.gz 2.25 GB），但**下载后约 18 GB、解包后再约 19 GB，共需预留 ~37 GB** |

⚠️ 评测数据这 37 GB **绝不能落在 Overlay（`/workspace/...` 之外的根盘只剩 2.16 GiB）**，必须写 NAS 路径如 `/workspace/project-nas-1000073/liangjiaqi/GroundAnything/data/eval`。

❌ **NAS 上没有既存副本可省这一步**：`Grounding-EvalData` / `ScreenSpot-Pro` / `OSWorld-G` / `RefSpatial-Bench` / `coco.tar.gz` / `doclaynet.tar.gz` / `FSCD_test.jsonl` / `_annotations` 等特征名在 `/workspace/project-nas-1000073/liangjiaqi` 与 `/workspace/nas-data` 下**零命中**。NAS 上现存的只有游戏视频 YOLO 数据集（`/workspace/nas-data/dataset/` 下 cs2/lol/pubg 等）与 `LocateAnything` 代码副本。同理 GroundingDINO / Rex-Omni 权重只在 `/workspace/nas-data/models/` 里有 Rex-Omni（142 GB 的模型根），**没有 GroundingDINO/RLv2**。

评测数据的正规接法（出处：该 dataset 自己的 README，已实测可达、带 `SHA256SUMS`）：

```bash
hf download Skywalker0410/Grounding-EvalData --repo-type dataset --local-dir ./Grounding-EvalData
python Grounding-EvalData/scripts/prepare_data.py \
  --data-root /abs/path/to/grounding-eval-data \
  --code-root /abs/path/to/GroundAnything     # 可重复多次
# 全量配方：--suite groundanything30，模板选 configs/eval/{dlm,dlm_speculative,vlm}.yaml
.venv-eval/bin/python Grounding-EvalData/scripts/create_eval_config.py \
  --code-root . --template configs/eval/dlm.yaml \
  --data-root /abs/path/to/grounding-eval-data \
  --suite groundanything30 --output configs/eval/full30.local.yaml
```

- `prepare_data.py` 会校验 `SHA256SUMS`、解全部 19 个归档、写 `configs/datasets.local.yaml`，且**保护既有源/config 文件、拒绝覆盖不同数据**
- ⚠️ PyArrow 必须 **21.0.0**；19.0.0 与部分 RefSpatial/RoboSpatial parquet 元数据不兼容（报 `Repetition level histogram size mismatch`）
- 该 dataset 覆盖 34 个条目，GroundAnything 用其中 **30 个**（`--suite groundanything30`）；Rex 口径与原始 benchmark 条数不同（如 COCO box 用 4,952 条、ICDAR2015 用 496 条）
- 实测 HTTP 可达：`coco.tar.gz` → 200，`content-length: 3880477371`（与 API 报的 3,880,477,371 一致）

## 入口与命令（出处：仓库 README 与 eval/README.md）

在开发机、仓库根目录执行：

| 想做什么 | 命令 | 出处 |
| --- | --- | --- |
| 下载 DLM 权重 | `hf download GroundingPI/GroundAnything --local-dir weights/dlm_bundle` | README:119 |
| 准备推理环境 | `python3 run.py setup serve` | README:120 |
| 起 DLM 服务（熵引导） | `python3 run.py serve --decoder denoise` | README:121 |
| 起 DLM 服务（self-speculative） | 停掉旧服务后 `python3 run.py serve --decoder speculative` | README:133 |
| 下载并起 VLM 服务 | `hf download GroundingPI/GroundAnything-VLM --local-dir weights/vlm` → `python3 run.py serve --config configs/release/vlm_sglang.yaml` | README:143-144 |
| vLLM 路线（仅 VLM） | `python3 run.py setup vllm --platform gpu` → `python3 run.py serve --engine vllm --platform gpu` | README:216-217 |
| 准备评测环境 | `python3 run.py setup eval` | README:360 |
| 评测 dry-run（查缺失输入） | `.venv-eval/bin/python scripts/evaluate.py <cfg>.yaml --dry-run` | eval/README.md:160 |
| 跑 30-benchmark | `python3 run.py eval --config configs/eval/full_suite_denoise.yaml` | eval/README.md:170 |
| 准备训练环境 | `python3 run.py setup train` | README:391 |

服务地址：DLM `http://127.0.0.1:8101/v1`（model id `groundinganything`），VLM `http://127.0.0.1:8102/v1`（`groundinganything-vlm`）。

## ✅ 开发机 serve 环境已部署完成（2026-10-09，位置于 2026-10-10 改为持久 NAS）

**结论：SGLang 可以在这台机器上部署，已跑通。**

### 最终位置（2026-10-10 定稿）

```
/workspace/cpfs-data/ga-serve-venv          <- 真实数据，NAS(NFS)，持久
        │ bind mount
        └──> <repo>/.venv-serve             <- 所有工具期望的路径
```

- 仓库内的 `.venv-serve` 是**真实目录 + bind mount**（不是软链）——因为 `run.py:44-48` 与
  `scripts/config_contract.py:19-21` 会 `resolve()` 后校验 `is_relative_to(root)`，
  **软链指向项目外会被拒**（`path escapes project through a symlink`）。
- 重启后恢复：`bash scripts/restore_serve_env_mount.sh`（幂等，只重挂不重装，已实测）
- `.gitignore:9` 的 `.venv*/` 已覆盖该路径，不会污染仓库。

### 环境矩阵（实测）

| 组件 | 版本 | 校验 |
| --- | --- | --- |
| torch | **2.9.1+cu128**，CUDA 12.8 | `torch.cuda.is_available()=True`，`device_count=8` |
| cuDNN | **9.16.0.29**（加载版本 91600） | `check_serve_environment.py` **通过**，`unexpected_errors: []` |
| transformers | **5.5.4** | 符合 `requirements/serve.txt` 锁定 |
| sglang | **0.5.6.post2**（= vendored 自定义引擎） | 路径实测指向 `vendor/sglang/sglang`，非 PyPI 版 |
| sgl-kernel | **0.3.20** | 与 serve 锁一致 |
| flashinfer-python / -cubin | **0.5.3** | 与 serve 锁一致 |
| triton | **3.5.1** | 与 serve 锁一致 |
| grounding-anything | 0.1.0 | 可 `from grounding_anything import GroundingAnything` |
| 包数 | **176** 个 dist-info | `pip check` 仅剩已记录的 cuDNN 元数据覆盖 |
| 文件数 | **75,219** | — |
| GPU | NVIDIA A800-SXM4-80GB × 8 | — |

`python3 run.py serve --decoder denoise --dry-run` 已能正确解析出
`scripts/run.py configs/release/dlm_sglang_denoise.yaml`。

⚠️ `torchrun --version` / `sglang --version` 会报错是**正常的**（这两个 CLI 不支持 `--version`）；
`torchrun` 无参、`sglang` 无参都能正常打印 usage，控制台脚本 shebang 均指向新路径。

### ⚠️ 部署时踩到的坑（按重要性）

**1. 同为 NAS，不同挂载点的元数据性能差 96 倍——选错要多花几十小时**

| 路径（都是 NAS） | 创建 100 个小文件 | 75,219 文件预计 |
| --- | --- | --- |
| `/workspace/project-nas-1000073/...`（项目卷，nfs4） | **15,427 ms**（6.5 files/s） | **约 3.2 小时**（实测更慢，见下） |
| `/workspace/cpfs-data/...`（另一个 NAS 导出） | **160 ms**（625 files/s） | — |
| `/dev/shm`（tmpfs 内存） | 4 ms | 秒级 |

实测 tar 解包到**项目卷**只有 **1.2 files/s**，75,219 个文件预估 **61 天**，只能放弃。
最终搬到 `/workspace/cpfs-data` 用 `cp -a`：**9 分 13 秒完成，约 136 files/s**。
顺序写两边都不慢（项目卷 80–449 MB/s），**瓶颈纯粹是元数据**。

**2. 工具链的两条约束互相打架**

- `scripts/setup_environment.py:30` 拒绝**任何已存在**的 venv 目录
- `mount --bind` 又要求挂载点**必须先存在**
- 绕过方式：先建挂载点 → `mount --bind` → 再按 `setup_environment.py` 的**同等命令序列**安装
  （见 `scripts/install_serve_env_tmpfs.sh`，仅跳过那句存在性检查）。容器内是 root 且有 `CAP_SYS_ADMIN`。

**3. pip 下大 wheel 会莫名卡住，curl 反而快 1000 倍**

- `pypi.org` 索引页 16.2 s vs aliyun 0.14 s；但 aliyun **没有 cuDNN 9.16**，且下大 wheel 只有 34 kB/s
- pip 直连 pypi 下 cuDNN 时卡在 **10 kB/s**；同一 URL 用 `curl` 跑到 **11 MB/s**（617.7 MB / 45 s）
- **大 wheel 别迷信 pip，必要时 curl 下来本地 `pip install <file>.whl`**

**4. cuDNN 9.16 覆盖不是可选项**

`scripts/check_serve_environment.py:35` 硬性要求 `torch.backends.cudnn.version() >= 91500`，
而 torch 2.9.1 自带 9.10.2.21。`pip check` 会一直报一条已知元数据冲突——
**这是预期的**，`check_serve_environment.py:14-16` 专门把它归类为 `known_metadata_override`。

### 复现 / 恢复命令

```bash
# 容器重启后恢复（只重挂，秒级）
bash scripts/restore_serve_env_mount.sh

# 从头重建（装到 /dev/shm，快但易失；装完再 cp -a 到 NAS 持久化）
bash scripts/install_serve_env_tmpfs.sh
```

### 尚未做的下一步

环境已就绪且**服务已跑通**（见下节）。起服务前必须重跑 `nvidia-smi` 复核（这台机共享）。

## 🎯 推理演示（Gradio）已上线（2026-10-10）

仓库原本**没有**交互式 demo（只有 `examples/predict.py` 命令行版），本会话补了一个：`demo/app.py`。

**架构是客户端/服务分离**——demo 不加载模型，只发 HTTP 请求到 SGLang 服务。
这与参考实现 `LocateAnything/game_finetune/app/gradio_app.py` 不同（那个用 transformers 进程内直载）。

```
demo/app.py (Gradio 6, :7860)  ──HTTP──>  SGLang 服务 (:8101, GPU 4)
     不占 GPU                              权重 9.69 GB，加载约 6 分钟
```

**七种模式**（几何类型 × 语义族群的正交组合，覆盖协议层全部能力）：
检测框 `dense_bbox` / 检测点 `dense_point` / 指代框 `refer_bbox` / 指代点 `refer_point` /
OCR / 版面 `layout` / GUI。
> 内置客户端 `GroundingAnything.predict()` 只覆盖 refer 两个族且每 phrase 只回一个框，
> 所以 demo 直接驱动 `grounding_anything/protocol.py`。

**核心设计：宁可报错，不画错。** 所有输出过 `validate_prompt_answer_pair(prompt, answer, family)`
（与评测同一门），不通过时把错误 + 模型原始输出一起显示，**绝不宽松解析后画框**。

### 实测记录（GPU 4，A800-80GB）

| 图 | 模式 | 结果 | 耗时 |
| --- | --- | --- | --- |
| `teaser.png` 3559×3246 | 检测框 person | ✅ **31 个目标**，人工像素抽样 5/5 命中 | 3.6 s |
| `teaser.png` | 指代点 the person | ✅ 30 个点 | 2.2 s |
| `demo-poster.jpg` 1920×1080 | 检测点 person | ✅ 9 个目标，5/5 命中 | 0.8 s |
| `demo-poster.jpg` | OCR | ✅ 检出 "Ground"/"Anything" | 0.7 s |
| `demo-poster.jpg` | 检测框 dog, cat | ❌ 模型用 `,`（无空格）分隔 entry | 0.7 s |
| `teaser.png` | 指代框 the person | ❌ 重复坐标元组（退化） | 3.3 s |

**发现三个模型行为问题**（demo 保留了原始输出可复现）：
① entry 分隔符有时是 `,` 而非协议要求的 `, `；② `refer_bbox` 会输出重复坐标元组；
③ `refer_point` 曾生成 3841 tokens 才停（点被反复输出），而同图 `detect_point` 只要 0.8 s。
**并行解码的速度优势很直观：3559×3246 大图出 31 个框只要 3.6 秒。**

### 启动与访问

```bash
# 终端 1：模型服务
CUDA_VISIBLE_DEVICES=4 .venv-serve/bin/python -u -m infer.serve_sglang \
  --model weights/dlm_bundle --decoder denoise --output outputs/sglang/denoise --port 8101
# 就绪标志："The server is fired up and ready to roll!"（首次约 6 分钟）

# 终端 2：界面
PYTHONPATH=$PWD/demo:$PWD /workspace/cpfs-data/ga-demo-venv/bin/python demo/app.py --port 7860
```

Windows 本地访问用隧道：`ssh -L 7860:127.0.0.1:7860 notebook` → `http://127.0.0.1:7860`。
demo venv 在 `/workspace/cpfs-data/ga-demo-venv`（gradio 6.30 + pillow + numpy，**不含 torch**，与模型栈隔离）。
Gradio 6 把 `theme`/`show_api` 从 `Blocks()` 移到 `launch()`，代码用 `try/except TypeError` 兼容 4/5/6。

## ⚠️ 本地 Windows 不是"GPU 不足"，是代码级阻断

- ✅ 三个 setup profile 在 Windows 被源码硬拒：`scripts/setup_environment.py:25-28` 对 `platform: Linux x86_64` 的 train/eval 直接 `parser.error`；`scripts/setup_vllm_environment.py:44-45` 强制 `sys.platform=="linux"`。serve 无 platform 字段，但依赖树面向 CUDA。
- ✅ 任何真实执行都被 `run.py:98` 的 `venv/bin/python`（POSIX 布局）挡住。
- ✅ 本地唯一能跑的真实入口是 `--dry-run`：`run.py:125` 只在非 dry-run 时才校验 venv，所以 `python run.py serve --decoder denoise --dry-run`、`python run.py eval --dry-run` 在本机可用（需 PyYAML）。
- ✅ 全局单点阻塞 `vendor/` 不存在（三个 tar.gz 都在 `third_party/`：transformers 7.9 MB / sglang 3.0 MB / lmms_eval 117 KB）。`models/dependency_contract.py:14` 会抛 `prepare dependency snapshot first`；`scripts/evaluate.py:176,262` 的 PYTHONPATH 又硬指 `vendor/lmms-eval`。恢复：`python scripts/prepare_dependencies.py --apply`（纯 stdlib+哈希，本机可跑）。
  - ⚠️ 依赖锁的 Transformers 是 **5.7.0**，本机装的是 **5.14.1**，对不上。

## ⚠️ 评测链路已知会掩盖问题的地方

- `scripts/evaluate.py:186-187` **静默跳过**坏掉的指标 JSON
- `scripts/evaluate.py:232` `code = code or (2 if missing_tasks else 0)` 让子进程非零码盖掉"任务缺失"信号
- `scripts/evaluate.py:90` 报错文案说 tasks 来自 `configs/eval/tasks.json`，但 `task_inventory()`（`:26-37`）实际 `rglob eval/**/*.yaml`——**`tasks.json` 全代码库从未被读取**（死文件）。三者口径不一：tasks.json 42 项 / README:343 说 30 个 benchmark / `eval/task_config.py` 39 项
- `configs/eval/full_suite_{denoise,speculative,vlm}.yaml` 被 `eval/README.md:170-176` 引用，但**仓库不发布**，只给生成 heredoc（`eval/README.md:82-144`），且 `open("x")` 拒绝覆盖

## ⚠️ 破坏性操作

- 最险：`eval/eval_runner.py:738-745` **默认 `rmtree(<log_root>/log/gpt_response)`**。YAML 前端恒传 `--skip_cleanup`（`scripts/evaluate.py:150`）所以走 YAML 路线安全；**手工直调 runner 会真删目录**。
- `train/tokenizer/extend_tokens.py:276-288` 原地 rename 模型目录（默认留备份，加 `--delete-backup-after-validation`（:299）才删）
- 代码/config 里几乎没有硬编码绝对路径（只有 `docs/VLLM.md:129` 的容器挂载示例是合理的）

## ⚠️ 文档自相矛盾

- `README.md:124` 说已发布 DLM bundle **不需要** `prepare-model`；`README.zh-CN.md:52` 却要求先跑 `prepare-model`。以英文版 + 权重实测（bundle 已含 `base_model.*` 前缀与 chat template）为准：**不需要**。

## 数据与配置口径

- 配置入口：`configs/eval/*.yaml`（模板 `limit: 8` 是冒烟用，全量须 `limit: null`）、`configs/datasets.yaml`（数据集相对路径，相对 `data_root`）、`configs/eval/tasks.json`（任务注册表，42 个 recipe 里 **30 个**属于论文主表，12 个是 `*_Labelless`/`*_Boxonly`/humanref/visual 变体）
- `data_root: data/eval`，产物落 `outputs/eval/<run_id>/`，含 `run.json`、`responses.jsonl`、`summary.json`、`log_eval/`、`cache/`
- 评测脚本只给**逐任务**结果，**不**算论文级总平均，也不做论文的 10 次重复统计
- ✅ devbox 上 `weights/`、`data/`、`outputs/` **都不存在**，仓库尚未进入可跑状态

## 术语表

| 词 | 在本项目里的确切含义 |
| --- | --- |
| DLM | 双向 blockwise 去噪的 GroundAnything 主检查点（区别于自回归 `-VLM`） |
| GAM | 评测协议/模式名，用原生 spatial token + 固定 0–999 网格；三个已发布检查点都用 `mode: GAM` |
| entropy-guided decoding | 主表口径的解码器，`--decoder denoise`；评测配置里对应 `decoder: denoise` |
| self-speculative | 可选加速解码，`--decoder speculative`，论文报 4.51× 加速、COCO F1mIoU 降 0.74 pp |
| F1mIoU | 检测类任务的指标（F1 @ mIoU） |

## 未决问题

- [x] ~~是否给 `.gitignore` 追加 `/GroundAnything-Model/`~~ → **已执行并实测生效**，见上方"已修复"节
- [ ] 是否下载 VLM 检查点（9.09 GB）与 Grounding-EvalData（~37 GB）到 devbox —— 用户明确选择"**先不推进，保持当前只读状态**"，未下载
- [ ] 是否在 devbox 跑 `run.py setup serve` 建环境并冒烟 —— **用户已两次明确指令"什么都不做，等发话"，未执行**。恢复工作时从这里开始（见下方"下次开工的第一条命令"）
- [ ] devbox 上 `weights/dlm_bundle` 是否要重下，还是直接复用本地已校验的 `GroundAnything-Model/`（本地副本 = 同一个 HF revision，可用 scp/rsync 省 9.09 GB 流量）
- [x] ~~是否有 `Grounding-EvalData` 已在 NAS 其他路径就位~~ → **已核查：零命中，没有既存副本**
- [ ] 项目专用 venv 位置，以及 `run.py setup serve|eval|train` 各自会装什么、装到哪（已知目前**完全没有**项目 venv）
- [ ] `GroundAnything-Model/` 是否应该重命名/软链成 README 约定的 `weights/dlm_bundle`
- [x] ~~三份大文件未读完（`models/dlm/hybrid.py`、`train/dlm/train_dlm.py`、`infer/dlm/hierarchy.py`、`eval/metrics/detection_metrics.py`）~~ → **仍未读完**（保持一致：这几份仍是未读状态，见 `docs/investigation/repo-survey.md` 第 4 节）
- [ ] `grounding_anything/protocol.py` 的 `parse_answer(require_canonical=True)` 语义未确认；`train/rl/distributed/multiroute_56/64.py` 无 argparse，真实启动方式未确认
- [x] ~~本机 unittest/pytest 一个都没跑过~~ → **云端已跑**：`PYTHONPATH=$R python3 -m pytest tests/ -q` → 73 passed / 1 failed（既有失败，见"云端测试结果"节）；本机仍未跑
- [ ] **待你定**：`tests/test_gam_cache_builder.py::test_read_rows_is_strict_about_invalid_json` 这个既有失败怎么修（报错带上出错行内容 / 放宽断言）
- [ ] **待你定**：`CONTEXT.md` 与 `docs/` 三份调研文档目前是未跟踪状态，要不要提交进仓库

## 下次开工的第一条命令

用户暂停了"实际跑起来"这一步（2026-10-09，两次明确指令）。恢复时按顺序做，**每步都先 `nvidia-smi` 复核**（这台机器共享且动态，15 分钟内卡的状态就变过）：

```bash
# 0. 复核卡与空间（不要相信任何历史快照）
ssh notebook "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; df -h /workspace/project-nas-1000073 | tail -1"

# 1. 权重落地：不要再下 9.09 GB，用本地已校验的副本
#    本地 D:\Desktop\GroundAnything\GroundAnything-Model\ 与 HF GroundingPI/GroundAnything 同一 revision
rsync -av --info=progress2 /path/to/local/GroundAnything-Model/ \
  notebook:/workspace/project-nas-1000073/liangjiaqi/GroundAnything/weights/dlm_bundle/
#    落盘后核对 sha256 == 3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2

# 2. 建推理环境（当前 devbox 无 sglang/vllm，transformers 4.57.1 不符）
cd /workspace/project-nas-1000073/liangjiaqi/GroundAnything && python3 run.py setup serve

# 3. 先在 CPU 侧验证链路，再起服务
python3 run.py serve --decoder denoise --dry-run
python3 run.py serve --decoder denoise          # 起服务，日志落 weights 同级 run 目录

# 4. 冒烟：一条真实推理（此时还不需要评测数据）
python3 examples/predict.py --image <一张图> --prompt "the red car"

# 5. 只有在拿到 Grounding-EvalData（~37 GB，NAS 无既存副本）之后才谈评测
#    hf download Skywalker0410/Grounding-EvalData --repo-type dataset --local-dir <NAS 路径>
```

**先读 `docs/investigation/remote-devbox.md` 的 §9 与 `docs/papers/claim-code-map.md` 的 §2**：前者列了 10 项未确认，后者列了 12 条"仓库未实现/未开源"（MTP 对照不存在、30-benchmark 聚合分无实现等），避免重复踩坑。

## 仓库同步与干净化（2026-10-09 完成）

### 结论：本地、云端、你的 fork 三端现已完全一致

| 端 | HEAD | 状态 |
| --- | --- | --- |
| 本地 `D:\Desktop\GroundAnything` | `793ee9a` | 无已跟踪改动 |
| 云端 `/workspace/project-nas-1000073/liangjiaqi/GroundAnything` | `793ee9a` | 无已跟踪改动，0 个"假修改" |
| GitHub `kuangdang123/GroundAnything`（origin） | `793ee9a` | 已推送 |

三端 `git rev-list --left-right --count origin/main...HEAD` 均为 `0 0`。

### 两个 remote 的关系（回答"是不是 clone 再上传"）

`kuangdang123/GroundAnything` 与 `groundingpi/GroundAnything` 返回**同一个 commit**；GroundingPI 也是（`0047d451`）。即 `kuangdang123` 就是官方上游（fork 或 GitHub 重定向），**不是两个不同的源**。

但本仓库**不是干净的上游克隆**——带着 `guangdang` 的 4 个提交（2026-10-05）：`c5f7840` Game UI 评测、`7ff9b49` VLM LoRA 微调、`7712493` Arrow cache 导出器、`4c92fcd` LocateAnything→GAM 转换器。

这 4 个提交与上游新增的 2 个提交都从 `5534c00` 分叉，**内容互补、零冲突**：上游有 `scripts/prepare_eval_data.py` 等评测数据工具，但**没有**上面那 4 个文件。

### 已执行的合并

`1185fb6 Merge upstream main (8a3b43a) into fork main` —— **自动合并成功，零冲突**，两边工作全保留。合并前的安全网：

- 本地 tag `backup/fork-c5f7840-20261009` → `c5f7840`（**已推到 origin**）
- 云端 bundle：`/workspace/project-nas-1000073/liangjiaqi/GroundAnything-git-backup-20261009/fork-c5f7840.bundle`（19.9 MB）

⚠️ 合并前 origin 上**只有** `main`，那 4 个提交此前只存在于本地与云端，**GitHub 上没有任何备份**。

### 行尾问题的真正根因（前两轮的判断都不准，此处更正）

- 本地 Windows `core.autocrlf` = **true**（来自 `C:/Program Files/Git/etc/gitconfig`）
- devbox `core.autocrlf` **三级配置全为空**（`git config --show-origin` 确认）

同一份提交内容因此在云端表现为 **341 个"已修改"文件**。**可靠判据**：逐文件取 `HEAD` blob 与 worktree 内容、剥掉 `\r` 后比对 → **358 个文本文件全一致**；7 个二进制文件用 `git hash-object` 比对 → **7/7 一致**。**内容从未损坏，纯粹配置不对称。**

⚠️ `git diff --ignore-cr-at-eol` 在这套 git 2.43 上**对这批文件不生效**（仍报 341），**不要再拿它当判据**。

**结构性修复**：新增 `.gitattributes`（`* text=auto eol=lf`，显式把 png/jpg/mp4/pdf/gz/safetensors 标为 `binary`，py/md/yaml/json 标为 `text eol=lf`），提交 `793ee9a`；devbox 同步设 `core.autocrlf false` + `core.eol lf` 并规范化工作区 → 341 → **0**。

### 顺带清理

删掉本地 `.git/` 下一个 0 字节**陈旧 `index.lock`**（10:09:52 生成，是我第 1 轮 kill 掉的 dry-run 作业留下的，当时阻塞了所有 git 写操作；已确认无 git 进程、index 本身 38031 字节完好）。

### 云端测试结果（合并后验证）

devbox 装 pytest 9.1.1 后 `PYTHONPATH=$R python3 -m pytest tests/ -q`：

**1 failed, 73 passed, 39 subtests passed**

- 唯一失败 `tests/test_gam_cache_builder.py::test_read_rows_is_strict_about_invalid_json` 是**既有问题，与合并无关**：该文件 blob 与 fork 原版**逐字节相同**（`5f77b07b…`），且在合并前 tip `c5f7840` 的临时 worktree 里**同样失败**
  - 原因：测试要求错误信息含原始行内容（`match="not-json"`），而 `tools/build_gam_cache.py:95` 只抛 `f"{input_path}:{line_number}: {error}"`
  - **未修** —— 属你 fork 的代码/测试口径不一致，两种修法（报错带上出错行内容 / 放宽断言）需你定
- ⚠️ 本仓库**没有 CI，requirements 里也没有 pytest 声明**，`tests/` 无 `__init__.py`，所以 `python3 -m unittest tests/x.py` 会报 `ModuleNotFoundError`（调用方式问题，非测试失败）

## 变更记录

| 日期 | 做了什么 | 产物/影响 |
| --- | --- | --- |
| 2026-10-09 | 只读侦察（Lead + 3 个队友并行）：核实本地与 devbox 仓库逐文件一致、DLM 权重下载完成并通过上游 sha256（两端独立实测 27/27 OK）、确认 devbox 缺 venv/weights/data 且无 sglang/vllm、确认 NAS 无评测数据既存副本、确认 devbox git diff 为 CRLF 假象、精读论文 61 页并做 claim→代码映射 | `CONTEXT.md`（本文件）；`docs/investigation/repo-survey.md`（881 行）、`docs/investigation/remote-devbox.md`（778 行）、`docs/papers/2026-yu-groundanything.md`（536 行）、`docs/papers/claim-code-map.md` |
| 2026-10-09 | arXiv 交叉核对：`arxiv.org/abs/2609.39600` 摘要、作者列表（22 人，Qize Yu 一作）、页数（61 pages）与本地 PDF **完全一致**，关键数字 72.42 / 71.35 / 61.75 / 53.32 / 4.51× / 0.74 pp 逐项吻合。arXiv 为 v1，30 Sep 2026，18,482 KB | 无新增文件；精读笔记的来源可靠性得到外部确认 |
| 2026-10-09 | **权重结构逐张量验证**（只读 safetensors header）：569 张量全 BF16、参数总和 **4,843,669,504 与论文表 9 完全一致（偏差 0.0%）**、论文表 8 的 12 项规格 12/12 通过、视觉塔 27 block + projector 形状吻合。同时发现 `config.json` 的 `vocab_size`(152670) 与 `embed_tokens`(152671) 差 1 的陷阱 | `CONTEXT.md` 新增"权重结构已逐张量验证"节；`docs/papers/claim-code-map.md` §3 第 485 行由"⚠️ 可近似"升级为"✅ 已 bit-exact 验证"，并在 §6 追加权重级验证表 |
| 2026-10-09 | **仓库同步与干净化**：确认 `kuangdang123` 即官方上游（与 `groundingpi` 同 commit）；发现本仓库带 4 个 fork 提交、上游已前进 2 个提交、自 `5534c00` 分叉；打 tag `backup/fork-c5f7840-20261009` + 云端 git bundle 备份；**合并上游零冲突**（`1185fb6`）；新增 `.gitattributes` 统一 LF（`793ee9a`）；修正云端 `core.autocrlf` 并规范化工作区使 341 个假修改归零；删除陈旧 `index.lock`；三端推平至 `793ee9a` | `.gitattributes`（新）；`.gitignore`（含上游条目）；commit `1185fb6`、`793ee9a`；origin 新增 tag `backup/fork-c5f7840-20261009`；云端 bundle `GroundAnything-git-backup-20261009/fork-c5f7840.bundle`；测试 73 passed / 1 既有失败 |
| 2026-10-10 | **serve 环境改为持久 NAS 存储**：实测同为 NAS 的不同挂载点元数据性能差 96 倍（项目卷 6.5 files/s，cpfs-data 625 files/s），据此把 11 GB / 75,219 文件的 venv 从 `/dev/shm` 搬到 `/workspace/cpfs-data/ga-serve-venv`（`cp -a` 9 分 13 秒），bind mount 回项目内 `.venv-serve`；新增 `scripts/restore_serve_env_mount.sh`（幂等，容器重启后只重挂不重装，已实测卸载→重挂→校验全通过）；释放 /dev/shm 11 GB | `scripts/restore_serve_env_mount.sh`（新）；`.venv-serve` 现为 NAS 路径的 bind mount；`CONTEXT.md` 部署节重写 |

### 队友侦察报告索引

| 文件 | 内容 |
| --- | --- |
| `docs/investigation/repo-survey.md` | 本地仓库 A→H 全量侦察：40 条命令速查表、缺失项 M-01~M-04、风险 R-01~R-10、12 条未确认项 |
| `docs/investigation/remote-devbox.md` | 云端 10 节侦察：权重校验原始输出、git 假象判据、环境矩阵、GPU 动态采样、磁盘、未确认清单 §9 |
| `docs/papers/2026-yu-groundanything.md` | 单篇精读笔记：3 条核心 claim + 证据位置、全部关键数字（注表号/图号）、红旗清单、术语表 |
| `docs/papers/claim-code-map.md` | claim→代码映射；**12 条"仓库未实现/未开源"（N1–N12）**，含 MTP 对照不存在、二次自推测未实现、数据引擎未开源、30-benchmark 聚合分无实现 |

### 复现前必须先处理的两处论文↔代码差异（详见 claim-code-map.md）

- **D1（高）**：论文 §B.1 式 24 与模型卡都写熵在"可生成词表、**排除 mask token**"上算；代码 `infer/decode/reliability.py:37-46` 对**全词表** logits 算熵，调用方 `infer/decode/sglang_algorithm.py:315-337` 也无裁剪 → 同一个 τ 语义不同。
- **D3（中）**：发布 profile 对 medium/dense 用 T=0.1、ocr_medium 用 T=0.3（`infer/decode/configs/task_profiles.json` → tiers），论文正文**未报告任何温度设置**。
