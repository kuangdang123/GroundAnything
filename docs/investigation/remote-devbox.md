# 远端 devbox + NAS 只读侦察报告（GroundAnything）

- 侦察员：teammate `remote-scout`（task-1）
- 侦察时间：2026-10-09 10:03–10:22 CST（远端 UTC 02:03–02:22）
- 目标主机：`notebook-4f33l8-13456281-84959b4df8-kbv7x.hy`，用户 `root`，SSH 别名 `notebook`
- 项目根：`/workspace/project-nas-1000073/liangjiaqi/GroundAnything`
- **纪律**：全程只读。未执行任何 kill / rsync / scp / git 写操作；未在远端仓库或 NAS 创建、删除、改名任何文件。唯二的写操作是在 **overlay `/tmp`**（非 NAS）落两个校验输出文件，见文末「临时文件说明」。
- 证据口径：每条结论都给出真实命令与原始输出片段；凡未实测的一律标 **未确认**。

---

## 0. 摘要（先看这里）

| # | 结论 | 置信度 |
|---|---|---|
| 1 | 远端 HEAD = 本地 HEAD = `c5f7840a4d8a288f96e99654ccb1e836565a8e3b`，**远端不比本地领先** | 实测 |
| 2 | 远端 341 个"已修改"文件是 **CRLF↔LF 行尾假象**，内容零差异（`git diff --ignore-cr-at-eol --stat` 输出 0 行） | 实测 |
| 3 | 远端 main 比 `origin/main` **领先 4 个提交**（未推送） | 实测 |
| 4 | `GroundAnything-Model/model.safetensors` sha256 **通过**（与 `checksums.sha256` 期望值逐字符一致） | 实测 |
| 5 | 同目录 `sha256sum -c checksums.sha256` **27/27 全部 OK** | 实测 |
| 6 | `assets/` 下 `.gitattributes` 声明的 **9 个 LFS 文件全部存在** | 实测 |
| 7 | 开发机 **8× A800-SXM4-80GB**，但**与他人共享、占用动态变化**，无独占保证 | 实测 |
| 8 | `/opt/venv` **缺 sglang 与 vllm**，且 torch/transformers/datasets 版本与仓库锁定值不符 | 实测 |
| 9 | **NAS 上没有 GroundAnything 评测数据集**（ScreenSpot-Pro / OSWorld-G / RefSpatial-Bench / `_annotations` / `coco.tar.gz` 等特征名均无命中） | 实测（受搜索深度限制，见 §5.4） |
| 10 | `GroundingPI` 是**同一套代码骨架的另一个模型/论文**，不是 GroundAnything 的另一版本 | 实测 |

**一句话**：权重与代码都已就位且校验通过；缺的是 **评测数据集** 和 **可用的训练/推理环境（sglang/vllm + 版本匹配的 torch/transformers）**。

---

## 1. 仓库清单与 git 状态

### 1.1 命令

```bash
ssh notebook "cd /workspace/project-nas-1000073/liangjiaqi/GroundAnything && pwd && ls -la && du -sh -- */ | sort -h"
```

### 1.2 顶层清单（原始输出节选）

```
total 19976
drwxr-xr-x  6 root root     4096 Oct  5 13:40 configs
drwxr-xr-x  3 root root     4096 Oct  5 13:41 docs
drwxr-xr-x  2 root root     4096 Oct  5 13:41 environments
-rw-r--r--  1 root root      240 Oct  4 23:16 environment.yml
drwxr-xr-x 13 root root     4096 Oct  5 13:41 eval
drwxr-xr-x  3 root root     4096 Oct  5 13:41 examples
drwxr-xr-x  7 root root     4096 Oct  9 10:02 .git
-rw-r--r--  1 root root      810 Oct  4 23:16 .gitignore
drwxr-xr-x  3 root root     4096 Oct  5 14:06 GroundAnything-Model
-rw-r--r--  1 root root 20281962 Oct  4 23:10 GroundAnything Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed.pdf
drwxr-xr-x  3 root root     4096 Oct  5 13:41 grounding_anything
drwxr-xr-x  7 root root     4096 Oct  5 13:41 infer
-rw-r--r--  1 root root    11558 Oct  4 23:16 LICENSE
drwxr-xr-x  2 root root     4096 Oct  5 13:41 licenses
drwxr-xr-x  5 root root     4096 Oct  5 13:41 models
drwxr-xr-x  2 root root     4096 Oct  5 13:42 __pycache__
-rw-r--r--  1 root root      498 Oct  4 23:16 pyproject.toml
-rw-r--r--  1 root root    30232 Oct  4 23:16 README.md
-rw-r--r--  1 root root      6289 Oct  4 23:16 README.zh-CN.md
drwxr-xr-x  2 root root     4096 Oct  5 13:41 requirements
-rw-r--r--  1 root root      257 Oct  4 23:16 requirements.txt
-rw-r--r--  1 root root      8693 Oct  4 23:16 run.py
drwxr-xr-x  3 root root     4096 Oct  5 13:41 scripts
drwxr-xr-x  3 root root     4096 Oct  5 13:41 tests
drwxr-xr-x  3 root root     4096 Oct  5 13:41 third_party
-rw-r--r--  1 root root      5387 Oct  4 23:16 THIRD_PARTY_NOTICES.md
drwxr-xr-x  3 root root     4096 Oct  5 13:41 tools
drwxr-xr-x 10 root root     4096 Oct  5 13:42 train
```

目录体积（`du -sh -- */ | sort -h`）：

```
20K	requirements/      44K examples/     64K tools/      108K tests/
112K	grounding_anything/  124K scripts/   172K configs/   412K models/
640K	infer/             848K eval/        1.2M train/    7.2M docs/
12M	third_party/       9.1G GroundAnything-Model/
```

### 1.3 git 跟踪 vs 未跟踪

```bash
git ls-files | wc -l                                  # → 365
git ls-files | cut -d/ -f1 | sort -u
git status --porcelain --untracked-files=all | grep '^??'
```

- **跟踪文件总数 365**，跟踪的顶层条目（节选）：`configs docs environments environment.yml eval examples .gitignore grounding_anything infer LICENSE licenses models pyproject.toml README.md README.zh-CN.md requirements requirements.txt run.py scripts tests third_party THIRD_PARTY_NOTICES.md tools train`
- **未跟踪（`??`）只有 2 项，展开后 36 个文件**：
  - `GroundAnything Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed.pdf`（20,281,962 B）
  - `GroundAnything-Model/` 整棵树（含 `model.safetensors`、`assets/*`、tokenizer、modeling 代码、`checksums.sha256`）
- 顶层另有 `__pycache__/`（被 .gitignore 覆盖，未出现在 status 中）
- **`git ls-files GroundAnything-Model` 输出为空** → 权重目录完全未被父仓库跟踪，是一份独立的 HF 仓库下载（自带 `.gitattributes`）

与本地 `D:\Desktop\GroundAnything` 对比（本地执行 `git --no-optional-locks status --porcelain`）：

```
?? "GroundAnything Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed.pdf"
?? GroundAnything-Model/
?? docs/investigation/
```

本地只有 3 项未跟踪、**零修改**；远端有 341 项"修改"。差异见 §2。

---

## 2. git 状态详情：远端是否领先本地？

### 2.1 命令与输出

```bash
git rev-parse HEAD
git rev-parse --abbrev-ref HEAD
git log --oneline -12
git branch -a ; git remote -v
git status --short | wc -l
git diff --stat | tail -3
```

```
=== REV ===
c5f7840a4d8a288f96e99654ccb1e836565a8e3b
main
=== LOG ===
c5f7840 feat(eval): add shared game UI grounding evaluation
7ff9b49 feat(train): add GroundAnything VLM LoRA fine-tuning recipe
7712493 feat(data): add GroundAnything Arrow cache exporter
4c92fcd feat(data): add LocateAnything to GAM training converter
5534c00 Merge pull request #7 from Skywalker-yqz/remove-entropy-header-badge
753cd21 Remove redundant README header badge
1b3d1c2 Merge pull request #6 from Skywalker-yqz/readme-title-artwork
58d6cfa Balance the feature badge rows
6ad0a47 Use supplied title artwork in README headers
2f72d34 Merge pull request #5 from Skywalker-yqz/news-webpage-release
92d1cc5 Announce project webpage release on October 4
267bdf5 Merge pull request #4 from Skywalker-yqz/paper-wording
=== BRANCHES/REMOTES ===
* main
  remotes/origin/HEAD -> origin/main
  remotes/origin/main
origin	https://github.com/kuangdang123/GroundAnything.git (fetch)
origin	https://github.com/kuangdang123/GroundAnything.git (push)
=== STATUS COUNT ===
343
=== DIFF STAT TAIL ===
 train/tokenizer/verify_weights.py                  |   352 +-
 341 files changed, 64748 insertions(+), 64748 deletions(-)
```

```bash
git rev-parse origin/main
git status --porcelain | cut -c1-2 | sort | uniq -c
```

```
=== origin/main ===
5534c0043564632120c663e4b64b7c700e624e6e
=== porcelain kind counts ===
      2 ??
    341  M
```

**→ 远端 main 比 `origin/main` 领先 4 个提交**（`4c92fcd → 7712493 → 7ff9b49 → c5f7840`），这 4 个提交**未推送**到 GitHub。

### 2.2 关键判据：341 个修改全是 CRLF 假象

```bash
git diff --numstat -- .gitignore README.md configs/eval/gam.yaml run.py
git diff --ignore-cr-at-eol --stat | wc -l
head -c 80 .gitignore | od -c
git config --get core.autocrlf ; git config --get core.eol
```

```
=== NUMSTAT SAMPLE ===
57	57	.gitignore
486	486	README.md
15	15	configs/eval/gam.yaml
145	145	run.py
=== IGNORE-CR-AT-EOL DIFFSTAT (lines) ===
0
=== od of .gitignore first bytes ===
0000060   p   u   t   s  \r  \n   /   v   e   n   d   o   r  /  \r  \n
=== git config autocrlf/eol ===
(end autocrlf)
(end eol)
```

判读：

1. 每个文件的增/删行数**完全相等**（57/57、486/486、15/15、145/145），且合计 `64748 增 / 64748 删` —— 典型的整文件重写特征。
2. **`git diff --ignore-cr-at-eol --stat | wc -l` = 0** —— 忽略行尾差异后，**没有任何 diff**。这是决定性证据。
3. `od -c` 直接看到工作区文件是 `\r\n` 行尾。
4. `core.autocrlf` 与 `core.eol` **均未设置**（空）。

**→ 结论：这 341 个文件的"修改"100% 是 CRLF↔LF 行尾差异，文件内容与 `c5f7840` 逐字节等价（忽略行尾）。远端是"行尾脏工作区"，不是"代码领先"。**

### 2.3 远端是否比本地领先？

| 维度 | 本地 `D:\Desktop\GroundAnything` | 远端 devbox | 结论 |
|---|---|---|---|
| HEAD | `c5f7840a4d8a288f96e99654ccb1e836565a8e3b` | `c5f7840a4d8a288f96e99654ccb1e836565a8e3b` | **完全一致** |
| 工作区修改 | 0 | 341（仅 CRLF） | 无实质差异 |
| 未跟踪 | PDF、`GroundAnything-Model/`、`docs/investigation/` | PDF、`GroundAnything-Model/` | 等价 |
| 相对 origin | 领先 4（**未确认本地是否同样领先**，未查本地 `origin/main`） | 领先 4 | — |

**结论：远端不比本地领先。两端在同一提交 `c5f7840`，且远端不存在本地没有的提交或未跟踪代码。**

> 与 Lead 转述的"397 个文件逐字节一致"对照：本报告实测的跟踪文件数是 **365**，标记修改的是 **341**（另 2 项未跟踪）。口径不同（可能含未跟踪文件或计数方式不同），但**"内容一致"这一结论两方一致**，且本报告给出了可复现的判据命令。

---

## 3. 权重完整性校验

### 3.1 单文件 sha256（**通过**）

```bash
ssh notebook "cd /workspace/project-nas-1000073/liangjiaqi/GroundAnything/GroundAnything-Model && \
  ls -l model.safetensors checksums.sha256 && grep -n 'model.safetensors' checksums.sha256 && \
  setsid nohup sha256sum model.safetensors > /tmp/ga_sha256_model.txt 2>&1 < /dev/null &"
```

```
=== EXPECTED LINE ===
12:3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2  model.safetensors
-rw-r--r-- 1 root root       2361 Oct  4 23:25 checksums.sha256
-rw-r--r-- 1 root root 9687419736 Oct  5 00:09 model.safetensors
```

`sha256sum` 起于 **UTC 02:03:29**，回读时 **UTC 02:04:09** 已完成：

```
3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2  model.safetensors
```

| 项 | 值 |
|---|---|
| 文件大小 | 9,687,419,736 字节（与预期一致） |
| 实测 sha256 | `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2` |
| 期望 sha256 | `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2` |
| **判定** | ✅ **通过（逐字符一致）** |
| 耗时 | 约 40 秒（NAS 缓存热） |

### 3.2 全目录逐项校验（**27/27 全部 OK**）

```bash
cd .../GroundAnything-Model && setsid nohup sha256sum -c checksums.sha256 > /tmp/ga_checksums_verify.txt 2>&1 &
wc -l checksums.sha256    # → 27
```

UTC 02:04:42 启动，02:12:58 回读输出：

```
LICENSE: OK
README.md: OK
added_tokens.json: OK
chat_template.jinja: OK
config.json: OK
configuration_groundinganything.py: OK
configuration_groundinganything_vision.py: OK
generation_config.json: OK
image_processing_groundinganything.py: OK
media_utils.py: OK
merges.txt: OK
model.safetensors: OK
modeling_groundinganything.py: OK
modeling_groundinganything_vision.py: OK
preprocessor_config.json: OK
processing_groundinganything.py: OK
requirements.txt: OK
special_tokens_map.json: OK
streammind_gate.py: OK
tokenizer.json: OK
tokenizer_config.json: OK
vocab.json: OK
LICENSE-Apache-2.0: OK
LICENSE-Kimi-K3: OK
assets/demo.mp4: OK
assets/demo-poster.jpg: OK
assets/fig1-teaser.png: OK
```

- `checksums.sha256` 共 **27 行**，输出恰好 **27 行 `OK`，零 `FAILED`**。
- 校验入口就是官方 `checksums.sha256`（`sha256sum -c`），无需额外脚本。
- 已核对的 3 个资产（`demo.mp4`、`demo-poster.jpg`、`fig1-teaser.png`）均在清单内且 OK。

> 与 Lead 本地实测互为独立双端验证：**两端哈希一致且都与上游 `checksums.sha256` 相符。**

### 3.3 `GroundAnything-Model/` 目录清单

```
model.safetensors                9,687,419,736
tokenizer.json                      11,604,906
vocab.json                           2,776,833
merges.txt                           1,671,853
special_tokens_map.json                141,421
tokenizer_config.json                  191,605
modeling_groundinganything.py           77,519
modeling_groundinganything_vision.py    54,576
README.md                               39,093
added_tokens.json                       19,637
media_utils.py                          13,844
configuration_groundinganything_vision.py 11,498
image_processing_groundinganything.py    8,111
configuration_groundinganything.py       7,845
streammind_gate.py                       5,102
processing_groundinganything.py          5,367
config.json                              3,942
LICENSE-Kimi-K3                          3,065
checksums.sha256                         2,361
generation_config.json                     226
LICENSE-Apache-2.0                      11,358
LICENSE                                  1,014
requirements.txt                            75
chat_template.jinja                      1,017
preprocessor_config.json                   953
tokenizer_config.json  (见上)
assets/                                    58 MB（9 个文件）
```

---

## 4. 缺失资产核对：`assets/` vs `.gitattributes`

`.gitattributes` 位于 **`GroundAnything-Model/.gitattributes`**（不是仓库根；仓库根**没有** `.gitattributes`，`cat .gitattributes` → `No such file or directory`）。

声明的 LFS 资产共 9 个，`ls -la GroundAnything-Model/assets` 实测：

| LFS 声明文件 | 远端 | 大小（字节） |
|---|---|---|
| `assets/decoding.mp4` | ✅ 有 | 8,080,597 |
| `assets/demo.mp4` | ✅ 有 | 45,075,106 |
| `assets/demo-poster.jpg` | ✅ 有 | 724,970 |
| `assets/fig1-teaser.png` | ✅ 有 | 4,459,677 |
| `assets/fig2-architecture.png` | ✅ 有 | 413,160 |
| `assets/fig4-attention-mask.png` | ✅ 有 | 371,605 |
| `assets/fig6-self-speculative-decoding.png` | ✅ 有 | 280,935 |
| `assets/fig7-grounding-performance.png` | ✅ 有 | 819,418 |
| `assets/logo.png` | ✅ 有 | 416,379 |

**→ 远端有 9/9，远端无 0/9。无缺失资产。**

`.gitattributes` 还声明了 `*.safetensors`、`tokenizer.json` 等为 LFS —— 两者均存在且校验 OK（§3.2）。

---

## 5. 兄弟目录 `GroundingPI` 与数据/权重就位情况

### 5.1 `GroundingPI` 是什么（一句话）

**`GroundingPI` 是复用同一套代码脚手架的另一个模型和另一篇论文**（"A Grounding Foundation Model towards Physical Intelligence with Visual Primitives"，4B 模型，HF `GroundingPI/GroundingPI`，arXiv 2609.39601，GitHub `groundingpi/GroundingPI`），**不是 GroundAnything 的另一版本**。

证据（`ls -la` + `du -sh` + `head -30 README.md`）：

```
drwxr-xr-x 18 root root 4096 Oct  5 14:10 .
drwxr-xr-x  5 configs      drwxr-xr-x  3 docs        drwxr-xr-x  2 environments
drwxr-xr-x 12 eval         drwxr-xr-x  2 examples    drwxr-xr-x  7 .git
drwxr-xr-x  2 grounding_pi drwxr-xr-x  3 GroundPI-Model
drwxr-xr-x  3 infer        drwxr-xr-x  3 models      drwxr-xr-x  2 scripts
drwxr-xr-x  2 tests        drwxr-xr-x  3 third_party drwxr-xr-x  7 train
drwxr-xr-x  4 vla
-rw-r--r--  1 root root 14529796 Oct  4 23:10 GroundingPI A Grounding Foundation Model towards Physical Intelligence with Visual Primitives.pdf
9.2G  GroundingPI/GroundPI-Model
```

```
<h1 align="center"><img src="docs/assets/readme-title.svg" width="420" alt="GroundingPI" /></h1>
<h3 align="center">A Grounding Foundation Model towards Physical Intelligence with Visual Primitives</h3>
  ... 4B Model ... HF Model/GroundingPI ... arxiv.org/abs/2609.39601 ... github.com/groundingpi/GroundingPI
```

`GroundPI-Model/` 内含 2 分片权重：`model-00001-of-00002.safetensors` 4,995,018,360 B、`model-00002-of-00002.safetensors` 4,692,384,808 B（合计约 9.2 GB）。

> 目录结构（`configs/docs/environments/eval/examples/infer/models/scripts/tests/third_party/train`）与 GroundAnything 高度同构，说明二者出自同一产品线/同一脚手架；但**模型、论文、权重都不同**。（是否同作者/同仓库分叉 **未确认**，本轮未查其 `.git` remote。）

### 5.2 评测数据契约（代码要求什么）

`eval/utils/eval_data_root.py` 明确：数据集路径相对 `GAM_EVAL_DATA_ROOT`（默认 `data/eval`）解析，配置在 `configs/datasets.yaml`；`missing_data_inputs()` 用 `glob` 检查存在性。

`configs/datasets.yaml`（节选）要求的数据根内容：

```
images: images
screenspot_pro: ScreenSpot-Pro
screenspot_v2: ScreenSpot-v2
osworld_g: OSWorld-G
refspatial: RefSpatial-Bench
refcoco: RefCOCO        refcocoplus: RefCOCOplus     refcocog: RefCOCOg
icdar2015_dontcare: annotations/ocr/Challenge4_Test_Task1_GT.zip
gam_coco: annotations/box/COCO.jsonl            gam_lvis: annotations/box/LVIS.jsonl
gam_refcocog_test/val, gam_humanref, gam_m6doc, gam_doclaynet, gam_hiertext,
gam_totaltext, gam_icdar2015, gam_sroie, gam_visdrone, gam_dense200  → annotations/box/*.jsonl
gam_rex_point_* → annotations/point/*.jsonl
gam_robospatial_context: RoboSpatial-Home/data/context-00000-of-00001.parquet
gam_osworld_g: OSWorld-G/OSWorld-G.json
gam_screenspot_pro: ScreenSpot-Pro/annotations/*.json
gam_screenspot_v2_1..3: ScreenSpot-v2/screenspot_*_v2.json
gam_visual_coco/lvis/dense200 → annotations/visual_prompt/*.jsonl
gam_fsc147: annotations/visual_prompt/FSCD_test.jsonl
```

而**仓库根下 `data/`、`weights/`、`outputs/`、`runs/` 四个目录全部不存在**：

```
ls: cannot access '/workspace/project-nas-1000073/liangjiaqi/GroundAnything/data': No such file or directory
ls: cannot access '/workspace/project-nas-1000073/liangjiaqi/GroundAnything/outputs': No such file or directory
ls: cannot access '/workspace/project-nas-1000073/liangjiaqi/GroundAnything/runs': No such file or directory
```

### 5.3 评测数据集是否已在 NAS？（**否**）

```bash
ls -d /workspace/project-nas-1000073/liangjiaqi/*[Ee]val* /workspace/nas-data/*[Ee]val* 2>/dev/null | head
find /workspace/project-nas-1000073/liangjiaqi -maxdepth 3 \( -name 'ScreenSpot-Pro' -o -name 'coco.tar.gz' -o -name 'OSWorld-G' \) 2>/dev/null | head
find /workspace/project-nas-1000073/liangjiaqi /workspace/nas-data -maxdepth 5 -type d \
  \( -name 'ScreenSpot-Pro' -o -name 'ScreenSpot-v2' -o -name 'OSWorld-G' -o -name 'RefSpatial-Bench' \
  -o -name 'RoboSpatial-Home' -o -name 'RefCOCO*' -o -name 'FSC147*' -o -name 'VisDrone*' \
  -o -name 'Dense200*' -o -name 'DocLayNet*' -o -name 'M6Doc*' \) 2>/dev/null | head -40
find /workspace/project-nas-1000073/liangjiaqi /workspace/nas-data/models /workspace/nas-data/LocateAnything-evalbase \
  -maxdepth 4 -type d \( -iname '*groundingdino*' -o -iname '*rex*' -o -iname '*locateanything*' \
  -o -iname '*rlv2*' -o -iname '*rl-v2*' -o -iname '*gam*' \) 2>/dev/null | head -40
```

输出：

```
--- eval dirs ---
/workspace/nas-data/LocateAnything-evalbase
/workspace/project-nas-1000073/liangjiaqi/eval_baselines_pergame_on_merged.py
/workspace/project-nas-1000073/liangjiaqi/eval_unified.py
--- feature names ---
(空)
=== BASELINE-ISH WEIGHT DIRS ===
(空)
=== EVAL-DATA ROOT CANDIDATES (dirs) ===
(空)
```

**逐项核对：**

| 要找的东西 | 结果 |
|---|---|
| `Grounding-EvalData` / `Grounding-Eval*` 类副本 | ❌ **未找到** |
| `ScreenSpot-Pro` / `ScreenSpot-v2` / `OSWorld-G` / `RefSpatial-Bench` / `RoboSpatial-Home` | ❌ 未找到（maxdepth 5） |
| `RefCOCO*` / `FSC147*` / `VisDrone*` / `Dense200*` / `DocLayNet*` / `M6Doc*` | ❌ 未找到（maxdepth 5） |
| `coco.tar.gz` / `doclaynet.tar.gz` | ❌ 未找到（maxdepth 3） |
| `FSCD_test.jsonl` / `_annotations/` 目录 | ❌ 未找到 |
| baseline 权重 `*groundingdino*` / `*rex*` / `*locateanything*` / `*rlv2*` / `*gam*` 目录 | ❌ 无命中（maxdepth 4，覆盖 `liangjiaqi`、`nas-data/models`、`nas-data/LocateAnything-evalbase`） |

**→ 结论：GroundAnything 的评测数据集在 NAS 上不存在，需要从零准备/下载。** 这也是 Lead 提到的"用户明确说暂时不下"的直接后果——目前**没有既存副本可利用**。

**已知存在但与本评测无关的东西：**

- `/workspace/nas-data/dataset/`（69 项）全是**游戏视频 YOLO 数据集**：`cs2_v1/v2`、`dota2_v1/v2`、`pubg_v1/v2`、`wangzherongyao_v1..v4`、`guandan_v1/v2`、`lolgame_firstview_and_match_detect_data_voc2yolo_class53`、`sanjiaozhou`、`yuanshen` 等 —— 属于 `all-games-yolo-detection` 课题，**不是** GroundAnything 的 grounding 评测集。
- `/workspace/nas-data/LocateAnything-evalbase/` 是 **LocateAnything 的代码仓库副本**（`eval.sh`、`evaluation/`、`game_finetune/`、`locateanything_worker.py` 等），**不是** GroundAnything 的数据根。
- `/workspace/project-nas-1000073/liangjiaqi/dataset/` = `FullScreen` / `LocateAnything` / `YOLO_Detections`，同样是游戏检测数据。

**已就位的模型权重（供参考，非 GroundAnything 评测 baseline）：**

| 路径 | 大小 |
|---|---|
| `liangjiaqi/models/Qwen3.8-27B` | 52 G（18 分片） |
| `liangjiaqi/models/Qwen3-VL-32B-Instruct` | 63 G（14 分片） |
| `liangjiaqi/models/Qwen3-VL-8B-Instruct` | 17 G |
| `liangjiaqi/models/YOLOv12` | 232 M |
| `nas-data/models/` | `CLIP`、`Qwen3.8-27B`、`Qwen3-VL-32B/8B-Instruct`、`Rex-Omni`、`YOLOv12`（各自大小**未实测**） |
| `liangjiaqi/Rex-Omni/` | Rex-Omni **代码仓库**（含 `model/` 子目录，权重是否下载**未确认**） |
| `liangjiaqi/LocateAnything/` | LocateAnything **代码仓库**（含 `runs/`） |
| `GroundingPI/GroundPI-Model/` | 9.2 G（2 分片） |
| `GroundAnything/GroundAnything-Model/` | 9.1 G（**已校验通过**） |

**GroundingDINO / RLv2 权重：未找到（未确认是否以其它命名存在）。**

### 5.4 本节的搜索局限（重要）

- 定向 `find` 的深度上限为 **3–6 层**；NAS 树极深时可能漏检。
- 一次 `maxdepth 6` 的特征名全量搜索被 **中途终止**（Lead 要求立即停止长命令），**未完成**。
- 因此 §5.3 的"未找到"应理解为 **"在上述深度与命名范围内无命中"**，不等于"绝对不存在"。**深度搜索未确认。**

---

## 6. 开发机 GPU 与环境矩阵

### 6.1 GPU（**8× A800-SXM4-80GB**，但为共享机）

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

**采样 1（UTC 02:04:20 / CST 10:04:20）**

```
NVIDIA-SMI 550.163.01             Driver Version: 550.163.01     CUDA Version: 12.6
|   0  NVIDIA A800-SXM4-80GB  ...  8065MiB /  81920MiB  |  0%  Default | MIG Disabled
|   1  NVIDIA A800-SXM4-80GB  ...     4MiB /  81920MiB  |  0%  Default |
|   2  NVIDIA A800-SXM4-80GB  ... 10945MiB /  81920MiB  |  0%  Default |
|   3  NVIDIA A800-SXM4-80GB  ...     4MiB /  81920MiB  |
|   4  NVIDIA A800-SXM4-80GB  ...   421MiB /  81920MiB  |
|   5  NVIDIA A800-SXM4-80GB  ...   421MiB /  81920MiB  |
|   6  NVIDIA A800-SXM4-80GB  ...   421MiB /  81920MiB  |
|   7  NVIDIA A800-SXM4-80GB  ...     4MiB /  81920MiB  |
Processes: （表为空）
=== QUERY COMPUTE APPS ===
pid, process_name, used_gpu_memory [MiB]
496016, [Not Found], 8058 MiB
250101, [Not Found], 10938 MiB
268945, [Not Found], 414 MiB
268946, [Not Found], 414 MiB
268947, [Not Found], 414 MiB
```

**采样 2（UTC 02:19 / CST 10:19，约 15 分钟后）**

```
--- gpu mem ---
0, 8065 MiB
1, 54587 MiB
2, 4 MiB
3, 4 MiB
4, 4 MiB
5, 4 MiB
6, 4 MiB
7, 4 MiB
```

**判读（重点）：**

1. 硬件充足：**8 张 A800-80GB，合计 640 GB 显存**，MIG 已禁用。
2. **占用是动态的**：GPU 1 在 15 分钟内从 `4 MiB` 涨到 `54,587 MiB`；GPU 2/4/5/6 又回落到 4 MiB。说明**其它租户/任务正在同一台机器上启停**。
3. `nvidia-smi` 进程表为空，而 `--query-compute-apps` 能列出 PID（`process_name` 显示 `[Not Found]`）——典型**跨 PID namespace / 容器不可见**，因此**无法从本机确认占用者身份（未确认）**。
4. **纪律遵守：本次侦察未 kill 任何进程。**
5. 提醒：Lead 转述的"卡 1/2/7 空闲、卡 0/3/4/5/6 被占"与**采样 1** 接近（除卡 4/5/6 只占 421 MiB），但到**采样 2** 时已变成"仅卡 0/1 被占"。**GPU 空闲状态是随时间变化的快照，排产前必须重新采样。**

### 6.2 磁盘

```bash
df -h /workspace/project-nas-1000073 /workspace/nas-data / /tmp
df -B1 /workspace/project-nas-1000073 /workspace/nas-data / /tmp
```

```
Filesystem                                       Size  Used  Avail Use% Mounted on
...nas.ivolces.com:/ai-data-service/project/1000073   10P  635T  9.4P   7% /workspace/project-nas-1000073
...nas.ivolces.com:/ai-data-service/workspace/liangjiaqi 10P 635T  9.4P   7% /workspace/nas-data
overlay                                           10G  7.9G  2.2G  79% /
overlay                                           10G  7.9G  2.2G  79% /tmp
```

```
Filesystem                                             1B-blocks            Used        Available Use%
...:/ai-data-service/project/1000073        11258999068426240  698157278167040  10560841790259200   7%
...:/ai-data-service/workspace/liangjiaqi   11258999068426240  698157278167040  10560841790259200   7%
overlay                                          10737418240     8415272960      2322145280  79%
```

| 挂载 | 人类可读 | 精确字节（Available） | 备注 |
|---|---|---|---|
| `/workspace/project-nas-1000073` | 10P 总 / 635T 已用 / **9.4P 可用（7%）** | **10,560,841,790,259,200 B ≈ 9.60 PiB** | NAS，可写大文件 |
| `/workspace/nas-data` | 10P 总 / 635T 已用 / **9.4P 可用（7%）** | **10,560,841,790,259,200 B ≈ 9.60 PiB** | **同一个 NAS 卷的另一个导出**（`/ai-data-service/workspace/liangjiaqi`），数字完全相同 |
| `/`（overlay） | 10G 总 / 7.9G 已用 / **2.2G 可用（79%）** | **2,322,145,280 B ≈ 2.16 GiB** | 根盘，`/tmp` 也在其上 |
| `/tmp` | 同 overlay | 同上 | **不要在此放大文件** |

**→ NAS 空间充裕（9.6 PiB 可用）；overlay 根盘只剩 2.16 GiB，任何大文件（权重、数据集、日志）都必须落 NAS。**

### 6.3 Python / 框架版本

```bash
which -a python python3 ; python3 -V
ls -la /opt/venv/bin/python*
python3 -c 'import torch;print(1,torch.__version__,torch.version.cuda,torch.cuda.device_count(),torch.cuda.get_device_name(0))'
python3 -c 'import transformers;print(1,transformers.__version__)'
python3 -c 'import vllm;print(1,vllm.__version__)'
python3 -c 'import sglang;print(1,sglang.version.__version__)'
python3 -c 'import flash_attn;print(1,flash_attn.__version__)'
/opt/venv/bin/pip list | grep -Ei 'torch|transformers|vllm|sglang|flash|accelerate|deepspeed|peft|trl|datasets|lmms|opencv|pillow'
conda env list
find / -maxdepth 6 -type d -name 'sglang' ; find / -maxdepth 6 -type d -name 'vllm'
```

```
=== PYTHON ===
/opt/venv/bin/python
/opt/venv/bin/python3
/usr/bin/python3
/bin/python3
Python 3.12.3
=== TORCH ===
1 2.7.1+cu126 12.6 8 NVIDIA A800-SXM4-80GB
=== TRANSFORMERS ===
1 4.57.1
=== VLLM ===            → ModuleNotFoundError: No module named 'vllm'
=== SGLANG ===          → ModuleNotFoundError: No module named 'sglang'
=== FLASH_ATTN ===
1 2.8.3.post1
=== sglang dirs anywhere ===   (空)
=== vllm dirs anywhere ===     (空)
=== CONDA ENVS ===
bash: line 1: conda: command not found
```

`/opt` 内容与解释器解析：

```
drwxr-xr-x node-v24.19.0   drwxr-xr-x nvidia   drwxr-xr-x venv
lrwxrwxrwx /opt/venv/bin/python   -> python3
lrwxrwxrwx /opt/venv/bin/python3  -> /usr/bin/python3
lrwxrwxrwx /opt/venv/bin/python3.12 -> python3
```

**Python 解释器绝对路径：`/opt/venv/bin/python`（→ `/usr/bin/python3`，Python 3.12.3）。**

`pip list` 关键包：

```
accelerate      1.5.2        datasets        5.0.1        deepspeed      0.15.4
flash_attn      2.8.3.post1  open_clip_torch 3.3.0        opencv-python  5.0.0.93
opencv-python-headless 5.0.0.93  peft  0.12.0  pillow  12.2.0
torch  2.7.1+cu126  torchaudio 2.7.1  torchvision 0.22.1+cu126  transformers 4.57.1
```

### 6.4 GroundAnything 专用 venv：**不存在**

```bash
ls -d /workspace/project-nas-1000073/liangjiaqi/GroundAnything/.venv* \
      /workspace/project-nas-1000073/liangjiaqi/GroundAnything/venv* \
      /workspace/project-nas-1000073/liangjiaqi/*/.venv* \
      /workspace/project-nas-1000073/liangjiaqi/GroundAnything/weights 2>&1
```

```
ls: cannot access '/workspace/.../GroundAnything/.venv*': No such file or directory
ls: cannot access '/workspace/.../GroundAnything/venv*': No such file or directory
ls: cannot access '/workspace/.../liangjiaqi/*/.venv*': No such file or directory
ls: cannot access '/workspace/.../GroundAnything/weights': No such file or directory
```

**→ 项目级 venv 不存在；`weights/` 目录不存在（权重实际放在 `GroundAnything-Model/`）。全机唯一可用环境是系统级 `/opt/venv`。**

### 6.5 版本冲突表（仓库锁定 vs 实际）

```bash
cat requirements.txt requirements/serve.txt requirements/vllm.txt requirements/eval.txt requirements/train.txt environment.yml pyproject.toml
grep -rn 'transformers' requirements/ requirements.txt environment.yml pyproject.toml environments/
grep -rn 'sglang' requirements/ requirements.txt environment.yml pyproject.toml environments/ scripts/
```

仓库侧锁定值：

```
requirements/serve.txt: torch==2.9.1  transformers==5.5.4  triton==3.5.1
                        sgl-kernel==0.3.20  flashinfer-python==0.5.3  flashinfer-cubin==0.5.3
requirements/vllm.txt:  # "The isolated vLLM environment installs the bundled Transformers 5.7.0 fork."
                        huggingface-hub==1.32.0  regex==2026.9.10
requirements/eval.txt:  transformers>=4.39.2,<5   datasets>=2.19,<4   numpy==1.26.4
                        opencv-python-headless<4.12  sentence-transformers==5.1.2  pycocotools/shapely/scipy/rapidfuzz/editdistance
                        torch @ .../torch-2.5.1+cpu-cp312-...  torchvision @ .../torchvision-0.20.1+cpu-cp312-...
requirements/train.txt: torch==2.10.0+cu129  transformers==5.7.0  torchvision==0.25.0+cu129  triton==3.6.0
                        datasets==5.0.0  accelerate==1.13.0  deepspeed==0.19.0  flash-attn==2.8.3.post1
                        flash-linear-attention==0.5.2  qwen-vl-utils==0.0.14  pyarrow==24.0.0
environments/install.yaml:27:  - sglang
environment.yml: python=3.12 (conda bootstrap，conda 本机不可用)
```

对照表：

| profile | 仓库锁定 | `/opt/venv` 实际 | 判定 |
|---|---|---|---|
| **train** | torch 2.10.0+cu129, transformers **5.7.0**, deepspeed 0.19.0, accelerate 1.13.0, datasets 5.0.0, torchvision 0.25.0+cu129 | torch 2.7.1+cu126, transformers **4.57.1**, deepspeed 0.15.4, accelerate 1.5.2, datasets 5.0.1, torchvision 0.22.1+cu126 | ❌ **不满足** |
| **serve（SGLang）** | torch 2.9.1, transformers 5.5.4, triton 3.5.1, sgl-kernel 0.3.20, flashinfer 0.5.3, **sglang** | torch 2.7.1+cu126, transformers 4.57.1, **sglang 缺失**, sgl-kernel/flashinfer 未见 | ❌ **不满足** |
| **vllm** | 需继承 vLLM 0.18.x 平台镜像 + transformers 5.7.0 fork 隔离环境 | **vllm 缺失** | ❌ **不满足** |
| **eval** | transformers>=4.39.2,<5；datasets>=2.19,<4；numpy==1.26.4；opencv-python-headless<4.12；CPU 版 torch | transformers 4.57.1 ✅；datasets 5.0.1 ❌；opencv-python-headless 5.0.0.93 ❌；numpy 未测；torch 为 GPU 版 | ⚠️ **部分满足** |

**关键点：仓库依赖锁定的 `transformers` 三条线分别是 5.5.4（serve）/ 5.7.0（train、vllm fork）/ `>=4.39.2,<5`（eval），实测 `/opt/venv` 是 `4.57.1`。** 即：**当前环境最多只能勉强对齐 eval profile 的 transformers 约束，train 与 serve/vllm 均不满足。**

---

## 7. 历史跑批痕迹

```bash
find /workspace/project-nas-1000073/liangjiaqi -maxdepth 3 -type d \( -name runs -o -name outputs -o -name logs -o -name checkpoints \)
find /workspace/project-nas-1000073/liangjiaqi -maxdepth 3 -type f -newermt '2026-09-28' -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n' | sort -r | head -30
```

**GroundAnything 仓库本身：`runs/`、`outputs/`、`logs/` 均不存在（§5.2），即该仓库在 devbox 上从未跑过训练/评测。**

兄弟课题的跑批痕迹（存在）：

```
/workspace/project-nas-1000073/liangjiaqi/LocateAnything/runs
/workspace/project-nas-1000073/liangjiaqi/LocateAnything-Label-Generator/runs
/workspace/project-nas-1000073/liangjiaqi/all-games-yolo-detection/{logs,runs}
/workspace/project-nas-1000073/liangjiaqi/locany-runs/*/logs      (18 个 <game>-vllm-qwen3-vl-8b-opc8-rc4-desc_grounding_zh)
```

**最近修改的 10 个产物（2026-09-28 起，`sort -r` 截取）：**

| # | 时间 | 大小 (B) | 路径 |
|---|---|---|---|
| 1 | 2026-10-09 10:02 | 38,031 | `GroundAnything/.git/index`（只读 `git status` 刷新索引的副作用，非产物） |
| 2 | 2026-10-09 09:49 | 8,397 | `all-games-yolo-detection/final_report.md` |
| 3 | 2026-10-09 09:49 | 6,103 | `all-games-yolo-detection/unified_vs_pergame_strict449.json` |
| 4 | 2026-10-09 09:49 | 6,096 | `all-games-yolo-detection/unified_vs_pergame_union336.json` |
| 5 | 2026-10-09 09:49 | 3,218 | `all-games-yolo-detection/unified_vs_pergame_strict449.md` |
| 6 | 2026-10-09 09:49 | 3,217 | `all-games-yolo-detection/unified_vs_pergame_union336.md` |
| 7 | 2026-10-05 04:05 | 52 | `GroundAnything/.git/COMMIT_EDITMSG` |
| 8 | 2026-10-05 04:04 | 2,875 | `GroundAnything/docs/GAME_UI_EVALUATION.md` |
| 9 | 2026-10-05 04:04 | 2,785 | `GroundAnything/tests/test_game_ui_metrics.py` |
| 10 | 2026-10-05 03:54 | 1,497 | `GroundAnything/tests/test_vlm_launcher.py` |

**判读**：devbox 上最近活跃的课题是 **`all-games-yolo-detection`（YOLO 游戏目标检测）**，与 GroundAnything 无关。GroundAnything 的最后动作是 **2026-10-05 的代码提交**（`c5f7840`），**之后没有任何 GroundAnything 的跑批产物**。

---

## 8. 可用性检查：能否直接跑 GPU 任务？

**分三层结论：**

| 层 | 结论 |
|---|---|
| **硬件** | ✅ 8× A800-80GB，显存合计 640 GB，驱动 550.163.01 / CUDA 12.6。**但机器是与他人共享的**：15 分钟内 GPU 1 从 4 MiB 涨到 54,587 MiB（§6.1）。**无独占保证，排产前必须重新采样。** 无法确认占用者身份（跨 PID namespace，`[Not Found]`）。 |
| **环境** | ❌ **不能直接跑**。`/opt/venv` 缺 **sglang**、缺 **vllm**（全机 `find / -maxdepth 6` 也无 sglang/vllm 目录）；torch 2.7.1+cu126 ≠ train 锁定的 2.10.0+cu129 / serve 的 2.9.1；transformers 4.57.1 ≠ train 的 5.7.0 / serve 的 5.5.4。**项目 venv 不存在**，conda 不可用。需按 `run.py setup` / `requirements/*.txt` 建环境（vLLM 还需平台镜像）。 |
| **数据** | ❌ **评测数据集不在 NAS 上**（§5.3）。`Grounding-EvalData` 类副本、ScreenSpot-Pro/OSWorld-G/RefSpatial-Bench/`_annotations`/`coco.tar.gz` 等特征名均无命中。跑 eval 前必须准备数据。 |
| **权重** | ✅ GroundAnything 权重已就位且**27/27 校验通过**（§3）；`assets/` 9/9 齐备（§4）。 |

**可以立刻做的事（无需新数据）**：权重/配置级别的离线检查、tokenizer/processor 加载冒烟、以及不依赖评测数据集的代码级测试（`tests/`）。

**被阻塞的事**：任何需要 `sglang`/`vllm` 的 serve 路径；任何需要 `data/eval` 的 eval 路径；任何要求 torch 2.9+/2.10 的 train 路径。

---

## 9. 未确认 / 未跑项（显式清单）

1. **NAS 深层（>6 层）是否存在 GroundAnything 评测数据** —— 全量 `maxdepth 6` 特征名搜索被中途终止，**未确认**。
2. **`/workspace/nas-data/models/Rex-Omni`、`CLIP` 等目录的具体大小与内容完整性** —— **未实测**。
3. **`liangjiaqi/Rex-Omni/model/` 下是否已下载权重** —— **未确认**。
4. **GroundingDINO、RLv2 权重是否以其它命名存在于更深处** —— **未确认**。
5. **GroundingPI 的 `.git` remote / HEAD / 与 GroundAnything 的仓库关系** —— **未查**（Lead 要求不深挖）。
6. **`/workspace/cpfs-data`、`/workspace/project-cpfs-1000073` 两个挂载下是否有相关数据** —— **未查**。
7. **GPU 占用进程的真实属主与作业名** —— 跨 PID namespace，`process_name=[Not Found]`，**未确认**。
8. **本地 `D:\Desktop\GroundAnything` 的 `origin/main` 位置**（远端领先 origin 4 个提交，本地是否同样） —— **未查**。
9. **`sha256sum -c` 进程是否已完全退出** —— 27 行 `OK` 已全部产出（与清单行数相等），但**进程退出状态未复核**。
10. **远端仓库的 `git status` 是否因本次侦察而变化** —— `.git/index` 在 2026-10-09 10:02 被刷新（只读命令 `git status` 的正常副作用）；**未修改任何被跟踪文件的内容**。

---

## 10. 附录

### 10.1 临时文件说明（唯一的两处写操作）

两个输出文件写在 **overlay `/tmp`**（非 NAS，非仓库），用于让长耗时的 `sha256sum` 后台运行而不阻塞 ssh：

| 路径 | 内容 | 处置 |
|---|---|---|
| `/tmp/ga_sha256_model.txt` | `model.safetensors` 单文件 sha256 结果 | 未删除（遵守"不删远端文件"的只读纪律） |
| `/tmp/ga_checksums_verify.txt` | `sha256sum -c checksums.sha256` 的 27 行结果 | 未删除（同上） |

两者合计 < 2 KB，位于 overlay 根盘，**未触碰 NAS、未触碰仓库**。

### 10.2 本次侦察执行过的全部远端命令（只读）

```
ls -la / ls -1 / du -sh / wc -l / cat / head / grep / od -c / file
git rev-parse HEAD | --abbrev-ref HEAD | log --oneline -12 | branch -a | remote -v
git status --short | --porcelain | --untracked-files=all
git diff --stat | --numstat | --ignore-cr-at-eol --stat
git ls-files | config --get core.autocrlf / core.eol
sha256sum model.safetensors ; sha256sum -c checksums.sha256
nvidia-smi ; nvidia-smi --query-compute-apps / --query-gpu
which -a python python3 ; python3 -V ; pip list ; python3 -c 'import ...'
find (maxdepth 3–6，只读)
df -h / df -B1
```

**未执行**：rsync、scp、git add/commit/checkout/reset/clean、rm、mv、cp、kill、pkill、任何 pip install、任何写 NAS 的操作。
