# GA Current5 baseline 与单卡速度评估

## 状态与证据边界

本工具为 2026-10-13 新实现；CPU 指标/客户端测试与 LA 原纯函数差分已验证。
**没有启动 GA GPU 作业，没有得到新 baseline 分数或性能数字。**
使用现有 GA DLM bundle；`causal` 指同一 DLM 的因果分支，**不是独立的 GroundAnything-VLM checkpoint**。
两卡 baseline 是两个单卡副本分片，并非 TP=2 或 batch=2。第三卡速度依次运行三模式，不同时占三份权重。

## 1. 相同数据与指标

默认读取 notebook 现有目录：

```text
/workspace/nas-data/Feishu-Codex-Agent/workspaces/oc_9adceb0944844d02e4dbfcd75331573d/analysis/current5_eval_20260927
```

只读 `protocol.json` 和 `_samples/<form>/<game>.jsonl`，**不重抽样、不修改 LA 输出**。
逐文件校验 manifest 的 SHA256 与样本数，图片来自各 game `run_dir/val/images`。
本轮 notebook CPU 实读校验：18 游戏、1396 请求、88 个非空 game/form cell。
category=216、description=360、multiclass=180、multi_instance=292、negative=348；两卡分片为692/704请求。
部分 multi_instance 数量不足32或为空，以既有实际清单为准。

| form | 指标 |
|---|---|
| category / multiclass | 标签归一化后，label-aware micro-F1；IoU=0.50,0.55,...,0.95 各自累计TP/FP/FN，最后平均10个F1 |
| description / multi_instance | 忽略标签，所有框最大二分匹配；同样10阈值micro-F1、request all-correct@0.5、count MAE、best IoU、框数分桶 |
| negative | LA 原定义 none_accuracy、幻觉框/请求、显式 None 与 empty 率 |
| 总汇总 | 每 form 内按游戏宏平均；5 form 等权宏平均；再按原 protocol.train_weights 汇总，及最差游戏 |

不能把全请求微平均替代游戏宏平均，也不能把缺失 cell 当0。未完成的覆盖不输出完整总分。
LA 的 `none_accuracy` 把无框非法/空回答也计成功；这里保留该历史定义，但单列 canonical_valid_rate、transport_errors、quality。
HTTP故障不是合法None：保留错误行，完整汇总覆盖中不算完成，可重试传输失败。

### 提示、坐标与解析

- 默认 `--prompt-mode la_zh` **保留每条样本原中文句子**，所有请求都是 all-target，不对multi_instance强制单目标。
- `--prompt-mode ga_native` 使用 GA 原生英文 all-instance 模板，查询正文仍中文；必须新run，属于单独prompt消融，不能冒充完全同prompt比较。
- baseline 默认与 LA cloud-b 外层采样一致：T=0.1 / top_p=0.9 / max output=512。GA denoise 内层亦显式传入相同设置，seed=42。
- GA 原生坐标 `[0,999]` 用 `/999*(W-1)` 映射像素；LA GT 的 `[0,1000]` 按原parser用 `/1000*W`。不把 LA 答案数字直接当GA token。
- 主分数从**完整可解析框**恢复（与LA宽松恢复可比），保留重复框和FP，不做NMS、不去重、不靠GT修正。
- 同时运行 GA canonical grammar 检查并保存原始回答、finish_reason；格式错误不隐藏。
- LA 的指标源文件 SHA 在 `eval/current5_metrics.py:SOURCE_SHA256`。既有 protocol 中的历史 evaluator SHA 与当前脚本 SHA 不同；记录两者，不改原manifest。

## 2. 两卡 baseline 的启动方式

**本次不执行以下真实启动命令。** 卡号只是规划，1/2/3 当前有 LA 作业，必须先等释放并核对 `nvidia-smi`。
默认每服务 TP=1，max_running_requests=1、无CUDA Graph、BF16。启动器拒绝已有compute进程的卡和已占端口。

```bash
cd /workspace/project-nas-1000073/liangjiaqi/GroundAnything

# 无GPU访问/服务启动的计划预览
.venv-serve/bin/python scripts/run_current5.py baseline \
  --gpus 1,2 --output-dir outputs/eval/current5_ga_denoise_gpu1-2_b1_zh512_v1 --dry-run

# 仅在用户确认卡空闲后执行：两个副本端口8111/8112
.venv-serve/bin/python -u scripts/run_current5.py baseline \
  --gpus 1,2 --output-dir outputs/eval/current5_ga_denoise_gpu1-2_b1_zh512_v1
```

每副本先10条真实冒烟，传输成功才放全量；canonical有效率打印供检查，不因格式无效悄悄改变预测。
整game/form cell按序round-robin分片，不重复计算，最终由一个汇总进程聚合。
长作业请用受管理的后台job启动；脚本自己管理子服务生命周期，只停止自己启动的进程组。
先记录模型配置哈希和权重大小（**非完整权重SHA验证**）、Git HEAD、包版本、GPU UUID/占用、启动参数。

### 中断恢复（已有服务仍在时）

```bash
.venv-serve/bin/python -m eval.current5 --output-dir <run>/baseline \
  --num-shards 2 --shard-index 0 --base-url http://127.0.0.1:8111/v1
.venv-serve/bin/python -m eval.current5 --output-dir <run>/baseline \
  --num-shards 2 --shard-index 1 --base-url http://127.0.0.1:8112/v1
# 两个client结束后，唯一writer汇总
.venv-serve/bin/python -m eval.current5 --output-dir <run>/baseline \
  --num-shards 2 --summarize
```

配置、服务有效参数、源样本hash与已经完成的图片hash改变会拒绝续跑。JSONL坏尾行报错，不自动截断。
启动器要求新版本目录；续跑客户端与服务恢复需显式操作，不自动覆盖旧run。

## 3. 第三卡速度：同样样本、单卡 batch=1 / 并发=1

```bash
# 先预览，不启动
.venv-serve/bin/python scripts/run_current5.py speed \
  --gpus 3 --output-dir outputs/eval/current5_ga_modes_gpu3_b1_zh512_v1 --dry-run

# 用户之后明确允许、卡3空闲时运行
.venv-serve/bin/python -u scripts/run_current5.py speed \
  --gpus 3 --output-dir outputs/eval/current5_ga_modes_gpu3_b1_zh512_v1
```

按 `denoise -> causal -> speculative` 顺序起停，同一张卡、端口8113。
共享 sample_manifest，默认每个非空game/form取1条：本轮真实清单 dry-run 为88条×3次=264测量/模式。
先3次warmup不计入测量。三模式共792次测量，另有9次warmup。
可用 `--speed-per-cell 2 --speed-repeats 5` 扩大；每种模式固定温度0/top_p1、512 output预算。
纯速度三模式采用贪心，是因为自推测验证是greedy；不要用这里T=0的数字解释T=0.1 baseline。
预热后测量不含模型加载/服务启动；加载状态和日志另存。

报告：
- E2E latency（RGB解码、PNG/base64准备、HTTP生成、解析）mean/p50/p90/p95；HTTP往返另列。
- sequential requests/s（sum durations），实际测试wall throughput若支持另列，不能混用。
- completion tokens/s（总completion tokens/总耗时），token长度缺失显式标明；**跨LA/GA词表不可直接当公平主指标**。
- 有效/无效/截断/传输失败计数；all-attempt速度与success-only分开，不能剔除慢失败让吞吐虚高。
- 每form、GT框数分桶；保留每条原答案/usage/时延，便于比较模式输出长度和对象数量。
- GPU UUID、前后nvidia-smi显存采样；不是精确CUDA峰值。外来PID出现会停止后续测试，不kill其他作业。

### 与 LA 的公平对比

LA eval_current5.py 已保存 per-request latency 和mean/p95，但没warmup/吞吐/输出token测试。
LA现有耗时范围是 PIL加载之后，worker detect生成+解析；GA HTTP包含序列化和服务开销。
所以本工具报告GA端到端与HTTP范围，**不称为已实现跨后端同范围纯模型计时**。
真正GA/LA应用级速度对照需在LA端补同一RGB输入准备起点、同一sample_manifest、warmup、单卡batch1/并发1、同输出预算。
分别记录LA generation_mode fast/hybrid/slow与GA三模式，不能凭名字认定算法一一对应。
共享卡速度只能标为受争用观测；启动器要求速度卡无外来compute PID。
notebook 实测 NVML 返回宿主PID，`/proc`仅显示容器PID，不能直接比对。启动器对**启动前空卡、就绪后唯一不可见driver PID**进行固定记录，并在每次请求前后拒绝新增/替换PID；每mode记录 `gpu-identity-<mode>.json`，服务停后再确认卡释放。这是空卡时序证据，**不是证明宿主/容器PID映射**，外部竞争窗口仍存在；多PID或可见但不属于服务时拒绝继续。其他GPU同时跑baseline仍可能有CPU/NAS争用，严格最终基准最好再单独复测。

## 4. 产物与 CPU 验证

每新run用 `..._vN`（已有目录不覆盖），含plan/environment/gpu-before/gpu-after、服务console、逐shard command/server/config、预测JSONL、summary。
`eval.current5_speed` 单独输出计划、配置、warmup、逐测量JSONL和统计。

```bash
python -B -m unittest discover -s tests -p 'test_current5*.py' -v
python -B scripts/verify_current5_metrics.py \
  --la-evaluator /workspace/nas-data/LocateAnything/game_finetune/evaluate/eval_current5.py \
  --la-common /workspace/nas-data/LocateAnything/game_finetune/core/common.py \
  --output <new-run>/metric-parity.json
```

本轮实测：本地65项CPU测试通过；notebook65项通过（0.838s）；两端原函数差分500个cell/1000个parser答案一致。
notebook证据保存在 `outputs/eval/current5_cpu_alignment_20261013_v1/metric-parity.json`、`cpu-tests.log`、`speed-plan/`。
差分验证只用AST提取原指标/解析纯函数，用人工box/fake图片/fake worker对照，**不是模型推理实测**。
不导入LA GPU模块，不复制checkpoint，不执行LA finalizer（它会重建/清理TensorBoard目录）。
