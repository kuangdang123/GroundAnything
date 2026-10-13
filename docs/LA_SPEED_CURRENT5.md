# LA 三模式单卡速度测量

## 本轮新增范围

`eval/current5_la_speed.py` 是 GA 项目中的测速客户端，调用远端已安装的 LA 基础模型原始 worker，不复制/修改 LA 模型源码。
基础权重默认 `/workspace/nas-data/LocateAnything/nvidia/LocateAnything-3B`，运行解释器 `/opt/venv/bin/python`，与 GA `.venv-serve` 隔离。
**当前只完成CPU测试和调度设置；尚无这一新增实验的LA速度结果。**

模式由实际 `modeling_locateanything.py:347-353` 和 worker 接口核对：
- fast：MTP only；
- hybrid：MTP优先，错误框切AR，到box_end再切MTP；
- slow：纯AR。
它们不是GA三种模式的一一同名对应。

## 固定实验条件

- 单张A800，batch_size=1（原生generate也assert1）、并发=1；
- 同一基础checkpoint加载一次，fast→hybrid→slow顺序测量；
- 与GA相同Current5原始中文查询、每非空game/form第一条，总88样本；
- 每模式预热3次，正式重复3次，264次/模式；
- T=0、top_p=1、top_k=0、repetition_penalty=1、max_new_tokens=512、seed=42；
- 同步GPU后记录时钟，不把加载时间纳入速度；加载秒数另存；
- 每请求前后检查GPU外来PID，发现新进程停止，不kill其他作业；
- 计数取原生generate的actual generated_ids统计，不重分词输出文本猜token数。

## TPS / BPS / 计时窗口

原生generate `verbose=True` 返回的stats包括 num_tokens、num_boxes、forward_step、prefill_time。
原生 BPS = literal `<box>` wrapper count / generate time，因此也可能含None或非法框，**另报可解析有效框/s**。

本工具分开输出：
1. E2E：RGB加载、同GA PNG/base64 CPU准备、LA worker预处理、模型生成、解析；没有GA的HTTP传输开销，不能宣称后端完全相同。
2. synchronized generate：视觉编码＋前缀prefill＋解码＋反分词；TPS/BPS主计时分母。
3. decode-window：首个LM forward结束后至generate返回；**仍包含首token选择**，且分子为所有output tokens，不冒充严格排除首token的paper decode-only TPS。
4. 原生verbose stats和LM forward计数保留，输出tokens/LM forward；不是视觉塔forward计数。

使用verbose会有每步历史token解码开销，三模式同样启用，不隐去。按总tokens/总时间计算吞吐，不平均每请求TPS。
GA论文附录C.2严格draft round计时始于prefix cache就绪、结束于commit，和本工具generate范围不同；本实验借鉴单卡条件控制，**不是论文4.51×或MTP drafter对照的复现**。
GA旧速度输出是HTTP/E2E范围，LA新增native计时必须分别展示，跨LA/GA首选同范围应用延迟、valid boxes/s，token词表不同不能只用TPS比较。

## 调度

原GA接续任务将扩展：
- 等LA的93项验证结果完整、三条queue结束、GPU1/2/3空闲；
- GPU1/2新run重跑GA baseline；
- GPU3先完成GA speculative测速，服务退出并释放后再运行LA fast/hybrid/slow；
- LA加载原始基础权重，不取正在验证的任意LoRA/full checkpoint；
- 使用新状态目录，旧等待调度器仅在确认未launch且无子任务时停止，保留其日志；
- 输出JSONL即时flush，配置/代码hash/实际环境/GPU观测/原答案/实际token与box计数齐全。

调度器 `scripts/after_la_current5.py --la-speed-output <new-path>` 接入。独立手动命令（本轮不执行）：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD /opt/venv/bin/python -u -m eval.current5_la_speed \
  --gpu 3 --output-dir outputs/eval/current5_la_base_modes_gpu3_b1_zh512_v1
```

先 `--dry-run` 核验样本/权重配置/worker代码hash，不加载torch/model、不探GPU。
真实运行只接受新输出目录；失败记录保留，不重复混入旧测速数据。

## 验证级别

CPU测试覆盖原生stats解析、ratio-of-totals TPS/BPS、模型timer/forward wrapper恢复、调度第三GPU与LA专用解释器映射。
这些测试不构成基础LA模型前向验证。排队后仍需真实预热/请求才知道三模式是否全部可跑。
