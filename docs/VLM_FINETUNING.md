# GroundAnything VLM 微调

GroundAnything 的 VLM SFT 入口是 `vlm-train`，底层训练循环由
`ms-swift` 提供，项目负责 recipe、路径契约和冻结策略审计。LA 的 LoRA
配方可以直接迁移为：冻结 vision tower，冻结 LLM base，在 `q/k/v/o` 和
`gate/up/down` 上注入 LoRA，同时解冻 GA 的视觉对齐器（projector）。

## 数据

使用 `tools/convert_locany_to_gam.py` 生成 canonical GAM JSONL。每条记录
必须有 `messages`、`images`、`source` 和唯一 `id`，图片占位符数量必须与
`images` 数量相同。VLM Swift 路由直接读取 JSONL；Arrow cache 是 DLM
packing 的输入，不能把 cache 目录误当成 Swift 的 JSONL dataset。

## LoRA 配方

`configs/train/vlm_lora.yaml` 对应 LA 的 `train_lora.sh`：

- `lora_rank=64`，`alpha=128`，`dropout=0.05`；
- LLM base 与 vision tower 冻结；
- `freeze_aligner=false`，训练 GA 的 projector；
- target modules 为 QKV/O 与 MLP projections；
- 每个 run 在输出目录写入 `trainable_parameter_policy.json`。

全参对照实验只需将 `tuner_type` 改为 `full`，并把
`freeze_llm/freeze_vit/freeze_aligner` 都设为 `false`。全参训练应使用
DeepSpeed ZeRO-2/ZeRO-3，并单独记录显存、吞吐和 checkpoint 大小。

## 启动

先只检查命令（不导入 Swift 或 GPU）：

```powershell
python scripts/run.py configs/release/vlm_lora_train.yaml --dry-run
```

实际训练需要安装与 Transformers 版本匹配的 `ms-swift`。launcher 会在
启动前检查该依赖，缺失时直接报错，不会产生一个看似成功的空 checkpoint。
