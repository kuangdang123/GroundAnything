# GroundAnything 本地仓库只读侦察报告

> 侦察对象：`D:\Desktop\GroundAnything`（clone of https://github.com/kuangdang123/GroundAnything.git，HEAD `c5f7840`）
> 侦察方式：只读（read / grep / 只读 pwsh 命令）。未运行任何训练、推理、评测命令，未安装依赖，未下载任何内容。
> 本机事实：Windows，**无 GPU**；本地解释器 `C:\Users\Huya\AppData\Local\Programs\Python\Python312\python.exe`（3.12.10）。
> 本文件是本次侦察唯一写入的文件。所有结论都标注 `文件:行号`；无法确认的一律写「未确认」。
> 进度：A ✅ / B ✅ / C ✅ / D ✅ / E ✅ / F ✅ / G ✅ / H ✅ / 速查表 ✅ / 缺失项 ✅ / 风险 ✅（收尾见文末「未确认」节）

---

## 0. 一句话定位与运行位置

GroundAnything 是一个「并行解码（DLM）+ 精确视觉 grounding」的统一模型仓库，代码面覆盖 5 条相互独立的运行路线：**DLM 训练**、**VLM LoRA 微调**、**SGLang/vLLM 服务**、**lmms-eval 评测**、**RL（RLV2 / GRPO）**。所有路线都假定 **Linux + CUDA**，缺乏 Windows 支持（见风险清单 R-01）。

| 项 | 值 | 出处 |
| --- | --- | --- |
| 项目根 | `D:\Desktop\GroundAnything` | `run.py:8` |
| 统一的源码树入口 | `python run.py {setup,train,serve,eval,prepare-model}` | `run.py:12-39` |
| 真实的 YAML 启动器 | `python scripts/run.py <config.yaml>`（服务/训练）、`python scripts/evaluate.py <config.yaml>`（评测） | `scripts/run.py:15`、`scripts/evaluate.py:21` |
| 评测引擎 | `python -m eval.eval_runner`（由 `scripts/evaluate.py:147-151` 组装） | `eval/eval_runner.py:814-863` |
| 依赖 | `pyproject.toml`（仅 498 B，只是元数据）、`requirements/{train,serve,eval,vllm}.txt`、`environments/install.yaml` | 见 F 节 |
| 数据根 | 由评测 YAML 的 `data_root` 指定，默认 `data/eval`；**本地完全不存在** | `eval/utils/eval_data_root.py:16-21,48` |
| 输出根 | 由评测 YAML 的 `output_root` 指定，`outputs/eval/<run_id>/`；**本地不存在** | `scripts/evaluate.py:76`、`eval/eval_runner.py:39` |

---

## A. 入口与参数契约（最高优先级）

### A1. `run.py`（8693 B，145 行）——源码树短命令前端

- **职责**：把 `setup/train/serve/eval/prepare-model` 五个动作翻译成下游 `scripts/*.py` 或 `-m <module>` 命令，并 `os.execv` 执行（`run.py:143`）。
- **顶层常量**：`ROOT = Path(__file__).resolve().parent`（`run.py:8`）。
- **参数契约**（`parser_for`，`run.py:10-39`）：

| 子命令 | 参数 | 类型/取值 | 默认 | help |
| --- | --- | --- | --- | --- |
| `setup` | `profile`（位置） | `train\|serve\|eval\|vllm` | 必填 | install a task environment（`run.py:14`） |
| `setup` | `--platform` | `gpu\|ppu` | 无 | required for the vLLM environment（`run.py:15`） |
| `train` | `--config` | str（互斥组） | 无 | project-relative training launch YAML（`run.py:18`） |
| `train` | `--resume` | flag（互斥组） | False | resume from the configured training checkpoint（`run.py:19`） |
| `serve` | `--engine` | `sglang\|vllm` | 无→`sglang` | default: sglang; vllm serves the VLM checkpoint only（`run.py:21`） |
| `serve` | `--platform` | `gpu\|ppu` | 无 | vLLM platform; defaults to its environment runtime.json（`run.py:22`） |
| `serve` | `--config` | str（互斥组） | 无 | project-relative service launch YAML（`run.py:24`） |
| `serve` | `--decoder` | `denoise\|causal\|speculative`（互斥组） | 无 | SGLang decoder (default: entropy-guided denoise)（`run.py:25`） |
| `eval` | `--config` | str（互斥组） | 无 | project-relative evaluation YAML（`run.py:28`） |
| `eval` | `--decoder` | `denoise\|causal\|speculative`（互斥组） | 无 | request policy matching the running service (default: denoise)（`run.py:29`） |
| `prepare-model` | `--base` | str | `weights/base_model` | （`run.py:32`） |
| `prepare-model` | `--source` | str | `weights/dlm` | （`run.py:33`） |
| `prepare-model` | `--output` | str | `weights/dlm_bundle` | new project-relative model directory（`run.py:34`） |
| 全部子命令 | `--venv` | str | `.venv-<profile>` | project-relative environment directory（`run.py:37`） |
| 全部子命令 | `--dry-run` | flag | False | print the downstream command without executing it（`run.py:38`） |

- **默认配置选择逻辑**（`run.py:112-122`）：
  - `train`：`--resume` → `configs/release/dlm_resume_h800_smoke.yaml`；否则 `configs/release/dlm_train_h800_smoke.yaml`（`run.py:115`）。
  - `serve`：vLLM → `configs/release/vlm_vllm_<platform>.yaml`；SGLang → `configs/release/dlm_sglang_<decoder|denoise>.yaml`（`run.py:117`）。
  - `eval`：`decoder ∈ {None, denoise}` → `configs/eval/dlm.yaml`，否则 `configs/eval/dlm_<decoder>.yaml`（`run.py:119`）。
  - 下游脚本：`eval` → `scripts/evaluate.py`，其余 → `scripts/run.py`（`run.py:123`）。
- **I/O 约定**：所有路径必须是工程内相对路径；`relative()`（`run.py:41-46`）拒绝绝对路径、`..`、以及解析后逃出 root 的路径，并返回 POSIX 风格字符串。
- **⚠️ 关键风险（见 R-01）**：`python = str(root / venv / 'bin/python')`（`run.py:98`）是 **POSIX 布局**；Windows 下对应 `Scripts\python.exe`，因此 `run.py` 在本地必然报 `environment missing`（`run.py:125-129`）。

### A2. `scripts/run.py`（7762 B，137 行）——服务/训练的 YAML 启动器

- **职责**：读一个 launch YAML，做 CPU-only 校验，拼出真实命令并 `subprocess.call`（`scripts/run.py:135`）。
- **顶层常量**：`ROOT`（`scripts/run.py:11`）、`TARGETS` 映射表（`scripts/run.py:15`）：

| entrypoint | 目标模块 |
| --- | --- |
| `dlm-sglang` | `-m infer.serve_sglang` |
| `vlm-sglang` | `-m infer.serve_sglang_vlm` |
| `vlm-vllm` | `-m infer.serve_vllm` |
| `dlm-train` | `-m train.dlm.train_dlm` |
| `vlm-train` | `-m train.vlm.train_vlm_lora` |
| `dlm-serve` | `-m infer.dlm.qwen3_openai_server` |
| `eval` | `-m eval.eval_runner` |
| `eval-yaml` | `scripts/evaluate.py` |
| `rl-24` | `-m train.rl.current.trainer` |
| `rl-56` | `-m train.rl.distributed.multiroute_56` |
| `rl-64` | `-m train.rl.distributed.multiroute_64` |

- **参数契约**（`scripts/run.py:90-93`）：`config`（位置，工程相对 YAML）、`--dry-run`（flag）。**只有一个位置参数**，没有其它开关。
- **launch YAML 允许的键**（白名单，`scripts/run.py:96`）：`entrypoint, args, env, inputs, outputs, native_config, distributed, checkpoint`；出现其它键直接 `ValueError`。
- **`distributed` 子键白名单**（`scripts/run.py:52`、`141-143`）：`nproc_per_node, nnodes, node_rank, master_addr, master_port`。
- **`checkpoint` 子键白名单**（`scripts/config_contract.py:98`）：`initial_dlm_checkpoint, resume_from_checkpoint`，二者互斥（`config_contract.py:100-101`），且只允许训练类 entrypoint（`config_contract.py:102-103`）。
- **数据流**：`yaml.safe_load` → 键白名单 → `validate_tree` 路径合规 → 拼命令 → 打印 `{"command","env","missing_inputs"}` JSON（`scripts/run.py:123`）→ `--dry-run` 时直接返回 → 非 dry-run 且 `missing_inputs` 非空则 `SystemExit("Provide missing inputs before execution.")`（`scripts/run.py:125`）→ 否则 `subprocess.call`。
- **训练路线的强约束**：
  - 训练参数**只能**来自 `native_config`（`scripts/run.py:44-45` 禁止 `args` 覆盖）。
  - `native_config` 必须是工程相对 YAML（`scripts/run.py:46-48`）。
  - `expected_world_size` 与 `nproc_per_node*nnodes` 必须相等（`scripts/run.py:57-59`）。
  - `model.family` 必须是 `groundinganything_qwen3`（DLM）或 `groundinganything_vlm`（VLM）（`scripts/run.py:64-66`）。
  - DLM 走 `torch.distributed.run`，单机自动加 `--standalone`（`scripts/run.py:79-81`）；VLM 直接把拓扑参数传给 `train_vlm_lora.py`（`scripts/run.py:68-76`）。
  - RL 路线（`rl-24/56/64`）**强制 world_size = 24/56/64**（`config_contract.py:126,145-147`），数据路径只能来自 `native_config.data.{grounding_path,ocr_path,multiroute_path}`（`config_contract.py:113-121,136-137`）。
- **I/O 约定**：`env` 里的 `IMAGE_MAX_TOKEN_NUM` 若与 native runtime 不一致会报错，否则被 native 的值覆盖（`scripts/run.py:110-116`）。真实执行时 `PYTHONPATH = ROOT + os.pathsep + 原值`（`scripts/run.py:128`）。
- **安全边界自述**：`config_contract.py:1` 明确写「CPU-only configuration checks. These checks are not a filesystem sandbox.」

### A3. `scripts/evaluate.py`（16175 B，277 行）——评测的唯一严格前端

- **职责**：校验评测 YAML、构造 `eval.eval_runner` 命令与全套 `GAM_*` 环境变量、监督子进程、写 `summary.json`。
- **顶层常量**：
  - `MODES = ('VLM','GAM','DLM','RLV2','REXOMNI','LOCATEANYTHING','GROUNDINGDINO')`（`scripts/evaluate.py:21`）
  - `FIELDS` = 评测 YAML 允许的 21 个键（`scripts/evaluate.py:22-24`）：`datasets, mode, model_path, model_type, api_url, data_root, output_root, tasks, run_id, limit, concurrency, coordinate_mode, locate_generation_mode, rex_tokenizer_path, base_port, model_id, max_tokens, service_contract, decoder`
  - 必填键（`scripts/evaluate.py:42`）：`mode, model_path, api_url, data_root, output_root, tasks, run_id`
- **参数契约**（`scripts/evaluate.py:246-249`）：`config`（位置，工程相对 YAML）、`--dry-run`（flag）。**没有其它开关**；所有实验设置都在 YAML 里。
- **关键校验**（逐条给行号）：
  - `model_id` 需匹配 `[A-Za-z0-9][A-Za-z0-9_./-]{0,199}`（`:46`），默认取 `model_path`（`:45`）。
  - `max_tokens` ∈ 1..65536（`:49`）。
  - `service_contract ∈ {openai, native}`，默认 `openai`（`:51-53`）。
  - `decoder` 仅允许 GAM/DLM/RLV2 + `openai` 契约 + `model_type=qwen3vl` + `api_url` 以 `/v1` 结尾（`:57-68`）；`denoise` 不允许与 `max_tokens` 同用（`:69-70`）。
  - `run_id` 需匹配 `[A-Za-z0-9][A-Za-z0-9_-]{0,79}`（`:72`）。
  - **输出目录必须与输入目录不重叠**（`:78-81`、`:126-129`）。
  - `tasks` 必须是 `eval/**/*.yaml` 里 `task:` 字段的去重非空子集（`:87-90`、`task_inventory` 在 `:26-37`）。
  - `concurrency` ∈ 1..256（默认 4，`:94,96-97`）；`base_port` ∈ 1024..65535-len(tasks)（默认 12345，`:95,98-99`）。
  - `coordinate_mode`：VLM/GROUNDINGDINO **必须显式**给 `qwen2|qwen3|norm01|abs`，GROUNDINGDINO 只能是 `qwen3`；其它模式必须省略（`:100-107`）。
  - `locate_generation_mode ∈ {fast,slow,hybrid}` 且仅 LOCATEANYTHING 可用（`:108-113`）。
  - `rex_tokenizer_path` 仅 REXOMNI 可用且必填（`:114-120`）。
  - `model_type ∈ {qwen25vl,qwen3vl,qwen35,deepseekvl2}`，默认 `qwen3vl`（`:121-123`）。
- **写入的产物**（I/O 约定）：
  - 输出目录 `output_root/run_id`，**必须不存在**，用 `mkdir(exist_ok=False)`（`scripts/evaluate.py:272`）。
  - `run.json`：`{"config","plan","service"}`（`:273`）。
  - `summary.json`：先写 `.summary.json.tmp` 再 `replace`（原子替换，`:239-241`）；字段 `records/missing_tasks/returncode/status/termination_signal/error_type/runner_cleanup_timed_out/mode/evaluation_scope/config_sha256`（`:235-238`）。
  - `responses.jsonl`：响应审计文件，路径由 `GAM_EVAL_RESPONSE_AUDIT_PATH` 指定（`:178`）。
  - `run.json`/`summary.json` 在 `summarize()` 里被显式跳过，不当作指标来源（`:185`）。
- **状态机**（`execute_evaluation`，`:194-242`）：`interrupted`（收到信号）→ `completed`（code 0）→ `failed`；清理超时则强制 `cleanup_failed` + code 1（`:234`）。`code == 0` 但 `missing_tasks` 非空时返回 2（`:232`）。
- **环境变量契约**（`plan()` 生成，`:130-146`）：`EVAL_DATA_PATHS`（JSON）、`GAM_EVAL_DATA_ROOT`、`GAM_EVAL_LOG_ROOT`、`GAM_EVAL_API_CONCURRENCY_PER_TASK`、`GAM_EVAL_MODE`、`HF_HOME=outputs/cache/huggingface`、`GAM_EVAL_MODEL_ID`、`GAM_OPENAI_BASE_URLS`、`GAM_IMAGE_ROOT`、`GAM_MAX_IMAGE_PIXELS=2560*28*28`、`GAM_METRIC_MODULE=eval/metrics/detection_metrics.py`；条件项 `GAM_EVAL_MAX_TOKENS_OVERRIDE`、`GAM_EVAL_LIMIT`、`GAM_COORD_MODE`、`GAM_LOCATEANYTHING_GENERATION_MODE`、`GAM_REXOMNI_TOKENIZER_PATH`、`GAM_EVAL_DECODER`（denoise 时追加 `GAM_DLM_DECODE_PROFILE=task_profiles`、`GAM_DLM_SINGLE_TARGET_RULE=1`）。
- **环境清理规则**（`evaluation_environment`，`:169-179`）：继承的进程环境里所有以 `GAM_ / EVAL_ / ROBOSPATIAL_ / SCREENSPOT_ / LMMS_EVAL_` 开头的键以及 `LM_HARNESS_CACHE_PATH` 全部剔除，再套用 YAML 的值——即 **YAML 是唯一权威**。`PYTHONPATH` 固定为 `vendor/lmms-eval:<root>`（`:176`）。
- **前置依赖检查**（`main()`，`:260-270`）：`verify_dependency('lmms_eval')` → 要求 `vendor/lmms-eval` 在 `sys.path` → `importlib.util.find_spec` 检查 `lmms_eval` 与 `accelerate` → `get_model('async_openai')` → `eval_preflight.preflight()`。**任一步失败都在建输出目录之前退出**（`:268-272`）。
- **`--dry-run` 产物**：只打印 `prepared` JSON，不建目录（`:252-253`）。

### A4. `eval/eval_runner.py`（34529 B，867 行）——并行评测引擎

- **职责**：把 N 个任务各起一个 `lmms-eval` 子进程，分配端口/CPU，监控，汇总退出码。
- **顶层常量**：`GAM_EVAL_ROOT`（`:38`）、`EVAL_LOG_ROOT = $GAM_EVAL_LOG_ROOT` 或 `<root>/outputs/eval`（`:39`）。
- **主要类/函数签名**：
  - `@dataclass TaskProcess(task_name, process, port, cpu_count, log_file, log_fd, start_time)`（`:41-50`）
  - `class ProcessMonitor(tasks, check_interval=60.0)`（`:53`），`start_monitoring/stop_monitoring/_monitor_loop/print_status`（`:63-97`）
  - `apply_smoke_generation_overrides(gen_kwargs) -> Dict`（`:126`）
  - `apply_mode_generation_contract(gen_kwargs, mode) -> Dict`（`:147`）
  - `apply_model_family_generation_contract(...)`（`:298`）
  - `run_kill_eval_script() -> int`（`:341`）
  - `setup_environment(...)`（`:383`）
  - `parse_tasks(tasks_str, prompt_optim=False) -> List[str]`（`:445`）
  - `calculate_cpu_allocation(tasks, model_eval_config, available_cpu) -> Dict[str,int]`（`:461`）
  - `start_eval_process(model_type, task, port, cpu_count, model_path, log_dir, log_dir_timestamp, cache_dir, env, model_eval_config, include_path=None) -> (Popen, str)`（`:507-519`）
  - `cleanup_task_processes(tasks, grace_seconds=2.0)`（`:633`）
  - `run_parallel_eval(...)`（`:659`）、`main()`（`:814`）
- **参数契约**（`main()`，`:816-844`）：

| 参数 | 类型 | 默认 | help |
| --- | --- | --- | --- |
| `--model_type` | str | `qwen3vl` | 模型类型 |
| `--model_path` | str | None | 模型路径 |
| `--tasks` | str | `ocrbench` | 逗号分隔的任务列表 |
| `--job_id` | str | None | 任务ID，用于日志目录命名 |
| `--base_port` | int | 12345 | 基础端口号 |
| `--cpu_ratio` | float | 0.8 | CPU使用比例（0-1） |
| `--start_delay` | float | 2.0 | 启动任务之间的延迟（秒） |
| `--api_url` | str | None | OpenAI-compatible API base URL；外部凭据应由安全代理持有 |
| `--prompt_optim` | flag | False | 使用与训练数据对齐的优化 prompt |
| `--include_path` | str | None | lmms_eval --include_path |
| `--skip_cleanup` | flag | False | 跳过任务后的进程清理 |
| `--mode` | str.upper，choices=`VALID_EVAL_MODES` | `VLM` | VLM/GAM/DLM/RLV2/REXOMNI/LOCATEANYTHING/GROUNDINGDINO 评测协议 |

- **I/O 约定**：
  - 每个任务一个日志：`<log_dir_timestamp>/<task>.log`（`:527`）。
  - 每任务一个 lmms-eval sqlite 缓存：`<cache_dir>/sqlite_cache_<task><suffix>.db`，suffix 由 mode/decoder/locate 模式拼接（`:533-542`）——**同一 `job_id` 复用会命中旧缓存**，代码注释明确说 GAM 用独立文件名规避（`:530-532`）。
  - 像素范围：`task_pixel_min_max`（`:545-551`）。
  - 生成参数：`generation_kwargs`，缺失时回退 `DEFAULT_GENERATION_KWARGS`（`:553-556`）。
  - `GAM_EVAL_API_CONCURRENCY_PER_TASK` 会**覆盖** CPU 分配数（`:521-525`）。
  - `GAM_EVAL_CPU_WEIGHT_OVERRIDES` 接受 JSON dict 覆盖每任务 CPU 权重（`:467-482`）。
- **未读部分**：`:100-440`（`apply_*` 三个 generation contract 的具体分支）、`:560-813`（进程启动/监控/清理细节）。结论不依赖它们，但**具体采样参数覆盖表未逐条核对**。

### A5. `eval/task_config.py`（7199 B，184 行）——逐任务 CPU/像素/生成参数表

- **职责**：给出 39 个 `gam_*` 任务的 `cpu_num`、`task_pixel_min_max`、`generation_kwargs`。
- **顶层常量**：
  - `_TEXT_STOP = ["\n\n"]`（`:20`）
  - `_LOW_DENSITY_GEN`：`repetition_penalty=1.0, top_p=1.0, temperature=0.0, max_tokens=4096, until=["\n\n"], enable_thinking=False`（`:22-29`）
  - `_HIGH_DENSITY_GEN`：`repetition_penalty=1.05, top_p=0.9, temperature=0.2, max_tokens=8192, until=["\n\n"], enable_thinking=False`（`:31-38`）
  - 像素档位：`_DETECTION_PIXELS=[12544, 2007040]`（`:55`，=16\*28²..2560\*28²）、`_POINTING_PIXELS=[100352, 4000014080]`（`:56`）
  - 兜底：`DEFAULT_CPU_NUM=1`、`DEFAULT_TASK_PIXEL_MIN_MAX=[1,40014080]`、`DEFAULT_GENERATION_KWARGS`（`:163-173`）
  - `MODELS_TASK_CONFIG`：`qwen25vl/qwen3vl/qwen35/deepseekvl2` 全部共享 `qwen3vl_task_config`（`:179-184`）
- **参数契约**：本文件无 CLI；`_task_cfg(cpu_num, pixel_range, generation_kwargs, **generation_overrides)`（`:41`）是唯一构造器。
- **任务分组**：Referring/Grounding/Dense（cpu 8）、Robo/Referring/Grounding/Dense point-in-mask（cpu 2）、GUI（cpu 2，`num_beams=1`）、OCR（cpu 4，`until=[]` + `skip_special_tokens=False`）、Layout（cpu 4）、VisualPrompt（cpu 4，高密度档）。逐项在 `:107-161`。
- **注意**：`top_k` 被刻意禁用（`:13-14` 注释：缺省即 None/不截断，避免发送 JSON null）。

### A6. `eval/process_control.py`（2711 B，87 行）——协作式取消

- **职责**：SIGINT/SIGTERM → `TerminationRequested` 异常；临界区延迟投递；清理期屏蔽信号。
- **签名**：`TerminationRequested(signum)`（`:11`）、`termination_handlers()`（`:17`）、`defer_cancellation()`（`:43`）、`finish_cleanup()`（`:65`）、`cancellation_exit(function)`（`:79`）。
- **I/O**：无文件 IO。非主线程直接空操作（`:19-21`、`:68-70`）。
- **自述边界**：`process_control.py:1` — 「SIGKILL and machine loss cannot run cleanup.」
- **Windows 相关性**：`os.killpg`（`scripts/evaluate.py:222`）在 Windows 不可用；`start_new_session=True`（`:203`）在 Windows 语义不同 → 见 R-02。

### A7. `grounding_anything/client.py`（5303 B，118 行）——最小 HTTP 客户端

- **职责**：连一个已在运行的服务，发 OpenAI 兼容 chat/completions 请求，解析成 `Result`。
- **顶层常量/数据类**：`Prediction(label, coordinates)`（`:16-19`）、`Result(task, raw_output, predictions, finish_reason, usage, parse_error)` + `valid` 属性（`finish_reason=="stop"` 且无 `parse_error`）（`:22-36`）。
- **签名**：
  - `parse_response(response: dict, task: str) -> Result`（`:39`）
  - `GroundingAnything.__init__(base_url="http://127.0.0.1:8101/v1", model="groundinganything", *, api_key=None, timeout=120)`（`:65`）
  - `GroundingAnything.predict(image: str|Path, phrase: str, *, task="bbox", max_tokens=4096) -> Result`（`:77`）
  - `visualize(image, result) -> PIL.Image`（`:101`）
- **参数契约**：无 CLI；构造函数即契约。`task ∈ {bbox, point}`（`:40,78`）；图片扩展名只允许 JPEG/PNG/WebP（`:85-86`）；`timeout` 必须有限且为正（`:70-71`）；`base_url` 不允许 credentials/query/fragment（`:67-69`）。
- **I/O 约定**：读本地图片文件并 base64 内联（`:87-92`）；写：无。坐标使用 **canonical 0..999 网格**（`:102`、`:111`）。
- **严格性**：`finish_reason != "stop"` 时预测强制清空（`:55-58`）——杜绝「部分输出被当成成功预测」。

### A8. `grounding_anything/protocol.py`（32808 B）——prompt/解析协议（仅概括）

- **职责**：构造 refer bbox / refer point prompt 并解析模型输出为结构化实体（`client.py:13` 只用到 `build_refer_bbox_prompt`、`build_refer_point_prompt`、`parse_answer`）。
- **未逐行读**（32808 B）。`parse_answer(raw, task=..., require_canonical=True)` 在 `client.py:49` 调用，`require_canonical` 语义未确认。

### A9. `eval/utils/eval_data_root.py`（4538 B，102 行）——数据集根路径解析

- **职责**：把 `configs/datasets.yaml` 的键解析成绝对路径，并支持 `${EVAL_DATA:<key>}` 令牌展开。
- **关键常量**：`ROOT`（`:12`）、`TOKEN = \$\{EVAL_DATA:([a-zA-Z0-9_]+)\}`（`:13`）。
- **签名**：`input_path(base,value)->Path`（`:16`，**允许绝对路径与 `~`**）、`load_data_paths(root,data_root,config_path='configs/datasets.yaml')->dict`（`:24`）、`configured_paths()->dict`（`:44`，优先读 `EVAL_DATA_PATHS` 环境变量）、`dataset_path(key,*parts)->Path`（`:51`）、`expand_data_paths(value)`（`:58`）、`task_data_inputs(root,tasks,inventory,paths)->list[str]`（`:74`）、`missing_data_inputs(inputs)->list[str]`（`:101`）。
- **强制单路径的键**（`:35`）：`images, screenspot_pro, screenspot_v2, osworld_g, refspatial, refcoco, refcocoplus, refcocog, icdar2015_dontcare`。
- **隐式键注入**：`gam_osworld_g` → `osworld_g`；`gam_icdar2015` → `icdar2015_dontcare`；任务 YAML 里出现 `eval/rexomni_jsonl.py` → `images`（`:91-95`）。
- **安全边界**：任务 `include` 不能逃出 `eval/`（`:81-82`）；`input_path` **不**做工程内限制（`datasets.yaml:1` 注释也说「Absolute paths are also accepted」）。
- **注意**：`missing_data_inputs` 用 `glob.glob` 判断存在性（`:102`），因此 `configs/datasets.yaml:34` 的 `ScreenSpot-Pro/annotations/*.json` 是通配符语义。

### A10. `configs/datasets.yaml`（2026 B，41 行）——数据集键 → 相对 data_root 的路径

全部键值见文件本身；要点：
- `images: images`（`:3`）是默认图片根。
- 两个嵌套 key：`gam_screenspot_v2_1/2/3`（`:35-37`）。
- 通配符：`gam_screenspot_pro: ScreenSpot-Pro/annotations/*.json`（`:34`）。
- Parquet/JSON 混合：`gam_robospatial_context` 用 `.parquet`（`:32`）。
- 注释声明「no dataset is downloaded or modified」（`:2`）——仓库**不下载数据**。

---

## B. 推理层 `infer/`

**总览**（`infer/README.md:5-9`）：`decode/` = DecodeV4 请求参数/任务分层/结构约束；`dlm/` = 原生 DLM HTTP 服务与 causal/speculative/hierarchy 解码及缓存；`engines/` = 默认服务用的自定义 SGLang 模型/采样/执行适配器。**模型权重另行提供**（`infer/README.md:9`）。

### B1. 服务入口

| 文件 | 职责 | CLI 参数（全部默认值） |
| --- | --- | --- |
| `infer/serve_sglang.py`（109 行） | 启动 bundled SGLang + DLM 插件（默认服务） | `--model`(必填,prepared DLM bundle)、`--decoder`(`denoise`)、`--host`(`127.0.0.1`)、`--port`(`8101`)、`--served-model-name`(`groundinganything`)、`--output`(`outputs/sglang`)（`:60-65`） |
| `infer/serve_sglang_vlm.py`（91 行） | 用同一 SGLang 起 **VLM causal** 检查点（8102） | `--model`(必填)、`--output`(`outputs/sglang/vlm`)、`--host`(`127.0.0.1`)、`--port`(`8102`)、`--served-model-name`(`groundinganything-vlm`)（`:17-21`） |
| `infer/serve_vllm.py`（88 行） | 用平台 vLLM 起 VLM（gpu/ppu） | `--model`(必填)、`--overlay`(`outputs/vllm-vlm`)、`--platform`(必填,`ppu\|gpu`)、`--host`(`127.0.0.1`)、`--port`(`8102`)、`--served-model-name`(`groundinganything-vlm`)、`--max-model-len`(`16384`)、`--max-num-seqs`(choices=(1,),`1`)、`--gpu-memory-utilization`(`0.7`)、`--tensor-parallel-size`(choices=(1,),`1`)（`:16-25`） |
| `infer/sglang_server.py`（36 行） | monkey-patch bundled SGLang 的 `/server_info` 就绪性后启动 | 无自有 argparse；透传 `sys.argv[1:]` 给 `prepare_server_args`（`:28`） |

**B1 关键事实**

- `serve_sglang.build_launch(model, output, decoder, mask, vocab, eos, paths, inherited, host, port, served_model_name)`（`infer/serve_sglang.py:12-13`）**不加载 CUDA、不写文件**（`:14` 注释），纯拼命令+环境，便于 CPU 预检。
- 环境清洗规则：继承环境里所有 `GAM_*`、`SGLANG_*` 被剔除（`infer/serve_sglang.py:20-21`），再显式注入 20 余个变量（`:22-38`），其中 `SGLANG_EXTERNAL_MODEL_PACKAGE="sglang_gam"`、`GAM_SGLANG_ALGORITHM="decode_v2"`（denoise 时）、`HF_HUB_OFFLINE="1"`、`TRANSFORMERS_OFFLINE="1"`、`GAM_SGLANG_DISABLE_CUDA_GRAPH="1"`、`GAM_SGLANG_VISION_ATTN="eager"`。`decoder=speculative` 时额外写 `outputs/sglang/speculative-trace.jsonl`（`:34`）；`denoise` 时强制 `GAM_DLM_DECODE_PROFILE=task_profiles`、`GAM_DLM_SINGLE_TARGET_RULE=1`（`:37`）。
- 启动命令行（`infer/serve_sglang.py:42-54`）：`--device cuda --dtype bfloat16 --tensor-parallel-size 1 --trust-remote-code --enable-multimodal --attention-backend triton --sampling-backend pytorch --grammar-backend none --mem-fraction-static 0.65 --max-total-tokens 32768 --max-running-requests 1 --max-queued-requests 2 --disable-radix-cache --chunked-prefill-size 16384 --disable-cuda-graph --disable-overlap-schedule`，并读 `model/chat_template.jinja`（`:47`）。**这些是硬编码，不可通过 CLI 改。**
- 模型入场校验（`infer/serve_sglang.py:82-89`）：tokenizer 必须有原子 `|<MASK>|` token；`model.safetensors` 里 `base_model.model.language_model.embed_tokens.weight` 的行数必须等于 `len(tokenizer)`，且 `mask_id == vocab-1`。
- 写盘产物：`<output>/engine_runtime.json`（`:103`）与 `<output>/<decoder>.yaml`（算法配置，`:17`、`:97`）；最后 `os.execve`（`:105`）。
- `serve_sglang_vlm` 复用 `build_launch(..., "causal", -1, vocab, ...)`（`:64-68`）并覆写 `GAM_SGLANG_WEIGHT_PREFIX=""`、`GAM_SGLANG_MODEL_KIND="vlm"`、`GAM_SGLANG_LOGITS_VOCAB_SIZE=vocab`（`:69-71`）。
- `serve_vllm` 硬门槛：`transformers.__version__ == "5.7.0"` 且 import 路径在 `vendor/transformers/src` 下（`:60-61`）；`vllm` 版本必须 `0.18.*` 且 ppu/gpu 与版本名一致（`:63-64`）；`torch.cuda.is_available()` 必须为真（`:65-66`）；设备名含 "PPU" 必须与 `--platform ppu` 一致（`:67-69`）。**这三条决定了本机（无 GPU）绝对无法起 vLLM。**
- `serve_vllm.build_command` 强制 `--model-impl transformers --enforce-eager --limit-mm-per-prompt {"image":1,"video":0}`（`infer/serve_vllm.py:36-43`）。
- overlay 产物：`<overlay>/engine_runtime.json`（`infer/serve_vllm.py:82`）；overlay 目录本身由 `infer/overlay/prepare_vllm_overlay.py` 的 `source_files/create_overlay/verify_overlay` 维护（`:29/67/107`）。

### B2. 契约层（无模型依赖，可在 CPU 上跑）

- **`infer/decoding.py`（64 行）**：公共 decoder 选择。
  - `ALGORITHMS = {'denoise': 'GAMDecodeV2Block', 'causal': None, 'speculative': 'GAMSpeculativeBlock'}`（`:14-15`）。
  - `PROFILE_PATH = ROOT/infer/decode/configs/task_profiles.json`（`:16`）。
  - `decoder_settings(decoder)`（`:19`）：denoise 读 task_profiles.json 的 `shared_decode` 并强制 `token_shift=1, temperature=0.0, top_p=1.0`（`:27`），返回 `task_profile_sha256`（`:30`）；speculative 固定 `{'block_size':32,'token_shift':1,'debug':False}`（`:32`）。
  - `evaluation_generation(task, generation, decoder)`（`:36`）：denoise 要求环境变量 `GAM_DLM_DECODE_PROFILE == 'task_profiles'`，否则报错（`:41-42`）。
  - `validate_server_decoder(decoder, info)`（`:54`）：要求 SGLang `/server_info` 暴露 `dllm_algorithm` 且等于期望值（`:57-61`）。
- **`infer/request_contract.py`（18 行）**：`validate_thinking(body, non_thinking)`（`:3`，`chat_template_kwargs` 只允许布尔 `enable_thinking`）、`validate_budget(requested, cap, prompt_tokens=None, context=None)`（`:13`，`prompt+max_tokens` 不得超 `max_model_len`）。
- **`infer/decode/config.py`（179 行）**：`@dataclass(frozen=True) DecodeConfig`，**41 个字段**，默认值全在 `:29-70`：`block_size=32, sub_block_size=4, denoise_steps=4, token_shift=1, temperature=0.0, top_p=1.0, top_k=0, sampling_seed=None, acceptance_policy="confidence", confidence_threshold=0.99, entropy_threshold=None, position_penalty=0.0, repetition_mode="none", ..., enable_eos_early_stop=False, ..., expected_cardinality=None, box_*/coord_* 5 个 token id, coordinates_per_target=None, legacy_force_fill_all=False, task=None, profile=None`。
  - **优先级**：先 server YAML，再 request 的 `custom_params.gam_dlm_decode`（`:5-6`、`:82`）。
  - 别名兼容：`threshold` → `confidence_threshold`（`:85-86`）；`debug`、`use_AR_for_first_token` 被忽略（`:87`）。
  - `ACCEPTANCE_POLICIES = {confidence, entropy, entropy_position, entropy_position_repetition}`（`:16-23`）；`REPETITION_MODES = {none, ngram, block}`（`:24`）。
  - `request_decode_mapping(custom_params)`（`:169`）**不接受别名**（`:170`）。
- **`infer/decode/request_contract.py`（65 行）**：lmms-eval 的逗号分隔 `--gen_kwargs` 只能传标量，故用 `PREFIX = "gam_dlm_decode_"` + 30 个 `SCALAR_FIELDS`（`:8-45`）桥接成 SGLang custom object；`pack_decode_custom_params`（`:48`）、`unpacked_generation_fields`（`:59`，遇到非白名单字段直接报错 `:63-64`）。
- **`infer/dlm/config.py`（87 行）**：`@dataclass(frozen=True) DLMInferenceConfig`，`mode ∈ {hierarchy_step, hierarchy_dynamic, speculative}`（默认 `speculative`），字段默认见 `:22-33`；`validate(trained_block_size)`（`:35`）逐条约束（`block_size ∈ [2, trained]`、`denoise_steps ∈ [1, block_size-1]`、prefix cache / full block / 结构化终止仅 Hierarchy 模式可用等）。
- **`infer/dlm/model_contract.py`（11 行）**：`validate_qwen3_base_config(model_dir)` 要求 `config.json` 的 `model_type ∈ {groundinganything, groundinganything_vlm}` 且 `text_config.model_type == "qwen3"`（`:9-10`）。
- **`infer/dlm/prompt_contract.py`**：`_user_texts`（`:22`）、`infer_coordinate_group_width(messages)`（`:43`）。
- **`infer/dlm/stop_contract.py`**：`normalize_openai_stop(stop)`（`:9`）、`single_token_stop_ids`（`:28`）、`token_stop_sequences`（`:50`）、`first_token_sequence_match`（`:92`）。

### B3. `infer/decode/` 算法与遥测

| 文件 | 职责 | 关键符号（行号） |
| --- | --- | --- |
| `sglang_algorithm.py`（25753 B） | 默认 `denoise` 算法主体 | `class GAMDecodeV2Block(DllmAlgorithm)`（`:37`） |
| `sampling.py` | 采样 | `sample_tokens`（`:12`）、`best_non_token`（`:51`） |
| `reliability.py` | 置信度/熵可靠性 | `RawReliability`（`:12`）、`raw_reliability`（`:20`）、`raw_reliability_masked`（`:60`）、`adjusted_entropy`（`:129`）、`normal_acceptance`（`:158`）、`most_reliable_position`（`:172`） |
| `repetition.py` | 重复风险 | `repetition_risks`（`:60`） |
| `structure.py` | 结构约束 | `ordered_eos_cut`（`:10`）、`pending_eos_cut`（`:31`）、`cardinality_cutoff`（`:50`） |
| `decode_ops.py` | 张量移位 | `shifted_local_logits`（`:14`）、`shifted_argmax`（`:41`） |
| `telemetry.py`（12527 B） | 遥测 | `BlockTelemetry`（`:17`）、`TelemetryWriter`（`:234`）、`telemetry_from_environment`（`:287`） |
| `task_profiles.py`（6896 B） | 任务 → profile | `decoder_profile(name, task)`（`:213`） |
| `configs/task_profiles.json`（2839 B） | DecodeV4 profile 表 | `schema_version, name, revision="DecodeV4", checkpoint_scope, activation, shared_decode, tiers, invariants`；`shared_decode = {acceptance_policy:"entropy", entropy_threshold:0.8, top_k:0, position_penalty:0.0, repetition_mode:"none", repetition_weight:0.0, enable_eos_early_stop:false, block_size:32, sub_block_size:4, denoise_steps:4, legacy_force_fill_all:false, sampling_seed:42}` |
| `configs/default.yaml`（938 B） | 框架默认（保守） | `block_size:32, sub_block_size:4, denoise_steps:4, token_shift:1, temperature:0.0, top_p:1.0, top_k:0, acceptance_policy:confidence, confidence_threshold:0.99, repetition_mode:none`；`repetition_ignore_token_ids` 显式列出 `7 '('、8 ')'、11 ','、58 '['、60 ']'、151646-151648 (object_ref/box start/end)` |
| `configs/compatibility.yaml`（693 B） | 仅供 plumbing 回归的 A0 兼容路线（注释 `:1-3`） | 与 default 同构，但 `legacy_force_fill_all: true` |

### B4. `infer/dlm/` 原生 DLM 服务与解码（大体量）

| 文件 | 大小 | 关键符号 |
| --- | --- | --- |
| `hierarchy.py` | 54591 B | `_trace_hierarchy_logits`(23)、`_enforce_single_bbox_grammar`(80)、`_prefix_safe_grammar_acceptance`(176)、`_suppress_invalid_terminations`(221)、`_suppress_repeated_ngrams`(430)、`class HierarchyBlockDecoder`(494) |
| `prefix_cache.py` | 31828 B | `_prefix_flex_attention_kernel`(46)、`PrefixCacheContractError`(69)、`LinearPrefixState`(180)、`CleanAttentionPrefixState`(189)、`PrefixCacheMutationStamp`(198)、`ForkedLinearPrefixState`(210)、`HybridBlockPrefixCache`(219)、`linear_state_map`(429)、`Qwen35PrefixDraftRunner`(441) |
| `build_public_model_bundle.py` | 22890 B | CLI：`--kind {vlm,dlm}`(必填)、`--label`(必填)、`--source`(必填)、`--base`、`--destination`(必填)（`:484-488`）；函数 `sha256`(79)、`sanitize_json_value`(100)、`build_vlm`(173)、`build_dlm`(258)、`validate_bundle`(309)、`sensitive_scan`(394)、`write_release_docs`(408)、`write_checksums`(474) |
| `openai_server.py` | 19343 B | `DEFAULT_MODE = os.environ.get("DLM_INFERENCE_MODE","speculative")`(34)、`BLOCK_SIZE = int(os.environ.get("DLM_INFERENCE_BLOCK_SIZE","32"))`(36)、`run_inference`(196)、`main`(384) |
| `legacy_backend.py` | 21558 B | `load_model`(98)、`_decode_image_url`(162)、`_prepare_inputs`(189)、`run_inference`(209)、`class Handler(BaseHTTPRequestHandler)`(300)、`main`(486) |
| `qwen3_openai_server.py` | 5490 B | `load_qwen3_model`(33)、`prepare_qwen3_request`(106)、`main`(115)（`dlm-serve` entrypoint） |
| `selection.py` | 5306 B | `filter_logits`(10)、`predictions_and_confidence`(58)、`step_acceptance`(83)、`dynamic_acceptance`(112)、`entropy_acceptance`(131) |
| `engine.py` | 1803 B | `class DLMInferenceEngine`(13) |
| `causal.py` | 4549 B | `causal_generate_cached(...)`(13) |
| `qwen3_inputs.py` | 1753 B | `prepare_qwen3_inputs(processor, messages)`(5) |

> **未读**：本报告未逐行读 `hierarchy.py`、`prefix_cache.py`、`openai_server.py`、`legacy_backend.py`、`build_public_model_bundle.py` 的算法细节。已读范围仅限文件大小、符号表与 CLI 契约。

### B5. `infer/engines/` 自定义 SGLang 适配层

- `dlm_task_profiles.py`（26328 B）：denoise 路线的任务分层生成契约。核心 `build_dlm_task_config`（`:569`）、`profile_summary`（`:645`）；内部分层命名 `_three_tier_profile_name`(253)、`_four_tier`(276)、`_five_tier`(300)；环境变量读取器 `_referring_max_tokens`(358)、`_single_target_rule_enabled`(369)、`_optional_float`(376)、`_optional_int`(389)、`_apply_dlm_sampling_overrides`(402)、`_decode_profile`(418)。
- `sglang_gam/__init__.py`（10980 B）：`_mask_id_from_tokenizer`(35)、`configure_dllm`(50)、`register_algorithm(name)`(68)、`patch_dllm_request_progress`(90)、`patch_dllm_custom_params_transport`(143)、`patch_dllm_single_target_completion`(173)。
- `sglang_gam/gam_qwen3.py`（35229 B）：`GroundAnythingVLMTwoLayerProjector`(86)、`GroundAnythingVLMVisionModel`(117)、`class FastDVLMForConditionalGeneration`(146)（服务启动时 `--json-model-override-args` 指定的架构名与此一致，见 `infer/serve_sglang.py:24,46`）。
- `sglang_gam/gam_hierarchy.py`（26649 B）：`class GAMHierarchyBlock(DllmAlgorithm)`(29)、`class GAMReliableBlock(GAMHierarchyBlock)`(516)。
- `sglang_gam/gam_speculative.py`（26253 B）：`class GAMSpeculativeBlock(DllmAlgorithm)`(32)。
- `sglang_gam/compat.py`（30813 B）：`install_triton_dllm_graph_patch`(69)、`install_dllm_packed_mask_replay_patch`(327)、`fast_eager_context`(414)、`install_sgl_kernel_compat`(458)。
- `sglang_gam/dlm_sampling.py`（5544 B）：`transfer_schedule(mask_count, denoise_steps)`(24)、`sample_tokens`(45)、`select_transfer_positions`(112)。
- `sglang_gam/execution_ops.py`（2855 B）：`cached_extend_positions`(7)、`longest_prefix_and_correction`(45)。
- `sglang_gam/qwen3_processor.py`（4278 B）：`class GAMQwen3ImageProcessor(BaseMultimodalProcessor)`(18)。
- `sglang_gam/gam_mineru.py`（10811 B）：`class GAMMinerUBlock(GAMHierarchyBlock)`(32)。
- `sglang_gam/triton_verify_graph.py`（7630 B）：`install_triton_verify_graph_patch`(29)、`replay_verifier`(103)。
- `sglang_gam/token_audit.py`（1818 B）：`install_token_audit`(8)。
- **未读**：上述文件的算法实现细节均未逐行读，仅取符号表。

### B6. `infer/overlay/prepare_vllm_overlay.py`（7462 B）

- 职责：为 vLLM 生成一个可被 `--model` 指向的「overlay」目录（把自定义模型代码复制/校验进去）。
- 签名：`source_files(model_dir, root)`(29)、`create_overlay(model_dir, output_dir, root=PROJECT_ROOT)`(67)、`verify_overlay`(107)、`main()`(130)；CLI `--model`(必填,Path)、`--output`(必填,Path)（`:132-133`）。
- **注意**：`serve_vllm.py:53` 会**先**调用 `source_files(model, ROOT)`，注释说这样做能「在尝试导入 accelerator 之前拒绝 DLM 模型」（`:51`）。

---

## C. 模型层 `models/`

**总览**（`models/README.md:3-8`）：`vlm/` = VLM 模型/配置/processor；`vlm_compat.py` = checkpoint 加载与推理兼容；`dlm/` = 训练与推理共用的 DLM wrapper；`dependency_contract.py` = 依赖源码完整性校验。
**关键约定**（`models/README.md:8`）：「Serialized class names and configuration identifiers in existing checkpoints are used during model loading and must match the model code」——即 **checkpoint 里序列化的类名必须与代码一致**，`scripts/prepare_model_code.py` 负责把缺失的模型代码补进模型目录。

### C1. `models/dependency_contract.py`（24 行）

- `verify_dependency(name, root=None)`（`:9`）→ 读 `third_party/manifest.json` 的 `dependencies[name]`，逐文件校验 sha256（`:15-20`），并**禁止任何未登记的 `.py`**（`:21-23`），返回 `source_revision`（`:24`）。
- **这是全仓库最关键的完整性闸门**：`vendor/` 一旦有额外 `.py` 文件就会 fail closed。

### C2. `models/vlm_compat.py`（367 行）——纯源码变换的 checkpoint 兼容层

- 顶层常量：`CONFIG_MODULE / KIMI_MODEL_MODULE / MODEL_MODULE / PROCESSOR_MODULE`（`:58-61`）、`VL_VISION_ATTENTION_FUNCTIONS`（`:229`）。
- 签名：`_accept_prepared_source(module)`（`:8`，装饰器）、`_patched_config_source`(65)、`_patched_model_source`(172)、`_patched_kimi_model_source`(193)、`_patched_processor_source`(241)。
- **设计要点**：用 AST dump 比对（`:34`、`:43-47`）——远程/checkpoint 自带源码必须与仓库 `models/vlm/` 下的 canonical 源码**AST 完全等价**，否则抛 `unsupported model source`（`:51`）。已在 `prepared_markers`（`:17-32`）里识别「已经是完整 serving 兼容版」的源码，不重复打补丁。

### C3. `models/vlm/manifest.json`（14 行）——VLM 代码文件哈希清单

- `family: groundinganything_vlm_qwen3`（`:2`），9 个文件（含 `LICENSE`）逐一记 sha256（`:3-13`）。

### C4. `models/vlm/` 模型定义（未逐行读）

| 文件 | 关键类/函数（行号） |
| --- | --- |
| `modeling_groundinganything.py`（79318 B） | `GroundAnythingVLMVisionEmbeddings`(225)、`GroundAnythingVLMVisionEncoder`(558)、`Siglip2MultiheadAttentionPoolingHead`(651)、`GroundAnythingVLMTwoLayerProjector`(962)、`GroundAnythingVLMVisionModel`(992)、`GroundAnythingVLMBaseModel`(1014)、`GroundAnythingVLMBaseForConditionalGeneration`(1367)、`GroundAnythingVLMModel`(1775)、`GroundAnythingVLMForConditionalGeneration`(1779)、`GroundAnythingModel`(1792)、`GroundAnythingForConditionalGeneration`(1798) |
| `modeling_groundinganything_vision.py`（55921 B） | MoonViT 系列：`MoonVision3dPatchEmbed`(313)、`Rope2DPosEmbRepeated`(368)、`MoonViTEncoderLayer`(488)、`MoonViT3dEncoder`(594)、`tpool_patch_merger`(648)、`MoonViT3dPretrainedModel`(676)、`GroundAnythingBackboneForConditionalGeneration`(921) |
| `configuration_groundinganything.py`（8016 B） | `GroundAnythingVLMVisionConfig`(11)、`GroundAnythingVLMConfig`(59)、`GroundAnythingConfig`(168) |
| `configuration_groundinganything_vision.py`（11783 B） | `GroundAnythingBackboneLinearConfig`(6)、`GroundAnythingBackboneVisionConfig`(159)、`GroundAnythingBackboneConfig`(229) |
| `processing_groundinganything.py`（5511 B） | `GroundAnythingVLMProcessor`(10)、`GroundAnythingProcessor`(143) |
| `image_processing_groundinganything.py`（8319 B） | `GroundAnythingVLMImageProcessor`(20) |
| `media_utils.py`（14220 B） | `ImageInput`(12)、`TransparentBgConfig`(21)、`navit_resize_image`(109)、`ensure_media_type`(196)、`image_to_np`(213)、`navit_patchify`(315) |
| `streammind_gate.py`（5234 B） | `PreNet`(11)、`PostNet`(20)、`SSMConfig`(30)、`VideoMamba`(35)、`Qwen3ForCausalLMCls`(54)、`ClsNet`(68)、`StreamMindGate`(90) |

### C5. `models/dlm/`

| 文件 | 关键符号 |
| --- | --- |
| `hybrid.py`（97121 B，仓库最大源码文件） | `MASK_TOKEN = "|<MASK>|"`(25)、`STABLE_BLOCK_MASK_ENV`(26)、`FLEX_CUDAGRAPHS_ENV`(27)、`ATTENTION_PROFILE_ENV`(28)、`_flex_attention_kernel`(40)、`compute_response_blocks(labels, block_size)`(70)、`initialize_mask_token(tokenizer, model)`(99)、`DLMOutput`(152)、`class GAMQwen35DLM(nn.Module)`(163) |
| `vlm.py`（15826 B） | `PACKED_CLEAN_ATTENTION_BACKEND_ENV`(23)、`class GAMQwen3DLM(GAMQwen35DLM)`(26) |

### C6. `models/qwen35_runtime.py`（477 行）——Qwen3.5 快路径安装

- **职责**：显式安装 GatedDeltaNet 卷积快路径（`fla_triton`），并在上游方法变化时 **fail closed**（`:1-8` 注释）。
- 顶层常量：`BACKEND_ENV="GAM_QWEN35_FASTPATH_BACKEND"`(24)、`FLA_TRITON_BACKEND="fla_triton"`(25)、`STABLE_L2NORM_ENV`(26)、`STABLE_CAUSAL_CONV_ENV`(27)、两个 `inspect.getsource` 摘要 `VISION_ROT_POS_EMB_SOURCE_SHA256`(32-34)、`GATED_DELTA_FORWARD_SOURCE_SHA256`(35-37)。
- **参数契约**（`:468-471`）：只有 `--check`（flag，help: install in this process and report provenance）。**无其它 CLI 参数。**
- 签名：`Qwen35FastPathError`(40)、`_require`(44)、`qwen35_stable_l2norm_fwd`(49)、`qwen35_stable_causal_conv1d_fn`(63)、`_install_stable_l2norm`(123)、`_install_fla_triton`(137)、`_install_cached_chunk_extension`(255)、`_install_vision_rot_pos_emb_patch`(360)、`install_qwen35_fastpath`(438)、`main`(468)。
- **含义**：这里锁的是 Transformers **5.2.0 released wheel** 的源码摘要（`:28-31` 注释），而 `vendor/transformers` 实际记的是 5.7.0（`infer/dependency_guard.py:25`、`infer/serve_vllm.py:60`）——**两个不同的 Transformers 版本共存**，属高风险点 R-06。

---

## D. 训练层 `train/`

**总览**（`train/README.md:5-10`）：`data/` = JSONL 行校验与 bbox/point 空间 token 格式（**明确声明「Raw-data-to-cache conversion is not included」**）；`dlm/` = DLM 数据整理/混合损失/自定义训练循环；`launcher/` = 模型/数据/checkpoint/分布式设置的一致性检查；`rl/` = 可选 DLM RL，**只有 RLV2 启用**，活跃实现/共享代码/禁用分支分目录保存；`runtime/` = cache 加载与媒体路径；`tokenizer/` = 空间 token 扩展与词表一致性检查。

### D1. 训练阶段链 `configs/training_stages.yaml`（26 行）

| stage | launch | sampled_rows | epochs | initialization |
| --- | --- | --- | --- | --- |
| general | `configs/release/general_train.yaml` | 2997407 | 2 | VLM base |
| specialist | `configs/release/specialist_train.yaml` | 29996394 | 2 | general DLM weights, fresh optimizer |
| sft1 | `configs/release/sft1_train.yaml` | 4739428 | 1 | specialist DLM weights, fresh optimizer |
| sft2 | `configs/release/sft2_train.yaml` | 302678 | 1 | sft1 DLM weights, fresh optimizer |

`rl: {enabled_algorithm: RLV2, included_in_default_chain: false}`（`:24-26`）——**RL 不在默认链里**。

### D2. 训练入口

| 文件 | CLI 参数（全部，`:行号`） |
| --- | --- |
| `train/dlm/train_dlm.py`（80544 B） | `--mode`(choices=("DLM",),required)(1044)、`--config`(Path,required)(1045)、`--model-path`(1046)、`--output-dir`(1047)、`--deepspeed-config`(Path)(1048)、`--batch-size`(int)(1049)、`--gradient-accumulation-steps`(int)(1050)、`--expected-global-batch-size`(int)(1051)、（1052 多行）、`--gradient-checkpointing`(BooleanOptionalAction)(1058)、`--vision-gradient-checkpointing`(1059)、`--language-checkpoint-stride`(int)(1060)、`--packing-language-checkpoint-stride`(int)(1061)、`--padding-free-packing`(1062)、`--packing-manifest`(1063)、`--packing-linear-kernel`(choices=("chunk",))(1064)、（1065 多行）、`--packing-lm-head-loss-chunk-tokens`(int)(1069)、`--packing-activation-cpu-offload`(1070)、`--packing-offload-threshold-mib`(1071)、`--packing-offload-min-tokens`(1072)、`--packing-offload-layer-stride`(1073)、（1074 多行）、`--max-steps`(int, None)(1079)、`--stop-after-step`(int)(1080)、`--disable-checkpointing`(1081)、`--resume-from-checkpoint`(1082)、`--initial-dlm-checkpoint`(Path)(1083) |
| `train/vlm/train_vlm_lora.py`（7450 B） | `--config`(required)(130)、`--resume-from-checkpoint`(131)、`--nproc-per-node`(int,1)(132)、`--nnodes`(int,1)(133)、`--node-rank`(int,0)(134)、`--master-addr`("127.0.0.1")(135)、`--master-port`(int,29500)(136)、`--dry-run`(137) |
| `train/rl/current/trainer.py`（25675 B） | `--config`(Path,required)、`--grounding-data`、`--ocr-data`、`--multiroute-data`、`--model-path`(required)、`--initial-checkpoint`(required)、`--resume-checkpoint`、`--output-dir`(required)、`--run-kind`(choices=(smoke,formal),required)、`--stable-gate`(Path,required)、`--final-gate`(Path,required)、`--max-optimizer-steps`(int)、`--local_rank`(int,-1)（`:52-64`） |
| `train/rl/branches/trace/trainer.py`（37279 B） | 同上 + `--temperature`、`--replay-token-budget`(24576)、`--maximum-replay-batch-size`(1)、`--temperature-adjustment-enabled`(choices=(0,1),0)、`--restart-index`(choices=(0,1),0)（`:54-68`） |
| `train/rl/distributed/multiroute_56.py` | 无 argparse（未确认其启动方式） |
| `train/rl/distributed/multiroute_64.py` | 无 argparse（未确认其启动方式） |

- **注意**：`train_dlm.py` 的 `--model-path/--output-dir/...` 等可选项都由 `scripts/run.py` 经 `train/release_parameters.py` 的 `dlm_control_args(native)` 注入（`scripts/run.py:62,78`），**手工直接调用极易漏参数**。

### D3. `train/release_parameters.py`（5169 B）与 `train/launcher/`

- `train/release_parameters.py` 暴露 `validate_training_fields(native)` 与 `dlm_control_args(native)`，被 `scripts/run.py:62` 在 `project_context()` 内导入（延迟导入，避免 CPU 预检时拉起重依赖）。
- `train/launcher/common.py`（25649 B）、`train/launcher/manifest_contract.py`（23427 B，CLI `--manifest` required + 一个多行参数，`:558-559`）是分布式/清单一致性检查。
- **未读**：`train/` 下 40+ 个文件的实现细节。`train/dlm/` 里的 9 个 audit/build 脚本（`build_sampling_manifest.py`、`build_packing_manifest.py`、`build_selected_lengths.py`、`audit_*.py`、`drop_actual_length_rejections.py`、`materialize_pass_actual_length_workload.py`、`summarize_response_block_distribution.py`、`validate_dlm_checkpoint*.py`）的 CLI 参数已从全局 grep 取得，语义未逐条核对。
- **RL 数据契约**（`configs/rl/default.yaml:32-34`）：`data.grounding_path: data/rl/grounding.jsonl`、`data.ocr_path: data/rl/ocr.jsonl`；`causal_justgrpo` 关键值：`num_generations: 8`、`temperature: 0.7`、`max_completion_tokens: 1024`、`clip_epsilon: 0.2`、`learning_rate: 2.0e-06`、`total_optimizer_steps: 3653`、`save_steps: 281`、`seed: 20260812`、`deepspeed_zero_stage: 1`、`freeze_vision_encoder: true`。
- **注意**：`config_contract.py:115-121` 要求 `rl-24` 的 `native_config.data` 恰好含 `{grounding_path, ocr_path}`，`rl-56/64` 恰好含 `{multiroute_path}`——**keys 集合必须精确相等**。

### D4. 训练 native config 的形状（以 `configs/train/dlm_h800_smoke.yaml` 为例，69 行）

顶层键：`mode: DLM`(1)、`meta{experiment,description}`(2-4)、`model{family,path,output_dir}`(5-8)、`data{sampling_manifest,sampled_rows}`(9-11)、`training{...}`(12-57)、`runtime{...}`(58-69)。
- `model.family` 必须是 `groundinganything_qwen3`（`scripts/run.py:64-66`）。
- `runtime.expected_world_size` 必须等于 `nproc_per_node*nnodes`（`scripts/run.py:57-59`）。
- `training.expected_global_batch_size` 必须等于 `world_size*per_device_train_batch_size*gradient_accumulation_steps`（`config_contract.py:90-93`）。
- smoke 配置规模：`max_steps: 2`(57)、`expected_world_size: 1`(61)、`image_max_token_num: 512`(62)、`max_length: 2048`(63)、`max_joint_length: 4096`(64)、`deepspeed_config: configs/train/zero2_cpu_smoke.json`(69)。
- `runtime` 下还要求 3 份审计报告：`selected_tail_actual_length_audit`、`selected_image_integrity_audit`、`selected_response_block_audit`（`:65-67`）——**本机 `data/smoke/...` 全部不存在**（见缺失项 M-04）。

### D5. launch YAML 的形状（`configs/release/*.yaml`）

以 `configs/release/general_train.yaml`（13 行）为例：`entrypoint: dlm-train`、`native_config: configs/train/general.yaml`、`args: []`、`env: {}`、`inputs: []`、`outputs: [outputs/general]`、`distributed{nproc_per_node:8, nnodes:8, node_rank:0, master_addr:127.0.0.1, master_port:29500}`。
以 `configs/release/dlm_sglang_denoise.yaml`（18 行）为例：`entrypoint: dlm-sglang`，`args` 为 `--model weights/dlm_bundle --decoder denoise --output outputs/sglang/denoise --port 8101`，`env{HF_HOME: outputs/hf, OMP_NUM_THREADS:'4'}`，`inputs[weights/dlm_bundle, vendor/sglang]`，`outputs[outputs/sglang/denoise]`。
- **含义**：`inputs` 就是 `scripts/run.py:118-125` 的存在性闸门；`weights/dlm_bundle` 与 `vendor/sglang` 本地都不存在 → 任何 `serve`/`train` 的真实运行会在 `Provide missing inputs before execution.` 处停住（`scripts/run.py:125`）。

---

## E. 数据工具 `tools/`

| 文件 | 职责（docstring 行号） | CLI 参数（全部） |
| --- | --- | --- |
| `tools/build_gam_cache.py`（160 行） | 从 canonical GAM JSONL 构建 GroundAnything Arrow cache；通过 `datasets.Dataset.save_to_disk` 产出，保持与现有 DLM loader 兼容；文本长度=tokenizer 长度，字符近似回退**仅在显式 `--allow-approximate-length` 时可用**（`:1-7`） | `--input`(Path,required)(148)、`--cache`(Path,required)(149)、`--manifest`(Path,required)(150)、`--source-id`(required)(151)、`--tokenizer`(Path)(152)、`--allow-approximate-length`(flag)(153)、`--task`(choices=非 TRAJECTORY 的 TaskType 值,默认 `bbox`)(154) |
| `tools/convert_locany_to_gam.py`（243 行） | 把 LocateAnything ShareGPT 标注转成 GAM JSONL；**绝不修改输入文件**，把 LA 的 0..1000 坐标重写到 GAM 的 0..999 网格，逐行按 canonical row contract 校验（`:1-6`） | `--input`(Path,required)(231)、`--output`(Path,required)(232)、`--metadata-output`(Path)(233)、`--rejects-output`(Path)(234)、`--source`(235)、`--task`(默认 `bbox`)(236)、`--default-phrase`(默认 `"target"`)(237) |
| `tools/validate_gam_jsonl.py`（42 行） | 校验 GAM JSONL 行与 answer grammar（`:1`） | `--input`(Path,required)(35)、`--task`(默认 `bbox`)(36)；输出为 stdout JSON `{"rows","valid","invalid"}`（`:38`） |

- **I/O 约定**：`validate_file` 逐行 `json.loads` + `validate_uniform_row(row)` + 对每条 `role=="assistant"` 的 message 调 `validate_answer(content, task)`（`:22-26`）；遇到第一处错误立即 `raise ValueError(f"{path}:{line_number}: {error}")`（`:29`）——**不做「跳过坏行继续」**。
- **共享依赖**：三者都依赖 `grounding_anything.protocol`（`TaskType`、`validate_answer`、`GroundingEntry`、`canonicalize_phrase`、`serialize_answer`）与 `train/data/row_contract.validate_uniform_row`（`build_gam_cache.py:19-20`、`convert_locany_to_gam.py:16-22`、`validate_gam_jsonl.py:9-10`）。
- **数据配置文件**：`configs/datasets.yaml` 见 A10。

---

## F. 环境与依赖

### F1. 包元数据

| 文件 | 内容 |
| --- | --- |
| `pyproject.toml`（20 行） | `name="grounding-anything"`, `version="0.1.0"`, `requires-python=">=3.10"`, **唯一运行依赖 `PyYAML>=6,<7`**（`:11`）；可选 extra `visualization=["Pillow>=10"]`（`:14`）；打包只含 `grounding_anything*`（`:16-19`） |
| `requirements.txt`（4 行） | 只有一行 `-e .[visualization]`（`:4`）——「HTTP 客户端 + 可视化 + setup 命令」的依赖 |
| `environment.yml`（8 行） | 可选 Conda bootstrap：`name: grounding-anything`，`python=3.12` + `pip`（`:3-8`） |

### F2. 四个工作流 profile

`environments/install.yaml`（30 行）：

| profile | requirements | snapshots | platform / 特殊项 |
| --- | --- | --- | --- |
| `train` | `requirements/train.txt` | `transformers` | `platform: Linux x86_64`；`accelerator_packages`：torch 2.10.0+cu129、torchvision 0.25.0+cu129（**manylinux_2_28_x86_64 wheel + sha256**）；`source_extensions: flash-attn==2.8.3.post1`；`build_env: TORCH_CUDA_ARCH_LIST=9.0, MAX_JOBS=8, NVCC_THREADS=2, FLASH_ATTENTION_FORCE_BUILD=TRUE, FLASH_ATTN_CUDA_ARCHS=90` |
| `eval` | `requirements/eval.txt` | `lmms_eval` | `platform: Linux x86_64` |
| `serve` | `requirements/serve.txt` | `sglang` | 无 platform 字段；`post_install_overrides: nvidia-cudnn-cu12==9.16.0.29` |

`requirements/train.txt`：`datasets==5.0.0, pyarrow==24.0.0, accelerate==1.13.0, deepspeed==0.19.0, flash-linear-attention==0.5.2, fla-core==0.5.2, flash-attn==2.8.3.post1, torchvision==0.25.0+cu129, qwen-vl-utils==0.0.14, torch==2.10.0+cu129, triton==3.6.0, transformers==5.7.0`。
`requirements/serve.txt`：`torch==2.9.1, transformers==5.5.4, triton==3.5.1, sgl-kernel==0.3.20, flashinfer-python==0.5.3, flashinfer-cubin==0.5.3`。
`requirements/eval.txt`：`pycocotools, shapely, scipy, rapidfuzz, editdistance`，**torch/torchvision 直接指向 `download.pytorch.org/whl/cpu` 的 `cp312-linux_x86_64` wheel**（`:6-7`），`datasets>=2.19,<4`、`numpy==1.26.4`、`transformers>=4.39.2,<5`、`opencv-python-headless<4.12`、`sentence-transformers==5.1.2`、`openai>=1,<3`。
`requirements/vllm.txt`：只有 `huggingface-hub==1.32.0`、`regex==2026.9.10`（其余继承平台镜像，vLLM 0.18.x）。

> **⚠️ 结论（本地不可执行的根因）**：三个核心 profile 的依赖都是 **Linux x86_64** 专用。`scripts/setup_environment.py:25-28` 会对 `platform == 'Linux x86_64'` 的 profile 直接 `parser.error('train profile requires Linux x86_64 and Python 3.12 for its wheels')`。`scripts/setup_vllm_environment.py:44-45` 同样 `raise ValueError("vLLM runtime requires Linux Python 3.12")`。**因此 `run.py setup {train,eval,vllm}` 在 Windows 上必然失败。**

### F3. 安装器的参数契约与 I/O

| 脚本 | CLI 参数 | 关键 I/O |
| --- | --- | --- |
| `scripts/setup_environment.py`（74 行） | `--profile`(required)、`--venv`(required,新目录)、`--apply`(flag)、`--inherit-container`(flag)、`--index-url`(`https://pypi.org/simple`)（`:16-20`） | 读 `environments/install.yaml`（`:22`）；**已存在的 venv 目录直接 `parser.error` 拒绝覆盖**（`:30`）；`python = venv/bin/python`（`:33`，POSIX）；写 `<venv>/installed.freeze.txt`（`:71`）；打印 JSON 计划（`:53-54`，含 `gpu_verified: false`） |
| `scripts/setup_vllm_environment.py`（88 行） | `--venv`(required)、`--platform`(required, gpu\|ppu)、`--index-url`(`https://pypi.org/simple`)、`--apply`(flag)（`:34-37`） | `environment_plan()`（`:16`）拒绝已存在目录（`:18-19`）；用 `--system-site-packages`（`:24`）继承平台镜像；只 `prepare_dependencies --name transformers`（`:25`）；记录 PPU 的 ACEXT `lib/include` 路径（`:50-55`） |
| `scripts/prepare_dependencies.py`（67 行） | `--name`(可重复,默认全部)、`--apply`(flag)（`:50-51`） | 读 `third_party/manifest.json`（`:53`）；**先校验 tar.gz 的 sha256**（`:60-61`）再解包；解包时逐个成员核对文件清单与哈希（`:32-39`），**只允许常规文件**（`:35`）；用临时目录 + `os.replace` 原子落地（`:28,42`）；已存在则走 `verify_dependency` 返回 `already_verified`（`:22-26`）；输出 stdout JSON `[{name,files,destination,status}]`（`:62-64`） |
| `scripts/rebuild_dependency.py`（5177 B） | `--name`(required)、`--base-archive`(Path,required)、`--output`(Path,required)（`:109-111`） | 从上游 tar.gz 重建发布快照；**「Existing output files are never overwritten」**（`third_party/README.md:26`） |
| `scripts/prepare_model_code.py`（40 行） | `--model`(required)、`--apply`(flag)（`:13-14`） | 读模型 `config.json` 要求 `model_type=="groundinganything_vlm"` 且 `text_config.model_type=="qwen3"`（`:18-19`）；按 `models/vlm/manifest.json` 校验并补齐 9 个文件；已存在但哈希不符 → `ValueError('existing model code differs; no files replaced')`（`:29-30`）；写用**独占创建 `open('xb')`**（`:37`） |
| `scripts/check_serve_environment.py`（40 行） | 无参数 | 跑 `pip check`，把 `torch 2.9.1 → nvidia-cudnn-cu12==9.10.2.21 vs 9.16.0.29` 归为**已知覆盖**（`:14-17`），其余算错误；写 `<sys.prefix>/serve-dependency-check.json`（`:30`）；还要求 `torch.backends.cudnn.version() >= 91500`（`:35-36`） |
| `scripts/request_image.py`（48 行） | `--image`(Path,required)、`--base-url`(required,需以 /v1 结尾)、`--model`(required)、`--prompt`(required)、`--max-tokens`(int,256)、`--timeout`(float,120)（`:28-33`） | **纯标准库**（`:1`）；只接受 PNG/JPEG/WebP（`:13`）；POST `<base_url>/chat/completions`，固定 `temperature: 0, skip_special_tokens: False`（`:18-19`）；stdout 打印 `{model,content,finish_reason,usage}`（`:42-44`） |
| `scripts/eval_preflight.py`（58 行） | 无 CLI（库） | `get_json(url)` 用 `OPENAI_API_KEY` 环境变量加 Bearer 头（`:9-13`）；`validate_service` 要求 `model_id` 出现在 `/v1/models`（`:18-19`）；native 契约要求 `/health` 暴露 `service_contract=="native-v1"` 与整数 `max_new_tokens`/`max_model_len`（`:23-30`）；预算检查 `n > cap or n >= context` 直接报错（`:31-32`）；`mode ∈ {GAM,DLM,RLV2}` 要求 `non_thinking is True`（`:33-34`）；`preflight` 还会请求 `/server_info` 校验 decoder（`:50-54`），并断言 preflight 与 worker 端点一致（`:56-57`） |

### F4. `examples/`

| 文件 | CLI 参数 | 产物 |
| --- | --- | --- |
| `examples/predict.py`（1238 B） | `--image`(Path,required)、`--phrase`(required)、`--task`(choices bbox\|point, `bbox`)、`--base-url`(`http://127.0.0.1:8101/v1`)、`--model`(`groundinganything`)、`--output`(Path, `outputs/prediction`)（`:10-15`） | `result.json` + `prediction.png`（`examples/README.md:11`；**每次预测必须用新输出目录**，`:11`） |
| `examples/batch_predict.py`（10698 B） | `--input`(required, JSONL；图片路径相对于该文件)、`--output`(required, append-only 结果 JSONL，**成功请求可续跑**)、`--base-url`(`DEFAULT_BASE_URL`)、`--model`(`DEFAULT_MODEL`)、`--max-tokens`(int,4096)、`--timeout`(float,180)、`--run-tag`(``，help：server 权重或解码设置变化时必须改)、`--api-key-env`(环境变量名)（`:197-204`） | 追加式 JSONL（`:198`） |

### F5. `licenses/` 与 `third_party/`

- `licenses/Kimi-K3.txt`（3116 B）、`licenses/Kimi-K3.source.json`（258 B）。
- `third_party/`：`manifest.json`（1080526 B，**`schema_version` + `dependencies` 两个顶层键**，3 个依赖）：
  | name | destination | files | source_revision | archive |
  | --- | --- | --- | --- | --- |
  | transformers | `vendor/transformers` | 2328 | `90dd8674248257e1115e531f2250fd85a32c3863` | `third_party/transformers.tar.gz`（7944881 B） |
  | sglang | `vendor/sglang` | 1525 | `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b` | `third_party/sglang.tar.gz`（2994935 B） |
  | lmms_eval | `vendor/lmms-eval` | 61 | `bbc1f57e892a6e5af586c71c7733411208c75b82` | `third_party/lmms_eval.tar.gz`（116700 B） |
- `third_party/patches/{transformers,sglang,lmms_eval}.json`：钉住「公开对比 commit + 源码 archive 校验和」，用于 `rebuild_dependency.py`（`third_party/README.md:16-18`）。
- **重要**：`third_party/*.tar.gz` 存在，但 `vendor/` 不存在 → `verify_dependency` 会抛 `prepare dependency snapshot first: <name>`（`models/dependency_contract.py:14`）。

---

## G. 测试 `tests/`（6 个文件）

**运行方式**：仓库**没有任何 pytest/unittest 配置**（`pyproject.toml` 里没有 `[tool.pytest]`），也没有 `conftest.py`。唯一被文档写出的运行命令是 `docs/BATCH_INFERENCE.md:90`：
```bash
python3 -m unittest discover -s tests -p test_batch_predict.py -v
```
两个文件用 pytest 风格（`tests/test_gam_cache_builder.py:4`、`tests/test_locany_to_gam.py:4` import pytest），**但 `pytest` 不在任何 requirements profile 里**——需要自行安装。

| 文件 | 测什么 | 依赖什么（import 行号） |
| --- | --- | --- |
| `tests/test_vllm_deployment.py`（12710 B，unittest） | **CPU-only** VLM 部署回归（`:1` 自述）；覆盖 `run.py plan()` 的 CLI 路由、overlay 创建/校验、`serve_vllm.build_command` 与 `setup_vllm_environment.environment_plan` | `infer/overlay/prepare_vllm_overlay`(19)、`infer/serve_vllm`(20)、`scripts/setup_vllm_environment`(21)；用 `importlib.util.spec_from_file_location` 直接加载 `run.py`(23-25)；`make_checkpoint` 造一个 **伪 `model.safetensors`（内容 `b"test-weight-link-only"`，注释明确说不是可加载张量）**(47-48) |
| `tests/test_batch_predict.py`（7701 B，unittest） | **CPU 集成**：起一个本地假 OpenAI 兼容 HTTP 服务（`ThreadingHTTPServer`），测 `examples/batch_predict.py` 的续跑/appending 行为（`:1-2`） | 标准库 http.server + `importlib.util` 加载 `examples/batch_predict.py` |
| `tests/test_vlm_launcher.py`（1497 B，unittest） | 测 VLM LoRA 启动参数拼装 | `train.vlm.train_vlm_lora` 的 `_swift_args`、`build_command`（`:7`） |
| `tests/test_game_ui_metrics.py`（2785 B，unittest） | 测 game UI 评测指标 | `eval.game_ui.metrics` 的 `aggregate_results, box_iou, evaluate_sample, normalize_box`（`:3`） |
| `tests/test_gam_cache_builder.py`（1940 B，pytest） | 测 `tools/build_gam_cache.py` 的 `normalize_row`、`read_rows`；覆盖「近似长度必须显式 opt-in」与「非法 JSON 严格报错」（`:29,36,43`） | `pytest`(4)、`tools.build_gam_cache`(6) |
| `tests/test_locany_to_gam.py`（2315 B，pytest） | 测 `convert_locany_to_gam` 的角色/grid/`<image>` 占位符转换，以及转换→校验往返（`:20-53`） | `pytest`(4)、`tools.convert_locany_to_gam`(6)、`tools.validate_gam_jsonl`(7) |

- **可本地跑性**：`tests/test_vllm_deployment.py` 与 `tests/test_batch_predict.py` 自称 CPU-only，理论上可在 Windows 跑；`test_gam_cache_builder` 需要 `transformers`（`tools/build_gam_cache.py:32`），本地未安装 → 未验证。
- **注意**：仓库存在 `tests/__pycache__/`（6 个 `.pyc`）与根 `__pycache__/run.cpython-312.pyc`——说明**这些测试/入口曾在本机被 Python 3.12 执行过**，但这不等于测试通过；`__pycache__` 被 `.gitignore:10` 排除。

---

## H. 文档与产物约定

### H1. `README.md`（30232 B，486 行）——可执行命令与数字声明

**可执行命令块**（逐条出处）：

| 行 | 命令 |
| --- | --- |
| 83-85 | `git clone https://github.com/groundingpi/GroundAnything.git` / `cd GroundAnything` / `python3 -m pip install -r requirements.txt huggingface_hub` |
| 119-121 | `hf download GroundingPI/GroundAnything --local-dir weights/dlm_bundle` → `python3 run.py setup serve` → `python3 run.py serve --decoder denoise` |
| 133 | `python3 run.py serve --decoder speculative` |
| 144 | `python3 run.py serve --config configs/release/vlm_sglang.yaml` |
| 183-187 | `python3 examples/predict.py …` |
| 216-218 | `python3 run.py setup vllm --platform gpu` / `python3 run.py serve --engine vllm --platform gpu` |
| 224-226 | （PPU 版 vLLM 命令） |
| 241-244 | `python3 examples/batch_predict.py …` |
| 360-364 | `python3 run.py setup eval` / `run.py eval --decoder denoise` / `--decoder speculative` / `--config configs/eval/vlm.yaml` |
| 391-397 | `python3 run.py setup train` / `.venv-train/bin/python scripts/run.py configs/release/sft1_train.yaml --dry-run` / `python3 run.py train --config configs/release/sft1_train.yaml` |

**数字与事实声明**（必须核对的论文相关声明）：

| 声明 | 出处 |
| --- | --- |
| 评测套件覆盖论文报告的 **30 benchmarks** | `README.md:343` |
| 共享评测器支持 **7 modes**（GAM/VLM/REXOMNI/LOCATEANYTHING/GROUNDINGDINO/DLM/RLV2） | `README.md:345-355` |
| 三个已发布 checkpoint 都用 **GAM** 模式 | `README.md:349,357` |
| 服务的 **Linux x86_64 + Python 3.12** 前提；serve profile 用 Torch 2.9.1 / Transformers 5.5.4 / Triton 3.5.1 / sgl-kernel 0.3.20 / FlashInfer 0.5.3 | `README.md:80` |
| 已测加速器：**NVIDIA B300, B200, H200, H800, PPU** | `README.md:92` |
| 论文与补充材料：**arXiv 2609.39600** | `README.md:422` |
| 训练配方针对 **H800 级 CUDA GPU**，源码编译 FlashAttention | `README.md:373` |
| 派发阶段模板按 **8 nodes × 8 devices** | `README.md:388` |
| 发布的 DLM checkpoint **已是完整 bundle，不需要 `prepare-model`** | `README.md:124` |
| vLLM 适配器**仅支持 GroundAnything-VLM**，只有 CPU 级检查，**端到端加速器验证尚未完成** | `README.md:104` |
| 出货评测配方是**8 样本 smoke**，不是全量 | `README.md:367` |

**目录约定**（`README.md:440-452`）：`run.py / grounding_anything/ / configs/ / infer/ / train/ / eval/ / models/ / examples/ / environments/ / docs/`。

### H2. `README.zh-CN.md`（6289 B，107 行）

命令块：`:12` `python -m pip install -r requirements.txt`；`:38-39` `python3 run.py setup train` + `python3 run.py train --config configs/release/general_train.yaml`；`:51-53` `run.py setup serve` + **`python3 run.py prepare-model`**；`:58` `python3 run.py serve`；`:64-66` `--decoder causal|speculative`；`:77-79` `run.py setup eval` + `run.py eval`；`:84-86` `eval --decoder causal|speculative`。
> **差异点**：中文 README **推荐先跑 `prepare-model`**（`:52`），而英文 README 明确说发布的 DLM bundle **不需要** `prepare-model`（`README.md:124`）。两者不一致，属文档漂移（见风险 R-08）。

### H3. `eval/README.md`（10254 B）——评测的唯一完整操作手册

- 环境与服务对照表（`:24-28`）：

| Checkpoint / 解码 | 起服务 | 评测模板 | API base URL | model ID |
| --- | --- | --- | --- | --- |
| GroundAnything / entropy-guided | `python3 run.py serve --decoder denoise` | `configs/eval/dlm.yaml` | `http://127.0.0.1:8101/v1` | `groundinganything` |
| GroundAnything / self-speculative | `python3 run.py serve --decoder speculative` | `configs/eval/dlm_speculative.yaml` | `http://127.0.0.1:8101/v1` | `groundinganything` |
| GroundAnything-VLM / autoregressive | `python3 run.py serve --config configs/release/vlm_sglang.yaml` | `configs/eval/vlm.yaml` | `http://127.0.0.1:8102/v1` | `groundinganything-vlm` |

- 30 任务清单（`:89-120`，`assert len(tasks) == 30 and len(set(tasks)) == 30` 在 `:121`）通过一个 **heredoc 脚本**（`:82-144`）生成 `configs/eval/full_suite_{denoise,speculative,vlm}.yaml`（`:126-128`），`limit: null`（`:139`），`run_id` 带 UTC 时间戳 + 8 位 uuid（`:130,139`），用 `open("x")` **拒绝覆盖**（`:141`，`:147` 也明确说「refuses to overwrite existing generated configurations」）。
- **这三个 `full_suite_*.yaml` 文件本地不存在**（见缺失项 M-05）——它们是**生成物**，不是随仓库发布的文件。
- 任务分组表（`:64-73`）：Detection 4、Referring boxes 5、Object pointing 6、Spatial pointing 4、GUI grounding 3、OCR 4、Document layout 2、Visual prompting 2。
- 输入检查命令（`:160-161`）：`.venv-eval/bin/python scripts/evaluate.py configs/eval/full_suite_denoise.yaml --dry-run`，并明确与 `python3 run.py eval --dry-run` 的区别（`:164`：前者检查配置与缺失输入，后者只预览下游命令）。
- 结果目录内容（`:189-194`）：`run.json` / `responses.jsonl` / `log_eval/` 与 `cache/` / `summary.json`。

### H4. 其余 `docs/*.md` 的命令清单

| 文档 | 命令（行号） |
| --- | --- |
| `docs/TRAINING.md`（58 行） | `python3 run.py setup train`(6)、`python3 run.py train --config configs/release/general_train.yaml`(25)、`.venv-train/bin/python scripts/run.py configs/release/general_train.yaml --dry-run`(31)、RL：`.venv-train/bin/python scripts/run.py configs/release/rl_train.yaml`(53)；产物写在 native config 的 `model.output_dir`，general 默认 `outputs/general/`(34) |
| `docs/DATA_PREPARATION.md`（122 行） | `python tools/build_gam_cache.py --input data/game/train.gam.jsonl --cache data/game/cache/train --manifest …/cache_manifest.json --source-id game_train --tokenizer weights/base_model`(10-16)、`.venv-train/bin/python -m train.tokenizer.validate_tokens weights/base_model --write-manifest weights/base_model/gam_tokenizer_manifest.json`(28-29)、`-m train.dlm.build_sampling_manifest --config data/sources.yaml --output-dir data/general/sampling --sample-size 2997407 --seed 32 --stage general`(49-51)、`-m train.dlm.build_selected_lengths --sampling-manifest … --output data/general/selected_lengths.npy --workers 8`(52-54)、`-m train.dlm.build_packing_manifest --sampling-manifest … --selected-lengths data/general/workload_lengths.npy --output-dir data/general/packing --epochs 2 --world-size 64 --pack-size 32 --packing-strategy balanced --seed 32`(86-90)、`-m train.dlm.audit_selected_actual_lengths --help`(72) |
| `docs/DATA_PREPARATION.md` 关键约定 | 缓存导出器**拒绝覆盖非空 cache**，图片保留为 `{"bytes": null, "path": "..."}`，`--allow-approximate-length` **只用于 dry run**(:18-21)；坐标 token `<0>`–`<999>`、`</c>`；bbox 4 坐标 / point 2 坐标；DLM checkpoint 需原子 mask token `\|<MASK>\|`(:23)；RL 数据量：两源配方需 **6,557 Grounding + 5,500 OCR** 行，选 4,400 OCR 行同行合并(:107)；multiroute 需 **17,700** 行(:116)；gate 文件 `stable_gate.json`（首次成功更新后）与 `final_gate.json`（结束时）默认写在 `outputs/rl/`(:118) |
| `docs/INFERENCE.md`（5446 B） | `python3 run.py setup serve`(10)、`python3 run.py prepare-model`(11)、`python3 run.py serve`(12)、`.venv-train/bin/python scripts/run.py configs/release/dlm_serve.yaml`(56) |
| `docs/VLLM.md`（8099 B） | `python3 -m pip install -r requirements.txt huggingface_hub`(14,136)、`hf download GroundingPI/GroundAnything-VLM --local-dir weights/vlm`(15,137)、`python3 run.py setup vllm --platform gpu`(18,138)、`python3 run.py serve --engine vllm --platform gpu`(19,139)、ppu 版(25-26)、`python3 run.py serve --engine vllm --platform gpu \`(95，多行)、`python3 run.py setup eval`(108)、`python3 run.py eval --config configs/eval/vlm_vllm.yaml`(109) |
| `docs/BATCH_INFERENCE.md`（7517 B） | `python3 examples/batch_predict.py \`(36)、`python3 -m unittest discover -s tests -p test_batch_predict.py -v`(90) |
| `docs/VLM_FINETUNING.md`（1622 B） | `python scripts/run.py configs/release/vlm_lora_train.yaml --dry-run`(34) |
| `docs/LOCANY_MIGRATION.md`（2374 B） | `python tools/convert_locany_to_gam.py \`(18)、`python tools/validate_gam_jsonl.py --input data\game\train.gam.jsonl`(24，**注意用了 Windows 反斜杠**)、`python tools/build_gam_cache.py \`(43) |
| `docs/GAME_UI_EVALUATION.md`（2875 B） | `python -m eval.game_ui.evaluate_game …`(44，**PowerShell 续行反引号 `**——见风险 R-09) |

### H5. `docs/assets/`

存在：`demo-poster.jpg`、`groundanything-title.png`、`readme-title.svg`、`teaser.png`。
被 `.gitignore:57` 排除的：`/docs/assets/demo.mp4`（注释说「the exact main demo is hosted on Hugging Face」）——**本地没有 demo.mp4 是预期行为**。

---

## I. 本机环境实测（2026-10-09 只读探测）

| 项 | 实测值 | 判定依据 |
| --- | --- | --- |
| 解释器 | `C:\Users\Huya\AppData\Local\Programs\Python\Python312\python.exe`，Python 3.12.10 | `python -V`；与 `environments/install.yaml` 要求的 3.12 一致 |
| torch | **2.13.0+cpu**，`torch.cuda.is_available() == False`，`device_count() == 0` | 实测 |
| transformers | **5.14.1** | 实测 |
| 已安装 | `yaml, torch, transformers, PIL, numpy, safetensors, httpx` | `importlib.util.find_spec` |
| 未安装 | `datasets, pytest, accelerate, sglang, vllm, lmms_eval, openai` | 同上 |
| 平台 | Windows；`scripts/setup_environment.py:27` 要求 `platform.system() == 'Linux'` | 源码 |

**关键含义**：
1. 本地 `transformers 5.14.1` **同时不满足三条路线**——serve 要 5.5.4（`requirements/serve.txt`）、train/vllm 要 5.7.0（`requirements/train.txt`、`infer/serve_vllm.py:60`）、eval 要 `<5`（`requirements/eval.txt`）。本机 transformers 只能用于纯 CPU 的源码/配置检查。
2. 本地无 GPU + 无 `vendor/sglang` → **任何 serve 路线都不可能在本机跑通**。
3. 但 `run.py` 的 `--dry-run` 不检查 venv（`run.py:125` 只在 `not args.dry_run` 时要求解释器存在），因此 **`python run.py {serve,eval} --dry-run` 在本机可以跑**（本次侦察按纪律未执行）。
4. `train/release_parameters.py:1` 自称「no GPU imports」，是少数可在 CPU 上安全导入的训练模块。

---

## 1. 命令速查表

> 「在哪跑」一列基于本机（Windows/无 GPU/无 vendor/无 venv）的事实判断。
> ✅本地 = 本机可直接跑；⚠️本地部分 = 需要先在本机装少量依赖或只在本机跑 dry-run；❌本地 = 本机不可跑，必须 devbox。

| 想做什么 | 命令 | 在哪跑 | 出处 |
| --- | --- | --- | --- |
| 装轻量客户端依赖 | `python -m pip install -r requirements.txt` | ✅本地 | `requirements.txt:4`、`environments/README.md:6` |
| 查看入口帮助 | `python run.py --help` | ✅本地 | `README.md:454`、`run.py:132-134` |
| **预览 serve 命令（不执行）** | `python run.py serve --decoder denoise --dry-run` | ✅本地（dry-run 不查 venv） | `run.py:38,125,139-141` |
| **预览 eval 命令（不执行）** | `python run.py eval --decoder denoise --dry-run` | ✅本地 | `run.py:38,119,125` |
| 预览 train 命令 | `python run.py train --config configs/release/sft1_train.yaml --dry-run` | ⚠️本地部分（需 `yaml`，已有；train 分支不导入重依赖） | `run.py:112-124`、`scripts/run.py:62` |
| 预览 DLM 评测计划 + 缺失输入 | `.venv-eval/bin/python scripts/evaluate.py configs/eval/dlm.yaml --dry-run` | ❌本地（POSIX venv 路径 + 需 `vendor/lmms-eval`，见 `scripts/evaluate.py:176,262`） | `eval/README.md:160-161` |
| 装 train 环境 | `python3 run.py setup train`（= `scripts/setup_environment.py --profile train --venv .venv-train --apply`） | ❌本地（`scripts/setup_environment.py:25-28` 强制 Linux x86_64） | `README.md:391`、`environments/README.md:27`、`environments/install.yaml` |
| 装 serve 环境 | `python3 run.py setup serve` | ❌本地（`run.py:98` 用 `venv/bin/python`；依赖树面向 CUDA） | `README.md:120`、`environments/README.md:28` |
| 装 eval 环境 | `python3 run.py setup eval` | ❌本地（同 Linux 强制；且 `requirements/eval.txt:6-7` 是 linux_x86_64 CPU wheel） | `README.md:360`、`environments/README.md:29` |
| 装 vLLM 环境 | `python3 run.py setup vllm --platform gpu`（或 `--platform ppu`） | ❌本地（`scripts/setup_vllm_environment.py:44-45` 强制 `sys.platform == "linux"`） | `README.md:216`、`docs/VLLM.md:18` |
| 抽依赖快照到 vendor/ | `python scripts/prepare_dependencies.py`（检查）/ `--apply`（解包） | ✅本地（纯 stdlib + 文件哈希；解包会写 `vendor/`，本次侦察未执行） | `third_party/README.md:8-9`、`scripts/prepare_dependencies.py:50-51` |
| 重建依赖发布包 | `python scripts/rebuild_dependency.py --name transformers --base-archive <upstream.tar.gz> --output <out.tar.gz>` | ⚠️本地（纯文件操作；需自备上游 archive） | `third_party/README.md:23`、`scripts/rebuild_dependency.py:109-111` |
| 把模型代码补进模型目录 | `python scripts/prepare_model_code.py --model <dir> --apply` | ⚠️本地（要求 `model_type == groundinganything_vlm`，**对当前 `GroundAnything-Model/` 会直接报错**，见 R-04） | `scripts/prepare_model_code.py:13-14,18-19` |
| 校验依赖完整性 | `python -c "from models.dependency_contract import verify_dependency; verify_dependency('sglang')"` | ❌本地（`vendor/` 不存在 → `prepare dependency snapshot first`） | `models/dependency_contract.py:9-14` |
| 下载 DLM 权重 | `hf download GroundingPI/GroundAnything --local-dir weights/dlm_bundle` | ❌（已由 NAS 提供，**本地已在 `GroundAnything-Model/`，见 M-01**） | `README.md:119` |
| 下载 VLM 权重 | `hf download GroundingPI/GroundAnything-VLM --local-dir weights/vlm` | ❌（本机缺失，见 M-02） | `docs/VLLM.md:15` |
| 起 DLM 服务（默认 denoise） | `python3 run.py serve --decoder denoise` | ❌devbox | `README.md:121`、`eval/README.md:26` |
| 起 DLM 服务（self-speculative） | `python3 run.py serve --decoder speculative` | ❌devbox | `README.md:133`、`eval/README.md:27` |
| 起 VLM 服务（SGLang causal） | `python3 run.py serve --config configs/release/vlm_sglang.yaml` | ❌devbox | `README.md:144`、`eval/README.md:28` |
| 起 VLM 服务（vLLM gpu/ppu） | `python3 run.py serve --engine vllm --platform gpu` | ❌devbox | `docs/VLLM.md:19,26` |
| 单图预测（客户端） | `python examples/predict.py --image X.jpg --phrase "the red car" --task bbox --output outputs/car` | ⚠️本地部分（需一个在跑的服务；`Pillow` 本地已有） | `examples/README.md:7` |
| 批量标注 | `python3 examples/batch_predict.py --input in.jsonl --output out.jsonl …` | ⚠️本地部分（需在跑的服务） | `README.md:241-244`、`examples/batch_predict.py:197-204` |
| 直连服务发一条请求 | `python scripts/request_image.py --image X.png --base-url http://127.0.0.1:8101/v1 --model groundinganything --prompt "…"` | ⚠️本地部分（纯 stdlib，不需 torch） | `scripts/request_image.py:28-33` |
| 跑 smoke 评测 | `python3 run.py eval --decoder denoise` | ❌devbox（连服务） | `README.md:362`、`configs/eval/dlm.yaml`（`limit: 8`） |
| 生成 30 任务全量配置 | `python3 - <<'PY' … PY`（heredoc，`eval/README.md:82-144`） | ⚠️本地部分（需 `yaml`，已有；会写 `configs/eval/full_suite_*.yaml`） | `eval/README.md:82-144` |
| 跑全量 30 任务 | `python3 run.py eval --config configs/eval/full_suite_denoise.yaml` | ❌devbox | `eval/README.md:170` |
| 训练（单阶段） | `python3 run.py train --config configs/release/general_train.yaml` | ❌devbox | `docs/TRAINING.md:25`、`README.md:397` |
| 训练 dry-run（子启动器） | `.venv-train/bin/python scripts/run.py configs/release/general_train.yaml --dry-run` | ❌本地（POSIX venv 路径） | `docs/TRAINING.md:31` |
| RL 训练（RLV2, 24 卡） | `.venv-train/bin/python scripts/run.py configs/release/rl_train.yaml` | ❌devbox | `docs/TRAINING.md:53`、`config_contract.py:126` |
| 构建 GAM Arrow 缓存 | `python tools/build_gam_cache.py --input data/game/train.gam.jsonl --cache data/game/cache/train --manifest data/game/cache/cache_manifest.json --source-id game_train --tokenizer weights/base_model` | ⚠️本地部分（需 `transformers` 精确版本；本机 5.14.1 未验证） | `docs/DATA_PREPARATION.md:10-16` |
| 校验 GAM JSONL | `python tools/validate_gam_jsonl.py --input data/game/train.gam.jsonl --task bbox` | ✅本地（仅 `grounding_anything` + `train.data.row_contract`） | `tools/validate_gam_jsonl.py:35-36`、`docs/LOCANY_MIGRATION.md:24` |
| LocateAnything → GAM 转换 | `python tools/convert_locany_to_gam.py --input la.json --output gam.jsonl` | ✅本地（同上） | `tools/convert_locany_to_gam.py:231-237`、`docs/LOCANY_MIGRATION.md:18` |
| 生成 tokenizer manifest | `.venv-train/bin/python -m train.tokenizer.validate_tokens weights/base_model --write-manifest weights/base_model/gam_tokenizer_manifest.json` | ❌本地（POSIX venv + 需权重目录） | `docs/DATA_PREPARATION.md:28-29` |
| 生成采样清单 | `.venv-train/bin/python -m train.dlm.build_sampling_manifest --config data/sources.yaml --output-dir data/general/sampling --sample-size 2997407 --seed 32 --stage general` | ❌devbox | `docs/DATA_PREPARATION.md:49-51` |
| 生成长度清单 | `.venv-train/bin/python -m train.dlm.build_selected_lengths --sampling-manifest … --output data/general/selected_lengths.npy --workers 8` | ❌devbox | `docs/DATA_PREPARATION.md:52-54` |
| 生成 packing manifest | `.venv-train/bin/python -m train.dlm.build_packing_manifest --sampling-manifest … --selected-lengths data/general/workload_lengths.npy --output-dir data/general/packing --epochs 2 --world-size 64 --pack-size 32 --packing-strategy balanced --seed 32` | ❌devbox | `docs/DATA_PREPARATION.md:86-90` |
| 准备发布 bundle | `python3 run.py prepare-model [--base weights/base_model --source weights/dlm --output weights/dlm_bundle]` | ❌本地（`run.py:98` POSIX venv；`infer/dlm/build_public_model_bundle.py:484-488`） | `README.zh-CN.md:52`、`docs/INFERENCE.md:11` |
| Game UI 评测（离线） | `python -m eval.game_ui.evaluate_game --input samples.jsonl [--output agg.json]` | ⚠️本地（需 `yaml`；`eval/game_ui/*` 无 GPU 依赖） | `docs/GAME_UI_EVALUATION.md:44`、`eval/game_ui/evaluate_game.py:15-16` |
| 跑批预测测试 | `python3 -m unittest discover -s tests -p test_batch_predict.py -v` | ⚠️本地（需服务假件由测试内建；无需 pytest） | `docs/BATCH_INFERENCE.md:90` |
| 跑 vLLM 部署回归 | `python -m unittest tests.test_vllm_deployment -v` | ⚠️本地（自称 CPU-only；但导入 `scripts.setup_vllm_environment` 需可 import，未见 GPU 依赖） | `tests/test_vllm_deployment.py:1,19-21` |
| 跑 pytest 风格测试 | `python -m pytest tests/test_gam_cache_builder.py tests/test_locany_to_gam.py` | ❌本地（`pytest` 未安装，且不在任何 requirements profile） | `tests/test_gam_cache_builder.py:4`、`tests/test_locany_to_gam.py:4` |

---

## 2. 缺失项清单

### M-01（重要更正）`GroundAnything-Model/` **已完整就位且权重校验通过**

任务派发时给的上下文说「model.safetensors 尚未就位，本地只有正在下载中的 `.filepart`」。**实测与此不符，请以本节为准**：

| 检查 | 结果 |
| --- | --- |
| `GroundAnything-Model/model.safetensors` | **存在**，`9,687,419,736` 字节，`LastWriteTime = 2026-10-05 00:09:22` |
| `*.filepart` 残留 | 全目录递归搜索 **无任何 `.filepart`** |
| SHA256 | `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`，与 `GroundAnything-Model/checksums.sha256` 中 `model.safetensors` 的记录**完全一致（MATCH）** |
| 用途确认 | `GroundAnything-Model/README.md:32`「This repository contains the GroundAnything **DLM** checkpoint」；`config.json` 的 `auto_map` 指向 `GroundAnythingModel/…ForConditionalGeneration` |
| SGLang bundle 关键键 | safetensors header 内 **569 个张量，全部以 `base_model.` 开头**；`base_model.model.language_model.embed_tokens.weight` **存在**（正是 `infer/serve_sglang.py:87` 读取的键） |
| 词表对齐 | `embed_tokens` 形状 `[152671, 2560]`；`added_tokens.json` 最大 id = 152670，内容为 `\|<MASK>\|` → 满足 `infer/serve_sglang.py:88` 的 `mask[0] == vocab - 1`；`<999>`=152668、`</c>`=152669 为倒数第二/第三个新增 token |
| `chat_template.jinja` | 存在（`infer/serve_sglang.py:47` 需要） |

**结论**：`GroundAnything-Model/` 就是一个**满足 SGLang DLM 入场校验的完整 bundle**，只是目录名不是 `weights/dlm_bundle`。
**但**：`weights/` 被 `.gitignore:4` 排除，**不能进 git**；`configs/release/dlm_sglang_*.yaml:4` 与 `configs/eval/dlm.yaml:2` 都硬指向 `weights/dlm_bundle`，因此要在 devbox 上用它必须**建立一份拷贝/软链**，或改 YAML（本次未做，属越界）。

### M-02 缺失目录（本地不存在）

| 路径 | 谁需要 | 后果 | 出处 |
| --- | --- | --- | --- |
| `vendor/`（含 `vendor/transformers`、`vendor/sglang`、`vendor/lmms-eval`） | serve / train / eval / vllm 全部 | 任何 `verify_dependency` 立刻失败；`scripts/evaluate.py:176,262` 与 `models/dependency_contract.py:14` 会报 `prepare dependency snapshot first` | `third_party/README.md:27-33`、`models/dependency_contract.py:14` |
| `weights/`（含 `weights/base_model`、`weights/dlm`、`weights/dlm_bundle`、`weights/vlm`） | train / serve / eval | `scripts/run.py:118-125` 的 `missing_inputs` 非空 → `Provide missing inputs before execution.` | `docs/DATA_PREPARATION.md:5`、`configs/release/dlm_sglang_denoise.yaml:15` |
| `data/`（含 `data/eval`、`data/smoke/**`、`data/rl/*.jsonl`） | train / eval | `data_root: data/eval`（`configs/eval/dlm.yaml:5`）不存在；训练审计报告 `data/smoke/lengths/with_workload.json` 等 3 份也不存在 | `configs/train/dlm_h800_smoke.yaml:65-67` |
| `outputs/` | 全部 | 评测要求输出目录**必须不存在**（`scripts/evaluate.py:272`），所以这不是「缺失」而是「预期为空」 | `scripts/evaluate.py:272` |
| `.venv-train / .venv-serve / .venv-eval / .venv-vllm` | 全部 | `run.py:125-129` 报 `environment missing; run: python3 run.py setup <profile>` | `run.py:91,125-129` |
| `configs/eval/full_suite_{denoise,speculative,vlm}.yaml` | 全量 30 任务评测 | 文档引用这 3 个文件（`eval/README.md:161,170,173,176`），但仓库里**只有生成脚本**（`eval/README.md:82-144`），**文件本身不发布** | `eval/README.md:125-128,141,147` |
| `docs/assets/demo.mp4` | README 演示 | **被 `.gitignore:57` 有意排除**，非缺陷 | `.gitignore:56-57` |

### M-03 本机环境缺口

- 缺 `datasets, pytest, accelerate, sglang, vllm, lmms_eval, openai`（实测 `importlib.util.find_spec`）。
- `torch 2.13.0+cpu` 无 CUDA（`torch.cuda.is_available() == False`）。
- 本地 `transformers 5.14.1` 与三个 profile 的要求（serve 5.5.4 / train·vllm 5.7.0 / eval `<5`）**都不匹配**。

### M-04 被 `.gitignore` 排除、因此「本地产物不会进 git」的路径

`.gitignore`（57 行）逐条：
`/vendor/`(2)、`/dist/`(3)、`/weights/`(4)、`/data/`(5)、`/outputs/`(6)、`/resources/`(7)、`/build/`(8)、`.venv*/`(9)、`__pycache__/`(10)、`.pytest_cache/`(11)、`*.py[cod]`(12)、`*.egg-info/`(13)、`/release/`(16)、`.env`/`.env.*`(19-20，带 3 个 `!` 白名单 21-23)、`credentials.json`/`credentials.*.json`(24-25)、`.netrc`(26)、`.ossutilconfig`(27)、`.aliyun/`(28)、`.aws/`(29)、`.ssh/`(30)、`**/.docker/config.json`(31)、`id_rsa`(32)、`id_ed25519`(33)、`*.key`(34)、`*.pem`(35)、`*.p12`(36)、`*.pfx`(37)、`secrets/`(38)、`logs/`(41)、`wandb/`(42)、`runs/`(43)、`checkpoints/`(44)、`profiler/`(45)、`*.log`(46)、`events.out.tfevents.*`(47)、`core`/`core.*`(48-49)、`*.prof`(50)、`*.nsys-rep`(51)、`*.ncu-rep`(52)、`*.trace.json`(53)、`*.trace.json.gz`(54)、`/docs/assets/demo.mp4`(57)。

实测 `git check-ignore -v`：`vendor/`(2)、`weights/`(4)、`data/`(5)、`outputs/`(6)、`resources/`(7)、`release/`(16)、`dist/`(3)、`build/`(8)、`.venv-eval/`(9) 全部命中。
**跨机同步含义**：`vendor/`、`weights/`、`data/`、`outputs/`、`.venv*/` **都不会随 git 走**，必须单独同步（见风险 R-07）。

`git status --porcelain` 当前只有 3 项：
```
?? "GroundAnything Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed.pdf"
?? GroundAnything-Model/
?? docs/investigation/
```
**注意 `GroundAnything-Model/` 并不在 `.gitignore` 里**（实测 `NOT IGNORED`）——一旦有人 `git add -A`，会把 **9.7 GB safetensors 与 45 MB demo.mp4 一起加进仓库历史**。这是本仓库最高危的误操作点之一（R-10）。

---

## 3. 风险清单

### R-01 入口脚本是 POSIX 专用（阻断级，但只在真实执行时触发）

| 文件:行 | 内容 |
| --- | --- |
| `run.py:98` | `python = str(root / venv / 'bin/python')` —— Windows 应为 `Scripts\python.exe` |
| `run.py:143` | `os.execv(...)` —— Windows 上 `execv` 行为不同 |
| `scripts/setup_environment.py:33` | `python = str(venv / 'bin/python')` |
| `scripts/setup_vllm_environment.py:22` | `python = str(target / "bin/python")` |
| `run.py:106-108` | 命令拼装时未做平台分支 |

**影响**：Windows 上任何非 `--dry-run` 的 `run.py` 调用都会在 `run.py:125-129` 以 `environment missing` 结束。**缓解**：本机只把 `run.py` 当「命令预览器」（`--dry-run`）。

### R-02 进程组控制依赖 POSIX 语义

| 文件:行 | 内容 |
| --- | --- |
| `scripts/evaluate.py:203` | `start_new_session=True`（Windows 语义不同） |
| `scripts/evaluate.py:222` | `os.killpg(process.pid, signal.SIGKILL)` |
| `eval/eval_runner.py:624` | worker 也 `start_new_session=True` |
| `eval/eval_runner.py:638,647,649` | `os.killpg(pid, sig)`（TERM → 0 探测 → KILL） |

**影响**：Windows 上 `os.killpg` 不可用；评测中断的清理会退化甚至抛异常。`eval/process_control.py:1` 自己承认「SIGKILL and machine loss cannot run cleanup」。

### R-03 破坏性删除（会真删数据）

| 文件:行 | 操作 | 触发条件与护栏 |
| --- | --- | --- |
| **`eval/eval_runner.py:738-745`** | `shutil.rmtree(<EVAL_LOG_ROOT>/log/gpt_response)` | 默认执行！仅 `--skip_cleanup` 可跳过（`:739`）。**`scripts/evaluate.py:150` 恒定传 `--skip_cleanup`**，所以走 YAML 前端时安全；**手工直接调 `eval.eval_runner` 会删掉共享的 `gpt_response` 目录** |
| `train/tokenizer/extend_tokens.py:276-288` | `model_dir.rename(backup)` → `staging.rename(model_dir)`，`--delete-backup-after-validation` 时 `shutil.rmtree(backup)`（`:288`） | **原地改写模型目录**；失败会尝试回滚（`:279-281`）。默认保留备份，加了该开关才删 |
| `train/tokenizer/extend_tokens.py:292` | `shutil.rmtree(staging, ignore_errors=True)` | 异常路径清理 |
| `infer/overlay/prepare_vllm_overlay.py:75-76,98,103` | 拒绝已存在目录（`:75-76`），临时目录 + `os.replace(temporary, output_dir)`（`:98`），`finally` 里 `shutil.rmtree(temporary)` 并**再次校验清理路径在 root 内**（`:101-102`） | 护栏最完整的一个；但 `os.replace` 是**整目录替换**语义 |
| `scripts/prepare_dependencies.py:28,42,44` | `tempfile.mkdtemp` → `os.replace(temporary, destination)`；`finally shutil.rmtree(temporary)` | 目标已存在时**不覆盖**，走 `already_verified`（`:22-26`） |
| `infer/dlm/build_public_model_bundle.py:497,513` | 写 `.bundle-in-progress` 标记（`:497`），结束时 `unlink`（`:513`） | 进程被杀会留下标记文件 |
| `infer/dlm/build_public_model_bundle.py:271` | `target.unlink()` | tokenizer 元数据调和时的删除，未逐行核对上下文（**未确认**） |
| `train/rl/branches/dare/materialize_*.py:161/238, 164/241` | `os.replace(staging, output)` + `finally shutil.rmtree(staging, ignore_errors=True)` | 输出目录整体替换 |
| `eval/utils/ensure_results_alias.py:56,58` | `os.replace(temporary, canonical)`；`temporary.unlink(missing_ok=True)` | 别名覆盖 |

**通用护栏**：`scripts/evaluate.py:272` 用 `mkdir(exist_ok=False)` **拒绝复用输出目录**；`eval/README.md:147` 说生成的配置拒绝覆盖；`third_party/README.md:26` 说重建工具「never overwrites existing output files」。整体上「覆盖式写入」被设计成**显式失败**而非静默覆盖，这是优点。

### R-04 `GroundAnything-Model/` 与 serve 路线的 model_type 不匹配

| 文件:行 | 要求 | 实际 |
| --- | --- | --- |
| `infer/serve_sglang.py:82-89` | 只校验 mask token 与词表对齐（**不查 `model_type`**） | ✅ `GroundAnything-Model` 满足（见 M-01） |
| `infer/serve_sglang_vlm.py:44-47` | `model_type == "groundinganything_vlm"` 且 `architectures == ["GroundAnythingVLMForConditionalGeneration"]` | ❌ 实际是 `"groundinganything"` / `["GroundAnythingForConditionalGeneration"]` |
| `infer/serve_sglang_vlm.py:51` | 需要 `model.safetensors.index.json` | ❌ 该文件**不存在**（单文件 safetensors） |
| `scripts/prepare_model_code.py:18-19` | `model_type == "groundinganything_vlm"` | ❌ 同一个不匹配 |
| `infer/dlm/model_contract.py:9-10` | 接受 `{groundinganything, groundinganything_vlm}` + `text_config.model_type == "qwen3"` | ✅ 满足 |
| `infer/serve_vllm.py:52-53` | `source_files(model, ROOT)` 先拒 DLM | VLM 路线需改用 `weights/vlm`（本地缺失，M-02） |

**影响**：**把 `GroundAnything-Model/` 当作 VLM 检查点喂给 `serve_sglang_vlm.py` / `prepare_model_code.py` 会在入场检查处直接失败**；它只适用于 DLM（`serve_sglang.py`）路线。

### R-05 静默失败 / 静默降级的路径

| 文件:行 | 行为 |
| --- | --- |
| `scripts/evaluate.py:186-187` | `summarize()` 里 `except (ValueError, OSError): continue` —— **坏掉的指标 JSON 被静默跳过**，只在 `missing_tasks` 里体现 |
| `scripts/evaluate.py:232` | `code = code or (2 if summary['missing_tasks'] else 0)` —— 只要子进程返回非零就用子进程码，**缺失任务被掩盖** |
| `eval/eval_runner.py:744-745` | 清理缓存失败只 `print_warn` |
| `eval/eval_runner.py:494-495,547-548,554-555` | 任务缺 CPU/像素/生成配置时只 `print_warn` 然后**用默认值继续** |
| `tools/validate_gam_jsonl.py:29` | 遇错立即 raise（**不静默**，属于正面） |
| `train/tokenizer/extend_tokens.py:292` | `ignore_errors=True` |
| `scripts/evaluate.py:178` | `GAM_EVAL_RESPONSE_AUDIT_PATH` 指向 `output/<responses.jsonl>` —— 若 runner 未写该文件，前端不会检查 |
| `eval/task_config.py:163-174` | 未在表里显式配置的任务**静默回退**到 `DEFAULT_*`；`eval/eval_runner.py:750-754` 未识别的 `model_type` 也静默回退 |

### R-06 两个 Transformers 版本共存，锁定的源码摘要互相矛盾

| 文件:行 | 声明版本 |
| --- | --- |
| `requirements/train.txt` | `transformers==5.7.0` |
| `requirements/serve.txt` | `transformers==5.5.4` |
| `requirements/eval.txt:9` | `transformers>=4.39.2,<5` |
| `models/qwen35_runtime.py:28-37` | 摘要锁的是 **Transformers 5.2.0 released wheel**（注释明说「Do not reuse the old 5.2.0.dev0 image digest」） |
| `infer/dependency_guard.py:12,25` | 要求 bundled **Transformers 5.7.0** |
| `infer/serve_vllm.py:60` | 要求 **5.7.0** 且在 `vendor/transformers/src` 下 |
| 本机实测 | `transformers 5.14.1` |

**影响**：`models/qwen35_runtime.py` 的 fail-closed 摘要校验（`:44-46`）很可能与 5.7.0 运行时冲突 → 一旦上游方法变过一次，快路径直接不可用。**未验证**（无 GPU、无 vendor）。

### R-07 跨机同步会丢东西

| 风险点 | 出处 |
| --- | --- |
| `vendor/`（约 3900 个文件、12 MB 压缩包）不进 git | `.gitignore:2`、`third_party/README.md:12` |
| `weights/`+`GroundAnything-Model/`（9.7 GB）不进 git**（若放 `weights/` 下）**；**但 `GroundAnything-Model/` 在仓库根，未被忽略** | `.gitignore:4` + 实测 `git check-ignore` |
| `data/`（评测集 + 训练 cache + RL JSONL）不进 git | `.gitignore:5`、`configs/datasets.yaml:1` |
| `outputs/`（评测结果、gate 文件）不进 git | `.gitignore:6`、`docs/DATA_PREPARATION.md:118` |
| `.venv*/` 不进 git，且**每个 profile 版本不同**（train torch 2.10 / serve torch 2.9.1 / eval torch 2.5.1-cpu） | `.gitignore:9`、`environments/install.yaml` |
| `.env`、`credentials*.json`、`secrets/`、`*.key/*.pem` 被排除 —— 如果 devbox 依赖这些，同步时会丢 | `.gitignore:19-38` |
| `GroundAnything-Model/` 是**未跟踪**状态；`git clean -xfd` 或 `git clean -fd` 会**直接删掉 9.7 GB 权重** | `git status --porcelain` 实测 |

### R-08 文档与实现不一致

| 文件:行 | 不一致 |
| --- | --- |
| `README.zh-CN.md:52` vs `README.md:124` | 前者让先跑 `prepare-model`，后者明说已发布的 DLM bundle「does **not** require prepare-model」 |
| `scripts/evaluate.py:90` | 报错文案说 tasks 来自 `configs/eval/tasks.json`，但 `task_inventory()`（`:26-37`）实际是 rglob `eval/**/*.yaml` 读 `task:` 字段；**`configs/eval/tasks.json` 在整个 Python 代码里从未被读取**（全局 grep 仅命中这一行文案） |
| `configs/eval/tasks.json` 有 **42** 个任务；`README.md:343` 说 **30** benchmarks；`eval/task_config.py:107-161` 只有 **39** 个 `gam_*` 配置项 | 三个数字口径不同，容易误判「配齐了没有」 |
| `eval/README.md:170-176` 引用 `configs/eval/full_suite_*.yaml`，但仓库不发布这些文件 | 必须先生成（`eval/README.md:82-144`） |
| `README.md:454` 说「Use `python3 run.py --help`」，但 `run.py:12` 要求子命令必填（`required=True`），单独 `--help` 可显示帮助但直接运行会报错 | 轻微 |
| `configs/release/dlm_sglang_denoise.yaml:15` 的 `inputs` 含 `vendor/sglang`，而 `weights/dlm_bundle` 本地并不存在 | `scripts/run.py:125` 会拦住 |

### R-09 平台/编码与脚本细节

| 文件:行 | 问题 |
| --- | --- |
| `docs/GAME_UI_EVALUATION.md:44` | 命令用 PowerShell 续行符 `` ` `` 且 `python -m eval.game_ui.evaluate_game` —— 在 bash 下不可用 |
| `docs/LOCANY_MIGRATION.md:24` | `--input data\game\train.gam.jsonl` 用 Windows 反斜杠 |
| `scripts/config_contract.py:16` | `relative()` **显式拒绝** `\`（`'\\' in value`）与 `~`、`${}`、`$()`、`://` —— 路径风格在 Windows 上会与文档冲突 |
| `tools/convert_locany_to_gam.py:194-196` | 输出用 `newline="\n"` 强制 LF（跨机 diff 友好） |
| `scripts/evaluate.py:273` | `run.json` 用 `json.dumps(..., ensure_ascii=False, indent=2)` 写文本但**未指定 `encoding`** —— 依赖平台默认编码（Windows 默认非 UTF-8，中文配置可能写出 GBK） |
| 同类：`scripts/evaluate.py:240`、`infer/serve_sglang.py:103`、`eval/game_ui/evaluate_game.py:30` 等都带 `encoding`/不带不一 | 需逐处确认（**未逐处核对**） |
| `eval/utils/eval_data_root.py:16-21` | `input_path` **故意允许绝对路径与 `~`**，与 `config_contract.relative()` 的严格形成对比 —— 数据集可来自仓库外，但这也意味着数据集路径不会被工程内约束 |
| `eval/utils/eval_data_root.py:102` | `missing_data_inputs` 用 `glob.glob` 判断存在性，`*` 通配符（`configs/datasets.yaml:34`）会掩盖「目录在但文件不对」的情况 |

### R-10 误提交 9.7 GB 权重的风险（本仓库最高危的 git 操作）

- `git status --porcelain` 实测 `?? GroundAnything-Model/`（**未被忽略**，`.gitignore` 里没有该规则）。
- `GroundAnything-Model/` 含 `model.safetensors` 9.69 GB + `assets/demo.mp4` 45 MB + `merges.txt` 1.67 MB + `tokenizer.json` 11.6 MB + `vocab.json` 2.78 MB。
- 一旦 `git add -A` / `git add .` 并 commit，仓库历史将被不可逆地膨胀。
- **缓解建议**（本次未执行，需 devbox/lead 决策）：把权重移入 `weights/`（已被 `.gitignore:4` 忽略），或在 `.gitignore` 显式加 `/GroundAnything-Model/`。

---

## 4. 未读 / 未确认项（不要当成已验证）

1. **未逐行读**的大文件：`models/dlm/hybrid.py`(97 KB)、`train/dlm/train_dlm.py`(80 KB)、`train/rl/rewards/grounding_bbox_reward.py`(76 KB)、`eval/metrics/detection_metrics.py`(71 KB)、`third_party/patches/sglang.json`(62 KB)、`eval/GUI/screenspot_pro/screenspot_utils.py`(59 KB)、`models/vlm/modeling_groundinganything_vision.py`(56 KB)、`eval/OCR/utils.py`(55 KB)、`infer/dlm/hierarchy.py`(54 KB)、`eval/utils/detection_utils.py`(37 KB)、`train/rl/branches/trace/trainer.py`(37 KB)、`infer/engines/sglang_gam/gam_qwen3.py`(35 KB)、`eval/eval_runner.py` 的 `:100-440` 与 `:560-813`。**它们的算法细节未被核对。**
2. `grounding_anything/protocol.py`（32 KB）只概括，`parse_answer(..., require_canonical=True)` 的语义**未确认**。
3. `train/launcher/common.py`(25 KB)、`train/launcher/manifest_contract.py`(23 KB)、`train/runtime/header_only_cache.py`(14 KB)、`train/dlm/data.py`(25 KB) 只取了符号表/CLI，**未读实现**。
4. `train/rl/distributed/multiroute_56.py` / `multiroute_64.py` **没有 argparse**，其真实启动方式**未确认**（可能是被 `scripts/run.py` 以模块方式调用，参数来自 `sys.argv` 或环境变量）。
5. `infer/dlm/openai_server.py` / `legacy_backend.py` / `qwen3_openai_server.py` 的 HTTP 端点与 `docs/INFERENCE.md` 声称的「reference backend」是否一致，**未核对**。
6. `configs/eval/*.yaml` 只精读了 `dlm.yaml`；`gam.yaml / rlv2.yaml / rexomni.yaml / locateanything.yaml / groundingdino.yaml / vlm.yaml / vlm_vllm.yaml / dlm_causal.yaml / dlm_speculative.yaml` 的字段值**未逐一核对**。
7. `configs/train/{general,specialist,sft1,sft2,vlm_lora}.yaml`、`configs/release/{rl_train,vlm_lora_train,dlm_serve,dlm_serve_single_bbox,vlm_vllm_gpu,vlm_vllm_ppu,vlm_sglang,dlm_train_h800_smoke,dlm_resume_h800_smoke}.yaml` 的字段值**未逐一核对**（只读了 `general_train.yaml`、`dlm_sglang_denoise.yaml`、`dlm_h800_smoke.yaml`）。
8. `eval/**/*.yaml` 42 个任务定义文件的 `task:` 名、`include:` 链、`generation_kwargs` **未读**。
9. `third_party/manifest.json` 只读了顶层结构、3 个依赖的 `destination/files 数/source_revision/archive`，**未核验任何单个文件哈希**（未跑 `verify_dependency`，因为 `vendor/` 不存在）。
10. 本地能否真正跑通 `python -m unittest tests.test_vllm_deployment`、`tests.test_batch_predict`、`tools/validate_gam_jsonl.py`：**本次按纪律未执行任何测试/工具**，只是静态判断为「理论上可本地跑」。
11. Windows 默认编码对 `scripts/evaluate.py:273` 等无 `encoding` 的 `write_text` 的具体影响：**未确认**。
12. `GroundAnything-Model/` 与 `weights/dlm_bundle` 在 `infer/serve_sglang.py` 下的**端到端可用性未验证**（本机无 GPU；只验证了入场检查所需的静态条件）。

---

*报告结束。全部结论均指向 `文件:行号`；无法确认处已标「未确认」。本文件是本次侦察唯一写入的产物。*
