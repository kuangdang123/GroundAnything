# LA / GA 游戏 UI Grounding 评估

`eval/game_ui` 提供 LocateAnything 和 GroundAnything 共用的离线评估口径。它接收一行一个样本的 JSONL，避免把 LA 的旧输出解析器和 GA 的 canonical parser 混在一起。

## 输入契约

```json
{
  "id": "sample-001",
  "image": "images/a.png",
  "width": 1920,
  "height": 1080,
  "split": "unseen",
  "game": "game-b",
  "category": "button",
  "ground_truth": {
    "coordinate_space": "la_1000",
    "boxes": [[100, 100, 300, 300]]
  },
  "prediction": {
    "status": "success",
    "coordinate_space": "ga_999",
    "predictions": [{"label": "button", "coordinates": [52, 93, 156, 278]}],
    "latency_ms": 42.5
  }
}
```

`coordinate_space` 必须明确填写：`pixel`、`la_1000`、`ga_999` 或 `normalized`。评估器不会根据数值大小猜测坐标来源。LA 的 `0..1000` 和 GA 的 `0..999` 都先变换到像素空间再计算 IoU；预测中无效框会计入非法坐标，不会静默截断。

`status != success` 是 parse error。成功响应中的合法框继续参加匹配，非法框被计数并从匹配中剔除。GT 的 `boxes: []` 表示 None 样本；显式的 `none` 字段可作为数据生产侧的冗余标记，但评估以规范化后的框集合为准。

## 指标与匹配

对每个 IoU 阈值 `0.50, 0.55, ..., 0.95`，按 label 分组执行一对一最大基数匹配（Kuhn 风格增广路径，候选按 IoU 降序访问），然后统计 TP/FP/FN、precision、recall 和 F1。一个预测框最多命中一个 GT 框；这与 LA `maximum_matches` 的 one-to-one 约束一致，但这里明确采用类别感知匹配。LA 报告中 `official_flat_matches` 的跨类别 flat 口径不要与本结果直接横比。

输出包含 `F1@0.50`、`F1@0.75`、`F1@0.90`、`F1@0.50:0.95`、mIoU、匹配框的 IoU@0.75/0.90、None accuracy、parse error rate、illegal coordinate rate、`false_positive`、`false_negative` 和每类指标。`false_positive`/`false_negative` 是 IoU@0.5 下的框级计数；None 统计区分 `none_correct`、`none_false_positive` 与 `none_false_negative`，None accuracy 的分母只包含 GT None 样本。

延迟从每行 `latency_ms` 读取，给出 P50/P95 与 `throughput_samples_per_second`（按总推理延迟计算）。输出还带有 `groups.split` 和 `groups.game`，可直接比较 unseen game/generalization；不要把训练集、已见游戏和未见游戏混成一个数字。

## 运行

```powershell
python -m eval.game_ui.evaluate_game `
  --input outputs/game_ui_predictions.jsonl `
  --output outputs/game_ui_metrics.json
```

输入可以由 LA `evaluate_game.py` 或 GA `examples/batch_predict.py` 的结果转换得到。GA 的 `predictions[].coordinates` 原生是 canonical `0..999`，转换时标记 `coordinate_space: "ga_999"`；LA 的旧解析结果标记 `la_1000`，禁止省略字段。
