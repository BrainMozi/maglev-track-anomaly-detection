# 联合单帧运行

YOLO 在固定业务 ROI 内检测 `person/vehicle`，开放集模块在 canonical 轨面内输出未知异物区域，两条链在统一 JSON 和叠加图中汇合。

```text
原始帧 ─┬─ ROI → YOLO26s → 已知目标框
        └─ 双侧定子带配准 → canonical采样 → WRN Layer2+Layer3正常记忆
                              → 高置信种子 → 像素正常差异 → Graph Cut
                              → 区域判定 → 异物mask
                                              ↓
                                      联合JSON与叠加图
```

## 1. 环境

在仓库根目录创建 Python 3.10+ 环境，按 `pyproject.toml` 安装 `inference` 依赖，GPU 机器构建 WRN 资产。

```bash
pip install -e ".[inference,dev]"
pytest -q
```

## 2. 开放集运行资产

`runtime/` 放置 ch07 YOLO 权重、轨面几何映射、参考帧、正常划分/配准清单和区域判定器。使用 `configs/runtime_build.example.json` 与 `runtime/manifests/development_frame_index.csv` 构建像素正常库、WRN 正常记忆及区域模型：

```bash
python scripts/build_open_set_artifacts.py \
  --config configs/runtime_build.example.json \
  --frame-index runtime/manifests/development_frame_index.csv \
  --out runtime/artifacts --device cuda
```

产物为 `pixel_normal_bank.npz`、`wrn_normal_memory.npz`、`region_classifier.joblib` 和 `artifacts_meta.json`（含各文件 SHA-256）。

## 3. 单帧或目录推理

`configs/unified.example.json` 已将模型及轨面资产指向 `runtime/`。以一张 ch07 图像为例：

```bash
python scripts/process_frames.py \
  --config configs/unified.example.json \
  --input data/annotations/ch07/ch07_20251104161952_20251112171033_000001.jpg \
  --out outputs/example --save-overlays alerts
```

`--input` 可传一张图片或目录；批量任务可用 `--shard-index/--shard-count` 切分，`hpc/process_frames_array.sbatch.example` 提供 Slurm 数组模板。汇总命令：

```bash
python scripts/summarize_frame_results.py outputs/example
```

输出颜色约定：绿色 `person`、橙色 `vehicle`、红色半透明区域为开放集异物。统一 JSON 的字段定义在 `schemas/unified_frame.schema.json`。
