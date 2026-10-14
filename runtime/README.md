# ch07 联合单帧运行资产

| 子目录 | 内容 |
| --- | --- |
| `weights/` | 冻结 YOLO26s 权重 |
| `geometry/` | canonical 轨面映射 |
| `reference/` | 平移配准参考帧 |
| `manifests/` | 正常划分、配准记录与区域开发索引 |
| `artifacts/` | 区域判定器与正常资产构建目录 |

现有 `region_classifier.joblib` 来源于开发区域候选。使用仓库 `configs/runtime_build.example.json` 和 `scripts/build_open_set_artifacts.py`，根据 `data/annotations/ch07/` 与清单生成 `pixel_normal_bank.npz`、`wrn_normal_memory.npz` 及资产哈希元数据。统一推理配置见 `configs/unified.example.json`。
