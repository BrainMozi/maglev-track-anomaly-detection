# 数据与运行资产

## 数据层

| 目录 | 内容 | 用途 |
| --- | --- | --- |
| `data/selected_frames/ch01,ch02,ch03,ch04,ch06,ch07,ch08/` | 七点位共3011张 ROI 变化筛选帧，按原视频帧名保存 | 场景数据浏览、后续选帧与标注 |
| `data/annotations/ch07/` | 282张原图、117份 X-AnyLabeling JSON、`classes.txt` | ch07 人工精确框标注 |
| `data/annotations/open_set_coarse/` | 124张 canonical 坐标人工圈定粗 mask 及完成记录 | ch07 异物区域粗标注 |
| `data/prelabels/yolo/images,labels/` | 已知目标任务按 train/val/test 划分的3116张图像及3116份AI预标注框；预标注用于后续人工标注 | `person/vehicle` 数据构建和复核 |
| `data/prelabels/open_set/images,masks/` | 异物任务按 train/val/test 划分的3275张图像及AI预标注mask；预标注用于后续人工标注 | 开放集数据构建和复核 |

`data/prelabels/` 的标签属于模型预标注；`data/annotations/ch07/` 保存人工精确框标注，`data/annotations/open_set_coarse/` 保存异物区域粗标注。两任务总量、各划分正常/异常比例和来源层级见 [`data/README.md`](../data/README.md)。文件名中的 `chXX_起录时间_生成时间_帧号` 保留机位与视频段信息。

原始监控视频经 1 fps 抽帧得到 49628 张全量底库；当前仓库数据层选用七点位 ROI 筛选帧。两任务划分数据由项目数据整理产生，和七点位目录可能包含相同源视频的不同抽帧，不按目录数量相加宣称唯一帧数。

## 运行资产

| 目录 | 内容 |
| --- | --- |
| `runtime/weights/` | ch07 YOLO26s 权重 |
| `runtime/reference/` | 配准参考帧 |
| `runtime/geometry/` | canonical 轨面映射 |
| `runtime/manifests/` | 正常划分、配准与开发索引 |
| `runtime/artifacts/` | 区域判定器与构建脚本输出的像素正常库、WRN正常记忆 |

`configs/runtime_build.example.json` 给出开放集正常库构建的本仓库路径，`configs/unified.example.json` 对应单帧联合推理的模型与几何资产。

## 输入输出契约

- 图片：JPEG/PNG，文件名与划分、配准表中的 `filename` 相对应。
- `normal_split.csv`：`filename`、`split`，split 为 `train_normal` 或 `val_normal`。
- `registration.csv`：`filename`、`dx_sub`、`dy_sub`、`status`。
- `canonical_map_v3_round2.npz`：`map_x`、`map_y`，开发场景形状 `2379×320`。
- 联合输出：逐帧 JSON 记录 YOLO 框、开放集 mask、配准状态与空间重合；格式参见 `schemas/unified_frame.schema.json`。

所有数据以原名保留，运行时读取原图，输出写入 `outputs/`。
