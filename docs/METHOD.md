# 方法与实现

固定机位磁浮监控由两条视觉链路处理同一张图像。已知目标链在保护区业务 ROI 内用 YOLO26s 检测人员和车辆；开放集链通过配准及 canonical 正则化，在轨面上建立逐位置正常参照并发现偏离正常的候选异物。

## 已知目标

`research/yolo/` 包含数据构建、训练与评估过程，`src/maglev_anomaly/known_yolo.py` 将冻结权重接入单帧处理。ROI 为原图 `(640, 270, 1920, 1080)`；统一输出记录类别、位置和置信度。

## 开放集异物

1. 通过参考帧与双侧定子带配准，将当前画面映射到 `2379×320` 的 canonical 轨面。
2. WideResNet50-2 的 Layer2、Layer3 中层网格特征与正常记忆逐位置比较，生成异常热力图；标准化异常分数达到 3.0 的位置形成候选种子。
3. 由正常轨面统计颜色、梯度的 Median-MAD 及位置分位数；像素差异为 Graph Cut 提供边界证据。
4. 在候选区域上汇总热图、像素、面积与边界特征，经区域判定器保留异物响应，输出 canonical/original mask。

实现位于 `src/maglev_anomaly/registration.py`、`wrn_memory.py`、`pixel_anomaly.py`、`graphcut.py`、`region.py`、`open_set.py`。

## 联合输出

`src/maglev_anomaly/unified.py` 将已知目标框与开放集 mask 汇合；可选叠加图与逐帧 JSON 提供两条链路的空间重合信息。交互界面位于 `demo/monitor/`，用于呈现态势、告警、回看与事件台账的操作过程。

## 数据划分

七点位去重候选帧在 `data/selected_frames/`；ch07 原图和初步人工标注在 `data/annotations/ch07/`；按 train/val/test 整理的两任务模型预标注数据在 `data/prelabels/`。运行参数与模型资产由 `configs/`、`runtime/` 给出，详情见 [DATA.md](DATA.md)。
