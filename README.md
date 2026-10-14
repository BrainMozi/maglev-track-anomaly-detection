# 磁域鹰眼

固定机位磁浮线路视觉监测：**YOLO26s 检测保护区人员、车辆；开放集方法检测轨面未知异物**。同一帧输出目标框、异物区域、叠加图和统一 JSON。

## 技术路线

```text
监控视频 → 抽帧 → 七点位场景图像
                  ├─ 保护区 ROI → YOLO26s → 人员/车辆检测框
                  └─ 轨面配准 → canonical 正则化
                     → WRN Layer2+Layer3 正常记忆 → 异常种子
                     → 颜色/梯度像素差异 → Graph Cut 轮廓细化
                     → 区域判定器 → 异物 mask
两条检测结果 → 统一 JSON 与叠加图
监控交互原型 → 线路态势、摄像头、告警、回看和事件台账
```

## 数据与标注

- **YOLO**：现有 3116 张图及对应标签，训练/验证/测试为 **2174/629/313**。ch07 的 **117 份人工精确框标注**在 `data/annotations/ch07/`；AI 生成的框仅作后续人工标注的预标注。
- **轨面异物**：3275 张，正常 **2151**、异常 **1124**，正常∶异常约 **1.91∶1**。**124 张人工圈定的异物区域 mask** 单独保存；AI 生成的 mask 仅作后续人工标注的预标注。
- 七点位去重候选帧共 **3011 张**。各划分比例与标注来源见 [`data/README.md`](data/README.md)。

## 文件入口

| 目录 | 内容 |
| --- | --- |
| `src/maglev_anomaly/`、`scripts/` | 双任务算法、资产构建、推理与汇总 |
| `research/yolo/` | YOLO 数据准备与训练 |
| `data/` | 七点位图像、人工标注、AI 预标注 |
| `runtime/`、`configs/` | 权重、正常参照、轨面映射、配置 |
| `demo/monitor/` | 监控系统交互原型 |
| `docs/`、`schemas/`、`tests/` | 方法说明、结果格式与测试 |

数据与运行资产作为本仓库 [v0.1.0 Release](https://github.com/BrainMozi/maglev-track-anomaly-detection/releases/tag/v0.1.0) 的分包发布。下载所需 ZIP 后解压到仓库根目录，即可得到表中的 `data/`、`runtime/` 结构；详细数量见 [`data/README.md`](data/README.md)。运行命令见 [`docs/UNIFIED_SINGLE_FRAME.md`](docs/UNIFIED_SINGLE_FRAME.md)。
