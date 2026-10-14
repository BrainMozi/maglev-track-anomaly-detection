# YOLO 已知目标链

已知目标链在固定机位业务 ROI 中识别 `person` 与 `vehicle`，与开放集轨面异常链在统一结果中汇合。

- ROI：`(640, 270, 1920, 1080)`；训练图像尺寸 1280，推理尺寸 960。
- 推理阈值：person 0.02、vehicle 0.15；NMS IoU 0.70。
- 冻结权重：`runtime/weights/yolo26s_ch07_final.pt`，SHA-256 `1974166f0e12beba1aa2c35f3c8ca0247fc2587c02ab54bad8787c872d30ae79`。
- 独立测试总体：Precision 0.481、Recall 0.864、F1 0.618、F2 0.745。

`prepare_dataset.py` 将 ch07 X-AnyLabeling 标注转换成 YOLO 训练数据，`train_evaluate.py` 实现训练、验证集阈值选择和测试。训练数据见 `data/annotations/ch07/` 与 `data/prelabels/yolo/`，运行模型由 `src/maglev_anomaly/known_yolo.py` 加载。
