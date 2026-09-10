"""成员 C · 检测核心模块

两个检测网络（YOLOv3 / PP-YOLOE-s）与两阶段检测流程共用的基础件：
数据管线、几何运算、损失函数、网络基础层、区域验证器、滑窗检测器。

所有实现在原生 Paddle 上完成，不依赖 PaddleDetection，
因此不需要安装 numpy<2 等与当前环境冲突的依赖（详见 ../README_det.md 第一节）。
"""

from . import boxes, data, det_ops, layers, utils, verifier, window_detector  # noqa: F401

__all__ = ["boxes", "data", "det_ops", "layers", "utils", "verifier",
           "window_detector"]
