# src/tools/metrics.py
"""公共指标库：分类 + 检测统一口径。"""

import json
import numpy as np

# ---------------- 分类 ----------------
def accuracy(y_true, y_pred) -> float:
    """返回准确率。"""
    raise NotImplementedError("空壳，待 D 实现")

def precision_recall_f1(y_true, y_pred, average: str = "macro") -> dict:
    """返回精确率、召回率、F1。"""
    raise NotImplementedError

def confusion_matrix(y_true, y_pred, num_classes: int = 6) -> np.ndarray:
    """返回混淆矩阵。"""
    raise NotImplementedError

# ---------------- 检测 ----------------
def compute_ap(recall, precision) -> float:
    """单类 AP。"""
    raise NotImplementedError

def compute_map(pred_boxes, gt_boxes,
                iou_threshold: float = 0.5,
                num_classes: int = 6) -> dict:
    """返回 mAP@0.5、每类 AP、精确率、召回率。"""
    raise NotImplementedError

def pr_curve(pred_boxes, gt_boxes, iou_threshold: float = 0.5) -> dict:
    """返回 PR 曲线数据。"""
    raise NotImplementedError

# ---------------- 通用 ----------------
def count_params(model) -> int:
    raise NotImplementedError

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)