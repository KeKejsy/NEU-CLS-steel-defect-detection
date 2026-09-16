"""成员 C · C 方案（学习式重排序）公共部分：特征构造、缓存读写、GT 对齐

## 这个方案要解决什么

诊断已经确认：真正卡住 mAP 的不是定位、也不是判别模型的准确率，而是**排序**。
两阶段方案的排序完全依赖一个**固定公式**（验证器与精修器概率的几何平均）
加一个**全局阈值**。C 方案用一个极小的学习式打分器替掉这一层：

    输入：两个模型的原始概率 pv/pr + 候选框与精修框的几何量
    输出：6 类各自的分数
    用途：分数直接当作该候选的置信度送进原有的阈值 + NMS 管线（后处理一行不改）

它**不碰** CNN、不动滑窗、不改评估口径，所以风险低、可解释、且与 0.2473 那条链
完全可比（同一个缓存里能同时算出「基线几何平均」与「重排序」两种分数）。

## 特征设计（26 维）

| 组 | 维度 | 理由 |
|---|---|---|
| 验证器 7 类概率 | 7 | 判别模型之一，分类头训练更充分 |
| 精修器 7 类概率 | 7 | 判别模型之二，见过真实候选分布（误差互补） |
| 精修框几何 | 6 | 宽、高、面积、对数长宽比、中心 x/y —— 大框更容易是误检 |
| 精修相对候选的位移 | 4 | 精修把框拉动了多少：拉动大说明候选本身不准 |
| 融合量 | 2 | 几何平均的缺陷概率与最大类概率（基线公式本身作为特征喂进去）|

## 标签

对每个窗口与每个类别 c：标签 1 当且仅当「精修框」与某个 c 类 GT 的 IoU ≥ 0.5。
训练只在**训练集**上做，评估只在**验证集**上做，绝不交叉。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import data as data_mod  # noqa: E402
from tools.tune_fused import get_gts  # noqa: E402

CN = data_mod.CLASS_NAMES
N_CLS = len(CN)
BG = N_CLS            # 背景在 7 类概率里的下标
FEATURE_NAMES = ([f"pv_{c}" for c in CN] + ["pv_bg"] +
                 [f"pr_{c}" for c in CN] + ["pr_bg"] +
                 ["box_w", "box_h", "box_area", "box_log_ar", "box_cx", "box_cy",
                  "d_w", "d_h", "d_cx", "d_cy",
                  "fuse_def", "fuse_cls_max"])
FEATURE_DIM = len(FEATURE_NAMES)


def load_cache(path):
    """读 rerank_cache.py 导出的 npz，返回 (每图数组列表, stems)"""
    z = np.load(path, allow_pickle=False)
    n = int(z["n"])
    refs = [z[f"r{i}"] for i in range(n)]
    pvs = [z[f"v{i}"] for i in range(n)]
    prs = [z[f"p{i}"] for i in range(n)]
    stems = [str(s) for s in z["stems"]]
    return refs, pvs, prs, stems


def build_features(ref, pv, pr, cand=None):
    """单张图内所有窗口的特征矩阵

    ref/pv/pr: (N,4) / (N,7) / (N,7)
    cand: 可选 (N,4) 候选框；不给时用 ref 自己（位移项为 0），用于老缓存
    """
    n = ref.shape[0]
    w = np.clip(ref[:, 2] - ref[:, 0], 1e-6, None)
    h = np.clip(ref[:, 3] - ref[:, 1], 1e-6, None)
    cx = (ref[:, 0] + ref[:, 2]) / 2
    cy = (ref[:, 1] + ref[:, 3]) / 2
    area = w * h
    log_ar = np.log(w / h)
    if cand is None:
        dw = dh = dcx = dcy = np.zeros(n, dtype="float32")
    else:
        cw = np.clip(cand[:, 2] - cand[:, 0], 1e-6, None)
        ch = np.clip(cand[:, 3] - cand[:, 1], 1e-6, None)
        dw = (w - cw) / cw
        dh = (h - ch) / ch
        dcx = ((ref[:, 0] + ref[:, 2]) - (cand[:, 0] + cand[:, 2])) / 2 / cw
        dcy = ((ref[:, 1] + ref[:, 3]) - (cand[:, 1] + cand[:, 3])) / 2 / ch
    fuse_def = np.sqrt(np.clip(1.0 - pr[:, BG], 1e-9, None) *
                       np.clip(1.0 - pv[:, BG], 1e-9, None))
    fuse_cls = np.sqrt(np.clip(pr[:, :N_CLS], 1e-9, None) *
                       np.clip(pv[:, :N_CLS], 1e-9, None)).max(-1)
    return np.stack([pv[:, 0], pv[:, 1], pv[:, 2], pv[:, 3], pv[:, 4], pv[:, 5], pv[:, 6],
                     pr[:, 0], pr[:, 1], pr[:, 2], pr[:, 3], pr[:, 4], pr[:, 5], pr[:, 6],
                     w, h, area, log_ar, cx, cy,
                     dw, dh, dcx, dcy, fuse_def, fuse_cls], axis=1).astype("float32")


def baseline_scores(ref, pv, pr):
    """基线的排序口径（交付版那一套）：几何平均缺陷分数 + 按 pv*pr 取类别"""
    def_p = np.sqrt(np.clip(1.0 - pr[:, BG], 1e-9, None) * np.clip(1.0 - pv[:, BG], 1e-9, None))
    labels = (np.clip(pv[:, :N_CLS], 1e-9, None) * np.clip(pr[:, :N_CLS], 1e-9, None)).argmax(-1)
    return def_p.astype("float32"), labels.astype("int64")


def iou_pairs(boxes, gt):
    """boxes (N,4) 与单框 gt (4,) 的 IoU（归一化 xyxy）"""
    lt = np.maximum(boxes[:, :2], gt[:2])
    rb = np.minimum(boxes[:, 2:], gt[2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    aa = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    ag = max((gt[2] - gt[0]) * (gt[3] - gt[1]), 1e-9)
    return inter / np.maximum(aa + ag - inter, 1e-9)


def labels_for_image(ref, gts_img, iou_th=0.5):
    """(N,4) 精修框与单图 GT -> (N,6) 多标签矩阵（IoU>=iou_th 即该类为正）"""
    n = ref.shape[0]
    y = np.zeros((n, N_CLS), dtype="float32")
    gb = gts_img["boxes"]
    gl = gts_img["labels"]
    for c in range(N_CLS):
        sel = [gb[i] for i in range(len(gl)) if int(gl[i]) == c]
        for g in sel:
            y[:, c] = np.maximum(y[:, c], (iou_pairs(ref, g) >= iou_th).astype("float32"))
    return y
