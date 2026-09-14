"""成员 C · 密集滑窗检测器（以区域验证器为检测核心）

## 为什么会有这个文件

本项目先实现了两个单阶段检测器（YOLOv3 / PP-YOLOE-s），但实测发现它们的**置信度排序**
在本数据集上不可用：

    - 框回归其实是准的：GT 所属特征点上的 IoU 达 0.742 / 0.757；
    - 但同一个框的 obj 分数只有 0.056，在 307 个候选里排第 135 位，被 top_k 截断淘汰；
    - 根因：每张图仅 2~3 个正样本 vs 上万条候选位置（占比 0.3%），
      obj 分支学不出判别力（试过 obj*cls、纯 cls、纯 obj、边缘能量等多种排序均无效）。

而「区域验证器」（core/verifier.py）学的是**稠密监督**的图块分类任务，实测非常可靠：

    - 验证集图块整体准确率 88.7%，缺陷类准确率 90.3%；
    - 在 GT 框上的缺陷概率均值 0.904（93.8% 超过 0.5），随机背景框仅 0.261，区分度 +0.643。

所以这里把验证器直接当作检测核心：**在多尺度滑窗上逐个窗口判「是不是缺陷、是哪类」**，
再用 NMS 去重。这样就不依赖任何「置信度排序」的近似假设。

## 与单阶段检测器的关系

- 检测器（YOLOv3 / PP-YOLOE-s）负责**定位**，本模块负责**判类与置信度**，二者互补；
- 本文件单独也能成检测器（纯滑窗），便于对比「网络结构 + 排序分支」与「稠密分类 + 滑窗」两条路线。

## 复杂度控制

滑窗数量 = 所有 (中心点 × 尺度) 的组合。本数据集图像仅 200×200 且缺陷偏大，
实测用较稀疏的网格即可（stride 24、3 个尺度、每图约 300 个窗口），
配合 batch 推理，单图耗时约 0.1 秒级。
"""

import itertools

import numpy as np
import paddle
from PIL import Image

from .data import CLASS_NAMES
from .verifier import BG_CLASS

# 默认滑窗尺度（相对图像短边的比例）、宽高比变体与网格步长
#
# **宽高比是本项目调优时最关键的一处**。各类缺陷的框形状差异极大
# （统计自全部 4189 个标注框，数值为占原图的比例）：
#
#     类别          宽中位   高中位   宽高比
#     Cr 龟裂       0.630   0.350    1.79   （横长）
#     In 夹杂       0.125   0.325    0.37   （细长竖条）
#     Pa 斑块       0.260   0.360    0.73
#     PS 麻点       0.625   0.955    0.76
#     RS 氧化铁皮压入 0.340   0.352    0.98
#     Sc 划痕       0.133   0.653    0.18   （极细长竖条）
#
# 最初只用 (0.6, 1.0, 1.7) 三档宽高比，**根本产生不出 In/Sc 需要的形状**，
# 因此这两类召回只有 0.49 / 0.51，而其余四类达 0.71~0.80。
# 加入 0.35 后（分层抽样实测）：In 的 AP 0.076 → 0.094、Sc 0.266 → 0.289，
# 整体 mAP 提升约 2%。更极端的 0.2 反而略降整体（候选数近乎翻倍带来的噪声
# 抵消了细长形状的收益），故不采用。
# **尺度上限也不要盲目加大**。曾因「PS 麻点的长边中位数达 0.962（几乎整幅图）、
# 而最大滑窗只有 0.65」而加入 0.9 与 1.0 两档大尺度，实测**反而明显变差**：
# 分层抽样 120 张上 mAP 0.2369 → 0.2061（加 0.9 档）、候选数 4624 → 5780。
# 原因是接近整幅图的窗口几乎必然覆盖到缺陷，验证器会给它们一致的高分，
# 只是徒增噪声而不带来有效定位信息。所以尺度上限保持 0.65。
DEFAULT_SCALES = (0.2, 0.3, 0.45, 0.65)
DEFAULT_ASPECTS = (0.35, 0.6, 1.0, 1.7)
DEFAULT_STRIDE = 16


def make_windows(img_w, img_h, scales=DEFAULT_SCALES, stride=DEFAULT_STRIDE,
                 aspects=DEFAULT_ASPECTS):
    """生成多尺度 + 多宽高比的滑窗（归一化 xyxy）

    尺度按「相对短边的比例」定义，因此与图像实际分辨率无关：
    原图 200x200 与放大到 416 的图会得到形状相同的归一化窗口集合。
    aspects 中的数值是「宽/高」：1.0 为正方形，<1 偏竖长，>1 偏横长。
    """
    short = min(img_w, img_h)
    wins = []
    for sc in scales:
        for ar in aspects:
            w = sc * short * (ar ** 0.5)     # 让不同宽高比的窗口面积大致相当
            h = sc * short / (ar ** 0.5)
            w, h = min(w, img_w), min(h, img_h)
            for cy in range(0, img_h, stride):
                for cx in range(0, img_w, stride):
                    wins.append([(cx - w / 2) / img_w, (cy - h / 2) / img_h,
                                 (cx + w / 2) / img_w, (cy + h / 2) / img_h])
    return np.asarray(wins, dtype="float32")


class DenseWindowDetector:
    """把区域验证器当检测器用：多尺度多宽高比滑窗 + 逐窗判类 + NMS"""

    def __init__(self, verifier_infer, scales=DEFAULT_SCALES, stride=DEFAULT_STRIDE,
                 aspects=DEFAULT_ASPECTS):
        self.verifier = verifier_infer
        self.scales = scales
        self.stride = stride
        self.aspects = aspects

    @paddle.no_grad()
    def detect(self, img: Image.Image, score_threshold=0.5, nms_iou=0.4,
               max_det=50, top_n_windows=500):
        """对单张图做检测

        返回 dict：boxes(归一化 xyxy)、scores(缺陷概率)、labels(类别 id)
        """
        W, H = img.size
        wins = make_windows(W, H, self.scales, self.stride, self.aspects)
        if len(wins) == 0:
            return {"boxes": np.zeros((0, 4), dtype="float32"),
                    "scores": np.zeros(0, dtype="float32"),
                    "labels": np.zeros(0, dtype="int64")}

        prob = self.verifier.score(img, wins)               # (N, 7)
        defect_p = 1.0 - prob[:, BG_CLASS]                  # 缺陷置信度
        labels = prob[:, :BG_CLASS].argmax(-1)              # 缺陷类别

        keep = defect_p >= score_threshold
        if not keep.any():
            return {"boxes": np.zeros((0, 4), dtype="float32"),
                    "scores": np.zeros(0, dtype="float32"),
                    "labels": np.zeros(0, dtype="int64")}
        b, s, l = wins[keep], defect_p[keep], labels[keep]

        # 先按分数截断，再做逐类 NMS —— 控制 NMS 的规模
        if len(s) > top_n_windows:
            o = np.argsort(-s)[:top_n_windows]
            b, s, l = b[o], s[o], l[o]

        from .boxes import batched_nms, clip_boxes_norm
        tb = paddle.to_tensor(b)
        ts = paddle.to_tensor(s).astype("float32")
        tl = paddle.to_tensor(l).astype("int64")
        idx = batched_nms(tb, ts, tl, len(CLASS_NAMES), iou_threshold=nms_iou,
                          top_k=min(max_det, int(tb.shape[0])))
        return {"boxes": clip_boxes_norm(tb[idx]).numpy(),
                "scores": ts[idx].numpy(),
                "labels": tl[idx].numpy()}
