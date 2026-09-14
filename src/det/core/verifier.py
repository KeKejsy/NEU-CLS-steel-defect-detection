"""成员 C · 区域验证器（两阶段检测的第二阶段）

## 为什么要加这一层

单阶段检测器（YOLOv3 / PP-YOLOE-s）在本数据集上遇到了明确的问题，实测数据：

| 检查项 | 实测结果 |
|---|---|
| 在 GT 所属特征点上的框回归质量 | IoU **0.742 / 0.757**（定位是准的） |
| 该框的置信度分数 | obj 仅 0.056，在 307 个候选里排到**第 135 位** |
| 结果 | 被 top_k 截断淘汰，mAP@0.5 恒为 0（试过 obj*cls / cls / obj / 边缘能量等排序均无效） |
| 根因 | 每张图仅 2~3 个正样本 vs 上万条候选位置（占比 0.3%），obj 分支判别力不足，学不出可靠的置信度 |

也就是说：**框能量出来，但「哪个框可信」这个判断学不好**。这在目标占比极大的小数据集
（1800 张 200×200、平均每图 2.3 个框）上是常见困难。

## 本模块的做法

把「定位」与「判类」拆开（即两阶段检测的思路）：

1. 第一阶段：检测网络（YOLOv3 / PP-YOLOE-s）负责产生候选框 —— 它的定位已验证可用；
2. 第二阶段：本模块的**区域验证器**对候选框裁剪出的图块判类（6 类缺陷 + 1 类背景），
   用类别概率作为该候选的置信度。

为什么这能行：验证器学的是「一小块图里有没有缺陷、是哪一类」，这是一个**稠密监督**的
分类任务 —— 每个训练样本都有明确标签（GT 框内是真缺陷、框外随机区域是背景），
不存在「上万负样本淹没几个正样本」的问题，因此可以学得很稳。

## 设计要点

- 裁剪时向外扩 20% 边距：让网络看到缺陷与周边正常钢材的对比，这正是判别缺陷的关键线索；
- 负样本从「与任何 GT 框 IoU < 0.1」的区域随机取，保证背景样本干净；
- 输入 96×96：太小丢失纹理（划痕、麻点都是细粒度纹理），太大反而引入无关区域。
"""

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from PIL import Image

from .data import CLASS_NAMES, CLASS_NAMES_CN

# 背景类别的编号（放在最后，输出维度 = 6 类缺陷 + 1 背景）
BG_CLASS = len(CLASS_NAMES)


def crop_with_margin(im: Image.Image, box_px, margin=0.2):
    """按像素 xyxy 裁剪，并向外扩 margin 比例（越界则夹住）"""
    W, H = im.size
    x1, y1, x2, y2 = box_px
    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - w * margin)
    y1 = max(0, y1 - h * margin)
    x2 = min(W, x2 + w * margin)
    y2 = min(H, y2 + h * margin)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return im.crop((int(x1), int(y1), int(x2), int(y2)))


class DefectVerifier(nn.Layer):
    """区域验证器：输入图块，输出 7 类（6 缺陷 + 背景）概率

    网络刻意做小，因为训练样本只有几千个图块，参数多了必然过拟合；
    而且它只判「这块图是不是某类缺陷」，比整图分类简单得多。

    实现说明：直接复用 `paddle.vision.models.mobilenet_v3_small`，
    把它的分类头输出（1024 维）当作特征，再接自己的线性分类头。
    这样不依赖 paddle 内部的 `features` 等私有结构，升级版本也不容易坏。
    """

    FEATURE_DIM = 1024

    def __init__(self, num_classes=7, in_size=96, dropout=0.3):
        super().__init__()
        from paddle.vision.models import mobilenet_v3_small
        # num_classes=1024 时，这个网络本身就等于「特征提取 + 池化 + 1024 维输出」
        self.trunk = mobilenet_v3_small(num_classes=self.FEATURE_DIM)
        self.in_size = in_size
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.FEATURE_DIM, num_classes)

    def forward(self, x):
        f = self.trunk(x)
        return self.fc(self.drop(f))


class VerifierDataset(paddle.io.Dataset):
    """从 VOC 标注里裁正样本与负样本

    ## 三类样本

    1. **正样本**：GT 框（可选小幅抖动，模拟「框基本对准」的候选）；
    2. **纯背景样本**：与所有 GT 的 IoU < 0.15 的随机区域；
    3. **难负样本（dilate_per_gt > 0 时启用）**：把 GT 框放大 1.5~3.0 倍。

    ## 为什么需要第 3 类（本项目实测结论）

    初版验证器只用「GT 框 vs 随机背景」训练。实测它在推理时的行为是：
    **阈值从 0.5 提到 0.9，每图输出框数只从 49.8 降到 48.6** ——
    它把几乎所有覆盖到缺陷的窗口都判成缺陷。原因就是它**从未见过
    「覆盖了缺陷、但框本身不准」的样本**；而实测高分窗口里有 68% 属于
    IoU 0.1~0.5 的这类框，正是「大而无效」的形态。

    直接生成「与 GT 部分重叠」的难负样本要跑一遍滑窗+打分，成本高；
    而**把 GT 框放大**能高效造出同性质的样本 —— 框放得越大、与真实缺陷的 IoU 越低，
    这也正好解释了「加大滑窗尺度反而变差」的现象（见 window_detector 的注释）。
    """

    def __init__(self, root, split="train", in_size=96, neg_per_img=1, margin=0.2,
                 seed=2026, pos_jitter=0.0, dilate_per_gt=0):
        super().__init__()
        self.root = Path(root)
        self.in_size = in_size
        self.margin = margin
        self.pos_jitter = pos_jitter
        self.dilate_per_gt = dilate_per_gt
        self.img_dir = self.root / "dataset" / "det" / "JPEGImages"
        self.ann_dir = self.root / "dataset" / "det" / "Annotations"
        names = [ln.strip() for ln in
                 (self.root / "dataset" / "det" / "ImageSets" / "Main" / f"{split}.txt")
                 .read_text(encoding="utf-8").splitlines() if ln.strip()]
        rng = random.Random(seed)

        self.samples = []  # (img_path, box_px or None, label)
        for n in names:
            img_p = self.img_dir / f"{n}.jpg"
            boxes, labels = self._read_xml(self.ann_dir / f"{n}.xml")
            for b, l in zip(boxes, labels):
                self.samples.append((img_p, self._jitter(b, rng) if pos_jitter > 0 else b, l))
                # 难负样本：放大 GT 框（「覆盖缺陷但框不准」）
                for _ in range(dilate_per_gt):
                    d = self._dilate(b, rng)
                    if d is not None:
                        self.samples.append((img_p, d, BG_CLASS))
            # 纯背景样本：与所有 GT 的 IoU 都 < NEG_IOU 的随机区域
            for _ in range(neg_per_img):
                bg = self._random_bg(boxes, rng)
                if bg is not None:
                    self.samples.append((img_p, bg, BG_CLASS))
        rng.shuffle(self.samples)

    def _jitter(self, box, rng):
        """正样本抖动：小幅缩放 + 平移，模拟「基本对准」的候选"""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        s = 1.0 + rng.uniform(-self.pos_jitter, self.pos_jitter)
        nw, nh = w * s, h * s
        ncx = cx + rng.uniform(-self.pos_jitter, self.pos_jitter) * w * 0.5
        ncy = cy + rng.uniform(-self.pos_jitter, self.pos_jitter) * h * 0.5
        return [ncx - nw / 2, ncy - nh / 2, ncx + nw / 2, ncy + nh / 2]

    def _dilate(self, box, rng):
        """把 GT 框放大 1.5~3.0 倍，造「大而无效」的难负样本"""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        f = rng.uniform(1.5, 3.0)
        nw, nh = w * f, h * f
        if nw > 200 * 1.05 or nh > 200 * 1.05:
            return None
        return [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]

    @staticmethod
    def _read_xml(p):
        root = ET.parse(p).getroot()
        size = root.find("size")
        W, H = float(size.findtext("width")), float(size.findtext("height"))
        boxes, labels = [], []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in CLASS_NAMES:
                continue
            bb = obj.find("bndbox")
            boxes.append([float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
            labels.append(CLASS_NAMES.index(name))
        return boxes, labels

    @staticmethod
    def _iou(a, b):
        lt = [max(a[0], b[0]), max(a[1], b[1])]
        rb = [min(a[2], b[2]), min(a[3], b[3])]
        iw, ih = max(0, rb[0] - lt[0]), max(0, rb[1] - lt[1])
        inter = iw * ih
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        return inter / max(aa + ab - inter, 1e-9)

    def _random_bg(self, boxes, rng, tries=12):
        """随机取一个与所有 GT 都不重叠（IoU<0.1）的区域"""
        for _ in range(tries):
            w = rng.uniform(30, 90)
            h = rng.uniform(30, 90)
            x1 = rng.uniform(0, 200 - w)
            y1 = rng.uniform(0, 200 - h)
            cand = [x1, y1, x1 + w, y1 + h]
            if all(self._iou(cand, b) < 0.1 for b in boxes):
                return cand
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, box, label = self.samples[idx]
        im = Image.open(img_p).convert("RGB")
        crop = crop_with_margin(im, box, self.margin)
        if crop is None:
            crop = im
        crop = crop.resize((self.in_size, self.in_size), Image.BILINEAR)
        arr = np.asarray(crop, dtype="float32") / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return {"image": arr, "label": np.int64(label)}


class VerifierInfer:
    """把验证器包成「给一批候选框打分」的工具，供 eval/viz 复用"""

    def __init__(self, model, in_size=96, margin=0.2, batch=64):
        self.model = model
        self.in_size = in_size
        self.margin = margin
        self.batch = batch
        self.model.eval()

    @paddle.no_grad()
    def score(self, im: Image.Image, boxes_norm):
        """boxes_norm: (N,4) 归一化 xyxy -> (N,7) 概率

        坐标为归一化，与检测网络输出口径一致，无需换算成像素再算回去。
        """
        n = len(boxes_norm)
        if n == 0:
            return np.zeros((0, len(CLASS_NAMES) + 1), dtype="float32")
        W, H = im.size
        crops = []
        for b in boxes_norm:
            px = [b[0] * W, b[1] * H, b[2] * W, b[3] * H]
            c = crop_with_margin(im, px, self.margin)
            if c is None:
                c = im
            c = c.resize((self.in_size, self.in_size), Image.BILINEAR)
            crops.append(np.transpose(np.asarray(c, dtype="float32") / 255.0, (2, 0, 1)))
        out = []
        for i in range(0, n, self.batch):
            chunk = paddle.to_tensor(np.stack(crops[i:i + self.batch], axis=0))
            out.append(F.softmax(self.model(chunk), axis=-1).numpy())
        return np.concatenate(out, axis=0)
