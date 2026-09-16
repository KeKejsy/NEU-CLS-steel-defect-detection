"""成员 C · 区域精修器（两阶段检测的第二阶段升级版）

## 为什么需要它

[瓶颈诊断](tools/bottleneck.py、bottleneck2.py) 得到的实测结论：

| 观察 | 数值 | 含义 |
|---|---|---|
| 滑窗候选数 | 2028 个/图 | 候选极多 |
| IoU>=0.5 的召回上限 | 70.1% | 另有 ~25% 的 GT 最佳 IoU 落在 0.3~0.5 |
| 验证器判「是不是缺陷」 | 95.1% | 很好 |
| 验证器判「是哪一类」 | 97.35% | 很好 |
| GT 最佳候选的排名中位数 | **194** | 正确框被淹没 |
| 阈值 0.5 -> 0.9 时每图框数 | 49.8 -> 48.6 | **阈值几乎无效** |

最后一行是关键：说明验证器把几乎所有覆盖缺陷的窗口都判成了缺陷。
原因是它训练时只见「GT 框（正）vs 与 GT 不重叠的随机区域（负）」，
**从未见过推理时这种密集重叠、部分覆盖缺陷的窗口**，所以无法区分它们。

## 本模块的做法

不再只判类，而是**同时回归框的位置**：

- 输入：一个候选窗口的裁剪图
- 输出：① 7 类概率（6 缺陷 + 背景，与验证器一致）② **框的修正量 (dx,dy,dw,dh)**

训练时以 GT 框为基准，随机抖动生成「模拟候选」（尺度、宽高比、中心偏移都按实测候选
分布采样），让网络学会把偏移的框拉回真值。这样：

- 原来 IoU 在 0.3~0.5 的候选可以被精修到 >0.5，直接提升召回上限；
- 精修后框与最匹配的 GT 高度重合，重叠窗口之间的一致性投票才有意义，
  排序问题也随之缓解。

坐标一律用「相对候选框自身」的归一化偏移，与图像分辨率无关。
"""

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from PIL import Image

from .data import CLASS_NAMES
from .verifier import BG_CLASS, crop_with_margin, prep_crop

# 训练用的「模拟候选」抖动范围。
#
# **这组数值经过 A/B 实测确定，不要凭直觉放宽**。
#
# 背景：实测（60 张验证图、约 10 万个高分窗口）发现，被当作预测输出的高分窗口里
# 只有 **8.3%** 与任何 GT 完全无重叠，而 **31.6% 落在 IoU 0.3~0.5、36.4% 落在 0.1~0.3** ——
# 绝大多数误检其实是「差一点就命中」的框。这提示可以通过框回归把它们拉过 0.5。
#
# 于是做了两组对比（tools/compare_refiners.py，同一批 90 张图、同一套窗口、同一后处理）：
#
#   抖动范围                        合成 IoU  推理精修增益        最终 mAP@0.5
#   尺度0.20~0.65 比例0.6~1.7 偏移25%   0.442   IoU +0.0805 命中+14.0pp   0.2385  <== 采用
#   尺度0.18~0.85 比例0.3~2.0 偏移40%   0.336   IoU +0.0697 命中+10.8pp   0.1289
#
# 结论：**抖动范围应与推理分布匹配，而不是越大越好**。滑窗实际产生的尺度就是
# 0.2~0.65（见 window_detector 的 DEFAULT_SCALES），放宽到 0.18~0.85 后合成任务
# 与真实分布脱节，训练 60 epoch 反而更差（mAP 几乎腰斩）。
# 注意「合成 IoU」这一训练日志指标会误导：放宽抖动时它偏低（0.336 vs 0.442），
# 但那只是任务更难，真正要看的指标是推理分布上的精修增益。
SCALE_RANGE = (0.20, 0.65)      # 与 DEFAULT_SCALES 对齐
ASPECT_RANGE = (0.6, 1.7)       # 与 DEFAULT_ASPECTS 的主干部分对齐
CENTER_JITTER = 0.25
# 与 GT 的 IoU 低于该值就当作背景（负样本），避免「半覆盖」被当成正样本
POS_IOU = 0.35
NEG_IOU = 0.15


def iou_xyxy(a, b):
    lt = [max(a[0], b[0]), max(a[1], b[1])]
    rb = [min(a[2], b[2]), min(a[3], b[3])]
    iw, ih = max(0.0, rb[0] - lt[0]), max(0.0, rb[1] - lt[1])
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-9)


class RefineDataset(paddle.io.Dataset):
    """样本构造：对每个 GT 生成若干「模拟候选」，标签是类别 + 框修正量

    同时生成背景样本（与所有 GT 的 IoU < NEG_IOU），让网络保留判别缺陷的能力。
    """

    def __init__(self, root, split="train", in_size=96, per_gt=6, neg_per_img=3,
                 margin=0.15, seed=2026, img_size=200, norm="none"):
        super().__init__()
        self.root = Path(root)
        self.in_size = in_size
        self.margin = margin
        self.img_size = img_size
        self.norm = norm
        self.img_dir = self.root / "dataset" / "det" / "JPEGImages"
        self.ann_dir = self.root / "dataset" / "det" / "Annotations"
        names = [ln.strip() for ln in
                 (self.root / "dataset" / "det" / "ImageSets" / "Main" / f"{split}.txt")
                 .read_text(encoding="utf-8").splitlines() if ln.strip()]
        rng = random.Random(seed)

        self.samples = []   # (img_path, cand_px, cls_id, target_px or None)
        for n in names:
            img_p = self.img_dir / f"{n}.jpg"
            gts, labels = self._read_xml(self.ann_dir / f"{n}.xml")
            if not gts:
                continue
            for g, l in zip(gts, labels):
                for _ in range(per_gt):
                    cand = self._jitter(g, rng)
                    self.samples.append((img_p, cand, l, g))
            for _ in range(neg_per_img):
                bg = self._random_bg(gts, rng)
                if bg is not None:
                    self.samples.append((img_p, bg, BG_CLASS, None))
        rng.shuffle(self.samples)

    def _jitter(self, gt, rng):
        """围绕 GT 生成一个偏移的候选框（归一化后按像素返回）"""
        x1, y1, x2, y2 = gt
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        S = self.img_size
        scale = rng.uniform(*SCALE_RANGE)
        ar = rng.uniform(*ASPECT_RANGE)
        nw = scale * S * (ar ** 0.5)
        nh = scale * S / (ar ** 0.5)
        ncx = cx + rng.uniform(-CENTER_JITTER, CENTER_JITTER) * w
        ncy = cy + rng.uniform(-CENTER_JITTER, CENTER_JITTER) * h
        return [ncx - nw / 2, ncy - nh / 2, ncx + nw / 2, ncy + nh / 2]

    def _random_bg(self, gts, rng, tries=12):
        S = self.img_size
        for _ in range(tries):
            w = rng.uniform(*SCALE_RANGE) * S
            h = rng.uniform(*SCALE_RANGE) * S
            x1 = rng.uniform(0, max(S - w, 1))
            y1 = rng.uniform(0, max(S - h, 1))
            cand = [x1, y1, x1 + w, y1 + h]
            if all(iou_xyxy(cand, g) < NEG_IOU for g in gts):
                return cand
        return None

    @staticmethod
    def _read_xml(p):
        root = ET.parse(p).getroot()
        gts, labels = [], []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in CLASS_NAMES:
                continue
            bb = obj.find("bndbox")
            gts.append([float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
            labels.append(CLASS_NAMES.index(name))
        return gts, labels

    @staticmethod
    def encode(cand, gt, S):
        """候选框 -> 修正目标：中心偏移与宽高对数比，按候选框自身尺度归一化

        取对数比而不是差值：宽高差值的量纲随框大小变化，取对数后不同尺度的框
        有相同的取值范围，回归更稳。
        """
        cx = (cand[0] + cand[2]) / 2 / S
        cy = (cand[1] + cand[3]) / 2 / S
        cw = (cand[2] - cand[0]) / S
        ch = (cand[3] - cand[1]) / S
        gcx = (gt[0] + gt[2]) / 2 / S
        gcy = (gt[1] + gt[3]) / 2 / S
        gw = (gt[2] - gt[0]) / S
        gh = (gt[3] - gt[1]) / S
        return np.asarray([
            (gcx - cx) / max(cw, 1e-6),
            (gcy - cy) / max(ch, 1e-6),
            float(np.log(max(gw, 1e-6) / max(cw, 1e-6))),
            float(np.log(max(gh, 1e-6) / max(ch, 1e-6))),
        ], dtype="float32")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, cand, cls_id, gt = self.samples[idx]
        im = Image.open(img_p).convert("RGB")
        crop = crop_with_margin(im, cand, self.margin)
        if crop is None:
            crop = im
        arr = prep_crop(crop, self.in_size, self.norm)
        if gt is None:
            delta = np.zeros(4, dtype="float32")
        else:
            delta = self.encode(cand, gt, self.img_size)
        return {"image": arr, "cls": np.int64(cls_id), "delta": delta,
                "has_gt": np.float32(0.0 if gt is None else 1.0),
                # 候选框与真值框（像素 xyxy）：用 IoU 系损失训练时需要，
                # 因为 IoU 必须在「解码后的框」上算，不能只看归一化偏移。
                "cand": np.asarray(cand, dtype="float32"),
                "gt": np.asarray(gt if gt is not None else [0.0, 0.0, 0.0, 0.0],
                                 dtype="float32")}


class RegionRefiner(nn.Layer):
    """共享主干 + 两个头：判类（7 类）+ 框精修（4 维偏移）

    主干沿用验证器的结构（MobileNetV3_small 到 1024 维），
    这样 `verifier_best.pdparams` 的主干权重可以直接迁移过来，训练更快也更稳。

    `pretrained=True`（A/B 实验开关，默认 False 保持历史行为）：主干加载 ImageNet
    预训练权重；与验证器同理，此时输入归一化必须设为 "imagenet"（见 prep_crop）。
    """

    FEATURE_DIM = 1024

    def __init__(self, num_classes=7, in_size=96, dropout=0.3, pretrained=False):
        super().__init__()
        from paddle.vision.models import mobilenet_v3_small
        self.trunk = mobilenet_v3_small(pretrained=pretrained,
                                        num_classes=self.FEATURE_DIM)
        self.pretrained = bool(pretrained)
        self.in_size = in_size
        self.drop = nn.Dropout(dropout)
        self.cls_head = nn.Linear(self.FEATURE_DIM, num_classes)
        self.reg_head = nn.Linear(self.FEATURE_DIM, 4)
        # 回归头初始化为接近 0，保证训练初期「精修≈不改变框」，不会把好框推坏
        self.reg_head.weight.set_value(paddle.zeros_like(self.reg_head.weight))
        self.reg_head.bias.set_value(paddle.zeros([4]))

    def forward(self, x):
        f = self.trunk(x)
        f = self.drop(f)
        return self.cls_head(f), self.reg_head(f)

    def load_from_verifier(self, path):
        """从验证器权重迁移主干（分类头形状一致时也一并加载）"""
        src = paddle.load(str(path))
        tgt = self.state_dict()
        loaded, skipped = [], []
        for k, v in src.items():
            if k in tgt and tuple(tgt[k].shape) == tuple(v.shape):
                tgt[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        self.set_state_dict(tgt)
        return loaded, skipped


def decode_refine(cand_px, delta, img_size):
    """把网络的偏移输出还原成精修后的像素框

    与 RefineDataset.encode 严格互为逆运算 —— 两者必须同步改，否则框会整体偏移。
    """
    cx = (cand_px[0] + cand_px[2]) / 2
    cy = (cand_px[1] + cand_px[3]) / 2
    w = cand_px[2] - cand_px[0]
    h = cand_px[3] - cand_px[1]
    ncx = cx + delta[0] * w
    ncy = cy + delta[1] * h
    nw = w * float(np.exp(np.clip(delta[2], -2.0, 2.0)))
    nh = h * float(np.exp(np.clip(delta[3], -2.0, 2.0)))
    return [ncx - nw / 2, ncy - nh / 2, ncx + nw / 2, ncy + nh / 2]


def decode_refine_batch(cand, delta):
    """`decode_refine` 的向量化张量版：cand/delta 均为 (N,4)，返回精修后框 (N,4) 像素

    用 IoU 系损失训练精修器时必须在解码后的框上算 IoU，所以需要可求导的批量版本。
    两处实现（numpy 版与张量版）必须保持一致，否则训练目标与推理行为会错位。
    """
    cx = (cand[:, 0] + cand[:, 2]) / 2
    cy = (cand[:, 1] + cand[:, 3]) / 2
    w = cand[:, 2] - cand[:, 0]
    h = cand[:, 3] - cand[:, 1]
    ncx = cx + delta[:, 0] * w
    ncy = cy + delta[:, 1] * h
    nw = w * paddle.exp(paddle.clip(delta[:, 2], -2.0, 2.0))
    nh = h * paddle.exp(paddle.clip(delta[:, 3], -2.0, 2.0))
    return paddle.stack([ncx - nw / 2, ncy - nh / 2,
                         ncx + nw / 2, ncy + nh / 2], axis=1)


class RefinerInfer:
    """把精修器包成「给一批候选框打分 + 精修」的工具"""

    def __init__(self, model, in_size=96, margin=0.15, batch=256, img_size=200,
                 norm="none"):
        self.model = model
        self.in_size = in_size
        self.margin = margin
        self.batch = batch
        self.img_size = img_size
        self.norm = norm
        self.model.eval()

    @paddle.no_grad()
    def run(self, im: Image.Image, boxes_px):
        """boxes_px: (N,4) 像素 xyxy -> (prob(N,7), refine_px(N,4))"""
        n = len(boxes_px)
        if n == 0:
            return (np.zeros((0, len(CLASS_NAMES) + 1), dtype="float32"),
                    np.zeros((0, 4), dtype="float32"))
        crops = []
        for b in boxes_px:
            c = crop_with_margin(im, b, self.margin)
            if c is None:
                c = im
            crops.append(prep_crop(c, self.in_size, self.norm))
        probs, deltas = [], []
        for i in range(0, n, self.batch):
            chunk = paddle.to_tensor(np.stack(crops[i:i + self.batch], axis=0))
            logit, d = self.model(chunk)
            probs.append(F.softmax(logit, axis=-1).numpy())
            deltas.append(d.numpy())
        prob = np.concatenate(probs, axis=0)
        delta = np.concatenate(deltas, axis=0)
        refined = np.asarray([decode_refine(boxes_px[i], delta[i], self.img_size)
                              for i in range(n)], dtype="float32")
        return prob, refined

    @paddle.no_grad()
    def run_norm(self, im: Image.Image, boxes_norm):
        """与 run 相同，但输入输出都用归一化坐标（与检测网络口径一致）"""
        W, H = im.size
        px = np.asarray([[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in boxes_norm],
                        dtype="float32")
        prob, ref_px = self.run(im, px)
        if len(ref_px) == 0:
            return prob, np.zeros((0, 4), dtype="float32")
        ref_norm = np.stack([ref_px[:, 0] / W, ref_px[:, 1] / H,
                             ref_px[:, 2] / W, ref_px[:, 3] / H], axis=1)
        return prob, np.clip(ref_norm, 0.0, 1.0).astype("float32")
