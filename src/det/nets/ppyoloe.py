"""成员 C · 检测网络：PP-YOLOE-s

网络二（与 YOLOv3 对比）。PP-YOLOE 相对 YOLOv3 的三处关键设计差异，本实现都保留了：

    1. **Anchor-free**：不再预设 9 个 anchor，每个特征点直接回归框。
       本数据集框宽高比跨度极大（细长划痕 Sc 到近方形麻点 PS），
       anchor 很难全覆盖，anchor-free 在这里有实际优势。
    2. **DFL 分布式回归**：不直接回归一个距离值，而是学「这条边落在哪个区间的分布」，
       取期望作为结果。对边界模糊的缺陷（龟裂 Cr）更稳。
    3. **ESE 通道注意力 + RepConv**：主干与 neck 里的结构增强。

结构（输入 640x640）：
    CSPResNet 主干  -> stride 8 (80x80) / 16 (40x40) / 32 (20x20)
    CSPPAN neck     -> 自顶向下 + 自底向上双向融合，输出 4 层
    PPYOLOEHead     -> 共享卷积塔 + cls 分支 + DFL reg 分支

正负样本分配用简化版 ATL：按框的等效边长选层，再取框中心附近 2.5 格内的锚点作为正样本。
本数据集每图平均仅 2.3 个框且 6 类互斥（一张图只标一类缺陷），
不需要 SimOTA 那种动态 k 与代价矩阵，简单阈值分配足够且更好定位问题。
"""

import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from core.boxes import batched_nms, clip_boxes_norm
from core.det_ops import (Distance2BBox, DistributionFocalLoss, IoULoss,
                          distribute_anchors, filter_valid_boxes)
from core.layers import ConvBNLayer, CSPRepLayer
from .cspresnet import CSPResNet


class PPYOLOEHead(nn.Layer):
    """PP-YOLOE 检测头：共享卷积塔 + cls 分支 + DFL reg 分支"""

    def __init__(self, in_ch, num_classes=6, hidden_ch=128, num_convs=2,
                 reg_max=16, act="silu"):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        stem = [ConvBNLayer(in_ch if i == 0 else hidden_ch, hidden_ch, 3, act=act)
                for i in range(num_convs)]
        self.stem = nn.Sequential(*stem)
        self.cls_conv = nn.Conv2D(hidden_ch, num_classes, 3, padding=1)
        self.reg_conv = nn.Conv2D(hidden_ch, 4 * (reg_max + 1), 3, padding=1)

        # 分类分支偏置初始化为负值：让训练初期所有位置的初始分数约 1%，
        # 否则海量负样本会在一开始就把正样本的梯度淹没。
        bias = -math.log((1 - 0.01) / 0.01)
        self.cls_conv.bias.set_value(paddle.full([num_classes], bias))

    def forward(self, feats):
        """feats: list[(B,C,H,W)] -> list[(cls_logits(B,NC,H,W), reg_logits(B,4*(reg_max+1),H,W))]"""
        outs = []
        for f in feats:
            h = self.stem(f)
            outs.append((self.cls_conv(h), self.reg_conv(h)))
        return outs


class PPYOLOES(nn.Layer):
    """PP-YOLOE-s 检测器

    im_size 必须与训练/推理时的输入边长一致：解码时用它把像素距离换算成归一化坐标。
    """

    def __init__(self, num_classes=6, im_size=640, reg_max=16,
                 base_channels=64, depths=(3, 3, 6, 3), act="silu",
                 box_loss="giou", cls_pos_weight=1.0, center_radius=2.5, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.im_size = int(im_size)
        self.reg_max = reg_max
        self.strides = [8, 16, 32, 64]
        self.center_radius = center_radius

        self.backbone = CSPResNet(base_channels=base_channels, depths=depths, act=act)
        in_chs = self.backbone.out_channels[-3:]

        # --- CSPPAN：自顶向下 + 自底向上，输出 4 层 ---
        # 通道链（实测，不是推导）：
        #   lat 把主干的 256/512/1024 统一压到 256
        #   -> p5/p4/p3 均为 256（融合是逐元素相加，通道不变）
        #   -> 自底向上 n3/n4/n5 仍为 256，n6 由 256 下采样得到也是 256
        # 所以四个 CSPRepLayer 的 in_ch 全是 256。
        # 之前误以为「两路拼接会变 512」而写成 512，结果 conv 权重形状与输入对不上。
        self.lat = nn.LayerList([ConvBNLayer(c, 256, 1, act=act) for c in in_chs])
        self.td = nn.LayerList([ConvBNLayer(256, 256, 3, act=act) for _ in range(2)])
        self.bu = nn.LayerList([ConvBNLayer(256, 256, 3, act=act) for _ in range(3)])
        self.pan_csp = nn.LayerList([
            CSPRepLayer(256, 256, num_blocks=3, expansion=0.5, act=act)
            for _ in range(4)
        ])
        self.downsample_p4 = ConvBNLayer(256, 256, 3, stride=2, act=act)

        self.head = PPYOLOEHead(256, num_classes=num_classes, reg_max=reg_max, act=act)

        # 回归损失用 reduction='none' 逐框返回，再由 forward_loss 按正样本掩码加权归约
        self.box_loss_fn = IoULoss(box_loss, reduction="none")
        self.dfl_loss_fn = DistributionFocalLoss()
        self.cls_pos_weight = cls_pos_weight

    # ------------------------------------------------------------------
    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        l3, l4, l5 = self.lat[0](c3), self.lat[1](c4), self.lat[2](c5)

        # 与 YOLOv3 同样的理由：必须按目标特征图尺寸缩放，不能用固定 scale_factor。
        # 本数据集原图 200x200，200/8=25、200/16=12(取整)、200/32=6，
        # 25->12 是 0.48 而不是 0.5，硬写 0.5 会得到 13 或 6 而错位。
        p5 = l5
        p4 = self.td[0](l4 + self._resize_to(p5, l4))
        p3 = self.td[1](l3 + self._resize_to(p4, l3))

        # 自底向上
        n3 = p3
        n4 = self.bu[0](p4 + self._resize_to(n3, p4))
        n5 = self.bu[1](p5 + self._resize_to(n4, p5))
        n6 = self.bu[2](self.downsample_p4(n5))

        feats = [self.pan_csp[0](n3), self.pan_csp[1](n4),
                 self.pan_csp[2](n5), self.pan_csp[3](n6)]
        return self.head(feats)

    @staticmethod
    def _resize_to(src, ref):
        """把 src 缩放到 ref 的空间尺寸（放大用最近邻，缩小也用最近邻以省算力）"""
        _, _, h, w = ref.shape
        if src.shape[2] == h and src.shape[3] == w:
            return src
        return F.interpolate(src, size=[h, w], mode="nearest")

    # ------------------------------------------------------------------
    def _flatten(self, raw):
        """各尺度 (cls, reg) 摊平成 (B,N,·)，并生成与之一一对应的锚点

        拼接顺序 = 逐尺度、尺度内按行优先。distribute_anchors 用同一顺序生成锚点，
        两边一旦不一致，框就会整体错位到别的像素点上，所以这里刻意不写两套逻辑。
        """
        cls_list, reg_list, sizes = [], [], []
        for cls_logits, reg_logits in raw:
            b, c, h, w = cls_logits.shape
            sizes.append((h, w))
            cls_list.append(cls_logits.reshape([b, c, h * w]).transpose([0, 2, 1]))
            reg_list.append(reg_logits.reshape([b, reg_logits.shape[1], h * w]).transpose([0, 2, 1]))
        cls = paddle.concat(cls_list, axis=1)
        reg = paddle.concat(reg_list, axis=1)
        anchors, strides = distribute_anchors(sizes, self.strides, self.reg_max)
        return cls, reg, anchors, strides

    def _level_of_size(self, eq_side_px: float) -> int:
        """按框的等效边长选特征层

        阈值取各层 stride 的 4 倍：小框交给高分辨率层，大框交给低分辨率层。
        这是 PP-YOLOE/ATL 的核心思想，写成显式阶梯比用 clip 换算更好读也更好调。
        """
        for li, s in enumerate(self.strides):
            if eq_side_px < 4.0 * s:
                return li
        return len(self.strides) - 1

    def _assign(self, targets, num_boxes, anchors, strides):
        """正样本分配，返回逐锚点的 (pos_mask, cls_tgt, dist_tgt)

        全部在 numpy 上做：GT 数量少（平均 2.3 个/图），循环开销可忽略，
        换来的好处是「哪个锚点被哪个 GT 认领」这件事一眼能看懂、能单步调试。
        关键点：同时记录 pos_gt_index，后续框回归损失直接按它取 GT，
        不需要再从坐标反推归属（那样既慢又容易出错）。
        """
        b, n, nc = targets.shape[0], anchors.shape[0], self.num_classes
        tgt_np = targets.numpy()
        nb_np = num_boxes.numpy()
        anc_np = anchors.numpy()

        pos_np = np.zeros([b, n], dtype=bool)
        cls_np = np.zeros([b, n, nc], dtype="float32")
        dist_np = np.zeros([b, n, 4], dtype="float32")
        gt_of_anchor = -np.ones([b, n], dtype="int64")

        for bi in range(b):
            nb = int(nb_np[bi])
            for gi in range(nb):
                cid = int(tgt_np[bi, gi, 0])
                gcx, gcy, gw, gh = [float(v) for v in tgt_np[bi, gi, 1:5]]
                px1 = (gcx - gw / 2) * self.im_size
                py1 = (gcy - gh / 2) * self.im_size
                px2 = (gcx + gw / 2) * self.im_size
                py2 = (gcy + gh / 2) * self.im_size

                # 选层
                eq_side = math.sqrt(max((gw * self.im_size) * (gh * self.im_size), 1.0))
                li = self._level_of_size(eq_side)

                # 该层锚点在全局索引里的起点
                offset = sum((self.im_size // self.strides[k]) ** 2 for k in range(li))
                s = self.strides[li]
                fh = fw = self.im_size // s

                cx_cell, cy_cell = gcx * fw, gcy * fh
                r = self.center_radius
                x0 = int(max(0, math.floor(cx_cell - r)))
                x1 = int(min(fw - 1, math.ceil(cx_cell + r)))
                y0 = int(max(0, math.floor(cy_cell - r)))
                y1 = int(min(fh - 1, math.ceil(cy_cell + r)))
                # 保底：中心格必须被选中，否则这个 GT 没有正样本
                gx = min(int(cx_cell), fw - 1)
                gy = min(int(cy_cell), fh - 1)
                if not (x0 <= gx <= x1):
                    x0 = x1 = gx
                if not (y0 <= gy <= y1):
                    y0 = y1 = gy

                for yy in range(y0, y1 + 1):
                    for xx in range(x0, x1 + 1):
                        idx = offset + yy * fw + xx
                        if idx >= n:
                            continue
                        pos_np[bi, idx] = True
                        cls_np[bi, idx, cid] = 1.0
                        gt_of_anchor[bi, idx] = gi
                        acx, acy = (xx + 0.5) * s, (yy + 0.5) * s
                        dist_np[bi, idx] = [(acx - px1) / s, (acy - py1) / s,
                                            (px2 - acx) / s, (py2 - acy) / s]

        return (paddle.to_tensor(pos_np), paddle.to_tensor(cls_np),
                paddle.to_tensor(dist_np), gt_of_anchor)

    def forward_loss(self, raw, targets, num_boxes):
        """分类用带正样本加权的 BCE；回归用 DFL + GIoU（均只算正样本）

        实现要点：**全程不做 masked_select**。
        一开始我用 masked_select 把正样本挑出来再算回归损失，前向只要 1ms，
        但反向高达 1.6s/步（8500x68 的中间张量展开 + 散射梯度），
        整个网络慢到 bs=8 都要 3.9s/步。改成「全量算损失 + 掩码加权归约」后，
        既没有 gather/scatter，梯度路径也变成规整的逐元素运算，速度回到正常量级。
        """
        cls, reg, anchors, strides = self._flatten(raw)
        b, n, nc = cls.shape

        pos_mask, cls_tgt, dist_tgt, gt_of_anchor = self._assign(
            targets, num_boxes, anchors, strides)

        pos_f = pos_mask.astype("float32")
        n_pos = paddle.clip(pos_f.sum(), min=1.0)

        # --- 分类：全部位置参与，正样本加权后按正样本数归一化 ---
        cls_loss_raw = F.binary_cross_entropy_with_logits(cls, cls_tgt, reduction="none")
        w = paddle.where(pos_mask.unsqueeze(-1),
                         paddle.full_like(cls_loss_raw, self.cls_pos_weight),
                         paddle.ones_like(cls_loss_raw))
        loss_cls = (cls_loss_raw * w).sum() / n_pos

        if int(pos_mask.astype("int64").sum()) == 0:
            zero = paddle.zeros([1])
            return {"loss_cls": loss_cls, "loss_dfl": zero, "loss_box": zero}

        # --- DFL：全量算 -> 掩码归约 ---
        dfl_all = self.dfl_loss_fn(reg, dist_tgt)  # (B,N,4)，逐边返回
        loss_dfl = (dfl_all.sum(-1) * pos_f).sum() / n_pos

        # --- GIoU：把全部锚点解码成框，再只在正样本位置上算损失 ---
        d2b = Distance2BBox(self.reg_max)
        anc_b = paddle.expand(anchors.unsqueeze(0), [b, n, 2])
        pred_boxes = d2b(reg, anc_b) / float(self.im_size)  # (B,N,4) 归一化 xyxy

        gt_boxes = self._build_gt_for_positions(targets, num_boxes, gt_of_anchor, b, n)
        giou_all = self.box_loss_fn(pred_boxes, gt_boxes)  # (B,N)，已按框逐元素返回
        loss_box = (giou_all * pos_f).sum() / n_pos

        return {"loss_cls": loss_cls, "loss_dfl": loss_dfl, "loss_box": loss_box}

    def _build_gt_for_positions(self, targets, num_boxes, gt_of_anchor, b, n):
        """构造与 pred_boxes 同形状 (B,N,4) 的 GT 张量，只有正样本位置有意义

        gt_of_anchor 是 _assign 里记录下来的「该锚点归属哪个 GT」，
        用它直接索引即可，不需要再从坐标反推归属（那样既慢又容易错）。
        """
        tgt_np = targets.numpy()
        nb_np = num_boxes.numpy()
        gt = np.zeros([b, n, 4], dtype="float32")
        for bi in range(b):
            nb = int(nb_np[bi])
            if nb == 0:
                continue
            for gi in range(nb):
                gcx, gcy, gw, gh = [float(v) for v in tgt_np[bi, gi, 1:5]]
                sel = (gt_of_anchor[bi] == gi)
                if not sel.any():
                    continue
                gt[bi, sel] = [gcx - gw / 2, gcy - gh / 2, gcx + gw / 2, gcy + gh / 2]
        return paddle.to_tensor(gt)

    # ------------------------------------------------------------------
    @paddle.no_grad()
    def postprocess(self, raw, im_shape, score_threshold=0.01, nms_threshold=0.5,
                    top_k=100):
        """解码 + 多类 NMS。PP-YOLOE 无 obj 分支，置信度就是类别分数"""
        cls, reg, anchors, strides = self._flatten(raw)
        b, n, nc = cls.shape
        scores_all = paddle.sigmoid(cls)

        d2b = Distance2BBox(self.reg_max)
        anc = paddle.expand(anchors.unsqueeze(0), [b, n, 2])
        boxes_norm = d2b(reg, anc) / float(self.im_size)

        results = []
        for bi in range(b):
            s = scores_all[bi]
            max_score = s.max(-1)
            keep = max_score > score_threshold
            if int(keep.astype("int64").sum()) == 0:
                results.append((paddle.zeros([0, 4]), paddle.zeros([0]),
                                paddle.zeros([0], dtype="int64")))
                continue
            bs, ss = boxes_norm[bi][keep], max_score[keep]
            ls = s[keep].argmax(-1)
            # 过滤退化框（负宽高/零面积），与 YOLOv3 走同一个函数，口径一致
            bs, ss, ls = filter_valid_boxes(bs, ss, ls)
            if bs.shape[0] == 0:
                results.append((paddle.zeros([0, 4]), paddle.zeros([0]),
                                paddle.zeros([0], dtype="int64")))
                continue
            idx = batched_nms(bs, ss, ls, self.num_classes,
                              iou_threshold=nms_threshold, top_k=top_k)
            results.append((clip_boxes_norm(bs[idx]), ss[idx], ls[idx]))
        return results
