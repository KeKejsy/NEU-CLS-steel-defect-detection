"""成员 C · 检测网络：YOLOv3

网络一（对比用）。完整实现 YOLOv3 的三大件：
    Darknet53 主干  ->  FPN 上采样融合  ->  三个尺度的检测头
    三尺度 stride 8/16/32，每个位置 3 个 anchor，输出 obj + 6 类分数

与常见复现的差别：
    1. 全程使用原生 paddle 算子（paddle.vision.ops.nms 等），不编译自定义 OP；
    2. 边框统一用「归一化 cxcywh」，不依赖固定的输入分辨率；
    3. 小目标层（stride 8）完整保留——本数据集框平均仅占全图 17.45%，
       砍掉这一层会明显掉点。

损失：obj 与 cls 用带正样本加权的 BCE，框回归用 GIoU（giou/diou/ciou 可配）。
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from core.boxes import batched_nms, clip_boxes_norm
from core.det_ops import (BCEWithLogitsLoss, IoULoss, decode_yolov3, decode_yolov3_scale,
                          filter_valid_boxes)
from core.layers import ConvBNLayer
from .darknet import Darknet53


class YOLOv3(nn.Layer):
    """YOLOv3 检测器

    参数
    ----
    num_classes : 类别数（本数据集为 6）
    anchors     : 三个尺度各自的 anchor 宽高，单位是「输入边长的比例」。
                  默认值是 YOLOv3 官方 anchor 除以 416 得到的归一化值，
                  这样与输入分辨率解耦：换到 320 输入时 anchor 不用改。
    strides     : 三个尺度的下采样倍数
    """

    # 官方 anchor (10,13)(16,30)(33,23) / (30,61)(62,45)(59,119) / (116,90)(156,198)(373,326)
    # 注意三个尺度的顺序与 strides 对应：stride 8（高分辨率）配小 anchor，
    # stride 32（低分辨率）配大 anchor。写反会让每个 GT 都分给错误的尺度。
    DEFAULT_ANCHORS = [
        [[10, 13], [16, 30], [33, 23]],       # stride 8
        [[30, 61], [62, 45], [59, 119]],      # stride 16
        [[116, 90], [156, 198], [373, 326]],  # stride 32
    ]

    def __init__(self, num_classes=6, anchors=None, strides=(8, 16, 32),
                 base_channels=32, norm_act="leaky", box_loss="giou",
                 label_smooth=0.0, obj_pos_weight=1.0, im_size=416,
                 score_mode="obj_cls", level_scale=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.strides = list(strides)
        self.num_anchors = 3
        # 置信度来源：
        #   obj_cls  —— 经典 YOLOv3 口径，分数 = sigmoid(obj) * sigmoid(cls)
        #   cls_only —— 只用类别分数（相当于把 obj 分支降级为辅助损失）
        # 两者都保留是因为本数据集上 obj 分支的正样本仅占候选位置的 0.3%，
        # 信号很弱、收敛慢；实测把它乘进分数会显著拖低排序质量。
        assert score_mode in ("obj_cls", "cls_only"), f"未知 score_mode: {score_mode}"
        self.score_mode = score_mode
        self.level_scale = float(level_scale)
        # 解码式 yolo 回归把像素坐标除以 im_size 归一化，所以必须知道基准边长。
        # 训练与推理必须传同一个值，否则框会整体缩放错。
        self.im_size = int(im_size)
        anc = anchors or self.DEFAULT_ANCHORS
        # 归一化：把像素 anchor 除以 416，得到与分辨率无关的比例
        self.register_buffer(
            "anchors_norm",
            paddle.to_tensor(anc, dtype="float32").reshape([3, 3, 2]) / 416.0)

        self.backbone = Darknet53(norm_act=norm_act, base_channels=base_channels)
        c3_ch, c4_ch, c5_ch = self.backbone.out_channels
        head_ch = [c3_ch, c4_ch, c5_ch]

        # FPN 侧向 1x1 卷积：把主干通道压到统一的 256
        self.lat5 = self._conv_bn(c5_ch, 256, 1, norm_act)
        self.lat4 = self._conv_bn(c4_ch, 256, 1, norm_act)
        self.lat3 = self._conv_bn(c3_ch, 256, 1, norm_act)
        # 上采样后的融合卷积（5 层 3x3，YOLOv3 原设置）
        self.smooth5 = self._make_smooth(norm_act)
        self.smooth4 = self._make_smooth(norm_act)
        self.smooth3 = self._make_smooth(norm_act)

        # 三个检测头，每个输出 3 * (5 + num_classes)
        out_ch = self.num_anchors * (5 + num_classes)
        self.head3 = nn.Conv2D(256, out_ch, 1)
        self.head4 = nn.Conv2D(256, out_ch, 1)
        self.head5 = nn.Conv2D(256, out_ch, 1)
        self._init_head_bias()

        self.box_loss_fn = IoULoss(box_loss, reduction="none")
        self.obj_loss_fn = BCEWithLogitsLoss(pos_weight=obj_pos_weight, label_smooth=0.0)
        self.cls_loss_fn = BCEWithLogitsLoss(pos_weight=1.0, label_smooth=label_smooth)

    def _init_head_bias(self):
        """给 obj / cls 分支的偏置设初值

        为什么必须做：一张图有上万条候选位置，正样本只有 2~3 个（占比约 0.3%）。
        若偏置从 0 开始，初始正负样本的损失几乎相等，优化器会优先把**所有**位置的
        obj 分数往下压 —— 因为负样本数量是正样本的 300 倍，压负样本的收益大得多。
        结果是退化解：obj logit 全为负（实测均值 -4.9），正样本的置信度只有 0.05~0.09。

        obj 偏置取 -2.0（sigmoid ≈ 0.12）是个折中：
        - 太负（如 -4.5）：正样本从 0.011 起爬，训练结束时才 0.07~0.20，
          排序上被大量「碰巧分数稍高」的邻近负样本挤掉；
        - 太正（如 0）：负样本初始损失大，训练初期不稳定。
        实测 -2.0 时正样本能爬到 0.5 以上，在 top_k 截断中存活。
        cls 偏置保持 -4.6（YOLOv5 常规做法），让类别分支从低概率起步。
        """
        for head in (self.head3, self.head4, self.head5):
            b = head.bias.numpy().reshape(self.num_anchors, 5 + self.num_classes)
            b[:, 0:2] = 0.0        # tx,ty：从 0 开始，sigmoid(0)=0.5 即网格中心
            b[:, 2:4] = 0.0        # tw,th：从 0 开始，exp(0)=1 即等于 anchor 尺寸
            b[:, 4] = -2.0         # obj
            b[:, 5:] = -4.6        # cls
            head.bias.set_value(paddle.to_tensor(b.reshape([-1]), dtype="float32"))

    @staticmethod
    def _conv_bn(in_ch, out_ch, k, act):
        return ConvBNLayer(in_ch, out_ch, k, act=act)

    def _make_smooth(self, act):
        return nn.Sequential(*[ConvBNLayer(256, 256, 3, act=act) for _ in range(5)])

    # ------------------------------------------------------------------
    # 前向：返回三个尺度的原始预测（训练用）
    # ------------------------------------------------------------------
    def forward(self, x):
        c3, c4, c5 = self.backbone(x)

        p5 = self.smooth5(self.lat5(c5))
        # 关键：按「目标特征图尺寸」上采样，而不是固定 scale_factor=2。
        # 原因：图像边长不是 32 的整数倍时（本数据集原图 200x200，200/32=6.25），
        # 三尺度尺寸会变成 25/13/7 —— 7 的两倍是 14 而目标层是 13，相加直接报错。
        # 用 size= 显式指定目标尺寸，对任意输入边长都成立。
        p4 = self.smooth4(self.lat4(c4) + self._upsample_to(p5, self.lat4(c4)))
        p3 = self.smooth3(self.lat3(c3) + self._upsample_to(p4, self.lat3(c3)))

        outs = []
        for feat, head in ((p3, self.head3), (p4, self.head4), (p5, self.head5)):
            b, _, h, w = feat.shape
            # (B, A*(5+NC), H, W) -> (B, A, H, W, 5+NC)
            o = head(feat).reshape([b, self.num_anchors, 5 + self.num_classes, h, w])
            outs.append(o.transpose([0, 1, 3, 4, 2]))
        return outs

    @staticmethod
    def _upsample_to(src, ref):
        """把 src 上采样到 ref 的空间尺寸（最近邻，语义不变）"""
        _, _, h, w = ref.shape
        return F.interpolate(src, size=[h, w], mode="nearest")

    # ------------------------------------------------------------------
    # 匹配：把 GT 框分配给「宽高 IoU 最大」的 anchor 与所在网格
    # ------------------------------------------------------------------
    def assign_targets(self, preds, targets, num_boxes):
        """把 GT 框分配给合适的尺度、anchor 与网格

        targets: (B, M, 5) = [cls, cx, cy, w, h]，坐标归一化，M 为 padding 后的最大框数
        num_boxes: (B,) 每张图真实框数

        **分配策略（关键，改前先看这段）**：
        先按「框的等效边长」把 GT 分到 stride 8 / 16 / 32 三层中更合适的一层，
        再在该层的 3 个 anchor 里挑宽高 IoU 最大的那个。
        一开始我写成「9 个 anchor 一起比宽高 IoU」，实测在 NEU-DET 上会退化成
        **所有 GT 都被吸到 stride 32 那一层**（因为大 anchor 的宽高 IoU 更容易占优），
        stride 8/16 完全没有正样本，细尺度白搭 —— 实测 23 个 GT 全部落在尺度 2，
        负责点上的预测框 IoU 均值只有 0.233。
        按大小分层能保证三层都拿到正样本，定位精度也随之改善。
        """
        b = preds[0].shape[0]
        anchors = self.anchors_norm  # (3 尺度, 3 anchor, 2) 归一化宽高

        out = []
        for s_idx, p in enumerate(preds):
            _, a, h, w, _ = p.shape
            obj_t = paddle.zeros([b, a, h, w], dtype="float32")
            cls_t = paddle.zeros([b, a, h, w, self.num_classes], dtype="float32")
            box_t = paddle.zeros([b, a, h, w, 4], dtype="float32")
            pos_m = paddle.zeros([b, a, h, w], dtype="bool")
            out.append([obj_t, cls_t, box_t, pos_m, h, w])

        tgt_np = targets.numpy()
        nb_np = num_boxes.numpy()
        anchors_np = anchors.numpy()

        for bi in range(b):
            n = int(nb_np[bi])
            for gi in range(n):
                cls_id = int(tgt_np[bi, gi, 0])
                gcx, gcy, gw, gh = [float(v) for v in tgt_np[bi, gi, 1:5]]

                # 第一步：按等效边长选层。
                # 阈值取各层 stride 的 level_scale 倍，由超参控制（默认 2.0）。
                # 取值越大，越多 GT 被推到低分辨率层（层内被打包），取值太小则细尺度
                # 正样本过多、回归目标跨度大。本数据集框偏大（平均占全图 17.45%），
                # 用 4.0 会让 12/12 个 GT 全落到 stride 32，细尺度白搭；
                # 2.0 时三层的正样本分布才比较均衡，实测对 mAP 明显更有利。
                eq = max(gw * self.im_size * gh * self.im_size, 1.0) ** 0.5
                s_idx = len(self.strides) - 1
                for li, st in enumerate(self.strides):
                    if eq < self.level_scale * st:
                        s_idx = li
                        break

                # 第二步：在该层的 anchor 里挑宽高 IoU 最大的
                best_a, best_iou = 0, -1.0
                for ai in range(anchors_np.shape[1]):
                    aw, ah = anchors_np[s_idx, ai]
                    inter = min(gw, aw) * min(gh, ah)
                    union = gw * gh + aw * ah - inter
                    iou = inter / max(union, 1e-9)
                    if iou > best_iou:
                        best_iou, best_a = iou, ai

                h, w = out[s_idx][4], out[s_idx][5]
                gi_x = min(int(gcx * w), w - 1)
                gi_y = min(int(gcy * h), h - 1)

                obj_t, cls_t, box_t, pos_m = out[s_idx][0], out[s_idx][1], out[s_idx][2], out[s_idx][3]
                obj_t[bi, best_a, gi_y, gi_x] = 1.0
                cls_t[bi, best_a, gi_y, gi_x, cls_id] = 1.0
                box_t[bi, best_a, gi_y, gi_x] = paddle.to_tensor(
                    [gcx, gcy, gw, gh], dtype="float32")
                pos_m[bi, best_a, gi_y, gi_x] = True

        return [(o[0], o[1], o[2], o[3]) for o in out]

    # ------------------------------------------------------------------
    # 损失
    # ------------------------------------------------------------------
    def forward_loss(self, preds, targets, num_boxes):
        assigns = self.assign_targets(preds, targets, num_boxes)
        anchors = self.anchors_norm

        total_box = paddle.zeros([1])
        total_obj = paddle.zeros([1])
        total_cls = paddle.zeros([1])

        for s_idx, (p, (obj_t, cls_t, box_t, pos_m)) in enumerate(zip(preds, assigns)):
            b, a, h, w, _ = p.shape
            stride = self.strides[s_idx]

            # --- 解码成归一化 xyxy ---
            # 注意：decode_yolov3_scale 会把 (H,W) 摊平成 N，返回 (B,N,·)，
            # 所以下面所有 masked_select 的掩码也必须摊平成同一个 (B,N,·) 布局，
            # 否则会出现「x 有 N 维、mask 还有 H,W 两维」的维度不匹配。
            pred_xyxy, _, _ = decode_yolov3_scale(p, anchors[s_idx], stride, self.im_size)

            # 摊平后的掩码与目标：(B,A,H,W) -> (B,N)
            pos_flat = pos_m.reshape([b, -1])
            box_flat = box_t.reshape([b, a * h * w, 4])

            # --- 框回归损失：只算正样本 ---
            n_pos = paddle.clip(pos_flat.astype("float32").sum(), min=1.0)
            n_pos_int = int(pos_flat.astype("int64").sum())
            if n_pos_int > 0:
                pred_pos = paddle.masked_select(pred_xyxy, pos_flat.unsqueeze(-1)).reshape([-1, 4])
                gt_pos = paddle.masked_select(box_flat, pos_flat.unsqueeze(-1)).reshape([-1, 4])
                total_box = total_box + self.box_loss_fn(pred_pos, gt_pos).sum() / n_pos

            # --- obj 损失：全部位置参与，正样本加权 ---
            # p[..., 4:5] 是 (B,A,H,W,1)，与 obj_t 的 (B,A,H,W,1) 逐元素对齐
            total_obj = total_obj + self.obj_loss_fn(p[..., 4:5], obj_t.unsqueeze(-1),
                                                     pos_m.unsqueeze(-1))
            # --- cls 损失：只算正样本 ---
            if n_pos_int > 0:
                cls_pred_pos = paddle.masked_select(p[..., 5:], pos_m.unsqueeze(-1)).reshape(
                    [-1, self.num_classes])
                cls_tgt_pos = paddle.masked_select(cls_t, pos_m.unsqueeze(-1)).reshape(
                    [-1, self.num_classes])
                total_cls = total_cls + self.cls_loss_fn(cls_pred_pos, cls_tgt_pos)

        # 三个尺度累加后各自收敛成标量：loss_box 按正样本数归一化，
        # obj/cls 的 BCE 内部已按正样本数归一化，这里对尺度取平均。
        n_scales = float(len(preds))
        return {
            "loss_box": total_box.sum() / n_scales,
            "loss_obj": total_obj.sum() / n_scales,
            "loss_cls": total_cls.sum() / n_scales,
        }

    # ------------------------------------------------------------------
    # 推理：解码 + 多类 NMS
    # ------------------------------------------------------------------
    @paddle.no_grad()
    def postprocess(self, preds, im_shape, score_threshold=0.01, nms_threshold=0.5,
                    top_k=100, im_size=416):
        """返回 list，每个元素是该图的 (boxes_xyxy_normalized, scores, labels)"""
        decoded = []
        for s_idx, p in enumerate(preds):
            decoded.append(decode_yolov3(
                [p], [self.anchors_norm[s_idx]], [self.strides[s_idx]],
                self.num_classes, im_size)[0])
        all_pred = paddle.concat(decoded, axis=1)  # (B, N, 5+NC)

        box = all_pred[..., 0:4]
        obj = all_pred[..., 4:5]
        cls = all_pred[..., 5:]
        if self.score_mode == "cls_only":
            # 只用类别分数排序（obj 仍参与训练，只是不进分数）
            scores_all = cls
        else:
            # 经典 YOLOv3：置信度 = obj * cls
            scores_all = obj * cls
        b = box.shape[0]

        results = []
        for bi in range(b):
            s = scores_all[bi]                      # (N, NC)
            max_score = s.max(-1)
            keep = max_score > score_threshold
            if int(keep.astype("int64").sum()) == 0:
                results.append((paddle.zeros([0, 4]), paddle.zeros([0]), paddle.zeros([0], dtype="int64")))
                continue
            bs = box[bi][keep]
            ss = max_score[keep]
            ls = s[keep].argmax(-1)
            # 过滤退化框（负宽高/零面积），否则会污染 mAP 与可视化
            bs, ss, ls = filter_valid_boxes(bs, ss, ls)
            if bs.shape[0] == 0:
                results.append((paddle.zeros([0, 4]), paddle.zeros([0]),
                                paddle.zeros([0], dtype="int64")))
                continue
            idx = batched_nms(bs, ss, ls, self.num_classes,
                              iou_threshold=nms_threshold, top_k=top_k)
            results.append((clip_boxes_norm(bs[idx]), ss[idx], ls[idx]))
        return results
