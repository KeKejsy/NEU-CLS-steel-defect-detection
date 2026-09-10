"""成员 C · 检测核心：损失函数与解码后处理

包含：
    - IoULoss        : 边框回归损失（GIoU/DIoU/CIoU 可选），YOLOv3 用
    - DistributionFocalLoss / bbox2distance / distance2bbox : PP-YOLOE 的 DFL 回归
    - decode_yolov3  : YOLOv3 三尺度输出 -> 归一化 xyxy 框
    - decode_ppyoloe : PP-YOLOE 输出 -> 归一化 xyxy 框

所有解码结果统一为「归一化 xyxy + 分数 + 类别」，交给 boxes.batched_nms 收尾。
"""

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .boxes import cxcywh_to_xyxy, box_iou

EPS = 1e-9


# --------------------------------------------------------------------------
# 边框回归损失
# --------------------------------------------------------------------------
class IoULoss(nn.Layer):
    """IoU 系列损失，pred/target 为同形状的 (..., 4) xyxy 框（支持 (N,4) 与 (B,N,4)）

    giou/diou/ciou 的区别在惩罚项：GIoU 看外接框，DIoU 看中心距，CIoU 再看长宽比。
    本数据集框平均只占全图 17.45%（属中小目标），中心距惩罚更稳，两种都支持由配置选。

    reduction='none' 时返回 (...,) 即逐框损失，便于调用方按掩码加权。
    """

    def __init__(self, loss_type: str = "giou", reduction: str = "mean"):
        super().__init__()
        assert loss_type in ("iou", "giou", "diou", "ciou"), f"不支持的 loss_type: {loss_type}"
        assert reduction in ("none", "mean", "sum")
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(self, pred: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        # 先展平成 (N,4) 统一处理，算完再还原前置维
        lead = pred.shape[:-1]
        n = 1
        for d in lead:
            n *= d
        p = pred.reshape([n, 4])
        t = target.reshape([n, 4])

        # 这里是「第 i 个预测框 vs 第 i 个目标框」的一一对应关系，
        # 所以直接逐元素算交并即可。**绝不能调用 box_iou**：那会构造 (N,N) 两两矩阵，
        # 在 (B,N,4) 输入下 N 可达数万（如 8x8500），矩阵元素上亿，直接爆显存
        # （实测申请 34GB 而显存只有 12GB）。
        lt = paddle.maximum(p[:, 0:2], t[:, 0:2])
        rb = paddle.minimum(p[:, 2:4], t[:, 2:4])
        iw = paddle.clip(rb[:, 0] - lt[:, 0], min=0)
        ih = paddle.clip(rb[:, 1] - lt[:, 1], min=0)
        inter = iw * ih

        ap = paddle.clip(p[:, 2] - p[:, 0], min=0) * paddle.clip(p[:, 3] - p[:, 1], min=0)
        at = paddle.clip(t[:, 2] - t[:, 0], min=0) * paddle.clip(t[:, 3] - t[:, 1], min=0)
        union = ap + at - inter
        iou = inter / paddle.clip(union, min=EPS)

        if self.loss_type == "iou":
            loss = 1 - iou
        else:
            cw = paddle.clip(paddle.maximum(p[:, 2], t[:, 2]) - paddle.minimum(p[:, 0], t[:, 0]), min=0)
            ch = paddle.clip(paddle.maximum(p[:, 3], t[:, 3]) - paddle.minimum(p[:, 1], t[:, 1]), min=0)
            area_c = cw * ch
            if self.loss_type == "giou":
                loss = 1 - (iou - (area_c - union) / paddle.clip(area_c, min=EPS))
            else:
                pc = (p[:, 0:2] + p[:, 2:4]) / 2
                tc = (t[:, 0:2] + t[:, 2:4]) / 2
                rho2 = ((pc - tc) ** 2).sum(-1)
                c2 = cw ** 2 + ch ** 2
                if self.loss_type == "diou":
                    loss = 1 - iou + rho2 / paddle.clip(c2, min=EPS)
                else:  # ciou
                    pw = paddle.clip(p[:, 2] - p[:, 0], min=EPS)
                    ph = paddle.clip(p[:, 3] - p[:, 1], min=EPS)
                    tw = paddle.clip(t[:, 2] - t[:, 0], min=EPS)
                    th = paddle.clip(t[:, 3] - t[:, 1], min=EPS)
                    v = (4 / math.pi ** 2) * (paddle.atan(tw / th) - paddle.atan(pw / ph)) ** 2
                    alpha = v / paddle.clip(1 - iou + v, min=EPS)
                    loss = 1 - iou + rho2 / paddle.clip(c2, min=EPS) + alpha * v

        loss = loss.reshape(list(lead))
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BCEWithLogitsLoss(nn.Layer):
    """带正样本权重与标签平滑的二分类交叉熵（YOLOv3 的 obj/cls 分支用）

    正样本极少（每图 2~3 个框 / 上千个候选位置），不加 pos_weight 会被负样本淹没。
    """

    def __init__(self, pos_weight: float = 1.0, label_smooth: float = 0.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.label_smooth = label_smooth

    def forward(self, logits, targets, positive_mask=None):
        """logits/targets 同 shape；positive_mask 为 True 的位置是正样本

        归一化口径：**正负样本各自取平均，再按 pos_weight 加权求和**
        （loss = mean_pos + w * mean_neg）。

        为什么不用简单的「全体取平均 + 正样本乘权重」：一张图有上万条候选位置、
        正样本只有 2~3 个（占比约 0.3%），全体取平均时损失完全由负样本主导，
        「所有位置都预测 0」几乎是零损失解 —— 实测 obj 分支就是这么退化的
        （logit 全为负，真正定位正确的框分数只有 0.056，在 307 个候选里排到第 135 位，
        被 top_k 截断淘汰，mAP 因此恒为 0，而 loss 曲线看着完全正常）。

        分开归一化后，正样本的增益与负样本的增益量级相当，模型才必须去分辨两者。
        权重取 1.0 意味着「正样本的平均误差」与「负样本的平均误差」同等重要。
        """
        if self.label_smooth > 0:
            targets = targets * (1 - self.label_smooth) + 0.5 * self.label_smooth
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        if positive_mask is None:
            return loss.mean()

        positive_mask = positive_mask.astype("bool")
        pos = paddle.masked_select(loss, positive_mask)
        neg = paddle.masked_select(loss, ~positive_mask)
        loss_pos = pos.mean() if int(pos.numel()) > 0 else paddle.zeros([1])
        loss_neg = neg.mean() if int(neg.numel()) > 0 else paddle.zeros([1])
        return loss_pos + self.pos_weight * loss_neg


# --------------------------------------------------------------------------
# PP-YOLOE 的 DFL 回归
# --------------------------------------------------------------------------
class DistributionFocalLoss(nn.Layer):
    """分布焦点损失：把框的每条边编码成长度为 reg_max+1 的离散分布

    相比直接回归一个数，DFL 学的是「这条边落在哪个区间」的分布，对小目标更友好。
    本数据集框宽高均值 71x95，属于中等偏小目标，DFL 正合适。
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_logits: paddle.Tensor, targets: paddle.Tensor,
                reduction: str = "none") -> paddle.Tensor:
        """pred_logits: (..., 4*(reg_max+1))   targets: (..., 4) 连续距离值（单位：stride）

        reduction='none' 时返回 (..., 4)：每条边的 DFL 损失，前置维原样保留，
        方便调用方按正样本掩码加权（与 IoULoss 的返回口径保持一致）。
        注意不要把前置维压平成 (B*N,4)，否则调用方拿到的形状与掩码对不上。
        """
        lead = list(pred_logits.shape[:-1])          # 如 [B, N]
        n = 1
        for d in lead:
            n *= d
        reg_max = pred_logits.shape[-1] // 4 - 1

        logits = pred_logits.reshape([-1, 4, reg_max + 1])   # (B*N, 4, reg_max+1)
        targets = targets.reshape([-1, 4])                   # (B*N, 4)

        # 目标离散化为左右两个整数桶 + 权重
        tl = paddle.floor(targets)
        tr = tl + 1
        wl = tr - targets
        wr = targets - tl
        tl = paddle.clip(tl, 0, reg_max).astype("int64")
        tr = paddle.clip(tr, 0, reg_max).astype("int64")

        flat_logits = logits.reshape([-1, reg_max + 1])
        tl_f = tl.reshape([-1])
        tr_f = tr.reshape([-1])
        wl_f = wl.reshape([-1]).astype(flat_logits.dtype)
        wr_f = wr.reshape([-1]).astype(flat_logits.dtype)

        loss_l = F.cross_entropy(flat_logits, tl_f, reduction="none") * wl_f
        loss_r = F.cross_entropy(flat_logits, tr_f, reduction="none") * wr_f
        loss = (loss_l + loss_r).reshape([-1, 4])            # (B*N, 4)
        loss = loss.reshape(lead + [4])                      # 还原前置维

        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        return loss.sum()


def bbox2distance(points: paddle.Tensor, bbox: paddle.Tensor,
                  reg_max: int, eps: float = 0.01) -> paddle.Tensor:
    """框 -> 四条边到锚点的距离（单位：stride 格）

    points: (N,2) 锚点中心；bbox: (N,4) xyxy 归一化框
    返回值裁剪到 (0, reg_max-1+eps)，避免超出分布范围导致 DFL 学不到。
    """
    lt = points - bbox[:, 0:2]
    rb = bbox[:, 2:4] - points
    return paddle.clip(paddle.concat([lt, rb], axis=-1),
                       min=0, max=reg_max - 1 + eps)


class Distance2BBox:
    """DFL 分布 -> 距离（期望值），再把距离还原成 xyxy 框"""

    def __init__(self, reg_max: int = 16):
        self.reg_max = reg_max
        self.project = paddle.arange(reg_max + 1, dtype="float32")

    def distance(self, logits: paddle.Tensor) -> paddle.Tensor:
        """logits: (..., 4*(reg_max+1)) -> (..., 4) 距离（对分布取期望）

        支持任意前置维度：既接受损失计算时的 (N,·)，也接受推理时的 (B,N,·)。
        实现上把前置维先摊平，算完再还原，避免为两种调用各写一份。
        """
        lead = logits.shape[:-1]            # 前置维，如 (N,) 或 (B,N)
        n = 1
        for d in lead:
            n *= d
        prob = F.softmax(logits.reshape([n, 4, self.reg_max + 1]), axis=-1)
        dist = F.linear(prob, self.project)  # (n,4)
        return dist.reshape(list(lead) + [4])

    def forward(self, logits: paddle.Tensor, points: paddle.Tensor) -> paddle.Tensor:
        """logits: (..., 4*(reg_max+1))；points: (..., 2) -> (..., 4) 归一化/像素 xyxy

        用 [..., 0:2] 这种带省略号的切片，而不是 d[:, 0:2]：
        后者在输入是 (B,N,4) 时会切错维度（把 N 当成 4 的那一维），
        只对 2D 输入才是对的。
        """
        d = self.distance(logits)
        x1y1 = points - d[..., 0:2]
        x2y2 = points + d[..., 2:4]
        return paddle.concat([x1y1, x2y2], axis=-1)

    def __call__(self, logits, points):
        return self.forward(logits, points)


def filter_valid_boxes(boxes: paddle.Tensor, scores: paddle.Tensor, labels: paddle.Tensor,
                       min_size: float = 1e-4):
    """丢掉退化框（x2<=x1 或 y2<=y1，或宽高过小）

    为什么必须做：训练初期模型可能吐出负宽高的框（实测出现 [58.8, 200.0, 0.0, 42.0]
    这种 x2<x1 的结果）。这类框面积算出来是 0，会被 NMS 当成互不重叠而全部留下，
    既污染 mAP（本来该算 FP 的变成无意义框），也会让可视化图上一堆零面积红点。
    这里统一过滤，训练、评估、可视化三条路径都走同一个函数，口径一致。
    """
    if boxes.shape[0] == 0:
        return boxes, scores, labels
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w > min_size) & (h > min_size)
    if int(keep.astype("int64").sum()) == boxes.shape[0]:
        return boxes, scores, labels
    idx = paddle.nonzero(keep).reshape([-1])
    return boxes[idx], scores[idx], labels[idx]


# --------------------------------------------------------------------------
# 解码：网络原始输出 -> 归一化 xyxy 框
# --------------------------------------------------------------------------
def decode_yolov3_scale(p: paddle.Tensor, anchors_norm: paddle.Tensor,
                        stride: int, im_size: int):
    """单个尺度的 YOLOv3 解码，返回归一化 xyxy（供损失与推理共用）

    p:            (B, A, H, W, 5+NC)，最后一维 [tx,ty,tw,th,obj_logit,cls_logits...]
    anchors_norm: (A, 2) 归一化 anchor 宽高（已按基准边长归一，与分辨率无关）
    stride:       该尺度下采样倍数
    im_size:      输入边长（像素）

    坐标链：网格 + sigmoid 偏移 -> 网格坐标（单位：格）-> 乘 stride 得像素 ->
           除以 im_size 得归一化。anchor 直接是归一化比例，无需再换算。
    """
    b, a, h, w, _ = p.shape
    grid_y, grid_x = paddle.meshgrid(paddle.arange(h), paddle.arange(w))
    grid = paddle.stack([grid_x, grid_y], axis=-1).astype(p.dtype).reshape([1, 1, h, w, 2])

    # 中心点：网格坐标（单位=格），归一化后 = (sigmoid + grid) * stride / im_size
    cxcy_norm = (paddle.sigmoid(p[..., 0:2]) + grid) * stride / float(im_size)
    # 宽高：exp 后乘归一化 anchor，直接就是归一化宽高
    wh_norm = paddle.exp(paddle.clip(p[..., 2:4], max=8.0)) * anchors_norm.reshape([1, a, 1, 1, 2])

    xyxy = cxcywh_to_xyxy(paddle.concat([cxcy_norm, wh_norm], axis=-1))
    obj = paddle.sigmoid(p[..., 4:5])
    cls = paddle.sigmoid(p[..., 5:])
    return xyxy.reshape([b, -1, 4]), obj.reshape([b, -1, 1]), cls.reshape([b, -1, cls.shape[-1]])


def decode_yolov3(preds, anchors, strides, num_classes: int, im_size: int):
    """YOLOv3 三尺度解码，返回 [(B, N, 4+1+NC), ...]

    末维为 [x1,y1,x2,y2, obj_score, cls_score...]，坐标全部归一化到 [0,1]。
    anchors 每个尺度一个 (A,2) 张量，为归一化宽高。
    """
    out = []
    for p, anc, stride in zip(preds, anchors, strides):
        xyxy, obj, cls = decode_yolov3_scale(p, anc, stride, im_size)
        out.append(paddle.concat([xyxy, obj, cls], axis=-1))
    return out


def decode_ppyoloe(pred_logits, pred_reg, anchors_points, stride: int,
                   num_classes: int, im_size: int, reg_max: int = 16):
    """PP-YOLOE 单尺度解码

    pred_logits: (B, N, NC) 已做 sigmoid 之前/之后的类别分数（本函数内做 sigmoid）
    pred_reg:    (B, N, 4*(reg_max+1))
    anchors_points: (N,2) 该尺度所有锚点中心（像素，基准 im_size）
    """
    b = pred_logits.shape[0]
    scores = paddle.sigmoid(pred_logits)  # (B,N,NC)

    d2b = Distance2BBox(reg_max)
    pts = anchors_points.astype("float32").reshape([1, -1, 2])
    pts = paddle.expand(pts, [b, anchors_points.shape[0], 2])
    boxes = d2b(pred_reg, pts) / float(im_size)  # 归一化 xyxy
    return boxes, scores


def distribute_anchors(feat_sizes, strides, reg_max: int = 16):
    """按特征图尺寸生成每个位置的锚点（中心，像素）与 stride 张量

    PP-YOLOE 是 anchor-free 的：每个位置一个锚点，不需要预设宽高。
    返回的锚点顺序必须与「网络输出 flatten 后的顺序」严格一致
    （逐尺度、尺度内按行优先），否则框会张冠李戴。
    """
    pts, strs = [], []
    for (h, w), s in zip(feat_sizes, strides):
        gy, gx = paddle.meshgrid(paddle.arange(h), paddle.arange(w))
        p = paddle.stack([gx, gy], axis=-1).astype("float32")  # (H,W,2) 网格坐标
        p = (p + 0.5) * s  # 网格中心 -> 像素坐标
        pts.append(p.reshape([-1, 2]))
        strs.append(paddle.full([h * w, 1], float(s), dtype="float32"))
    return paddle.concat(pts, axis=0), paddle.concat(strs, axis=0)


def xywh_norm_to_abs(boxes_norm, im_shape_hw):
    """归一化 xyxy -> 绝对像素 xyxy，im_shape_hw=(h,w)"""
    h, w = float(im_shape_hw[0]), float(im_shape_hw[1])
    scale = paddle.to_tensor([w, h, w, h], dtype=boxes_norm.dtype)
    return boxes_norm * scale
