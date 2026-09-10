"""成员 C · 检测核心：边框几何与 IoU/NMS

统一约定（全模块只用这一种，不再混用别的表示）：
    归一化坐标 (cx, cy, w, h)，数值范围 [0, 1]，相对所在图像宽高。
    输入图像会被缩放+padding 成变尺寸批次，所以绝不能用绝对像素坐标做中间表示。

用法：
    from core.boxes import cxcywh_to_xyxy, xyxy_to_cxcywh, box_iou, nms, batched_nms
"""

import paddle


def cxcywh_to_xyxy(boxes: paddle.Tensor) -> paddle.Tensor:
    """(cx,cy,w,h) -> (x1,y1,x2,y2)，末维 4

    用切片而不是 paddle.split，是为了兼容导出静态图时末维为动态 shape 的情况。
    """
    cx, cy = boxes[..., 0:1], boxes[..., 1:2]
    w, h = boxes[..., 2:3], boxes[..., 3:4]
    return paddle.concat([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def xyxy_to_cxcywh(boxes: paddle.Tensor) -> paddle.Tensor:
    """(x1,y1,x2,y2) -> (cx,cy,w,h)，末维 4"""
    x1, y1 = boxes[..., 0:1], boxes[..., 1:2]
    x2, y2 = boxes[..., 2:3], boxes[..., 3:4]
    return paddle.concat([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1)


def box_area(boxes: paddle.Tensor) -> paddle.Tensor:
    """边框面积，输入 (…, 4) xyxy，输出 (…, 1)。负宽高按 0 处理"""
    w = boxes[..., 2:3] - boxes[..., 0:1]
    h = boxes[..., 3:4] - boxes[..., 1:2]
    return paddle.clip(w, min=0) * paddle.clip(h, min=0)


def box_iou(boxes1: paddle.Tensor, boxes2: paddle.Tensor):
    """两两 IoU。输入 (N,4) 与 (M,4)（xyxy），返回 (N,M) 的 IoU 与 (N,M) 的并集面积

    并集面积一并返回，因为 GIoU/DIoU/CIoU 还要用，避免重复计算。
    """
    area1 = box_area(boxes1)  # (N,1)
    area2 = box_area(boxes2)  # (M,1)

    lt = paddle.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = paddle.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = paddle.clip(rb - lt, min=0)
    inter = wh[..., 0] * wh[..., 1]  # (N,M)

    union = area1 + area2.transpose([1, 0]) - inter
    return inter / paddle.clip(union, min=1e-9), union


def box_iou_batched(boxes1: paddle.Tensor, boxes2: paddle.Tensor):
    """逐样本算 IoU。输入 (B,N,4) 与 (B,M,4) -> (B,N,M)

    评估阶段用：每个样本 GT 数量不同，先 padding 成定长再批量矩阵算。
    """
    b1 = boxes1.unsqueeze(2)  # (B,N,1,4)
    b2 = boxes2.unsqueeze(1)  # (B,1,M,4)
    lt = paddle.maximum(b1[..., :2], b2[..., :2])
    rb = paddle.minimum(b1[..., 2:], b2[..., 2:])
    wh = paddle.clip(rb - lt, min=0)
    inter = wh[..., 0] * wh[..., 1]

    a1 = box_area(boxes1).squeeze(-1).unsqueeze(-1)  # (B,N,1)
    a2 = box_area(boxes2).squeeze(-1).unsqueeze(1)  # (B,1,M)
    union = a1 + a2 - inter
    return inter / paddle.clip(union, min=1e-9)


def nms(boxes: paddle.Tensor, scores: paddle.Tensor, iou_threshold: float = 0.5,
        top_k: int = 1000) -> paddle.Tensor:
    """单类 NMS，返回保留的索引（按分数降序）

    注意两点：
    1. paddle.vision.ops.nms 的形参顺序是 (boxes, iou_threshold, scores, ...)，
       scores 必须用关键字传，否则会撞上 iou_threshold 的位置；
    2. 该算子内部断言 top_k <= 框数，训练早期一张图可能只解码出几个框，
       直接传 100 会 AssertionError（实测在 epoch 2 验证时崩过）。
       所以这里把 top_k 夹到实际框数。
    """
    n = int(boxes.shape[0])
    if n == 0:
        return paddle.zeros([0], dtype="int64")
    keep = paddle.vision.ops.nms(boxes, iou_threshold=iou_threshold,
                                 scores=scores, top_k=min(int(top_k), n))
    return keep.astype("int64")


def batched_nms(boxes: paddle.Tensor, scores: paddle.Tensor, labels: paddle.Tensor,
                num_classes: int, iou_threshold: float = 0.5,
                top_k: int = 1000) -> paddle.Tensor:
    """多类 NMS：逐类独立抑制，返回保留的索引

    直接使用原生 nms 的 category_idxs/categories 参数做 batched NMS，
    比「类别号乘大偏移加到坐标上」的土办法更正确（后者会因坐标溢出失去精度）。
    同样需要把 top_k 夹到框数，原因见 nms() 的说明。
    """
    n = int(boxes.shape[0])
    if n == 0:
        return paddle.zeros([0], dtype="int64")
    keep = paddle.vision.ops.nms(
        boxes,
        iou_threshold=iou_threshold,
        scores=scores,
        category_idxs=labels.astype("int32"),
        categories=list(range(num_classes)),
        top_k=min(int(top_k), n),
    )
    return keep.astype("int64")


def clip_boxes_norm(boxes: paddle.Tensor, min_size: float = 1e-3) -> paddle.Tensor:
    """把归一化 xyxy 框夹到 [0,1]，并**保证结果仍是合法框**（宽高 >= min_size）

    关键：不能把四个坐标各自 clip 到 [0,1] —— 那样会把部分在画面外的框压成零宽/零高
    （例如 x1、x2 都超过 1，被同时压到 1），框退化成一条线，
    既污染 mAP（面积 0 的框在 NMS 里被当成互不重叠而全部保留），
    也让可视化图出现一堆看不见的点。实测这样做会留下 4082/27000 个退化框。

    正确做法：先夹「中心点 + 宽高」，再由中心重构四角。
    中心夹到 [w/2, 1-w/2] 等价于让框贴边，宽高不变，几何上恒合法。
    """
    cx = (boxes[..., 0:1] + boxes[..., 2:3]) / 2
    cy = (boxes[..., 1:2] + boxes[..., 3:4]) / 2
    w = paddle.clip(boxes[..., 2:3] - boxes[..., 0:1], min=min_size, max=1.0)
    h = paddle.clip(boxes[..., 3:4] - boxes[..., 1:2], min=min_size, max=1.0)
    # 注意：paddle.clip 的 min/max 只接受标量，这里是逐元素的边界，
    # 必须用 maximum/minimum 逐个夹，不能写成 clip(cx, min=w/2, max=1-w/2)
    cx = paddle.minimum(paddle.maximum(cx, w / 2), 1 - w / 2)
    cy = paddle.minimum(paddle.maximum(cy, h / 2), 1 - h / 2)
    return paddle.concat([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def box_wh_iou(wh1: paddle.Tensor, wh2: paddle.Tensor) -> paddle.Tensor:
    """只按宽高算 IoU（YOLOv3 选 anchor 用，与位置无关）

    wh1: (N,2)、wh2: (M,2) -> (N,M)
    """
    w1, h1 = wh1[:, 0:1], wh1[:, 1:2]
    w2, h2 = wh2[None, :, 0], wh2[None, :, 1]
    inter = paddle.minimum(w1, w2) * paddle.minimum(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    return inter / paddle.clip(union, min=1e-9)
