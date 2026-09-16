# src/tools/metrics.py
"""公共指标库：分类 + 检测统一口径（成员 D）

为什么要单独有这个文件
----------------------
分类（B）与检测（C）原来各自实现了一份指标计算：同一个名字（准确率 / mAP）
在两处各算一遍，口径一旦有细微差异，两个任务的数字就不能直接横向比较。
本模块把两边的口径收到一处，作为全组唯一的指标出口。

口径对齐说明
------------
* 分类：与 ``src/cls/_common.py`` 的 ``metrics_from_cm`` 一致 ——
  准确率、每类 P/R/F1、宏平均、按 support 加权平均、混淆矩阵。
* 检测：与 ``src/det/core/utils.py`` 的 ``evaluate_map`` 一致 ——
  逐类按分数全局排序匹配 GT（IoU >= 阈值且类别相同、GT 未被占用为 TP），
  逐点累计 precision/recall，用 VOC 2010+ 的 101 点插值法求 AP，mAP 取各类平均。

调用方如何接入
--------------
* B：``src/cls/_common.py`` 的 ``external_cls_metrics()`` 会按名字探测
  ``classification_metrics`` / ``cls_metrics`` / ``evaluate_classification`` /
  ``compute_cls_metrics`` / ``cls_report``。本模块已提供这些别名，命中后 B 会把
  结果额外记进 ``tools_metrics`` 字段做交叉核对（主口径仍是它自己的实现，
  已有结果不会变）。
* C：把 ``src/det/core/utils.py`` 里的 ``evaluate_map`` 换成
  ``from tools.metrics import evaluate_map`` 即可，入参顺序与返回结构完全一致。
"""

import json

import numpy as np

CLASS_NAMES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
NUM_CLASSES = len(CLASS_NAMES)
_EPS = 1e-12


# ============================== 分类 ==============================
def accuracy(y_true, y_pred) -> float:
    """返回准确率。"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true 与 y_pred 形状不一致：{y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """返回混淆矩阵，``cm[i, j]`` = 真实为 i 却被预测成 j 的数量。"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true 与 y_pred 形状不一致：{y_true.shape} vs {y_pred.shape}")

    cm = np.zeros((num_classes, num_classes), dtype="int64")
    for t, p in zip(y_true, y_pred):
        ti, pi = int(t), int(p)
        if not 0 <= ti < num_classes or not 0 <= pi < num_classes:
            raise ValueError(f"标签越界：true={ti}, pred={pi}, num_classes={num_classes}")
        cm[ti, pi] += 1
    return cm


def metrics_from_confusion_matrix(cm) -> dict:
    """由混淆矩阵算准确率、每类 P/R/F1 与宏平均 / 加权平均（口径同 B）。"""
    cm = np.asarray(cm, dtype="float64")
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"混淆矩阵必须是方阵，收到 {cm.shape}")

    n = cm.shape[0]
    names = CLASS_NAMES[:n] if n <= len(CLASS_NAMES) else [f"class_{i}" for i in range(n)]

    total = cm.sum()
    tp = np.diag(cm)
    rows = cm.sum(axis=1)          # 每类真实数量（support）
    cols = cm.sum(axis=0)          # 每类被预测的数量
    precision = tp / np.maximum(cols, _EPS)
    recall = tp / np.maximum(rows, _EPS)
    f1 = 2 * precision * recall / np.maximum(precision + recall, _EPS)
    weights = rows / max(rows.sum(), _EPS)

    return {
        "accuracy": float(tp.sum() / max(total, _EPS)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(rows[i]),
            }
            for i, name in enumerate(names)
        },
        "macro": {
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1": float(f1.mean()),
        },
        "weighted": {
            "precision": float((precision * weights).sum()),
            "recall": float((recall * weights).sum()),
            "f1": float((f1 * weights).sum()),
        },
        "confusion_matrix": cm.astype("int64").tolist(),
        "num_samples": int(total),
    }


def precision_recall_f1(y_true, y_pred, average: str = "macro") -> dict:
    """返回精确率、召回率、F1。

    ``average='macro'`` 各类等权；``'weighted'`` 按 support 加权；
    ``'per_class'`` 返回每类明细。
    """
    if average not in ("macro", "weighted", "per_class"):
        raise ValueError(f"average 只支持 macro / weighted / per_class，收到 {average!r}")

    full = metrics_from_confusion_matrix(confusion_matrix(y_true, y_pred))
    if average == "per_class":
        return {
            name: {k: v for k, v in detail.items() if k != "support"}
            for name, detail in full["per_class"].items()
        }
    return dict(full[average])


def classification_metrics(y_true, y_pred, num_classes: int = NUM_CLASSES) -> dict:
    """完整分类指标（B 的 ``external_cls_metrics()`` 探测入口）。

    返回结构与 B 本地实现 ``metrics_from_cm`` 完全一致，便于逐字段交叉核对。
    """
    return metrics_from_confusion_matrix(confusion_matrix(y_true, y_pred, num_classes))


# B 会按这几个名字逐个探测，任意一个命中即可，这里全部提供。
cls_report = classification_metrics
cls_metrics = classification_metrics
evaluate_classification = classification_metrics
compute_cls_metrics = classification_metrics


# ============================== 检测 ==============================
def _as_boxes(x) -> np.ndarray:
    boxes = np.asarray(x, dtype="float64")
    return boxes.reshape(-1, 4)


def _iou_one_to_many(box, boxes) -> np.ndarray:
    """单个框与一组框的 IoU（xyxy；归一化或像素坐标同尺度即可）。"""
    box = np.asarray(box, dtype="float64").reshape(4)
    boxes = _as_boxes(boxes)
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype="float64")

    lt = np.maximum(box[None, :2], boxes[:, :2])
    rb = np.minimum(box[None, 2:], boxes[:, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, 0] * wh[:, 1]

    area_a = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    side_b = np.clip(boxes[:, 2:] - boxes[:, :2], 0.0, None)
    area_b = side_b[:, 0] * side_b[:, 1]
    union = area_a + area_b - inter
    return np.where(union > _EPS, inter / np.maximum(union, _EPS), 0.0)


def iou_matrix(boxes_a, boxes_b) -> np.ndarray:
    """两组框的 IoU 矩阵，形状 ``(len(boxes_a), len(boxes_b))``。"""
    a = _as_boxes(boxes_a)
    b = _as_boxes(boxes_b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype="float64")

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]

    side_a = np.clip(a[:, 2:] - a[:, :2], 0.0, None)
    area_a = (side_a[:, 0] * side_a[:, 1])[:, None]
    side_b = np.clip(b[:, 2:] - b[:, :2], 0.0, None)
    area_b = (side_b[:, 0] * side_b[:, 1])[None, :]
    union = area_a + area_b - inter
    return np.where(union > _EPS, inter / np.maximum(union, _EPS), 0.0)


def compute_ap(recall, precision) -> float:
    """单类 AP，VOC 2010+ 的 101 点插值口径。"""
    rec = np.asarray(recall, dtype="float64").reshape(-1)
    prec = np.asarray(precision, dtype="float64").reshape(-1)
    if rec.size == 0 or prec.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    # precision 单调不增包络
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map(predictions, ground_truths, num_classes: int = NUM_CLASSES,
                 class_names=None, iou_threshold: float = 0.5,
                 max_dets: int = 100) -> dict:
    """计算 mAP@0.5 与每类 AP，并返回 PR 曲线数据（口径同 C）。

    参数
    ----
    predictions : list[dict]，每张图一项，含
                  ``'boxes'``(N,4 归一化 xyxy) / ``'scores'``(N,) / ``'labels'``(N,)
    ground_truths : list[dict]，每张图一项，含
                  ``'boxes'``(M,4 归一化 xyxy) / ``'labels'``(M,)
    iou_threshold : 判定 TP 的 IoU 阈值（本作业要求 mAP@0.5）

    返回 dict：``map`` / ``per_class`` / ``pr_curves`` / ``overall`` /
    ``iou_threshold`` / ``num_images``
    """
    n_images = len(predictions)
    if n_images != len(ground_truths):
        raise ValueError(f"预测数量({n_images})与标注数量({len(ground_truths)})不一致")

    names = list(class_names) if class_names is not None else CLASS_NAMES[:num_classes]

    per_class = {}
    pr_curves = {}
    aps = []
    tp_all = fp_all = fn_all = 0

    for c in range(num_classes):
        # ---- 1. 该类全部预测，按分数降序（每图先截 max_dets） ----
        entries = []
        for i in range(n_images):
            p = predictions[i]
            labels = np.asarray(p["labels"]).reshape(-1)
            sel = labels == c
            if not sel.any():
                continue
            boxes = _as_boxes(p["boxes"])[sel]
            scores = np.asarray(p["scores"], dtype="float64").reshape(-1)[sel]
            for o in np.argsort(-scores)[:max_dets]:
                entries.append((float(scores[o]), i, boxes[o]))
        entries.sort(key=lambda e: -e[0])

        # ---- 2. 该类全部 GT，按图号索引 ----
        gt_by_img = {}
        n_gt = 0
        for i in range(n_images):
            g = ground_truths[i]
            labels = np.asarray(g["labels"]).reshape(-1)
            sel = labels == c
            boxes = _as_boxes(g["boxes"])[sel] if sel.any() else np.zeros((0, 4))
            gt_by_img[i] = {"boxes": boxes, "used": np.zeros(len(boxes), dtype=bool)}
            n_gt += len(boxes)

        n_pred = len(entries)
        tp = np.zeros(n_pred)
        fp = np.zeros(n_pred)

        # ---- 3. 按分数从高到低匹配 GT ----
        for k, (_score, i, box) in enumerate(entries):
            gts = gt_by_img[i]
            if len(gts["boxes"]) == 0:
                fp[k] = 1
                continue
            ious = _iou_one_to_many(box, gts["boxes"])
            best = int(np.argmax(ious))
            if ious[best] >= iou_threshold and not gts["used"][best]:
                tp[k] = 1
                gts["used"][best] = True
            else:
                fp[k] = 1

        key = names[c] if c < len(names) else f"class_{c}"

        if n_gt == 0:
            # 该类在评估集里没有 GT：AP 记 0，仍计入平均（与 C 一致）
            per_class[key] = 0.0
            pr_curves[key] = {"recall": [0.0], "precision": [0.0], "ap": 0.0,
                              "n_gt": 0, "n_pred": int(n_pred)}
            aps.append(0.0)
            continue

        ctp = np.cumsum(tp)
        cfp = np.cumsum(fp)
        rec = ctp / n_gt
        prec = ctp / np.clip(ctp + cfp, _EPS, None)
        ap = compute_ap(rec, prec)

        per_class[key] = ap
        pr_curves[key] = {"recall": rec.tolist(), "precision": prec.tolist(),
                          "ap": ap, "n_gt": int(n_gt), "n_pred": int(n_pred)}
        aps.append(ap)

        tp_all += int(tp.sum())
        fp_all += int(fp.sum())
        fn_all += int(n_gt - tp.sum())

    overall_prec = tp_all / max(tp_all + fp_all, _EPS)
    overall_rec = tp_all / max(tp_all + fn_all, _EPS)
    overall_f1 = 2 * overall_prec * overall_rec / max(overall_prec + overall_rec, _EPS)

    return {
        "map": float(np.mean(aps)) if aps else 0.0,
        "per_class": per_class,
        "pr_curves": pr_curves,
        "overall": {
            "precision": float(overall_prec),
            "recall": float(overall_rec),
            "f1": float(overall_f1),
            "tp": tp_all, "fp": fp_all, "fn": fn_all,
        },
        "iou_threshold": iou_threshold,
        "num_images": n_images,
    }


def compute_map(pred_boxes, gt_boxes, iou_threshold: float = 0.5,
                num_classes: int = NUM_CLASSES) -> dict:
    """返回 mAP@0.5、每类 AP、PR 曲线与总体精确率 / 召回率。

    ``pred_boxes`` / ``gt_boxes`` 与 ``evaluate_map`` 的
    ``predictions`` / ``ground_truths`` 同构。
    """
    return evaluate_map(pred_boxes, gt_boxes, num_classes=num_classes,
                        iou_threshold=iou_threshold)


def pr_curve(pred_boxes, gt_boxes, iou_threshold: float = 0.5,
             num_classes: int = NUM_CLASSES) -> dict:
    """返回 PR 曲线数据（每类的 recall / precision / ap）。"""
    return evaluate_map(pred_boxes, gt_boxes, num_classes=num_classes,
                        iou_threshold=iou_threshold)["pr_curves"]


# ============================== 通用 ==============================
def count_params(model) -> int:
    """统计模型参数量（paddle / torch 的 Layer 均适用）。"""
    params = getattr(model, "parameters", None)
    if callable(params):
        total = 0
        for p in params():
            numel = getattr(p, "numel", None)
            total += int(numel()) if callable(numel) else int(np.prod(p.shape))
        return total
    if hasattr(model, "shape"):                      # 直接传 tensor / ndarray
        return int(np.prod(model.shape))
    raise TypeError(f"无法统计参数量的对象：{type(model)!r}")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
