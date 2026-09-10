"""成员 C · 检测任务公共工具：配置读取、指标计算、日志

指标部分（mAP@0.5 / 每类 AP / PR 曲线）在本文件内实现，不依赖外部包。
原因：成员 D 的 `src/tools/metrics.py` 尚未交付（0 字节），而 C 的评估不能停工等它。
接口按「D 交付后可直接替换」设计 —— 训练/评估脚本只用 evaluate_map() 这一个入口，
将来 D 的指标库就绪后，把本函数内部换成 `from tools.metrics import evaluate_map` 即可。

评测协议（与 VOC/COCO 一致，便于和别的复现结果对比）：
    1. 每张图每个类别只保留分数最高的若干预测，按分数全局排序；
    2. 按分数从高到低依次匹配 GT（IoU >= 阈值且类别相同、GT 未被占用）为 TP，否则 FP；
    3. 逐点累计 precision/recall，用 101 点插值法算 AP；
    4. mAP = 各类 AP 的平均。
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import paddle
import yaml

# 本文件位于 src/det/core/utils.py：parents[0]=core [1]=det [2]=src [3]=项目根。
# 注意与 src/data/*.py 不同（那些在 src/xxx/ 下只要 parents[2]），别照抄。
ROOT = Path(__file__).resolve().parents[3]
# 立刻校验根目录找对了：定位不到 label_list.txt 说明层级算错，早报错好过训练到一半才发现
if not (ROOT / "dataset" / "det" / "label_list.txt").exists():
    raise RuntimeError(
        f"项目根目录识别有误：{ROOT}\n"
        f"预期存在 {ROOT / 'dataset' / 'det' / 'label_list.txt'}，请检查 utils.py 的 ROOT 层级")


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
def load_config(path) -> dict:
    """读 yml 配置，并填入实验无关的默认值（便于配置文件保持精简）"""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"找不到配置文件：{p}")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    cfg.setdefault("num_classes", 6)
    cfg["train"] = {**{"epochs": 100, "batch_size": 8, "num_workers": 0,
                       "save_every": 10, "log_every": 50}, **cfg.get("train", {})}
    cfg["optimizer"] = {**{"type": "Momentum", "lr": 0.01, "momentum": 0.9,
                           "weight_decay": 0.0005, "warmup_epochs": 2,
                           "scheduler": "cosine"}, **cfg.get("optimizer", {})}
    cfg["data"] = {**{"im_size": 416, "augment": True, "mosaic_p": 0.5,
                      "close_mosaic_epoch": 10}, **cfg.get("data", {})}
    cfg["eval"] = {**{"score_threshold": 0.01, "nms_threshold": 0.5,
                      "top_k": 100, "iou_threshold": 0.5}, **cfg.get("eval", {})}
    cfg["output"] = {**{"weight_dir": "results/weights",
                        "log_dir": "results/logs",
                        "metric_dir": "results/metrics"}, **cfg.get("output", {})}
    if not cfg.get("name"):
        cfg["name"] = p.stem
    return cfg


def seed_everything(seed: int = 2026):
    """统一随机种子（项目铁律 2）。paddle/numpy/python random 三处都要设"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def resolve_dir(rel) -> Path:
    """把配置里的相对路径按项目根目录解析，并确保目录存在"""
    p = Path(rel)
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def pick_device(prefer_gpu: bool = True) -> str:
    """选设备并打印，让日志里能看出到底跑在哪"""
    if prefer_gpu and paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        paddle.set_device("gpu")
        return "gpu"
    paddle.set_device("cpu")
    return "cpu"


# --------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------
class CSVLogger:
    """把每个 epoch 的记录追加写入 CSV，方便 D 的 log2table.py 直接解析"""

    def __init__(self, path: Path, fields):
        self.path = Path(path)
        self.fields = list(fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(",".join(self.fields) + "\n", encoding="utf-8")

    def log(self, **kwargs):
        row = []
        for f in self.fields:
            v = kwargs.get(f, "")
            if isinstance(v, float):
                v = f"{v:.6g}"
            row.append(str(v))
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(",".join(row) + "\n")


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
# 指标：mAP@0.5 与每类 AP
# --------------------------------------------------------------------------
def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """VOC 2010+ 的 101 点插值 AP"""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map(predictions, ground_truths, num_classes, class_names,
                 iou_threshold: float = 0.5, max_dets: int = 100):
    """计算 mAP 与每类 AP，并返回 PR 曲线数据

    参数
    ----
    predictions : list[dict]，每张图一个，含
                  'boxes'(N,4 归一化 xyxy) / 'scores'(N,) / 'labels'(N,)
    ground_truths : list[dict]，每张图一个，含
                  'boxes'(M,4 归一化 xyxy) / 'labels'(M,)
    iou_threshold : 判定 TP 的 IoU 阈值（本作业要求 mAP@0.5）

    返回 dict：map、per_class（每类 AP）、pr_curves、以及总 TP/FP/FN 统计
    """
    n_images = len(predictions)
    if n_images != len(ground_truths):
        raise ValueError("预测数量与标注数量不一致")

    per_class = {}
    pr_curves = {}
    aps = []

    for c in range(num_classes):
        # 收集该类全部预测（含图号），按分数降序
        entries = []
        for i in range(n_images):
            p = predictions[i]
            sel = np.asarray(p["labels"]) == c
            if not sel.any():
                continue
            bx = np.asarray(p["boxes"])[sel]
            sc = np.asarray(p["scores"])[sel]
            order = np.argsort(-sc)[:max_dets]
            for o in order:
                entries.append((float(sc[o]), i, bx[o]))
        entries.sort(key=lambda e: -e[0])

        # 该类全部 GT，按图号索引
        gt_by_img = {}
        n_gt = 0
        for i in range(n_images):
            g = ground_truths[i]
            sel = np.asarray(g["labels"]) == c
            boxes = np.asarray(g["boxes"])[sel] if sel.any() else np.zeros((0, 4))
            gt_by_img[i] = {"boxes": boxes, "used": np.zeros(len(boxes), dtype=bool)}
            n_gt += len(boxes)

        n_pred = len(entries)
        tp = np.zeros(n_pred)
        fp = np.zeros(n_pred)

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

        if n_gt == 0:
            # 该类在评估集里没有 GT：AP 记 0（若也没预测则不算入平均）
            per_class[class_names[c]] = 0.0
            pr_curves[class_names[c]] = {"recall": [0.0], "precision": [0.0], "ap": 0.0}
            aps.append(0.0)
            continue

        ctp = np.cumsum(tp)
        cfp = np.cumsum(fp)
        rec = ctp / n_gt
        prec = ctp / np.clip(ctp + cfp, 1e-9, None)
        ap = _voc_ap(rec, prec)
        per_class[class_names[c]] = ap
        pr_curves[class_names[c]] = {
            "recall": rec.tolist(),
            "precision": prec.tolist(),
            "ap": ap,
            "n_gt": int(n_gt),
            "n_pred": int(n_pred),
        }
        aps.append(ap)

    return {
        "map": float(np.mean(aps)) if aps else 0.0,
        "per_class": per_class,
        "pr_curves": pr_curves,
        "iou_threshold": iou_threshold,
        "num_images": n_images,
    }


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """单个框与一组框的 IoU（纯 numpy，避免评估阶段占显存）"""
    if len(boxes) == 0:
        return np.zeros(0)
    lt = np.maximum(box[None, :2], boxes[:, :2])
    rb = np.minimum(box[None, 2:], boxes[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    a1 = max((box[2] - box[0]) * (box[3] - box[1]), 0)
    a2 = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return inter / np.clip(a1 + a2 - inter, 1e-9, None)


def dump_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
