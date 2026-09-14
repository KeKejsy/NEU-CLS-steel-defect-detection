"""成员 B · 分类任务公共模块（train / eval_cls / cam / export 共用）

只被 src/cls/ 下的脚本引用，不动别人的目录。

全组约定（见 README）：
    图片根目录  dataset/cls/images/                6 个子文件夹 Cr/In/Pa/PS/RS/Sc
    列表文件    dataset/cls/{train,val,test}.txt   每行 "Cr/crazing_1.jpg 0"
    类别顺序    Cr / In / Pa / PS / RS / Sc        与 dataset/det/label_list.txt 一致
    随机种子    2026
    测试集      最后才用一次，调参一律用验证集
"""

from __future__ import annotations

import csv
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import paddle
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]          # 项目根目录
CLS_DIR = Path(__file__).resolve().parent           # src/cls
IMG_ROOT = ROOT / "dataset" / "cls" / "images"

CLASS_NAMES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
CLASS_DESC = {"Cr": "裂纹", "In": "夹杂", "Pa": "斑块", "PS": "点蚀", "RS": "压入氧化皮", "Sc": "划痕"}
CLASS_EN = {"Cr": "crazing", "In": "inclusion", "Pa": "patches", "PS": "pitted", "RS": "rolled-in", "Sc": "scratches"}

MODEL_NAMES = ["resnet50_vd", "mobilenet_v3_small"]
MODEL_DESC = {
    "resnet50_vd": "ResNet50（paddle.vision 自带经典结构，约 2357 万参数）",
    "mobilenet_v3_small": "MobileNetV3-small（paddle.vision 自带轻量网络，约 153 万参数）",
}

SEED = 2026
NUM_CLASSES = len(CLASS_NAMES)

# ImageNet 均值方差：预训练权重就是按这个训练的，保持一致
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def set_seed(seed: int = SEED) -> None:
    """全组统一随机种子 2026，保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def dump_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def _jsonable(obj):
    """把 numpy / Path 之类的类型转成能写进 json 的类型"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def human_time(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}秒"
    if seconds < 3600:
        return f"{seconds // 60}分{seconds % 60}秒"
    return f"{seconds // 3600}小时{(seconds % 3600) // 60}分"


def rel_to_root(path: Path) -> str:
    """打印时用相对路径，报告里看起来干净"""
    try:
        return str(Path(path).relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# 配置与模型
# --------------------------------------------------------------------------- #
def load_hparams(model_name: str, config_path: str | None = None) -> dict:
    """读 src/cls/configs/<model_name>.yml，缺的键用下面的默认值补齐"""
    path = Path(config_path) if config_path else CLS_DIR / "configs" / f"{model_name}.yml"
    if not path.exists():
        raise SystemExit(f"找不到配置文件 {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg["model"] = model_name
    cfg["config_path"] = rel_to_root(path)
    cfg.setdefault("backbone", "resnet50" if model_name == "resnet50_vd" else "mobilenet_v3_small")
    cfg.setdefault("pretrained", True)
    cfg.setdefault("num_classes", NUM_CLASSES)
    cfg.setdefault("size", [200, 200])
    cfg.setdefault("epochs", 30)
    cfg.setdefault("batch_size", 32)
    cfg.setdefault("lr", 0.01)
    cfg.setdefault("momentum", 0.9)
    cfg.setdefault("weight_decay", 1e-4)
    cfg.setdefault("label_smoothing", 0.1)
    cfg.setdefault("warmup_epochs", 0)
    cfg.setdefault("lr_scheduler", "cosine")
    cfg.setdefault("seed", SEED)
    cfg.setdefault("num_workers", 0)          # Windows 上开多进程读图容易出岔子，用 0 最稳
    cfg.setdefault("log_every", 20)
    return cfg


def build_model(name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> paddle.nn.Layer:
    """按名字建网络。--model 的取值就是 MODEL_NAMES 里的两个。"""
    if name == "resnet50_vd":
        from paddle.vision.models import resnet50

        return resnet50(pretrained=pretrained, num_classes=num_classes)
    if name == "mobilenet_v3_small":
        from paddle.vision.models import mobilenet_v3_small

        return mobilenet_v3_small(pretrained=pretrained, num_classes=num_classes)
    raise SystemExit(f"未知网络：{name}（可选：{'、'.join(MODEL_NAMES)}）")


def count_params(model) -> int:
    return int(sum(int(np.prod(p.shape)) for p in model.parameters()))


def weights_path(model_name: str, which: str = "best") -> Path:
    """约定：results/weights/cls_<模型>_<best|last>.pdparams"""
    return ROOT / "results" / "weights" / f"cls_{model_name}_{which}.pdparams"


def load_trained_model(model_name: str, weights: str | Path | None = None, pretrained: bool = False):
    """建网络 + 读权重，返回 eval 模式的模型"""
    path = Path(weights) if weights else weights_path(model_name, "best")
    path = path if path.is_absolute() else ROOT / path
    if not path.exists():
        raise SystemExit(f"找不到权重文件 {path}，请先跑 src/cls/train.py --model {model_name}")
    model = build_model(model_name, pretrained=pretrained)
    state = paddle.load(str(path))
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.set_state_dict(state)
    model.eval()
    return model, path


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def read_split(split: str):
    """读 dataset/cls/<split>.txt，返回 [(图片路径, 标签id), ...]"""
    path = ROOT / "dataset" / "cls" / f"{split}.txt"
    if not path.exists():
        raise SystemExit(f"找不到 {path}：分类数据还没准备好（先让 A 跑 split_dataset.py）")
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rel, label = line.rsplit(" ", 1)
        items.append((IMG_ROOT / rel, int(label)))
    if not items:
        raise SystemExit(f"{path} 是空的")
    return items


def load_image(path, size=(200, 200)) -> np.ndarray:
    """读图 → float32 HWC、取值 0~1。
    图片内容其实是灰度，但文件是 RGB 三通道存的，统一转 3 通道，跟预训练权重对齐。"""
    img = Image.open(path).convert("RGB")
    if img.size != (size[1], size[0]):
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    return np.asarray(img, dtype="float32") / 255.0


def augment(arr: np.ndarray, pad: int = 16) -> np.ndarray:
    """训练期数据增强：随机裁剪 + 随机翻转 + 轻微亮度抖动。
    钢材缺陷没有固定朝向，所以左右、上下翻转都是安全且有效的。"""
    h, w = arr.shape[:2]
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    top = np.random.randint(0, 2 * pad + 1)
    left = np.random.randint(0, 2 * pad + 1)
    out = padded[top:top + h, left:left + w]
    if np.random.rand() < 0.5:
        out = out[:, ::-1]
    if np.random.rand() < 0.5:
        out = out[::-1, :]
    if np.random.rand() < 0.3:
        out = np.clip(out * np.random.uniform(0.9, 1.1), 0.0, 1.0)
    return np.ascontiguousarray(out)


def to_tensor(arr: np.ndarray) -> np.ndarray:
    """HWC 0~1 → CHW，并按 ImageNet 均值方差归一化"""
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1)).astype("float32")


class SteelClsDataset(paddle.io.Dataset):
    """NEU 钢材缺陷分类数据集，直接读 dataset/cls/<split>.txt"""

    def __init__(self, split: str, size=(200, 200), train: bool = False, limit: int | None = None):
        self.split = split
        self.size = tuple(size)
        self.train = train
        self.items = read_split(split)
        if limit:
            # 冒烟测试用：列表是按类别排好的，按步长抽样能覆盖到全部 6 类
            limit = int(limit)
            stride = max(1, len(self.items) // limit) if limit > 0 else 1
            self.items = self.items[::stride][:limit]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        arr = load_image(path, self.size)
        if self.train:
            arr = augment(arr)
        return to_tensor(arr), np.int64(label)


def build_loader(split: str, size, batch_size: int, train: bool = False, limit=None, num_workers: int = 0):
    dataset = SteelClsDataset(split, size=size, train=train, limit=limit)
    loader = paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=False,
        return_list=True,
    )
    return dataset, loader


# --------------------------------------------------------------------------- #
# 推理与指标
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion=None):
    """跑一遍数据，返回 (平均loss或None, 真实标签, 预测标签)"""
    model.eval()
    losses, ys, ps = [], [], []
    with paddle.no_grad():
        for x, y in loader:
            y = paddle.reshape(y, [-1]).astype("int64")
            logits = model(x)
            if criterion is not None:
                losses.append(float(criterion(logits, y)))
            ps.append(paddle.argmax(logits, axis=1).numpy())
            ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    mean_loss = float(np.mean(losses)) if losses else None
    return mean_loss, y_true, y_pred


def confusion_matrix(y_true, y_pred, num_classes: int = NUM_CLASSES) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype="int64")
    for t, p in zip(np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> dict:
    """由混淆矩阵算准确率、每类精确率/召回率/F1，以及宏平均、加权平均"""
    cm = np.asarray(cm, dtype="float64")
    eps = 1e-12
    total = cm.sum()
    tp = np.diag(cm)
    rows = cm.sum(axis=1)          # 每类真实数量（support）
    cols = cm.sum(axis=0)          # 每类被预测的数量
    precision = tp / np.maximum(cols, eps)
    recall = tp / np.maximum(rows, eps)
    f1 = 2 * precision * recall / np.maximum(precision + recall, eps)
    weights = rows / max(rows.sum(), eps)
    return {
        "accuracy": float(tp.sum() / max(total, eps)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(rows[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
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


def external_cls_metrics(y_true, y_pred):
    """如果 D 的公共指标库 src/tools/metrics.py 已经提供分类指标函数，就调用它。
    当前该文件还是空的，所以正常会返回 None，脚本自动退回本地实现。"""
    path = ROOT / "src" / "tools" / "metrics.py"
    if not path.exists() or path.stat().st_size == 0:
        return None
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        mod = importlib.import_module("src.tools.metrics")
    except Exception:
        return None
    for fn_name in ("classification_metrics", "cls_metrics", "evaluate_classification",
                    "compute_cls_metrics", "cls_report"):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            try:
                return fn(np.asarray(y_true), np.asarray(y_pred)), f"src/tools/metrics.py::{fn_name}"
            except Exception:
                return None
    return None


def evaluate_metrics(y_true, y_pred) -> dict:
    """指标主口径是本地实现；如果 D 的指标库可用，两边结果都存下来方便核对。"""
    result = metrics_from_cm(confusion_matrix(y_true, y_pred))
    ext = external_cls_metrics(y_true, y_pred)
    if ext is not None:
        result["tools_metrics"] = {"source": ext[1], "values": _jsonable(ext[0])}
    return result


def append_csv(path: Path, header, row) -> None:
    """追加一行 CSV，文件不存在时先写表头"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)


def print_metrics_table(metrics: dict, title: str = "") -> None:
    """在终端打印一张中文指标表"""
    if title:
        print(f"\n{title}")
    print("-" * 62)
    print(f"{'类别':<6}{'中文':<10}{'精确率':>10}{'召回率':>10}{'F1':>10}{'样本数':>8}")
    print("-" * 62)
    for name in CLASS_NAMES:
        m = metrics["per_class"][name]
        print(f"{name:<6}{CLASS_DESC[name]:<10}{m['precision']:>10.4f}{m['recall']:>10.4f}"
              f"{m['f1']:>10.4f}{m['support']:>8}")
    print("-" * 62)
    print(f"准确率(accuracy)      : {metrics['accuracy']:.4f}")
    print(f"宏平均  P/R/F1        : {metrics['macro']['precision']:.4f} / "
          f"{metrics['macro']['recall']:.4f} / {metrics['macro']['f1']:.4f}")
    print(f"加权平均 P/R/F1       : {metrics['weighted']['precision']:.4f} / "
          f"{metrics['weighted']['recall']:.4f} / {metrics['weighted']['f1']:.4f}")
    print(f"样本总数              : {metrics['num_samples']}")


def rebuild_summary_csv() -> Path:
    """把 B 的三个 JSON（训练结果 / 评估结果 / 导出测速）拼成一张表，D 画对比图直接用。

    输出：results/metrics/cls_summary.csv
    列：模型、参数量(M)、轮数、batch、lr、最好验证准确率、测试准确率、
        宏平均 P/R/F1、batch1 FPS、batch8 FPS、训练耗时
    """
    metrics_dir = ROOT / "results" / "metrics"
    eval_json = load_json(metrics_dir / "cls_eval_result.json")
    export_json = load_json(metrics_dir / "cls_export_result.json")
    header = ["model", "num_params_M", "epochs", "batch_size", "lr", "best_val_acc",
              "test_acc", "macro_precision", "macro_recall", "macro_f1",
              "fps_batch1", "fps_batch8", "train_time"]
    rows = []
    for name in MODEL_NAMES:
        train = load_json(metrics_dir / f"cls_train_result_{name}.json")
        runs = (eval_json.get("runs") or {})
        ev = runs.get(name, {})
        ex = (export_json.get("runs") or {}).get(name, {})
        if not train and not ev:
            continue
        hp = train.get("hyperparams", {})
        m = ev.get("metrics", {})
        rows.append([
            name,
            train.get("params_M", ""),
            hp.get("epochs", ""),
            hp.get("batch_size", ""),
            hp.get("lr", ""),
            round(train.get("best_val_acc", 0), 4) if train else "",
            round(m.get("accuracy", 0), 4) if m else "",
            round(m.get("macro", {}).get("precision", 0), 4) if m else "",
            round(m.get("macro", {}).get("recall", 0), 4) if m else "",
            round(m.get("macro", {}).get("f1", 0), 4) if m else "",
            ex.get("fps_batch1", ""),
            ex.get("fps_batch8", ""),
            train.get("train_time_human", ""),
        ])
    path = metrics_dir / "cls_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path
