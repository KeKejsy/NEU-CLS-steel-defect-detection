"""成员 C · 检测评估脚本（测试集只在这里用一次）

用法：
    python src/det/eval_det.py --model yolov3
    python src/det/eval_det.py --model ppyoloe_s
    python src/det/eval_det.py --model yolov3 --weights results/weights/yolov3_best.pdparams
    python src/det/eval_det.py --model yolov3 --split val     # 只看验证集（调参用）

产出：
    results/metrics/<name>_eval_result.json    mAP / 每类 AP / P / R / F1 / PR 曲线 / 混淆矩阵
    results/figures/<name>_pr_curve.png        每类 PR 曲线
    results/figures/<name>_per_class_ap.png    每类 AP 柱状图
    results/figures/<name>_confusion.png       混淆矩阵

评测口径（与 VOC 一致，可直接与同类复现结果对比）：
    IoU 阈值 0.5；每类按分数排序匹配；101 点插值算 AP；mAP = 6 类 AP 的平均。

项目铁律 1：测试集只许最后用一次、调参一律用验证集。
本脚本默认评估 test，并在输出里明确标注用的是哪个划分。
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import paddle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import data as data_mod  # noqa: E402
from core import utils  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ARCH_TO_CONFIG = {
    "yolov3": "src/det/configs/yolov3.yml",
    "ppyoloe_s": "src/det/configs/ppyoloe_s.yml",
}

# 每类固定颜色，保证两个网络的图可以直接对照着看
CLASS_COLORS = ["#378ADD", "#1D9E75", "#D85A30", "#534AB7", "#BA7517", "#0F6E56"]


def compute_prf(tp, fp, fn):
    """由 TP/FP/FN 算精确率、召回率、F1（除零时返回 0）"""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _iou(a, b):
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:], b[2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[0] * wh[1]
    aa = max((a[2] - a[0]) * (a[3] - a[1]), 0)
    ab = max((b[2] - b[0]) * (b[3] - b[1]), 0)
    return inter / max(aa + ab - inter, 1e-9)


def match_stats(preds, gts, num_classes, iou_threshold=0.5):
    """逐类统计 TP/FP/FN，并构建混淆矩阵

    匹配规则与 mAP 计算一致：按分数降序，IoU >= 阈值且类别相同、GT 未被占用才算 TP。
    混淆矩阵额外记录错配情况，便于看出哪些缺陷容易混。
    行=预测类别，列=真实类别，最后一行/列为背景（漏检 / 误检）。
    """
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    conf = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for i in range(len(preds)):
        g_boxes = np.asarray(gts[i]["boxes"])
        g_labels = np.asarray(gts[i]["labels"]).astype(int)
        p_boxes = np.asarray(preds[i]["boxes"])
        p_scores = np.asarray(preds[i]["scores"])
        p_labels = np.asarray(preds[i]["labels"]).astype(int)

        order = np.argsort(-p_scores)
        used = np.zeros(len(g_boxes), dtype=bool)

        for k in order:
            box, pc = p_boxes[k], p_labels[k]
            best, best_iou = -1, 0.0
            for j in range(len(g_boxes)):
                if used[j] or g_labels[j] != pc:
                    continue
                iou = _iou(box, g_boxes[j])
                if iou > best_iou:
                    best_iou, best = iou, j
            if best >= 0 and best_iou >= iou_threshold:
                used[best] = True
                tp[pc] += 1
                conf[pc, g_labels[best]] += 1
            else:
                fp[pc] += 1
                conf[pc, num_classes] += 1

        for j in range(len(g_boxes)):
            if not used[j]:
                fn[g_labels[j]] += 1
                conf[num_classes, g_labels[j]] += 1

    return tp, fp, fn, conf


def plot_pr_curves(pr_curves, class_names, out_path, title):
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    for i, c in enumerate(class_names):
        d = pr_curves.get(c)
        if not d:
            continue
        rec = np.asarray(d["recall"], dtype=float)
        prec = np.asarray(d["precision"], dtype=float)
        if len(rec) and rec[-1] < 1.0:
            rec = np.append(rec, 1.0)
            prec = np.append(prec, 0.0)
        ax.plot(rec, prec, color=CLASS_COLORS[i % len(CLASS_COLORS)], lw=1.8,
                label=f"{c}  AP={d['ap']:.3f}")
    ax.set_xlabel("召回率 Recall")
    ax.set_ylabel("精确率 Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, ls="--", lw=0.6)
    ax.legend(fontsize=9, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_class_ap(per_class_ap, class_names, out_path, title, map_value):
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    vals = [per_class_ap.get(c, 0.0) for c in class_names]
    bars = ax.bar(class_names, vals, color=CLASS_COLORS[:len(class_names)], width=0.62)
    ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.axhline(map_value, color="#444", ls="--", lw=1.2, label=f"mAP@0.5 = {map_value:.4f}")
    ax.set_ylabel("AP@0.5")
    ax.set_ylim(0, max(max(vals) * 1.25, 0.1))
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion(conf, class_names, out_path, title):
    labels = list(class_names) + ["背景"]
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=150)
    norm = conf / np.clip(conf.sum(axis=1, keepdims=True), 1, None)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("真实类别")
    ax.set_ylabel("预测类别")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if conf[i, j] == 0:
                continue
            ax.text(j, i, str(conf[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 0.5 else "#222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="行归一化比例")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="C · 检测评估（mAP@0.5 / 每类 AP / PR 曲线）")
    ap.add_argument("--model", required=True, choices=sorted(ARCH_TO_CONFIG),
                    help="网络名，用于定位默认配置与权重")
    ap.add_argument("--config", default=None, help="覆盖默认配置文件")
    ap.add_argument("--weights", default=None, help="权重路径，默认 <name>_best.pdparams")
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="评估划分；默认 test（调参请用 val）")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    cfg_path = args.config or ARCH_TO_CONFIG[args.model]
    cfg = utils.load_config(cfg_path)
    name = cfg["name"]
    num_classes = int(cfg["num_classes"])
    im_size = int(cfg["data"]["im_size"])

    weights = args.weights or f"results/weights/{name}_best.pdparams"
    wpath = Path(weights)
    if not wpath.is_absolute():
        wpath = utils.ROOT / wpath
    if not wpath.exists():
        raise SystemExit(
            f"找不到权重 {wpath}\n请先训练：python src/det/train.py --config {cfg_path}")

    dev = utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(2026)

    fig_dir = utils.resolve_dir("results/figures")
    metric_dir = utils.resolve_dir(cfg["output"]["metric_dir"])

    print("=" * 70)
    print(f"成员 C · 检测评估    {name}（{cfg['model']['arch']}）")
    print("=" * 70)
    print(f"设备     : {dev}")
    print(f"权重     : {wpath.relative_to(utils.ROOT)}")
    print(f"评估划分 : {args.split}" + ("   <- 测试集，仅此一次" if args.split == "test" else "   （调参用）"))
    print(f"输入尺寸 : {im_size}   IoU 阈值: {cfg['eval']['iou_threshold']}")
    if args.split == "test":
        print("\n提示：按项目铁律，测试集只用于最终评估。若在调参，请改用 --split val。")

    model = data_mod.build_model(cfg)
    model.set_state_dict(paddle.load(str(wpath)))
    model.eval()

    ds, loader = data_mod.build_loader(
        utils.ROOT, args.split, im_size, max(int(cfg["train"]["batch_size"]), 8),
        augment=False, shuffle=False, num_workers=0)
    print(f"\n评估样本数: {len(ds)}")

    preds, gts = data_mod.detect_loader(
        model, loader, num_classes,
        score_threshold=float(cfg["eval"]["score_threshold"]),
        nms_threshold=float(cfg["eval"]["nms_threshold"]),
        top_k=int(cfg["eval"]["top_k"]))

    n_gt_total = sum(len(g["labels"]) for g in gts)
    n_pred_total = sum(len(p["labels"]) for p in preds)
    print(f"标注框总数 {n_gt_total}，预测框总数 {n_pred_total}"
          f"（平均每图 {n_pred_total/max(len(preds),1):.1f} 个）")

    res = utils.evaluate_map(preds, gts, num_classes, ds.class_names,
                            iou_threshold=float(cfg["eval"]["iou_threshold"]))
    tp, fp, fn, conf = match_stats(preds, gts, num_classes,
                                   iou_threshold=float(cfg["eval"]["iou_threshold"]))

    print("\n" + "-" * 82)
    print(f"{'类别':<6}{'中文':<14}{'AP@0.5':>9}{'精确率':>9}{'召回率':>9}{'F1':>8}"
          f"{'TP':>6}{'FP':>6}{'FN':>6}")
    print("-" * 82)
    per_class_detail = {}
    for i, c in enumerate(ds.class_names):
        p, r, f1 = compute_prf(tp[i], fp[i], fn[i])
        ap_v = res["per_class"].get(c, 0.0)
        per_class_detail[c] = {"ap50": ap_v, "precision": p, "recall": r, "f1": f1,
                               "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i]),
                               "cn_name": data_mod.CLASS_NAMES_CN.get(c, "")}
        print(f"{c:<6}{data_mod.CLASS_NAMES_CN.get(c,''):<14}{ap_v:>9.4f}{p:>9.4f}"
              f"{r:>9.4f}{f1:>8.4f}{int(tp[i]):>6}{int(fp[i]):>6}{int(fn[i]):>6}")
    print("-" * 82)
    micro_p, micro_r, micro_f1 = compute_prf(tp.sum(), fp.sum(), fn.sum())
    print(f"{'总体':<20}{res['map']:>9.4f}{micro_p:>9.4f}{micro_r:>9.4f}{micro_f1:>8.4f}"
          f"{int(tp.sum()):>6}{int(fp.sum()):>6}{int(fn.sum()):>6}")
    print(f"\n>>> mAP@0.5 = {res['map']:.4f}")

    tag = "" if args.split == "test" else f" ({args.split})"
    plot_pr_curves(res["pr_curves"], ds.class_names, fig_dir / f"{name}_pr_curve.png",
                   f"{name} 各类 PR 曲线（mAP@0.5={res['map']:.4f}）{tag}")
    plot_per_class_ap(res["per_class"], ds.class_names, fig_dir / f"{name}_per_class_ap.png",
                      f"{name} 每类 AP@0.5{tag}", res["map"])
    plot_confusion(conf, ds.class_names, fig_dir / f"{name}_confusion.png",
                   f"{name} 混淆矩阵{tag}")

    report = {
        "model": name, "arch": cfg["model"]["arch"], "split": args.split,
        "weights": str(wpath.relative_to(utils.ROOT)), "im_size": im_size,
        "iou_threshold": float(cfg["eval"]["iou_threshold"]),
        "score_threshold": float(cfg["eval"]["score_threshold"]),
        "nms_threshold": float(cfg["eval"]["nms_threshold"]),
        "num_images": len(ds), "num_gt_boxes": int(n_gt_total),
        "num_pred_boxes": int(n_pred_total), "map50": res["map"],
        "per_class": per_class_detail,
        "overall": {"precision": micro_p, "recall": micro_r, "f1": micro_f1,
                    "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum())},
        "confusion_matrix": {"labels": list(ds.class_names) + ["background"],
                             "matrix": conf.tolist()},
        "pr_curves": {c: {"recall": res["pr_curves"][c]["recall"],
                          "precision": res["pr_curves"][c]["precision"],
                          "ap": res["pr_curves"][c]["ap"]} for c in ds.class_names},
        "figures": {"pr_curve": f"results/figures/{name}_pr_curve.png",
                    "per_class_ap": f"results/figures/{name}_per_class_ap.png",
                    "confusion": f"results/figures/{name}_confusion.png"},
    }
    out = utils.dump_json(report, metric_dir / f"{name}_eval_result.json")
    print(f"\n评估报告：{out.relative_to(utils.ROOT)}")
    print(f"PR 曲线  ：results/figures/{name}_pr_curve.png")
    print(f"每类 AP  ：results/figures/{name}_per_class_ap.png")
    print(f"混淆矩阵 ：results/figures/{name}_confusion.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
