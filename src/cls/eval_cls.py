"""成员 B · 任务1：分类评估（准确率 / 精确率 / 召回率 / F1 + 混淆矩阵）

⚠️ 铁律 1：测试集只许最后用一次。
    本脚本默认在测试集上评估，请只在两个网络都训练完之后再跑；
    调参阶段请加 --split val 用验证集。

用法：
    python src/cls/eval_cls.py --model mobilenet_v3_small
    python src/cls/eval_cls.py --model resnet50_vd
    python src/cls/eval_cls.py --model resnet50_vd --split val      # 调参时用验证集
    python src/cls/eval_cls.py --model resnet50_vd --weights results/weights/cls_resnet50_vd_best.pdparams

输出：
    results/metrics/cls_eval_result.json        两个网络的指标都追加在这里（给 D 汇总用）
    results/metrics/cls_summary.csv             超参 + 结果汇总表
    results/figures/cls_confusion_<模型>.png    混淆矩阵图
    results/logs/cls_test_usage.csv             测试集使用记录（证明测试集只用过一次）
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import paddle

matplotlib.use("Agg")                     # 只存图，不弹窗
import matplotlib.pyplot as plt           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CLASS_EN,
    CLASS_NAMES,
    MODEL_DESC,
    MODEL_NAMES,
    ROOT,
    append_csv,
    build_loader,
    dump_json,
    evaluate_metrics,
    load_hparams,
    load_json,
    load_trained_model,
    print_metrics_table,
    rebuild_summary_csv,
    rel_to_root,
    run_epoch,
    set_seed,
)


def plot_confusion(cm: np.ndarray, model_name: str, split: str, out_path: Path) -> None:
    """画混淆矩阵。图里用英文类别名，避免不同电脑缺中文字体时出现方框。"""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(cm, cmap="Blues")
    labels = [f"{c}\n{CLASS_EN[c]}" for c in CLASS_NAMES]
    ax.set_xticks(range(len(labels)), labels=labels, fontsize=9)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(f"{model_name} - confusion matrix ({split} set, n={int(cm.sum())})", fontsize=12)

    row_sum = cm.sum(axis=1, keepdims=True)
    recall = np.diag(cm) / np.maximum(row_sum[:, 0], 1)      # 每类召回率，标在左侧方便看
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]}\n{cm[i, j] / max(row_sum[i, 0], 1):.1%}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    for i, r in enumerate(recall):
        ax.text(-0.75, i, f"{r:.1%}", ha="right", va="center", fontsize=8, color="#c0392b")
    ax.text(-0.75, -1.0, "recall", ha="right", va="center", fontsize=8, color="#c0392b")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="NEU 钢材缺陷分类评估（B）")
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--weights", default=None, help="权重路径，默认用 results/weights/cls_<模型>_best.pdparams")
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="在哪份数据上评估；调参请用 val，测试集只跑一次")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_hparams(args.model)
    set_seed(cfg["seed"])
    batch_size = args.batch_size or int(cfg["batch_size"])

    print("=" * 66)
    print(f"NEU 钢材缺陷分类评估  |  {args.model}  |  数据集：{args.split}")
    print("=" * 66)
    if args.split == "test":
        print("⚠️  你在用测试集评估。铁律：测试集只在最后用一次，调参请改用 --split val。")
    print(f"网络    : {MODEL_DESC[args.model]}")

    model, weights = load_trained_model(args.model, args.weights)
    print(f"权重    : {rel_to_root(weights)}")

    dataset, loader = build_loader(args.split, cfg["size"], batch_size, train=False, limit=args.limit)
    criterion = paddle.nn.CrossEntropyLoss()
    loss, y_true, y_pred = run_epoch(model, loader, criterion)
    metrics = evaluate_metrics(y_true, y_pred)

    print(f"\n{args.split} 集共 {len(dataset)} 张，平均 loss = {loss:.4f}")
    print_metrics_table(metrics, f"[{args.model}] {args.split} 集指标")
    if "tools_metrics" in metrics:
        print(f"\n（已同时调用 D 的公共指标库：{metrics['tools_metrics']['source']}）")
    else:
        print("\n（D 的 src/tools/metrics.py 还是空的，本次用 src/cls/_common.py 的本地实现在算，口径见代码注释）")

    # ---- 混淆矩阵图 ----
    fig_path = ROOT / "results" / "figures" / f"cls_confusion_{args.model}.png"
    plot_confusion(np.asarray(metrics["confusion_matrix"]), args.model, args.split, fig_path)

    # ---- 结果 JSON（两个网络都往同一个文件里追加） ----
    result_path = ROOT / "results" / "metrics" / "cls_eval_result.json"
    data = load_json(result_path, default={})
    data.setdefault("task", "classification")
    data.setdefault("owner", "B")
    data["class_names"] = CLASS_NAMES
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data.setdefault("runs", {})
    data["runs"][args.model] = {
        "model": args.model,
        "num_params": None,
        "split": args.split,
        "num_images": len(dataset),
        "batch_size": batch_size,
        "loss": float(loss) if loss is not None else None,
        "weights": rel_to_root(weights),
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "figure": rel_to_root(fig_path),
    }
    dump_json(data, result_path)

    # ---- 测试集使用记录：一跑测试集就留痕，正好证明"只用过一次" ----
    usage_path = ROOT / "results" / "logs" / "cls_test_usage.csv"
    if args.split == "test":
        append_csv(usage_path,
                   ["time", "model", "split", "weights", "accuracy", "macro_f1", "note"],
                   [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.model, args.split,
                    rel_to_root(weights), f"{metrics['accuracy']:.6f}", f"{metrics['macro']['f1']:.6f}",
                    "正式评估（测试集仅此一次）"])
    else:
        append_csv(usage_path,
                   ["time", "model", "split", "weights", "accuracy", "macro_f1", "note"],
                   [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.model, args.split,
                    rel_to_root(weights), f"{metrics['accuracy']:.6f}", f"{metrics['macro']['f1']:.6f}",
                    "调参用验证集，未消耗测试集"])

    summary = rebuild_summary_csv()
    print(f"\n混淆矩阵：{rel_to_root(fig_path)}")
    print(f"结果 JSON：{rel_to_root(result_path)}")
    print(f"汇总表  ：{rel_to_root(summary)}")
    print(f"测试集使用记录：{rel_to_root(usage_path)}")


if __name__ == "__main__":
    main()
