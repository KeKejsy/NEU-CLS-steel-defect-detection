"""成员 B · 训练曲线对比图（两个分类网络的验证集准确率与损失随轮次变化）

用途：报告里说明"两个网络的精度上限一样，但收敛速度和训练成本差很多"。
      纯读 results/metrics/cls_train_result_*.json，不重新训练，秒出图。

用法：python src/cls/train_curves.py
输出：results/figures/cls_train_curves.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_NAMES, ROOT, load_json, rel_to_root  # noqa: E402


def main() -> None:
    runs = {}
    for name in MODEL_NAMES:
        data = load_json(ROOT / "results" / "metrics" / f"cls_train_result_{name}.json")
        if data and data.get("history"):
            runs[name] = data
    if not runs:
        raise SystemExit("找不到训练结果 json，请先跑 src/cls/train.py")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    styles = {"resnet50_vd": ("#1f77b4", "o"), "mobilenet_v3_small": ("#d62728", "s")}

    for name, data in runs.items():
        hist = data["history"]
        epochs = [h["epoch"] for h in hist]
        color, marker = styles.get(name, ("#333333", "o"))
        label = f"{name} ({data.get('params_M', '?')} M params)"
        axes[0].plot(epochs, [h["val_acc"] for h in hist], color=color, marker=marker, ms=3.5,
                     lw=1.8, label=label)
        axes[1].plot(epochs, [h["train_loss"] for h in hist], color=color, marker=marker, ms=3.5,
                     lw=1.8, label=label)
        axes[2].plot(epochs, [h["val_loss"] for h in hist], color=color, marker=marker, ms=3.5,
                     lw=1.8, label=label)

        best_epoch = data.get("best_epoch")
        if best_epoch:
            axes[0].scatter([best_epoch], [data.get("best_val_acc", 0)], s=70, facecolors="none",
                            edgecolors=color, linewidths=1.6, zorder=5)

    titles = ["Val accuracy", "Train loss", "Val loss"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("accuracy")
    axes[1].set_ylabel("cross entropy")
    axes[2].set_ylabel("cross entropy")
    axes[1].axhline(0.42, ls=":", color="gray", lw=1.2)
    axes[1].text(0.5, 0.435, "label smoothing floor ~0.42", fontsize=7, color="gray")
    fig.suptitle("Classification training curves (NEU steel surface defects, seed 2026)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = ROOT / "results" / "figures" / "cls_train_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"训练曲线已保存：{rel_to_root(out)}")


if __name__ == "__main__":
    main()
