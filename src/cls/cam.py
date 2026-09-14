"""成员 B · 任务1：Grad-CAM 热力图（看模型到底盯住了哪一块）

用法：
    python src/cls/cam.py --model mobilenet_v3_small
    python src/cls/cam.py --model resnet50_vd --num_per_class 3 --split val

说明：
    - 默认在验证集上出图：既能看到真实缺陷位置，又不占用"测试集只用一次"的额度。
    - 每类取前 N 张，画成 6 行 × N 列的网格，标题标注真实类别 / 预测类别 / 置信度。
    - 原理：取全局池化层之前那层卷积特征图，对预测类别的分数求梯度，
      梯度当权重做加权求和再 ReLU，得到"模型认为哪里最重要"的热力图。

输出：results/figures/cls_cam_<模型>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import paddle

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CLASS_EN,
    CLASS_NAMES,
    MODEL_DESC,
    MODEL_NAMES,
    ROOT,
    load_hparams,
    load_image,
    load_trained_model,
    read_split,
    rel_to_root,
    set_seed,
    to_tensor,
)


def gradcam(model, x):
    """返回 (热力图 HxW、预测类别)。注意这里不能包在 no_grad 里，否则拿不到梯度。"""
    captured = {}

    def pre_hook(layer, inputs):
        captured["feat"] = inputs[0]

    # 全局池化层的输入就是最后一层卷积特征图：ResNet 是 layer4 输出，
    # MobileNetV3 是最后一个卷积块输出，两个网络都能用同一个钩子拿到。
    handle = model.avgpool.register_forward_pre_hook(pre_hook)
    logits = model(x)
    handle.remove()

    feat = captured["feat"]
    pred = int(paddle.argmax(logits, axis=1)[0])
    grad = paddle.grad(logits[0, pred], feat)[0]           # 对预测类别求梯度
    weights = grad.mean(axis=[2, 3], keepdim=True)          # 每个通道一个权重
    cam = paddle.nn.functional.relu((weights * feat).sum(axis=1)).squeeze(0)
    cam = cam / (paddle.max(cam) + 1e-8)                    # 归一化到 0~1
    h, w = x.shape[2], x.shape[3]
    cam = paddle.nn.functional.interpolate(cam.reshape([1, 1, cam.shape[0], cam.shape[1]]),
                                           size=[int(h), int(w)], mode="bilinear", align_corners=False)
    prob = float(paddle.nn.functional.softmax(logits, axis=1)[0, pred])
    return cam.squeeze().numpy(), pred, prob


def main() -> None:
    ap = argparse.ArgumentParser(description="Grad-CAM 热力图（B）")
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="默认验证集；测试集只用一次，出图建议留在验证集上")
    ap.add_argument("--num_per_class", type=int, default=3, help="每个类别取几张")
    args = ap.parse_args()

    cfg = load_hparams(args.model)
    set_seed(cfg["seed"])
    size = tuple(cfg["size"])

    print("=" * 66)
    print(f"Grad-CAM 热力图  |  {args.model}  |  数据：{args.split} 集，每类 {args.num_per_class} 张")
    print("=" * 66)
    print(f"网络：{MODEL_DESC[args.model]}")

    model, weights = load_trained_model(args.model, args.weights)
    print(f"权重：{rel_to_root(weights)}")

    # 按类别分组挑图（每类取前 N 张）
    items = read_split(args.split)
    picks = {c: [] for c in CLASS_NAMES}
    for path, label in items:
        name = CLASS_NAMES[label]
        if len(picks[name]) < args.num_per_class:
            picks[name].append(path)
        if all(len(v) >= args.num_per_class for v in picks.values()):
            break

    n = args.num_per_class
    fig, axes = plt.subplots(len(CLASS_NAMES), n, figsize=(2.1 * n, 2.15 * len(CLASS_NAMES)))
    axes = np.atleast_2d(axes)

    for row, cls in enumerate(CLASS_NAMES):
        for col in range(n):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(picks[cls]):
                continue
            path = picks[cls][col]
            arr = load_image(path, size)
            x = paddle.to_tensor(to_tensor(arr)[None, ...])
            cam, pred, prob = gradcam(model, x)

            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            ax.imshow(cam, cmap="jet", alpha=0.45)
            ok = "OK" if pred == CLASS_NAMES.index(cls) else "WRONG"
            color = "#1a7f37" if ok == "OK" else "#c0392b"
            ax.set_title(f"{path.name}\ntrue={CLASS_EN[cls]} pred={CLASS_EN[CLASS_NAMES[pred]]} {prob:.2f} {ok}",
                         fontsize=7, color=color)
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"Grad-CAM  ({args.model}, {args.split} set)\nred = where the model focuses", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path = ROOT / "results" / "figures" / f"cls_cam_{args.model}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\n热力图已保存：{rel_to_root(out_path)}")
    print("（图里用英文类别名，避免别的电脑缺中文字体时显示成方框；类别对照见 README）")


if __name__ == "__main__":
    main()
