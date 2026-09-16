"""成员 C · A/B 对照：预训练主干 vs 随机初始化主干

背景
----
`README.md` 的「已知问题与后续改进」建议给检测任务换**有预训练权重的主干**。
本项目原实现里验证器/精修器的主干虽然是 `paddle.vision.models.mobilenet_v3_small`，
但一直用 `pretrained=False`（随机初始化），输入也只做 /255。

本脚本把两轮 `eval_fused.py` 的报告放在一起对照（同一划分、同一套滑窗与后处理），
用于判断「主干换 ImageNet 预训练 + 输入改 ImageNet 归一化」这一改动是否真的有效。

用法
----
    python src/det/tools/compare_norm_ab.py \
        --base results/metrics/val_fuse_final.json \
        --new  results/metrics/val_fuse_pre.json

两个报告来自：
    # 基线（历史权重，随机初始化主干）
    python src/det/eval_fused.py --split val --mode fuse --stride 12 --score-th 0.6 \
        --nms-iou 0.4 --top-k 150 --max-det 50 \
        --verifier-weights results/weights/verifier_best.pdparams \
        --refiner-weights  results/weights/refiner_best.pdparams \
        --norm none --tag val_fuse_final

    # 实验组（ImageNet 预训练主干）
    python src/det/eval_fused.py --split val --mode fuse --stride 12 --score-th 0.6 \
        --nms-iou 0.4 --top-k 150 --max-det 50 \
        --verifier-weights results/weights/verifier_pre_best.pdparams \
        --refiner-weights  results/weights/refiner_pre_best.pdparams \
        --norm imagenet --tag val_fuse_pre
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import utils  # noqa: E402
from core.data import CLASS_NAMES  # noqa: E402


def load(path):
    p = Path(path)
    if not p.is_absolute():
        p = utils.ROOT / p
    if not p.exists():
        raise SystemExit(f"找不到报告：{p}")
    return json.loads(p.read_text(encoding="utf-8")), p


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="C · 预训练主干 A/B 对照")
    ap.add_argument("--base", default="results/metrics/val_fuse_final.json",
                    help="基线报告（随机初始化主干）")
    ap.add_argument("--new", default="results/metrics/val_fuse_pre.json",
                    help="实验组报告（ImageNet 预训练主干）")
    args = ap.parse_args()

    b, bp = load(args.base)
    n, np_ = load(args.new)

    print("=" * 78)
    print("成员 C · A/B 对照：主干「随机初始化」 vs 「ImageNet 预训练」")
    print("=" * 78)
    print(f"  基线   {bp.relative_to(utils.ROOT)}")
    print(f"        归一化 {b.get('norm', 'none')} | 权重 {b.get('weights', 'verifier/refiner_best')}")
    print(f"  实验组 {np_.relative_to(utils.ROOT)}")
    print(f"        归一化 {n.get('norm', '?')} | 权重 {n.get('weights', '?')}")
    if b.get("split") != n.get("split"):
        print(f"\n⚠️ 两份报告的划分不同（{b.get('split')} vs {n.get('split')}），不可直接比较！")
    print(f"\n划分 {n.get('split')} | 图片 {n.get('num_images')} 张 | "
          f"滑窗步长 {n.get('params', {}).get('stride')} | "
          f"分数阈值 {n.get('params', {}).get('score_th')}")

    def delta(a, c):
        if a is None or c is None:
            return "—"
        d = c - a
        sign = "↑" if d > 0 else ("↓" if d < 0 else "=")
        return f"{d:+.4f} {sign}"

    print("\n【总指标】")
    print(f"  {'指标':<14}{'基线':>12}{'实验组':>12}{'变化':>16}")
    rows = [("mAP@0.5", b.get("map50"), n.get("map50"))]
    for k, label in (("precision", "精确率"), ("recall", "召回率")):
        rows.append((label, b.get("overall", {}).get(k), n.get("overall", {}).get(k)))
    for k, label in (("outputs_per_image", "每图输出框数"), ("candidates_per_image", "每图候选数"),
                     ("ms_per_image", "单图耗时(ms)")):
        rows.append((label, b.get(k), n.get(k)))
    for label, x, y in rows:
        print(f"  {label:<14}{fmt(x):>12}{fmt(y):>12}{delta(x, y):>16}")

    print("\n【每类 AP@0.5】")
    print(f"  {'类别':<8}{'基线':>10}{'实验组':>10}{'变化':>16}")

    def ap_of(rep, c):
        d = rep.get("per_class", {}).get(c) or {}
        return d.get("ap50", d.get("ap"))

    for c in CLASS_NAMES:
        x, y = ap_of(b, c), ap_of(n, c)
        print(f"  {c:<8}{fmt(x):>10}{fmt(y):>10}{delta(x, y):>16}")

    xm, ym = b.get("map50"), n.get("map50")
    if xm is not None and ym is not None:
        rel = (ym - xm) / max(xm, 1e-9) * 100
        verdict = "有效" if ym > xm else ("持平" if abs(ym - xm) < 1e-4 else "无效/更差")
        print(f"\n结论：mAP@0.5 {xm:.4f} -> {ym:.4f}（{ym - xm:+.4f}，相对 {rel:+.1f}%）=> {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
