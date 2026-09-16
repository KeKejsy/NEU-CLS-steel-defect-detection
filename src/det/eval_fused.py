"""成员 C · 评估 4.0：滑窗候选 + 验证器与精修器概率融合

用法：
    python src/det/eval_fused.py --split val                    # 迭代对比
    python src/det/eval_fused.py --split test                   # 最终评估（仅一次）
    python src/det/eval_fused.py --split val --mode refiner      # 消融：只用精修器
    python src/det/eval_fused.py --split val --mode verifier     # 消融：只用验证器

## 为什么融合有效

两个模型是独立训练的，看同一块图块时误差不同：

| 模型 | 图块分类准确率 | 在 GT 处窗口的判类准确率 |
|---|---|---|
| 验证器 | 88.72% | 95.92% |
| 精修器 | 78.07% | 96.43% |
| **两者平均** | — | **96.94%** |

验证器的优势是分类头训得更充分（背景样本更干净），
精修器的优势是看过真实候选分布（带抖动的模拟候选）。
两者平均后既保留了精修器更好的框，又借到了验证器更稳的判类 —— 实测 mAP 提升近一倍。

## 关键超参（实测最优点）

- **滑窗步长 12**：比 16 更密，候选 IoU>=0.5 的上限更高
- **NMS IoU 0.3**：比常规 0.5 更严 —— 本方案每图产出几十个框而真值只有 2~3 个，
  重叠框互相压制能显著提精度。实测 0.3 明显优于 0.45/0.6
- 分数阈值影响很小（0.5~0.95 几乎无差别），因为融合后的缺陷概率分布很集中
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import paddle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import data as data_mod          # noqa: E402
from core import utils                     # noqa: E402
from core.boxes import batched_nms, clip_boxes_norm  # noqa: E402
from core.refiner import RegionRefiner, RefinerInfer  # noqa: E402
from core.verifier import BG_CLASS, DefectVerifier, VerifierInfer  # noqa: E402
from core.window_detector import make_windows  # noqa: E402
from eval_det import compute_prf, match_stats, plot_confusion, plot_per_class_ap, plot_pr_curves  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
CN = data_mod.CLASS_NAMES


def get_gts(ds):
    out = []
    for _p, ann_p in ds.samples:
        gt = data_mod.parse_voc_xml(ann_p, CN)
        gtb = np.stack([np.array([x[1] - x[3] / 2, x[2] - x[4] / 2,
                                  x[1] + x[3] / 2, x[2] + x[4] / 2]) for x in gt]) \
            if len(gt) else np.zeros((0, 4), dtype="float32")
        out.append({"boxes": gtb,
                    "labels": gt[:, 0].astype("int64") if len(gt) else np.zeros(0, dtype="int64")})
    return out


def combine(pr, pv, mode):
    """按模式融合两个模型的类别概率与缺陷概率

    - fuse    : 类别概率与缺陷概率都取平均（本项目最优）
    - refiner : 只用精修器（消融基线）
    - verifier: 只用验证器（消融基线）
    """
    dpr = 1.0 - pr[:, BG_CLASS]
    dpv = 1.0 - pv[:, BG_CLASS]
    if mode == "refiner":
        return pr[:, :len(CN)], dpr
    if mode == "verifier":
        return pv[:, :len(CN)], dpv
    # 几何平均比算术平均更稳：任一模型给出低分时不会被另一个强行拉高，
    # 对「两个模型都同意才是真缺陷」这一直觉更贴合。
    cls_p = np.sqrt(np.clip(pr[:, :len(CN)], 1e-9, None) *
                    np.clip(pv[:, :len(CN)], 1e-9, None))
    def_p = np.sqrt(np.clip(dpr, 1e-9, None) * np.clip(dpv, 1e-9, None))
    return cls_p, def_p


def main():
    ap = argparse.ArgumentParser(description="C · 滑窗 + 验证器/精修器融合评估")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--mode", default="fuse", choices=["fuse", "refiner", "verifier"])
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--score-th", type=float, default=0.6)
    ap.add_argument("--nms-iou", type=float, default=0.4)
    ap.add_argument("--max-det", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=150,
                    help="送进 NMS 前保留的最高分框数（与 max_det 不同：先截 top_k "
                         "可避免低分框参与 NMS 把高分框误压掉）")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只评估前 N 张（调试用）")
    ap.add_argument("--refiner-weights", default="results/weights/refiner_best.pdparams",
                    help="精修器权重路径（便于 A/B 对比不同版本）")
    ap.add_argument("--verifier-weights", default="results/weights/verifier_best.pdparams",
                    help="验证器权重路径（便于 A/B 对比不同版本）")
    ap.add_argument("--norm", default="none", choices=["none", "imagenet"],
                    help="输入归一化，必须与被评估权重训练时一致："
                         "历史权重用 none；ImageNet 预训练主干用 imagenet")
    ap.add_argument("--in-size", type=int, default=96,
                    help="判别模型输入尺寸，必须与训练时一致")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(2026)

    rw = Path(args.refiner_weights)
    if not rw.is_absolute():
        rw = utils.ROOT / rw
    if not rw.exists():
        raise SystemExit(f"找不到精修器权重：{rw}")

    refiner = RegionRefiner(num_classes=len(CN) + 1, in_size=args.in_size)
    refiner.set_state_dict(paddle.load(str(rw)))
    refiner.eval()
    rinf = RefinerInfer(refiner, in_size=args.in_size, margin=0.15, batch=256, norm=args.norm)

    vw = Path(args.verifier_weights)
    if not vw.is_absolute():
        vw = utils.ROOT / vw
    if not vw.exists():
        raise SystemExit(f"找不到验证器权重：{vw}")
    verifier = DefectVerifier(num_classes=len(CN) + 1, in_size=args.in_size)
    verifier.set_state_dict(paddle.load(str(vw)))
    verifier.eval()
    vinf = VerifierInfer(verifier, in_size=args.in_size, batch=512, norm=args.norm)

    ds, _ = data_mod.build_loader(utils.ROOT, args.split, 416, 8,
                                  augment=False, shuffle=False)
    if args.limit > 0:
        ds.samples = ds.samples[:args.limit]
    gts = get_gts(ds)

    print("=" * 80)
    print(f"成员 C · 滑窗 + 融合评估    split={args.split}    {len(ds)} 张    模式={args.mode}")
    print("=" * 80)
    print(f"滑窗步长 {args.stride} | 分数阈值 {args.score_th} | NMS IoU {args.nms_iou} | "
          f"最多 {args.max_det} 框/图")
    print(f"权重：验证器 {vw.name} / 精修器 {rw.name} | 输入归一化 {args.norm}")

    preds = []
    t0 = time.time()
    n_cand = 0
    for k in range(len(ds)):
        p, _ = ds.samples[k]
        im = Image.open(p).convert("RGB")
        W, H = im.size
        wins = make_windows(W, H, stride=args.stride)
        if len(wins) == 0:
            preds.append({"boxes": np.zeros((0, 4), dtype="float32"),
                          "scores": np.zeros(0, dtype="float32"),
                          "labels": np.zeros(0, dtype="int64")})
            continue
        pr, ref = rinf.run_norm(im, wins)      # 精修器：类别 + 精修框
        pv = vinf.score(im, wins)              # 验证器：类别
        cls_p, def_p = combine(pr, pv, args.mode)
        labels = cls_p.argmax(-1)

        keep = def_p >= args.score_th
        n_cand += int(keep.sum())
        if not keep.any():
            preds.append({"boxes": np.zeros((0, 4), dtype="float32"),
                          "scores": np.zeros(0, dtype="float32"),
                          "labels": np.zeros(0, dtype="int64")})
            continue
        # 先按分数截 top_k，再进 NMS
        kb, ks, kl = ref[keep], def_p[keep], labels[keep]
        if len(ks) > args.top_k:
            o = np.argsort(-ks)[:args.top_k]
            kb, ks, kl = kb[o], ks[o], kl[o]
        tb = paddle.to_tensor(kb)
        ts = paddle.to_tensor(ks).astype("float32")
        tl = paddle.to_tensor(kl).astype("int64")
        good = (tb[:, 2] > tb[:, 0]) & (tb[:, 3] > tb[:, 1])
        gi = paddle.nonzero(good).reshape([-1])
        tb, ts, tl = tb[gi], ts[gi], tl[gi]
        if tb.shape[0] == 0:
            preds.append({"boxes": np.zeros((0, 4), dtype="float32"),
                          "scores": np.zeros(0, dtype="float32"),
                          "labels": np.zeros(0, dtype="int64")})
            continue
        idx = batched_nms(tb, ts, tl, len(CN), iou_threshold=args.nms_iou,
                          top_k=min(args.max_det, int(tb.shape[0])))
        preds.append({"boxes": clip_boxes_norm(tb[idx]).numpy(),
                      "scores": ts[idx].numpy(), "labels": tl[idx].numpy()})
    cost = time.time() - t0

    res = utils.evaluate_map(preds, gts, len(CN), CN, iou_threshold=0.5)
    tp, fp, fn, conf = match_stats(preds, gts, len(CN), iou_threshold=0.5)

    print(f"\n耗时 {cost:.0f}s（{cost/len(ds)*1000:.0f} ms/图）  "
          f"候选 {n_cand/len(ds):.0f}/图 -> 输出 "
          f"{sum(len(p['labels']) for p in preds)/len(ds):.1f}/图")
    print("\n" + "-" * 84)
    print(f"{'类别':<6}{'中文':<14}{'AP@0.5':>9}{'精确率':>9}{'召回率':>9}{'F1':>8}"
          f"{'TP':>6}{'FP':>6}{'FN':>6}")
    print("-" * 84)
    detail = {}
    for i, c in enumerate(CN):
        pp, rr, f1 = compute_prf(tp[i], fp[i], fn[i])
        ap_v = res["per_class"].get(c, 0.0)
        detail[c] = {"ap50": ap_v, "precision": pp, "recall": rr, "f1": f1,
                     "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i]),
                     "cn_name": data_mod.CLASS_NAMES_CN.get(c, "")}
        print(f"{c:<6}{data_mod.CLASS_NAMES_CN.get(c,''):<14}{ap_v:>9.4f}{pp:>9.4f}"
              f"{rr:>9.4f}{f1:>8.4f}{int(tp[i]):>6}{int(fp[i]):>6}{int(fn[i]):>6}")
    print("-" * 84)
    mp, mr, mf = compute_prf(tp.sum(), fp.sum(), fn.sum())
    print(f"{'总体':<20}{res['map']:>9.4f}{mp:>9.4f}{mr:>9.4f}{mf:>8.4f}"
          f"{int(tp.sum()):>6}{int(fp.sum()):>6}{int(fn.sum()):>6}")
    print(f"\n>>> mAP@0.5 = {res['map']:.4f}")

    tag = args.tag or f"fused_{args.mode}_{args.split}"
    fig_dir = utils.resolve_dir("results/figures")
    metric_dir = utils.resolve_dir("results/metrics")
    plot_pr_curves(res["pr_curves"], CN, fig_dir / f"{tag}_pr.png",
                   f"滑窗+融合({args.mode}) PR 曲线  mAP@0.5={res['map']:.4f}  [{args.split}]")
    plot_per_class_ap(res["per_class"], CN, fig_dir / f"{tag}_ap.png",
                      f"滑窗+融合({args.mode}) 每类 AP@0.5  [{args.split}]", res["map"])
    plot_confusion(conf, CN, fig_dir / f"{tag}_conf.png",
                   f"滑窗+融合({args.mode}) 混淆矩阵  [{args.split}]")

    report = {
        "pipeline": "window+verifier+refiner", "mode": args.mode, "split": args.split,
        "norm": args.norm,
        "weights": {"verifier": vw.name, "refiner": rw.name},
        "params": {"stride": args.stride, "score_th": args.score_th,
                   "nms_iou": args.nms_iou, "max_det": args.max_det},
        "num_images": len(ds), "ms_per_image": round(cost / len(ds) * 1000, 1),
        "candidates_per_image": round(n_cand / len(ds), 1),
        "outputs_per_image": round(sum(len(p["labels"]) for p in preds) / len(ds), 1),
        "map50": res["map"], "per_class": detail,
        "overall": {"precision": mp, "recall": mr, "f1": mf,
                    "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum())},
        "confusion_matrix": {"labels": list(CN) + ["background"], "matrix": conf.tolist()},
        "pr_curves": {c: {"recall": res["pr_curves"][c]["recall"],
                          "precision": res["pr_curves"][c]["precision"],
                          "ap": res["pr_curves"][c]["ap"]} for c in CN},
    }
    out = utils.dump_json(report, metric_dir / f"{tag}.json")
    print(f"\n报告：{out.relative_to(utils.ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
