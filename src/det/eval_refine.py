"""成员 C · 两阶段检测评估 3.0：滑窗候选 + 精修器（判类 + 框回归）

用法：
    python src/det/eval_refine.py --split val                 # 迭代对比（默认验证集）
    python src/det/eval_refine.py --split test                # 最终评估（仅一次）
    python src/det/eval_refine.py --no-refine                 # 消融：关掉框回归

与 eval_2stage.py 的区别：那一版只用验证器判类，候选框原样输出；
本版会先用精修器把候选框对齐到真实缺陷，再做 NMS —— 直接针对
「正确框被淹没」这一瓶颈（GT 最佳候选排名中位数 194）。
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
from core.verifier import BG_CLASS         # noqa: E402
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


def main():
    ap = argparse.ArgumentParser(description="C · 滑窗+精修器 评估")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--score-th", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=50)
    ap.add_argument("--top-windows", type=int, default=800,
                    help="送入精修器的候选上限（越大越慢、越全）")
    ap.add_argument("--no-refine", action="store_true", help="消融：不做框回归")
    ap.add_argument("--tag", default=None, help="输出文件名后缀")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(2026)

    model = RegionRefiner(num_classes=len(CN) + 1, in_size=96)
    wp = utils.ROOT / "results/weights/refiner_best.pdparams"
    if not wp.exists():
        raise SystemExit("找不到精修器权重，请先运行 python src/det/train_refiner.py")
    model.set_state_dict(paddle.load(str(wp)))
    model.eval()
    infer = RefinerInfer(model, in_size=96, margin=0.15, batch=256)

    ds, _ = data_mod.build_loader(utils.ROOT, args.split, 416, 8,
                                  augment=False, shuffle=False)
    gts = get_gts(ds)

    print("=" * 78)
    print(f"成员 C · 滑窗 + 精修器评估    split={args.split}    {len(ds)} 张")
    print("=" * 78)
    print(f"滑窗步长 {args.stride} | 分数阈值 {args.score_th} | NMS IoU {args.nms_iou} | "
          f"框回归 {'关闭（消融）' if args.no_refine else '开启'}")

    preds = []
    t0 = time.time()
    n_before = n_after = 0
    for k in range(len(ds)):
        img_p, _ = ds.samples[k]
        im = Image.open(img_p).convert("RGB")
        W, H = im.size
        wins = make_windows(W, H, stride=args.stride)
        if len(wins) == 0:
            preds.append({"boxes": np.zeros((0, 4), dtype="float32"),
                          "scores": np.zeros(0, dtype="float32"),
                          "labels": np.zeros(0, dtype="int64")})
            continue

        prob, refined = infer.run_norm(im, wins)        # (N,7), (N,4) 精修后的归一化框
        defect_p = 1.0 - prob[:, BG_CLASS]
        labels = prob[:, :len(CN)].argmax(-1)
        n_before += int((defect_p >= args.score_th).sum())

        keep = defect_p >= args.score_th
        if not keep.any():
            preds.append({"boxes": np.zeros((0, 4), dtype="float32"),
                          "scores": np.zeros(0, dtype="float32"),
                          "labels": np.zeros(0, dtype="int64")})
            continue
        # 按分数截断控制规模
        if int(keep.sum()) > args.top_windows:
            idx = np.argsort(-defect_p)[:args.top_windows]
            keep = np.zeros(len(wins), dtype=bool)
            keep[idx] = True

        use_boxes = wins if args.no_refine else refined
        tb = paddle.to_tensor(use_boxes[keep])
        ts = paddle.to_tensor(defect_p[keep]).astype("float32")
        tl = paddle.to_tensor(labels[keep]).astype("int64")
        good = (tb[:, 2] > tb[:, 0]) & (tb[:, 3] > tb[:, 1])
        if not bool(good.all()):
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
        n_after += len(idx)
    cost = time.time() - t0

    res = utils.evaluate_map(preds, gts, len(CN), CN, iou_threshold=0.5)
    tp, fp, fn, conf = match_stats(preds, gts, len(CN), iou_threshold=0.5)

    print(f"\n耗时 {cost:.0f}s（{cost/len(ds)*1000:.0f} ms/图）  "
          f"候选 {n_before/len(ds):.0f}/图 -> 输出 {n_after/len(ds):.1f}/图")
    print("\n" + "-" * 84)
    print(f"{'类别':<6}{'中文':<14}{'AP@0.5':>9}{'精确率':>9}{'召回率':>9}{'F1':>8}"
          f"{'TP':>6}{'FP':>6}{'FN':>6}")
    print("-" * 84)
    detail = {}
    for i, c in enumerate(CN):
        p, r, f1 = compute_prf(tp[i], fp[i], fn[i])
        ap_v = res["per_class"].get(c, 0.0)
        detail[c] = {"ap50": ap_v, "precision": p, "recall": r, "f1": f1,
                     "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i]),
                     "cn_name": data_mod.CLASS_NAMES_CN.get(c, "")}
        print(f"{c:<6}{data_mod.CLASS_NAMES_CN.get(c,''):<14}{ap_v:>9.4f}{p:>9.4f}"
              f"{r:>9.4f}{f1:>8.4f}{int(tp[i]):>6}{int(fp[i]):>6}{int(fn[i]):>6}")
    print("-" * 84)
    mp, mr, mf = compute_prf(tp.sum(), fp.sum(), fn.sum())
    print(f"{'总体':<20}{res['map']:>9.4f}{mp:>9.4f}{mr:>9.4f}{mf:>8.4f}"
          f"{int(tp.sum()):>6}{int(fp.sum()):>6}{int(fn.sum()):>6}")
    print(f"\n>>> mAP@0.5 = {res['map']:.4f}")

    tag = args.tag or f"refine_{args.split}" + ("_noregress" if args.no_refine else "")
    fig_dir = utils.resolve_dir("results/figures")
    metric_dir = utils.resolve_dir("results/metrics")
    plot_pr_curves(res["pr_curves"], CN, fig_dir / f"refine_pr_{tag}.png",
                   f"滑窗+精修器 PR 曲线  mAP@0.5={res['map']:.4f}  [{args.split}]")
    plot_per_class_ap(res["per_class"], CN, fig_dir / f"refine_ap_{tag}.png",
                      f"滑窗+精修器 每类 AP@0.5  [{args.split}]", res["map"])
    plot_confusion(conf, CN, fig_dir / f"refine_conf_{tag}.png",
                   f"滑窗+精修器 混淆矩阵  [{args.split}]")

    report = {
        "pipeline": "window+refiner", "split": args.split,
        "refine_enabled": not args.no_refine,
        "params": {"stride": args.stride, "score_th": args.score_th,
                   "nms_iou": args.nms_iou, "max_det": args.max_det,
                   "top_windows": args.top_windows},
        "num_images": len(ds), "ms_per_image": round(cost / len(ds) * 1000, 1),
        "candidates_per_image": round(n_before / len(ds), 1),
        "outputs_per_image": round(n_after / len(ds), 1),
        "map50": res["map"], "per_class": detail,
        "overall": {"precision": mp, "recall": mr, "f1": mf,
                    "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum())},
        "confusion_matrix": {"labels": list(CN) + ["background"], "matrix": conf.tolist()},
        "pr_curves": {c: {"recall": res["pr_curves"][c]["recall"],
                          "precision": res["pr_curves"][c]["precision"],
                          "ap": res["pr_curves"][c]["ap"]} for c in CN},
    }
    out = utils.dump_json(report, metric_dir / f"refine_eval_{tag}.json")
    print(f"\n报告：{out.relative_to(utils.ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
