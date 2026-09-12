"""成员 C · 精修器 A/B 对比（在同一推理分布上测量「精修增益」）

用法：
    python src/det/tools/compare_refiners.py --per-class 15

## 为什么需要这个脚本

精修器训练时用的「合成候选」抖动范围，与推理时真实滑窗的分布并不相同。
所以**训练日志里的「精修后 IoU」不能用来比较两个版本**（任务难度不同，
v1 合成 IoU 0.442、v2 0.330，但这不代表 v1 更好）。

真正该测的是：**在推理会输出的那批高分窗口上，精修能把 IoU 提升多少**。
本脚本就是测这个 —— 它加载不同版本的精修器，在完全相同的一批图上：
    1. 统计高分窗口（预测输出）与最近 GT 的 IoU 分布：精修前 vs 精修后；
    2. 报告 IoU>=0.5 的比例提升（这直接决定能救回多少漏检）；
    3. 顺带跑完整流程给出 mAP，作为最终裁决。

背景（tools 里实测得到）：高分窗口里只有 8.3% 与 GT 完全无重叠，
而 31.6% 落在 IoU 0.3~0.5 —— 说明大量误检其实是「差一点命中」的框，
精修若能把这批框推过 0.5，收益会比压掉它们更大。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import paddle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import data as data_mod  # noqa: E402
from core import utils  # noqa: E402
from core.boxes import batched_nms, clip_boxes_norm  # noqa: E402
from core.refiner import RefinerInfer, RegionRefiner  # noqa: E402
from core.verifier import BG_CLASS, DefectVerifier, VerifierInfer  # noqa: E402
from core.window_detector import make_windows  # noqa: E402
from tools.diagnose import iou_matrix  # noqa: E402
from tools.sweep_refine import stratified_sample  # noqa: E402

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


def best_iou_to_gt(boxes, gtb):
    if len(boxes) == 0 or len(gtb) == 0:
        return np.zeros(len(boxes))
    return iou_matrix(boxes, gtb).max(axis=1)


def main():
    ap = argparse.ArgumentParser(description="精修器 A/B 对比（推理分布上的精修增益）")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--score-th", type=float, default=0.6)
    ap.add_argument("--nms-iou", type=float, default=0.4)
    ap.add_argument("--max-det", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=150)
    ap.add_argument("--variants", default="refiner:v1,refiner_v2:v2(放宽抖动)",
                    help="逗号分隔的 name:label")
    ap.add_argument("--out", default="results/metrics/refiner_ab.json")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(2026)

    ds, _ = data_mod.build_loader(utils.ROOT, args.split, 416, 8,
                                  augment=False, shuffle=False)
    names = stratified_sample([p.stem for p, _ in ds.samples], args.per_class)
    keep_idx = [i for i, (p, _) in enumerate(ds.samples) if p.stem in set(names)]
    ds.samples = [ds.samples[i] for i in keep_idx]
    gts = get_gts(ds)

    print("=" * 90)
    print(f"C · 精修器 A/B 对比    分层抽样 {args.per_class}/类 = {len(ds)} 张 / "
          f"{sum(len(g['labels']) for g in gts)} GT")
    print("=" * 90)

    ver = DefectVerifier(num_classes=len(CN) + 1, in_size=96)
    ver.set_state_dict(paddle.load(str(utils.ROOT / "results/weights/verifier_best.pdparams")))
    ver.eval()
    vinf = VerifierInfer(ver, batch=512)

    # 预缓存窗口与验证器输出（两个变体共用，保证可比）
    cache = []
    for k in range(len(ds)):
        p, _ = ds.samples[k]
        im = Image.open(p).convert("RGB")
        W, H = im.size
        wins = make_windows(W, H, stride=args.stride)
        pv = vinf.score(im, wins)
        cache.append((ds.samples[k][0], wins, pv[:, BG_CLASS], pv[:, :len(CN)]))
    print(f"  已缓存 {len(cache)} 张图的窗口与验证器输出")

    results = {}
    for spec in args.variants.split(","):
        name, label = spec.split(":")
        wp = utils.ROOT / f"results/weights/{name}_best.pdparams"
        if not wp.exists():
            print(f"\n[跳过] {label}: 找不到 {wp}")
            continue
        model = RegionRefiner(num_classes=len(CN) + 1, in_size=96)
        model.set_state_dict(paddle.load(str(wp)))
        model.eval()
        rinf = RefinerInfer(model, in_size=96, margin=0.15, batch=256)

        iou_before, iou_after = [], []
        preds = []
        for k in range(len(cache)):
            img_p, wins, bg_p, cls_v = cache[k]
            im = Image.open(img_p).convert("RGB")
            pr, ref = rinf.run_norm(im, wins)
            def_p = np.sqrt(np.clip(1.0 - pr[:, BG_CLASS], 1e-9, None) *
                            np.clip(1.0 - bg_p, 1e-9, None))
            cls_p = np.sqrt(np.clip(pr[:, :len(CN)], 1e-9, None) *
                            np.clip(cls_v, 1e-9, None))
            labels = cls_p.argmax(-1)
            gtb = gts[k]["boxes"]

            keep = def_p >= args.score_th
            if keep.any():
                # 精修前 / 后的 IoU 都在「会被输出的那批窗口」上统计
                iou_before.extend(best_iou_to_gt(wins[keep], gtb).tolist())
                iou_after.extend(best_iou_to_gt(ref[keep], gtb).tolist())

            if not keep.any():
                preds.append({"boxes": np.zeros((0, 4), dtype="float32"),
                              "scores": np.zeros(0, dtype="float32"),
                              "labels": np.zeros(0, dtype="int64")})
                continue
            b, s, l = ref[keep], def_p[keep], labels[keep]
            if len(s) > args.top_k:
                o = np.argsort(-s)[:args.top_k]
                b, s, l = b[o], s[o], l[o]
            tb = paddle.to_tensor(b)
            ts = paddle.to_tensor(s).astype("float32")
            tl = paddle.to_tensor(l).astype("int64")
            gi = paddle.nonzero((tb[:, 2] > tb[:, 0]) & (tb[:, 3] > tb[:, 1])).reshape([-1])
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

        ib = np.asarray(iou_before)
        ia = np.asarray(iou_after)
        r = utils.evaluate_map(preds, gts, len(CN), CN, iou_threshold=0.5)
        results[label] = {
            "map50": r["map"], "per_class": r["per_class"],
            "iou_before_mean": float(ib.mean()), "iou_after_mean": float(ia.mean()),
            "hit_before_pct": float((ib >= 0.5).mean() * 100),
            "hit_after_pct": float((ia >= 0.5).mean() * 100),
            "n_windows": int(len(ib)),
        }
        print(f"\n[{label}]  权重 {wp.name}")
        print(f"  高分窗口数 {len(ib)}")
        print(f"  精修前 IoU 均值 {ib.mean():.4f}   IoU>=0.5 占比 {(ib>=0.5).mean()*100:.1f}%")
        print(f"  精修后 IoU 均值 {ia.mean():.4f}   IoU>=0.5 占比 {(ia>=0.5).mean()*100:.1f}%")
        print(f"  >>> 精修增益: IoU {ia.mean()-ib.mean():+.4f}   "
              f"命中率 {(ia>=0.5).mean()*100-(ib>=0.5).mean()*100:+.1f} 个百分点")
        print(f"  >>> 最终 mAP@0.5 = {r['map']:.4f}")
        print("  每类 AP: " + "  ".join(f"{c}={r['per_class'][c]:.3f}" for c in CN))

    if len(results) >= 2:
        ks = list(results.keys())
        print("\n" + "=" * 90)
        print("裁决（在同一批图、同一套窗口、同一后处理下比较）：")
        best = max(ks, key=lambda k: results[k]["map50"])
        for k in ks:
            d = results[k]
            print(f"  {k:<16} mAP={d['map50']:.4f}  精修增益 IoU {d['iou_after_mean']-d['iou_before_mean']:+.4f}  "
                  f"命中率 {d['hit_after_pct']-d['hit_before_pct']:+.1f}pp")
        print(f"  => 更优: {best}")

    utils.dump_json({"split": args.split, "per_class_sample": args.per_class,
                     "num_images": len(ds), "params": vars(args), "results": results},
                    utils.ROOT / args.out)
    print("=" * 90)


if __name__ == "__main__":
    main()
