"""成员 C · 确认实验：新增窄宽高比（0.35）对各类缺陷的增益

用法：
    python src/det/tools/ab_aspects.py --per-class 20

## 背景

用 `tools/analyze_result.py` 分析混淆矩阵发现：模型**几乎不把缺陷错分成别的类别**
（混淆矩阵的错分列全为 0），误差只有误检与漏检两种；而漏检集中在 In（召回 0.49）
与 Sc（0.51）两类，其余四类 0.71~0.80。

查目标框尺寸分布后原因明确：这两类是**细长形状**（宽高比 0.37 / 0.18），
而原滑窗最窄只到 0.6，**产生不出这种形状的窗口**，候选里根本没有贴合的框。

本脚本在**同一批分层抽样验证图**上做 A/B 对比，确认加入 0.35 是否真有增益。

> 注意：本项目曾因「用 ds.samples[:N] 取子集」而出过错结论
> （划分按类别前缀排序，前 80 张只有 Cr/In 两类）。本脚本沿用
> `sweep_refine.stratified_sample` 做每类等量抽样。
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
from tools.sweep_refine import stratified_sample  # noqa: E402

CN = data_mod.CLASS_NAMES
OLD_ASPECTS = (0.6, 1.0, 1.7)
NEW_ASPECTS = (0.35, 0.6, 1.0, 1.7)


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


def predict(rinf, vinf, ds, scales, aspects, stride, th, nms_iou, max_det, top_k):
    preds = []
    n_cand = []
    for k in range(len(ds)):
        p, _ = ds.samples[k]
        im = Image.open(p).convert("RGB")
        W, H = im.size
        wins = make_windows(W, H, scales=scales, stride=stride, aspects=aspects)
        pr, ref = rinf.run_norm(im, wins)
        pv = vinf.score(im, wins)
        cls_p = np.sqrt(np.clip(pr[:, :len(CN)], 1e-9, None) *
                        np.clip(pv[:, :len(CN)], 1e-9, None))
        def_p = np.sqrt(np.clip(1.0 - pr[:, BG_CLASS], 1e-9, None) *
                        np.clip(1.0 - pv[:, BG_CLASS], 1e-9, None))
        labels = cls_p.argmax(-1)
        n_cand.append(len(wins))

        keep = def_p >= th
        if not keep.any():
            preds.append({"boxes": np.zeros((0, 4), dtype="float32"),
                          "scores": np.zeros(0, dtype="float32"),
                          "labels": np.zeros(0, dtype="int64")})
            continue
        b, s, l = ref[keep], def_p[keep], labels[keep]
        if len(s) > top_k:
            o = np.argsort(-s)[:top_k]
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
        idx = batched_nms(tb, ts, tl, len(CN), iou_threshold=nms_iou,
                          top_k=min(max_det, int(tb.shape[0])))
        preds.append({"boxes": clip_boxes_norm(tb[idx]).numpy(),
                      "scores": ts[idx].numpy(), "labels": tl[idx].numpy()})
    return preds, n_cand


def main():
    ap = argparse.ArgumentParser(description="窄宽高比 A/B 确认实验")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--scales", default="0.2,0.3,0.45,0.65")
    ap.add_argument("--score-th", type=float, default=0.6)
    ap.add_argument("--nms-iou", type=float, default=0.4)
    ap.add_argument("--max-det", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=150)
    ap.add_argument("--out", default="results/metrics/ab_aspects.json")
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
    scales = tuple(float(s) for s in args.scales.split(","))
    print("=" * 92)
    print(f"C · 窄宽高比 A/B 确认    分层抽样 {args.per_class}/类 = {len(ds)} 张 / "
          f"{sum(len(g['labels']) for g in gts)} GT")
    print(f"尺度 {scales}  步长 {args.stride}  阈值 {args.score_th}  NMS {args.nms_iou}")
    print("=" * 92)

    refiner = RegionRefiner(num_classes=len(CN) + 1, in_size=96)
    refiner.set_state_dict(paddle.load(str(utils.ROOT / "results/weights/refiner_best.pdparams")))
    refiner.eval()
    rinf = RefinerInfer(refiner, in_size=96, margin=0.15, batch=256)
    ver = DefectVerifier(num_classes=len(CN) + 1, in_size=96)
    ver.set_state_dict(paddle.load(str(utils.ROOT / "results/weights/verifier_best.pdparams")))
    ver.eval()
    vinf = VerifierInfer(ver, batch=512)

    results = {}
    print(f"\n{'配置':<22}{'候选/图':>9}{'mAP@0.5':>10}"
          + "".join(f"{c:>7}" for c in CN) + f"{'召回':>8}{'精度':>8}")
    print("-" * 92)
    for label, aspects in (("A 原比例 (0.6/1.0/1.7)", OLD_ASPECTS),
                           ("B 加窄比例 (0.35/0.6/1.0/1.7)", NEW_ASPECTS)):
        preds, n_cand = predict(rinf, vinf, ds, scales, aspects, args.stride,
                                args.score_th, args.nms_iou, args.max_det, args.top_k)
        r = utils.evaluate_map(preds, gts, len(CN), CN, iou_threshold=0.5)
        pc = r["per_class"]
        n_gt = sum(len(g["labels"]) for g in gts)
        n_pd = sum(len(p["labels"]) for p in preds)
        # 粗略的整体精度/召回（按各类 TP/FP/FN 汇总，用 evaluate_map 的匹配结果近似）
        tp = sum(min(len(p["labels"]), len(g["labels"])) for p, g in zip(preds, gts))
        results[label] = {"map50": r["map"], "per_class": pc,
                          "candidates_per_image": float(np.mean(n_cand)),
                          "outputs_per_image": n_pd / len(ds)}
        print(f"{label:<22}{np.mean(n_cand):>9.0f}{r['map']:>10.4f}"
              + "".join(f"{pc[c]:>7.3f}" for c in CN)
              + f"{n_pd/max(n_gt,1):>8.2f}{n_pd/len(ds):>8.1f}")

    print("-" * 92)
    a = results["A 原比例 (0.6/1.0/1.7)"]
    b = results["B 加窄比例 (0.35/0.6/1.0/1.7)"]
    print(f"\n增益: 整体 mAP {a['map50']:.4f} -> {b['map50']:.4f} "
          f"({(b['map50']-a['map50'])/max(a['map50'],1e-9)*100:+.1f}%)")
    for c in CN:
        d = b["per_class"][c] - a["per_class"][c]
        flag = "  <== 明显改善" if d > 0.01 else ("  (下降)" if d < -0.01 else "")
        print(f"  {c:<4} {a['per_class'][c]:.3f} -> {b['per_class'][c]:.3f}  {d:+.3f}{flag}")

    utils.dump_json({"split": args.split, "per_class_sample": args.per_class,
                     "num_images": len(ds), "num_gt": sum(len(g['labels']) for g in gts),
                     "scales": list(scales), "stride": args.stride,
                     "results": results},
                    utils.ROOT / args.out)
    print("=" * 92)


if __name__ == "__main__":
    main()
