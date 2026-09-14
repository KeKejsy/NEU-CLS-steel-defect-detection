"""成员 C · 滑窗形状专项调优（针对细长缺陷 In / Sc）

用法：
    python src/det/tools/tune_windows.py --per-class 12

## 为什么需要这个脚本

从混淆矩阵分析（tools/analyze_result.py）发现：模型**几乎不把缺陷判成别的类别**
（错分列全为 0），所有误差只有两类 —— 误检与漏检（IoU 不够）。
而漏检集中在 In、Sc 两类（召回 0.49 / 0.51，其余四类 0.71~0.80）。

查目标框尺寸分布后原因清楚了：

| 类别 | 宽中位 | 高中位 | 宽高比 |
|---|---|---|---|
| In 夹杂 | 0.125 | 0.325 | **0.37** |
| Sc 划痕 | 0.133 | 0.653 | **0.18** |
| Cr 龟裂 | 0.630 | 0.350 | 1.79 |
| PS 麻点 | 0.625 | 0.955 | 0.76 |

原滑窗宽高比只有 (0.6, 1.0, 1.7)，**产生不出 In/Sc 那种细长形状的窗口**，
于是这两类的框永远对不准 —— 不是模型不会判，而是候选里根本没有合适的框。

本脚本扫描「宽高比 × 尺度」组合，找出能同时兼顾细长类与其他类的滑窗配置。
"""

import argparse
import itertools
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


def make_preds(cache, th, nms_iou, max_det, top_k):
    preds = []
    for ref, def_p, labels in cache:
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
    return preds


# 候选配置：从「当前」逐步加入更窄的宽高比与更大的尺度
CONFIGS = [
    ("当前 (0.6,1.0,1.7) 尺度0.2~0.65",
     (0.2, 0.3, 0.45, 0.65), (0.6, 1.0, 1.7)),
    ("加入窄比例 0.35",
     (0.2, 0.3, 0.45, 0.65), (0.35, 0.6, 1.0, 1.7)),
    ("加入极窄 0.2 + 窄 0.35",
     (0.2, 0.3, 0.45, 0.65), (0.2, 0.35, 0.6, 1.0, 1.7)),
    ("极窄 + 大尺度 0.9（覆盖 Sc 高度）",
     (0.2, 0.35, 0.5, 0.7, 0.9), (0.2, 0.35, 0.6, 1.0, 1.7)),
    ("极窄 + 大尺度 + 更细尺度",
     (0.15, 0.25, 0.35, 0.5, 0.7, 0.9), (0.2, 0.35, 0.6, 1.0)),
]


def main():
    ap = argparse.ArgumentParser(description="滑窗形状专项调优")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--stride", type=int, default=12)
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
    print("=" * 92)
    print(f"C · 滑窗形状专项调优    分层抽样 {args.per_class}/类 = {len(ds)} 张 / "
          f"{sum(len(g['labels']) for g in gts)} GT")
    print("=" * 92)

    refiner = RegionRefiner(num_classes=len(CN) + 1, in_size=96)
    refiner.set_state_dict(paddle.load(str(utils.ROOT / "results/weights/refiner_best.pdparams")))
    refiner.eval()
    rinf = RefinerInfer(refiner, in_size=96, margin=0.15, batch=256)
    ver = DefectVerifier(num_classes=len(CN) + 1, in_size=96)
    ver.set_state_dict(paddle.load(str(utils.ROOT / "results/weights/verifier_best.pdparams")))
    ver.eval()
    vinf = VerifierInfer(ver, batch=512)

    print(f"\n{'配置':<32}{'候选/图':>9}{'mAP@0.5':>10}{'Cr':>7}{'In':>7}"
          f"{'Pa':>7}{'PS':>7}{'RS':>7}{'Sc':>7}")
    print("-" * 92)
    best = (0, None)
    for label, scales, aspects in CONFIGS:
        cache = []
        for k in range(len(ds)):
            p, _ = ds.samples[k]
            im = Image.open(p).convert("RGB")
            W, H = im.size
            wins = make_windows(W, H, scales=scales, stride=args.stride, aspects=aspects)
            pr, ref = rinf.run_norm(im, wins)
            pv = vinf.score(im, wins)
            cls_p = np.sqrt(np.clip(pr[:, :len(CN)], 1e-9, None) *
                            np.clip(pv[:, :len(CN)], 1e-9, None))
            def_p = np.sqrt(np.clip(1.0 - pr[:, BG_CLASS], 1e-9, None) *
                            np.clip(1.0 - pv[:, BG_CLASS], 1e-9, None))
            cache.append((ref, def_p, cls_p.argmax(-1)))
        preds = make_preds(cache, 0.6, 0.4, 50, 150)
        r = utils.evaluate_map(preds, gts, len(CN), CN, iou_threshold=0.5)
        pc = r["per_class"]
        npb = np.mean([len(c[1]) for c in cache])
        if r["map"] > best[0]:
            best = (r["map"], (label, scales, aspects))
        print(f"{label:<32}{npb:>9.0f}{r['map']:>10.4f}"
              + "".join(f"{pc[c]:>7.3f}" for c in CN))

    print("-" * 92)
    print(f"\n最优: mAP@0.5 = {best[0]:.4f}   {best[1][0]}")
    print(f"  scales  = {best[1][1]}")
    print(f"  aspects = {best[1][2]}")
    print("=" * 92)


if __name__ == "__main__":
    main()
