"""成员 C · 融合方案的后处理精调（缓存候选，快速扫描）

用法：
    python src/det/tools/tune_fused.py --split val --limit 120

融合方案已把 mAP 从 0.071 提到 0.157，但精度仍然低（约 0.032），
原因是每图输出 44 个框而真值只有 2.4 个。本脚本缓存一次融合推理结果，
然后扫描后处理参数，找出最优组合。

为什么要缓存：融合推理约 2 秒/图，直接网格搜索会慢到不可用；
缓存后每次调参只需几毫秒。
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
    """按给定后处理参数生成预测

    top_k 是「送进 NMS 前保留多少个最高分框」。它与 max_det（NMS 后保留数）不同：
    先截 top_k 能避免低分框参与 NMS 把高分框误压掉，这在候选极多时很关键。
    """
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
        good = (tb[:, 2] > tb[:, 0]) & (tb[:, 3] > tb[:, 1])
        gi = paddle.nonzero(good).reshape([-1])
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


def main():
    ap = argparse.ArgumentParser(description="融合方案后处理精调")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--per-class", type=int, default=25)
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
    print("=" * 84)
    print(f"C · 融合后处理精调    分层抽样 {args.per_class}/类 = {len(ds)} 张 / "
          f"{sum(len(g['labels']) for g in gts)} GT")
    print("=" * 84)

    refiner = RegionRefiner(num_classes=len(CN) + 1, in_size=96)
    refiner.set_state_dict(paddle.load(str(utils.ROOT / "results/weights/refiner_best.pdparams")))
    refiner.eval()
    rinf = RefinerInfer(refiner, in_size=96, margin=0.15, batch=256)
    ver = DefectVerifier(num_classes=len(CN) + 1, in_size=96)
    ver.set_state_dict(paddle.load(str(utils.ROOT / "results/weights/verifier_best.pdparams")))
    ver.eval()
    vinf = VerifierInfer(ver, batch=512)

    print("缓存融合推理结果...")
    cache = []
    for k in range(len(ds)):
        p, _ = ds.samples[k]
        im = Image.open(p).convert("RGB")
        W, H = im.size
        wins = make_windows(W, H, stride=args.stride)
        pr, ref = rinf.run_norm(im, wins)
        pv = vinf.score(im, wins)
        cls_p = np.sqrt(np.clip(pr[:, :len(CN)], 1e-9, None) *
                        np.clip(pv[:, :len(CN)], 1e-9, None))
        def_p = np.sqrt(np.clip(1.0 - pr[:, BG_CLASS], 1e-9, None) *
                        np.clip(1.0 - pv[:, BG_CLASS], 1e-9, None))
        cache.append((ref, def_p, cls_p.argmax(-1)))
    print(f"  已缓存 {len(cache)} 张（候选 {np.mean([len(c[1]) for c in cache]):.0f}/图）")

    print(f"\n{'阈值':>6}{'NMS':>6}{'top_k':>7}{'max_det':>9}{'mAP@0.5':>10}{'框/图':>8}"
          f"{'PS':>8}{'In':>8}{'Sc':>8}")
    best = (0, None)
    for th, nms, tk, md in itertools.product(
            [0.3, 0.5, 0.6], [0.2, 0.3, 0.4, 0.5], [30, 60, 150], [10, 20, 50]):
        preds = make_preds(cache, th, nms, md, tk)
        r = utils.evaluate_map(preds, gts, len(CN), CN, iou_threshold=0.5)
        npb = sum(len(p["labels"]) for p in preds) / len(preds)
        pc = r["per_class"]
        if r["map"] > best[0]:
            best = (r["map"], (th, nms, tk, md))
            print(f"{th:>6}{nms:>6}{tk:>7}{md:>9}{r['map']:>10.4f}{npb:>8.1f}"
                  f"{pc['PS']:>8.3f}{pc['In']:>8.3f}{pc['Sc']:>8.3f}   <- 新最优")
    print(f"\n  最优: mAP@0.5 = {best[0]:.4f}")
    print(f"  参数: 阈值={best[1][0]}  NMS={best[1][1]}  top_k={best[1][2]}  max_det={best[1][3]}")
    print("=" * 84)


if __name__ == "__main__":
    main()
