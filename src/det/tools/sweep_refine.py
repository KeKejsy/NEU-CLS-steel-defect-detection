"""成员 C · 超参扫描（分层抽样，保证类别均衡）

用法：
    python src/det/tools/sweep_refine.py --per-class 20

**为什么要专门写这个脚本**：我最初直接用 `ds.samples[:N]` 取子集扫描，
结果是错的 —— 划分文件里图片是按类别前缀排序的，前 80 张只有 Cr 和 In 两类
（45 + 35），完全不代表整体，扫出来的 mAP（0.043）与全量评估（0.099）差一倍多。
本脚本改为**每类等量抽样**，并支持一次推理缓存候选、多次调参，兼顾正确性与速度。
"""

import argparse
import itertools
import sys
from collections import defaultdict
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

CN = data_mod.CLASS_NAMES
PREFIX = {"Cr": "crazing_", "In": "inclusion_", "Pa": "patches_",
          "PS": "pitted_surface_", "RS": "rolled-in_scale_", "Sc": "scratches_"}


def stratified_sample(names, per_class, seed=2026):
    """每类等量抽样，避免按文件名顺序取子集带来的类别偏置"""
    buckets = defaultdict(list)
    for n in names:
        for c, p in PREFIX.items():
            if n.startswith(p):
                buckets[c].append(n)
                break
    rng = np.random.RandomState(seed)
    out = []
    for c in CN:
        lst = buckets[c]
        k = min(per_class, len(lst))
        idx = rng.choice(len(lst), size=k, replace=False)
        out.extend(lst[i] for i in idx)
    return sorted(out)


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


def evaluate(rows, gts, th, nms_iou, max_det=50, use_refined=True):
    preds = []
    for wins, ref, dp, lb in rows:
        keep = dp >= th
        if not keep.any():
            preds.append({"boxes": np.zeros((0, 4), dtype="float32"),
                          "scores": np.zeros(0, dtype="float32"),
                          "labels": np.zeros(0, dtype="int64")})
            continue
        boxes = ref if use_refined else wins
        tb = paddle.to_tensor(boxes[keep])
        ts = paddle.to_tensor(dp[keep]).astype("float32")
        tl = paddle.to_tensor(lb[keep]).astype("int64")
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
    r = utils.evaluate_map(preds, gts, len(CN), CN, iou_threshold=0.5)
    npb = sum(len(p["labels"]) for p in preds) / max(len(preds), 1)
    return r["map"], npb, r["per_class"]


def main():
    ap = argparse.ArgumentParser(description="滑窗+精修器 超参扫描（分层抽样）")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--strides", default="12,16")
    ap.add_argument("--use-verifier", action="store_true",
                    help="用验证器代替精修器判类（对比用）")
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
    n_gt = sum(len(g["labels"]) for g in gts)
    from collections import Counter
    print("=" * 80)
    print(f"C · 超参扫描（分层抽样 {args.per_class}/类）    split={args.split}    "
          f"{len(ds)} 张 / {n_gt} GT")
    print("=" * 80)

    # ---- 模型 ----
    refiner = RegionRefiner(num_classes=len(CN) + 1, in_size=96)
    refiner.set_state_dict(paddle.load(str(utils.ROOT / "results/weights/refiner_best.pdparams")))
    refiner.eval()
    rinf = RefinerInfer(refiner, in_size=96, margin=0.15, batch=256)

    verifier = None
    if args.use_verifier:
        ver = DefectVerifier(num_classes=len(CN) + 1, in_size=96)
        ver.set_state_dict(paddle.load(str(utils.ROOT / "results/weights/verifier_best.pdparams")))
        ver.eval()
        verifier = VerifierInfer(ver, batch=512)

    strides = [int(s) for s in args.strides.split(",")]
    cache = {}
    for st in strides:
        rows = []
        for k in range(len(ds)):
            p, _ = ds.samples[k]
            im = Image.open(p).convert("RGB")
            W, H = im.size
            wins = make_windows(W, H, stride=st)
            if verifier is not None:
                prob = verifier.score(im, wins)
                ref = wins
            else:
                prob, ref = rinf.run_norm(im, wins)
            rows.append((wins, ref, 1.0 - prob[:, BG_CLASS],
                         prob[:, :len(CN)].argmax(-1)))
        cache[st] = rows
        print(f"  stride {st}: 候选 {np.mean([len(r[0]) for r in rows]):.0f}/图"
              + ("  （用验证器判类）" if verifier is not None else "  （用精修器判类+回归）"))

    print(f"\n{'stride':>7}{'阈值':>7}{'NMS':>6}{'mAP@0.5':>10}{'框/图':>8}"
          f"{'最好类':>10}{'最差类':>10}")
    best = (0, None)
    for st, th, nms in itertools.product(strides, [0.5, 0.7, 0.9, 0.95], [0.3, 0.45, 0.6]):
        mp, npb, pc = evaluate(cache[st], gts, th, nms)
        bestc = max(pc, key=pc.get)
        worstc = min(pc, key=pc.get)
        if mp > best[0]:
            best = (mp, (st, th, nms))
        print(f"{st:>7}{th:>7}{nms:>6}{mp:>10.4f}{npb:>8.1f}"
              f"{bestc+'='+format(pc[bestc],'.3f'):>10}{worstc+'='+format(pc[worstc],'.3f'):>10}")

    print(f"\n  最优: mAP@0.5 = {best[0]:.4f}   参数 stride={best[1][0]} "
          f"阈值={best[1][1]} NMS={best[1][2]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
