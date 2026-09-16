"""成员 C · 逐类分数阈值搜索（复用 tune_fused 的推理缓存，分钟级）

动机
----
融合方案目前用**单一全局阈值**决定「哪些候选留作预测」。但实测每类 AP 差异极大
（验证集上 PS 0.39 / In 0.10），说明各类的分数尺度并不一致，一个全局阈值不可能
对 6 类都最优。而 mAP 是按类独立计算的 —— 给每类各自选阈值，是零训练成本的收益点。

做法：坐标上升（coordinate ascent）。从全局阈值出发，逐类在候选阈值里挑「该类 AP 最高」
的值，多轮迭代直到不再变化。**只在验证集上搜**，测试集不参与。

用法：
    # 先用 tune_fused.py --all 生成缓存（已有则自动复用）
    python src/det/tools/tune_perclass_th.py --cache results/logs/tune_cache_val_270_imagenet_verifier_pre_best_refiner_pre_best.npz

风险提示：逐类阈值是在验证集上选的（6 个参数），val 数值会带轻微选择性偏差，
因此脚本同时报告搜索前后的 mAP 与各类变化，供人工判断是否值得采纳。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import data as data_mod  # noqa: E402
from core import utils  # noqa: E402
from tools.tune_fused import CN, get_gts, make_preds  # noqa: E402

CAND = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]


def load_cache(path):
    z = np.load(path)
    n = int(z["n"])
    return [(z[f"b{i}"], z[f"s{i}"], z[f"l{i}"]) for i in range(n)]


def map_of(cache, gts, th, nms, tk, md):
    preds = make_preds(cache, th, nms, md, tk)
    r = utils.evaluate_map(preds, gts, len(CN), CN, iou_threshold=0.5)
    npb = sum(len(p["labels"]) for p in preds) / len(preds)
    return r["map"], r["per_class"], npb


def main():
    ap = argparse.ArgumentParser(description="C · 逐类分数阈值搜索")
    ap.add_argument("--cache", required=True, help="tune_fused.py 生成的 npz 缓存")
    ap.add_argument("--split", default="val")
    ap.add_argument("--nms-iou", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=300)
    ap.add_argument("--max-det", type=int, default=50)
    ap.add_argument("--start-th", type=float, default=0.3, help="搜索起点（全局阈值）")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    utils.pick_device(prefer_gpu=True)
    ds, _ = data_mod.build_loader(utils.ROOT, args.split, 416, 8,
                                  augment=False, shuffle=False)
    gts = get_gts(ds)
    cache = load_cache(args.cache)
    print("=" * 84)
    print(f"C · 逐类分数阈值搜索    {args.split} {len(ds)} 张 / "
          f"{sum(len(g['labels']) for g in gts)} GT | 缓存 {len(cache)} 张")
    print(f"固定后处理：NMS {args.nms_iou} / top_k {args.top_k} / max_det {args.max_det}")
    print("=" * 84)

    base = np.full(len(CN), args.start_th, dtype="float32")
    m0, pc0, npb0 = map_of(cache, gts, base, args.nms_iou, args.top_k, args.max_det)
    print(f"起点（全局阈值 {args.start_th}）: mAP@0.5 = {m0:.4f}   框/图 {npb0:.1f}")

    best_th = base.copy()
    best_map = m0
    for rd in range(1, args.rounds + 1):
        changed = False
        for i, c in enumerate(CN):
            cur = float(best_th[i])
            best_i, best_v = cur, best_map
            for v in CAND:
                trial = best_th.copy()
                trial[i] = v
                m, _, _ = map_of(cache, gts, trial, args.nms_iou, args.top_k, args.max_det)
                if m > best_v + 1e-6:
                    best_i, best_v = v, m
            if best_i != cur:
                best_th[i] = best_i
                best_map = best_v
                changed = True
                print(f"  轮{rd}  {c}: 阈值 {cur} -> {best_i}   mAP {best_map:.4f}")
        print(f"轮 {rd} 结束：mAP@0.5 = {best_map:.4f}   阈值 = "
              f"{dict(zip(CN, [float(x) for x in best_th]))}")
        if not changed:
            print("  已收敛（本轮无变化）")
            break

    m1, pc1, npb1 = map_of(cache, gts, best_th, args.nms_iou, args.top_k, args.max_det)
    print("\n" + "-" * 84)
    print(f"{'类别':<6}{'起始AP':>10}{'逐类阈值后':>12}{'变化':>10}")
    for c in CN:
        print(f"{c:<6}{pc0[c]:>10.4f}{pc1[c]:>12.4f}{pc1[c] - pc0[c]:>+10.4f}")
    print("-" * 84)
    print(f"{'总体':<6}{m0:>10.4f}{m1:>12.4f}{m1 - m0:>+10.4f}")
    print(f"框/图 {npb0:.1f} -> {npb1:.1f}")

    out = {
        "split": args.split, "num_images": len(ds),
        "num_gt": int(sum(len(g["labels"]) for g in gts)),
        "postproc": {"nms_iou": args.nms_iou, "top_k": args.top_k, "max_det": args.max_det},
        "start": {"threshold": args.start_th, "map50": m0,
                  "per_class_ap50": {k: float(v) for k, v in pc0.items()},
                  "outputs_per_image": round(npb0, 1)},
        "perclass": {"thresholds": {c: float(best_th[i]) for i, c in enumerate(CN)},
                     "map50": m1,
                     "per_class_ap50": {k: float(v) for k, v in pc1.items()},
                     "outputs_per_image": round(npb1, 1)},
        "delta_map50": round(m1 - m0, 4),
        "candidates": CAND,
        "note": "逐类阈值在验证集上坐标上升搜索得到；测试集未参与，val 数值带选择性偏差。",
    }
    p = utils.ROOT / "results/metrics/perclass_threshold_val.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 results/metrics/perclass_threshold_val.json")
    print("=" * 84)


if __name__ == "__main__":
    main()
