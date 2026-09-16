"""成员 C · C 方案（学习式重排序）第 3 步：在验证集上评估

对照两组、同一套后处理：
    基线   —— 交付版的固定公式：几何平均缺陷分数 + 按 pv*pr 取类别（即 0.2473 那条链）
    重排序 —— 用 rerank_train.py 训出的小 MLP 打分，类别取其输出最大的那一类

关键设计：两组都跑**同一份候选、同一个 NMS 管线**，只有「分数怎么来的」这一层不同，
所以差异可以完全归因到排序层。后处理用 tools/tune_fused.py 的 make_preds 复用，
保证与 0.2473 的口径逐位一致。

用法：
    python src/det/tools/rerank_eval.py \
        --cache results/logs/rerank_cache_val270_i128.npz \
        --weights results/weights/rerank_mlp.pdparams \
        --meta results/metrics/rerank_mlp.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import paddle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import data as data_mod  # noqa: E402
from core import utils  # noqa: E402
from tools.rerank_common import (FEATURE_DIM, N_CLS, baseline_scores,  # noqa: E402
                                 build_features, load_cache)
from tools.rerank_train import ReRankMLP  # noqa: E402
from tools.tune_fused import make_preds  # noqa: E402

CN = data_mod.CLASS_NAMES
# 后处理网格：两组共用，保证公平（重排序器的分数尺度不同，必须允许各自选阈值）
GRID = [(th, nms, tk, md)
        for th in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        for nms in (0.3, 0.4)
        for tk in (150, 300)
        for md in (50,)]


def main():
    ap = argparse.ArgumentParser(description="C · 重排序评估")
    ap.add_argument("--cache", required=True, help="验证集特征缓存 npz")
    ap.add_argument("--weights", default="results/weights/rerank_mlp.pdparams")
    ap.add_argument("--meta", default="results/metrics/rerank_mlp.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(2026)

    refs, pvs, prs, stems = load_cache(args.cache)
    meta = json.loads((utils.ROOT / args.meta).read_text(encoding="utf-8"))
    print("=" * 84)
    print(f"C · 学习式重排序评估    {args.split} {len(refs)} 张 | 特征 {FEATURE_DIM} 维"
          f" | 模型 {meta['params']} 参数（第 {meta.get('best_epoch')} 轮）")
    print("=" * 84)

    ds, _ = data_mod.build_loader(utils.ROOT, args.split, 416, 8,
                                  augment=False, shuffle=False)
    gt_by_stem = {p.stem: ann_p for p, ann_p in ds.samples}

    def gts_of(stem):
        gt = data_mod.parse_voc_xml(gt_by_stem[stem], CN)
        boxes = (np.stack([np.array([x[1] - x[3] / 2, x[2] - x[4] / 2,
                                     x[1] + x[3] / 2, x[2] + x[4] / 2]) for x in gt])
                 if len(gt) else np.zeros((0, 4), dtype="float32"))
        labels = gt[:, 0].astype("int64") if len(gt) else np.zeros(0, dtype="int64")
        return {"boxes": boxes, "labels": labels}

    gts = [gts_of(s) for s in stems]

    # ---- 基线：几何平均 ----
    base_cache = []
    for i in range(len(refs)):
        def_p, labels = baseline_scores(refs[i], pvs[i], prs[i])
        base_cache.append((refs[i], def_p, labels))

    # ---- 重排序：小 MLP ----
    model = ReRankMLP(hidden=meta["hidden"])
    w = Path(args.weights)
    if not w.is_absolute():
        w = utils.ROOT / w
    model.set_state_dict(paddle.load(str(w)))
    model.eval()
    mu = np.asarray(meta["normalize"]["mean"], dtype="float32")
    sd = np.asarray(meta["normalize"]["std"], dtype="float32")

    rr_cache = []
    with paddle.no_grad():
        for i in range(len(refs)):
            X = (build_features(refs[i], pvs[i], prs[i]) - mu) / sd
            scores = paddle.nn.functional.sigmoid(
                model(paddle.to_tensor(X))).numpy()          # (N,6)
            labels = scores.argmax(-1).astype("int64")
            def_p = scores.max(-1).astype("float32")
            rr_cache.append((refs[i], def_p, labels))

    def best_of(cache, name):
        rows = []
        for cfg in GRID:
            preds = make_preds(cache, cfg[0], cfg[1], cfg[3], cfg[2])
            r = utils.evaluate_map(preds, gts, N_CLS, CN, iou_threshold=0.5)
            npb = sum(len(p["labels"]) for p in preds) / len(preds)
            rows.append((r["map"], cfg, npb, r["per_class"]))
        rows.sort(key=lambda x: -x[0])
        m, cfg, npb, pc = rows[0]
        print(f"\n【{name}】最优 mAP@0.5 = {m:.4f}   阈值 {cfg[0]} / NMS {cfg[1]}"
              f" / top_k {cfg[2]} / max_det {cfg[3]}   框/图 {npb:.1f}")
        for c in CN:
            print(f"    {c:<4}{pc[c]:>8.4f}")
        print(f"    前 5 名配置：" + " | ".join(
            f"{r[1][0]}/{r[1][1]}/{r[1][2]}={r[0]:.4f}" for r in rows[:5]))
        return m, cfg, npb, pc, rows[:5]

    mb, cb, nb, pcb, topb = best_of(base_cache, "基线：几何平均（交付版口径）")
    mr, cr, nr, pcr, topr = best_of(rr_cache, "重排序：学习式打分（C 方案）")

    print("\n" + "-" * 84)
    print(f"{'类别':<6}{'基线AP':>10}{'重排序AP':>12}{'变化':>10}")
    for c in CN:
        print(f"{c:<6}{pcb[c]:>10.4f}{pcr[c]:>12.4f}{pcr[c] - pcb[c]:>+10.4f}")
    print("-" * 84)
    print(f"{'总体':<6}{mb:>10.4f}{mr:>12.4f}{mr - mb:>+10.4f}")
    verdict = "有效" if mr > mb + 1e-4 else ("持平" if abs(mr - mb) <= 1e-4 else "无效/更差")
    print(f"\n结论：{verdict}（mAP@0.5 {mb:.4f} -> {mr:.4f}）")

    out = {
        "scheme": "C：学习式重排序（固定几何平均 -> 小 MLP 打分）",
        "split": args.split, "num_images": len(refs),
        "reranker": {"weights": w.name, "params": meta["params"],
                     "best_epoch": meta.get("best_epoch"), "hidden": meta["hidden"]},
        "postproc_grid": [list(g) for g in GRID],
        "baseline": {"map50": mb, "config": list(cb), "outputs_per_image": nb,
                     "per_class_ap50": {k: float(v) for k, v in pcb.items()},
                     "top5": [{"map50": r[0], "config": list(r[1])} for r in topb]},
        "rerank": {"map50": mr, "config": list(cr), "outputs_per_image": nr,
                   "per_class_ap50": {k: float(v) for k, v in pcr.items()},
                   "top5": [{"map50": r[0], "config": list(r[1])} for r in topr]},
        "delta_map50": round(mr - mb, 4), "verdict": verdict,
        "note": "两组共用同一份候选与同一套 NMS 管线，只有打分方式不同；"
                "重排序器只用训练集训练，验证集未参与拟合。",
    }
    p = utils.resolve_dir("results/metrics") / "rerank_eval_val.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 results/metrics/rerank_eval_val.json")
    print("=" * 84)


if __name__ == "__main__":
    main()
