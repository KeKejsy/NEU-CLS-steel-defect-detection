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

    `th` 可以是标量（全局阈值），也可以是长度 = 类别数 的向量（**逐类阈值**：
    每个候选按它被判成的类别取对应阈值）。逐类阈值用 tools/tune_perclass_th.py 搜。
    """
    th = np.asarray(th, dtype="float32")
    preds = []
    for ref, def_p, labels in cache:
        keep = def_p >= (th[labels] if th.ndim == 1 else th)
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
    ap.add_argument("--all", action="store_true",
                    help="用整个划分（val 全部 270 张），不做分层抽样；"
                         "这样扫出的最优值可直接与全量评估报告对比")
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--in-size", type=int, default=96,
                    help="判别模型输入尺寸，必须与训练时一致")
    ap.add_argument("--verifier-weights", default="results/weights/verifier_best.pdparams")
    ap.add_argument("--refiner-weights", default="results/weights/refiner_best.pdparams")
    ap.add_argument("--norm", default="none", choices=["none", "imagenet"],
                    help="必须与被评估权重训练时一致")
    ap.add_argument("--reuse-cache", action="store_true",
                    help="复用 results/logs/tune_cache_*.npz（跳过 6 分钟推理，只重扫网格）")
    ap.add_argument("--skip-grid", action="store_true",
                    help="只建缓存并评估固定配置（默认配置 + 逐类阈值起点），不扫网格；"
                         "后面还要接 tools/tune_perclass_th.py 时用它省时间")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(2026)

    ds, _ = data_mod.build_loader(utils.ROOT, args.split, 416, 8,
                                  augment=False, shuffle=False)
    if not args.all:
        names = stratified_sample([p.stem for p, _ in ds.samples], args.per_class)
        keep_idx = [i for i, (p, _) in enumerate(ds.samples) if p.stem in set(names)]
        ds.samples = [ds.samples[i] for i in keep_idx]
    gts = get_gts(ds)
    print("=" * 84)
    scope = "整个划分" if args.all else f"分层抽样 {args.per_class}/类"
    print(f"C · 融合后处理精调    {scope} = {len(ds)} 张 / "
          f"{sum(len(g['labels']) for g in gts)} GT")
    print(f"验证器 {Path(args.verifier_weights).name} | 精修器 {Path(args.refiner_weights).name}"
          f" | 归一化 {args.norm}")
    print("=" * 84)

    refiner = RegionRefiner(num_classes=len(CN) + 1, in_size=args.in_size)
    refiner.set_state_dict(paddle.load(str(utils.ROOT / args.refiner_weights)))
    refiner.eval()
    rinf = RefinerInfer(refiner, in_size=args.in_size, margin=0.15, batch=256, norm=args.norm)
    ver = DefectVerifier(num_classes=len(CN) + 1, in_size=args.in_size)
    ver.set_state_dict(paddle.load(str(utils.ROOT / args.verifier_weights)))
    ver.eval()
    vinf = VerifierInfer(ver, in_size=args.in_size, batch=512, norm=args.norm)

    print("缓存融合推理结果...")
    # 缓存持久化：第一次跑要 6 分钟（4624 候选/图），存盘后重扫网格只需几秒。
    # 缓存只依赖「权重 + 归一化 + 抽样图片 + 步长 + 输入尺寸」，与后处理参数无关，可安全复用。
    # 步长与输入尺寸也必须进文件名：否则换个 stride/in_size 重跑会误命中旧缓存，得到错误结论。
    tag = (f"{args.split}_{len(ds)}_{args.norm}_s{args.stride}_i{args.in_size}_"
           f"{Path(args.verifier_weights).stem}_{Path(args.refiner_weights).stem}")
    cache_path = utils.ROOT / "results" / "logs" / f"tune_cache_{tag}.npz"
    cache = []
    if args.reuse_cache and cache_path.exists():
        z = np.load(cache_path)
        for i in range(int(z["n"])):
            cache.append((z[f"b{i}"], z[f"s{i}"], z[f"l{i}"]))
        print(f"  复用已有缓存 {cache_path.name}（{len(cache)} 张）")
    else:
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
        np.savez_compressed(cache_path, n=len(cache),
                            **{f"{k}{i}": v for i, (b, s, l) in enumerate(cache)
                               for k, v in (("b", b), ("s", s), ("l", l))})
        print(f"  已缓存 {len(cache)} 张 -> {cache_path.name}")
    print(f"  候选 {np.mean([len(c[1]) for c in cache]):.0f}/图")

    print(f"\n{'阈值':>6}{'NMS':>6}{'top_k':>7}{'max_det':>9}{'mAP@0.5':>10}{'框/图':>8}"
          f"{'PS':>8}{'In':>8}{'Sc':>8}")
    best = (0, None)
    # 当前默认配置（写进交付文档的那套），单独跑一次做参照
    # 注意 make_preds 的形参顺序是 (th, nms_iou, max_det, top_k)
    th_d, nms_d, tk_d, md_d = 0.6, 0.4, 150, 50
    preds_d = make_preds(cache, th_d, nms_d, md_d, tk_d)
    map_d = utils.evaluate_map(preds_d, gts, len(CN), CN, iou_threshold=0.5)["map"]
    npb_d = sum(len(p["labels"]) for p in preds_d) / len(preds_d)
    print(f"{th_d:>6}{nms_d:>6}{tk_d:>7}{md_d:>9}{map_d:>10.4f}{npb_d:>8.1f}"
          f"   <- 当前默认配置")
    all_rows = []
    if args.skip_grid:
        print("\n（--skip-grid：跳过参数网格，仅报告两个固定配置）")
        for th in (0.6, 0.3):
            preds = make_preds(cache, th, 0.3, 50, 300)
            r = utils.evaluate_map(preds, gts, len(CN), CN, iou_threshold=0.5)
            npb = sum(len(p["labels"]) for p in preds) / len(preds)
            pc = r["per_class"]
            print(f"  阈值 {th} / NMS 0.3 / top_k 300 / max_det 50 : mAP@0.5 = {r['map']:.4f}  "
                  f"框/图 {npb:.1f}  PS {pc['PS']:.3f} In {pc['In']:.3f} Sc {pc['Sc']:.3f}")
        print(f"\n缓存已保存，可继续跑：python src/det/tools/tune_perclass_th.py "
              f"--cache results/logs/{cache_path.name}")
        print("=" * 84)
        return

    for th, nms, tk, md in itertools.product(
            [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.6],
            [60, 150, 300], [20, 50, 100]):
        preds = make_preds(cache, th, nms, md, tk)
        r = utils.evaluate_map(preds, gts, len(CN), CN, iou_threshold=0.5)
        npb = sum(len(p["labels"]) for p in preds) / len(preds)
        pc = r["per_class"]
        all_rows.append((r["map"], th, nms, tk, md, npb, pc))
        if r["map"] > best[0]:
            best = (r["map"], (th, nms, tk, md))
            print(f"{th:>6}{nms:>6}{tk:>7}{md:>9}{r['map']:>10.4f}{npb:>8.1f}"
                  f"{pc['PS']:>8.3f}{pc['In']:>8.3f}{pc['Sc']:>8.3f}   <- 新最优")
    print(f"\n  最优: mAP@0.5 = {best[0]:.4f}")
    print(f"  参数: 阈值={best[1][0]}  NMS={best[1][1]}  top_k={best[1][2]}  max_det={best[1][3]}")

    # 按 mAP 排序的前 10 名：单看「是否刷新最优」会掩盖整体走势，排行榜更直观
    all_rows.sort(key=lambda x: -x[0])
    print(f"\n  {'排名':>4}{'阈值':>7}{'NMS':>6}{'top_k':>7}{'max_det':>9}{'mAP@0.5':>10}"
          f"{'框/图':>8}{'PS':>7}{'In':>7}{'Sc':>7}")
    for i, (m, th, nms, tk, md, npb, pc) in enumerate(all_rows[:10], 1):
        mark = "  <- 默认" if (th, nms, tk, md) == (th_d, nms_d, tk_d, md_d) else ""
        print(f"{i:>4}{th:>7}{nms:>6}{tk:>7}{md:>9}{m:>10.4f}{npb:>8.1f}"
              f"{pc['PS']:>7.3f}{pc['In']:>7.3f}{pc['Sc']:>7.3f}{mark}")
    print("=" * 84)


if __name__ == "__main__":
    main()
