"""成员 C · 两阶段检测评估（检测器定位 + 区域验证器判类/打分）

用法：
    python src/det/eval_2stage.py --model yolov3                # 默认验证集
    python src/det/eval_2stage.py --model yolov3 --split test   # 测试集（仅一次）
    python src/det/eval_2stage.py --mode window                 # 纯滑窗模式（不用检测器）

## 为什么需要两阶段

本项目实测：单阶段检测器的**框回归可用**（GT 所属特征点上 IoU 达 0.742 / 0.757），
但**置信度排序不可用** —— 同一个框的 obj 分数只有 0.056，在 307 个候选里排第 135 位，
被 top_k 截断淘汰，mAP@0.5 恒为 0。根因是每张图只有 2~3 个正样本、候选位置上万条
（占比 0.3%），obj 分支学不出判别力。

而区域验证器（core/verifier.py）的图块分类是稠密监督任务，实测很可靠：
验证集整体准确率 88.7%、缺陷类 90.3%；GT 框上缺陷概率均值 0.904（93.8% > 0.5），
随机背景框仅 0.261。所以用它来判「哪个候选是真缺陷」，正好补上检测器缺的那一环。

## 两种模式

- `--mode detect`（默认）：检测网络产生候选框 -> 验证器重新打分与判类。
  用 top_n 限制候选数（默认 200，避免在检测器的低质量排序上浪费算力）。
- `--mode window`：不依赖检测器，直接用多尺度滑窗让验证器逐窗判类。
  作为对照，说明「检测器 + 验证器」相比「纯滑窗」是否真的有增益。

产出（results/metrics/ 与 results/figures/）：
    <name>_eval2stage_<mode>_<split>.json   完整指标
    <name>_pr2stage_<mode>_<split>.png      每类 PR 曲线
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

from core import data as data_mod  # noqa: E402
from core import utils  # noqa: E402
from core.boxes import batched_nms, clip_boxes_norm  # noqa: E402
from core.det_ops import decode_yolov3, filter_valid_boxes  # noqa: E402
from core.verifier import BG_CLASS, DefectVerifier, VerifierInfer  # noqa: E402
from core.window_detector import DenseWindowDetector  # noqa: E402
from eval_det import compute_prf, match_stats, plot_confusion, plot_per_class_ap, plot_pr_curves  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_verifier():
    path = utils.ROOT / "results/weights/verifier_best.pdparams"
    if not path.exists():
        raise SystemExit("找不到验证器权重，请先运行：\n  python src/det/train_verifier.py")
    ver = DefectVerifier(num_classes=len(data_mod.CLASS_NAMES) + 1, in_size=96)
    ver.set_state_dict(paddle.load(str(path)))
    ver.eval()
    return VerifierInfer(ver, batch=512)


@paddle.no_grad()
def detect_mode_candidates(model, batch, im_size, top_n, cand_th):
    """第一阶段：解码 -> 阈值筛 -> 取前 top_n -> NMS，返回每图候选框"""
    raw = model(batch["image"])
    decoded = [decode_yolov3([raw[i]], [model.anchors_norm[i]], [model.strides[i]],
                             model.num_classes, im_size)[0] for i in range(len(raw))]
    allp = paddle.concat(decoded, axis=1)
    box = allp[..., 0:4].numpy()
    obj = allp[..., 4].numpy()
    cls = allp[..., 5:].numpy()
    out = []
    for bi in range(box.shape[0]):
        sc = obj[bi] * cls[bi].max(-1)
        keep = sc > cand_th
        if not keep.any():
            out.append(np.zeros((0, 4), dtype="float32"))
            continue
        cb, cs = box[bi][keep], sc[keep]
        if len(cs) > top_n:
            o = np.argsort(-cs)[:top_n]
            cb = cb[o]
        b = paddle.to_tensor(cb)
        s = paddle.to_tensor(np.ones(len(cb), dtype="float32"))
        l = paddle.zeros([len(cb)], dtype="int64")
        b, s, l = filter_valid_boxes(b, s, l)
        out.append(b.numpy() if b.shape[0] else np.zeros((0, 4), dtype="float32"))
    return out


@paddle.no_grad()
def run_detect_mode(model, verifier, loader, ds, cfg, top_n, cand_th, bg_th,
                    nms_iou, max_det):
    """检测器给候选 -> 验证器打分判类 -> NMS"""
    im_size = int(cfg["data"]["im_size"])
    nc = int(cfg["num_classes"])
    preds, gts = [], []
    ii = 0
    for batch in loader:
        cands = detect_mode_candidates(model, batch, im_size, top_n, cand_th)
        for bi in range(batch["image"].shape[0]):
            cb = cands[bi]
            boxes = np.zeros((0, 4), dtype="float32")
            scores = np.zeros(0, dtype="float32")
            labels = np.zeros(0, dtype="int64")
            if len(cb):
                img = Image.open(ds.samples[ii][0]).convert("RGB")
                prob = verifier.score(img, cb)                  # (N,7)
                defect_p = 1.0 - prob[:, BG_CLASS]
                sel = defect_p >= bg_th
                if sel.any():
                    nb, dp = cb[sel], defect_p[sel]
                    vc = prob[sel][:, :BG_CLASS].argmax(-1)
                    tb = paddle.to_tensor(nb)
                    ts = paddle.to_tensor(dp).astype("float32")
                    tl = paddle.to_tensor(vc).astype("int64")
                    idx = batched_nms(tb, ts, tl, nc, iou_threshold=nms_iou,
                                      top_k=min(max_det, int(tb.shape[0])))
                    boxes = clip_boxes_norm(tb[idx]).numpy()
                    scores = ts[idx].numpy()
                    labels = tl[idx].numpy()
            preds.append({"boxes": boxes, "scores": scores, "labels": labels})
            t = batch["target"][bi].numpy()
            n = int(batch["num_boxes"][bi])
            t = t[:n]
            gts.append({"boxes": t[:, 1:5] if n else np.zeros((0, 4), dtype="float32"),
                        "labels": t[:, 0].astype("int64") if n else np.zeros(0, dtype="int64")})
            ii += 1
    return preds, gts


@paddle.no_grad()
def run_window_mode(verifier, ds, cfg, bg_th, nms_iou, max_det, stride, scales):
    """纯滑窗模式：完全不依赖检测网络"""
    nc = int(cfg["num_classes"])
    wd = DenseWindowDetector(verifier, scales=scales, stride=stride)
    preds, gts = [], []
    for i in range(len(ds)):
        img_p, ann_p = ds.samples[i]
        im = Image.open(img_p).convert("RGB")
        r = wd.detect(im, score_threshold=bg_th, nms_iou=nms_iou, max_det=max_det)
        preds.append({"boxes": r["boxes"], "scores": r["scores"],
                      "labels": r["labels"].astype("int64")})
        gt = data_mod.parse_voc_xml(ann_p, data_mod.CLASS_NAMES)
        gtb = np.stack([np.array([x[1] - x[3] / 2, x[2] - x[4] / 2,
                                  x[1] + x[3] / 2, x[2] + x[4] / 2]) for x in gt]) \
            if len(gt) else np.zeros((0, 4), dtype="float32")
        gts.append({"boxes": gtb, "labels": gt[:, 0].astype("int64") if len(gt)
                    else np.zeros(0, dtype="int64")})
    return preds, gts, nc


def main():
    ap = argparse.ArgumentParser(description="C · 两阶段检测评估")
    ap.add_argument("--model", default="yolov3", choices=["yolov3", "ppyoloe_s"],
                    help="第一阶段定位用的检测网络")
    ap.add_argument("--mode", default="detect", choices=["detect", "window"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--top-n", type=int, default=200, help="每图送入验证器的候选数")
    ap.add_argument("--cand-th", type=float, default=0.001, help="第一阶段候选阈值")
    ap.add_argument("--bg-th", type=float, default=0.5, help="验证器缺陷概率阈值")
    ap.add_argument("--nms-iou", type=float, default=0.4)
    ap.add_argument("--max-det", type=int, default=50)
    ap.add_argument("--stride", type=int, default=16, help="滑窗模式的网格步长")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(2026)
    cfg = utils.load_config(f"src/det/configs/{args.model}.yml")
    name = cfg["name"]

    print("=" * 74)
    print(f"成员 C · 两阶段检测评估    mode={args.mode}    {name}    划分={args.split}")
    print("=" * 74)
    if args.split == "test":
        print("注意：测试集按项目铁律只用于最终评估一次。")

    verifier = load_verifier()
    ds, loader = data_mod.build_loader(
        utils.ROOT, args.split, int(cfg["data"]["im_size"]),
        max(int(cfg["train"]["batch_size"]), 8), augment=False, shuffle=False)

    t0 = time.time()
    if args.mode == "detect":
        model = data_mod.build_model(cfg)
        model.set_state_dict(paddle.load(
            str(utils.ROOT / f"results/weights/{name}_best.pdparams")))
        model.eval()
        print(f"第一阶段候选数/图 {args.top_n}，候选阈值 {args.cand_th}，"
              f"验证器阈值 {args.bg_th}")
        preds, gts = run_detect_mode(model, verifier, loader, ds, cfg, args.top_n,
                                     args.cand_th, args.bg_th, args.nms_iou, args.max_det)
        nc = int(cfg["num_classes"])
    else:
        print(f"纯滑窗模式：步长 {args.stride}，验证器阈值 {args.bg_th}")
        preds, gts, nc = run_window_mode(verifier, ds, cfg, args.bg_th, args.nms_iou,
                                         args.max_det, args.stride,
                                         (0.2, 0.3, 0.45, 0.65))
    cost = time.time() - t0

    n_gt = sum(len(g["labels"]) for g in gts)
    n_pd = sum(len(p["labels"]) for p in preds)
    print(f"\n评估 {len(ds)} 张：标注框 {n_gt}，预测框 {n_pd}"
          f"（平均每图 {n_pd/max(len(preds),1):.1f}），总耗时 {cost:.0f}s "
          f"（{cost/max(len(ds),1)*1000:.0f} ms/图）")

    res = utils.evaluate_map(preds, gts, nc, ds.class_names,
                            iou_threshold=float(cfg["eval"]["iou_threshold"]))
    tp, fp, fn, conf = match_stats(preds, gts, nc,
                                   iou_threshold=float(cfg["eval"]["iou_threshold"]))

    print("\n" + "-" * 82)
    print(f"{'类别':<6}{'中文':<14}{'AP@0.5':>9}{'精确率':>9}{'召回率':>9}{'F1':>8}"
          f"{'TP':>6}{'FP':>6}{'FN':>6}")
    print("-" * 82)
    detail = {}
    for i, c in enumerate(ds.class_names):
        p, r, f1 = compute_prf(tp[i], fp[i], fn[i])
        ap_v = res["per_class"].get(c, 0.0)
        detail[c] = {"ap50": ap_v, "precision": p, "recall": r, "f1": f1,
                     "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i]),
                     "cn_name": data_mod.CLASS_NAMES_CN.get(c, "")}
        print(f"{c:<6}{data_mod.CLASS_NAMES_CN.get(c,''):<14}{ap_v:>9.4f}{p:>9.4f}"
              f"{r:>9.4f}{f1:>8.4f}{int(tp[i]):>6}{int(fp[i]):>6}{int(fn[i]):>6}")
    print("-" * 82)
    mp, mr, mf = compute_prf(tp.sum(), fp.sum(), fn.sum())
    print(f"{'总体':<20}{res['map']:>9.4f}{mp:>9.4f}{mr:>9.4f}{mf:>8.4f}"
          f"{int(tp.sum()):>6}{int(fp.sum()):>6}{int(fn.sum()):>6}")
    print(f"\n>>> mAP@0.5 = {res['map']:.4f}")

    fig_dir = utils.resolve_dir("results/figures")
    metric_dir = utils.resolve_dir("results/metrics")
    tag = f"{args.mode}_{args.split}"
    plot_pr_curves(res["pr_curves"], ds.class_names,
                   fig_dir / f"{name}_pr2stage_{tag}.png",
                   f"{name} 两阶段({args.mode}) PR 曲线  mAP@0.5={res['map']:.4f}  [{args.split}]")
    plot_per_class_ap(res["per_class"], ds.class_names,
                      fig_dir / f"{name}_ap2stage_{tag}.png",
                      f"{name} 两阶段({args.mode}) 每类 AP@0.5  [{args.split}]", res["map"])
    plot_confusion(conf, ds.class_names, fig_dir / f"{name}_conf2stage_{tag}.png",
                   f"{name} 两阶段({args.mode}) 混淆矩阵  [{args.split}]")

    report = {
        "pipeline": "two_stage", "mode": args.mode, "detector": name,
        "verifier": "results/weights/verifier_best.pdparams",
        "split": args.split, "num_images": len(ds),
        "num_gt_boxes": int(n_gt), "num_pred_boxes": int(n_pd),
        "ms_per_image": round(cost / max(len(ds), 1) * 1000, 1),
        "iou_threshold": float(cfg["eval"]["iou_threshold"]),
        "params": {"top_n": args.top_n, "cand_threshold": args.cand_th,
                   "verifier_threshold": args.bg_th, "nms_iou": args.nms_iou,
                   "max_det": args.max_det, "stride": args.stride},
        "map50": res["map"], "per_class": detail,
        "overall": {"precision": mp, "recall": mr, "f1": mf,
                    "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum())},
        "confusion_matrix": {"labels": list(ds.class_names) + ["background"],
                             "matrix": conf.tolist()},
        "pr_curves": {c: {"recall": res["pr_curves"][c]["recall"],
                          "precision": res["pr_curves"][c]["precision"],
                          "ap": res["pr_curves"][c]["ap"]} for c in ds.class_names},
    }
    out = utils.dump_json(report, metric_dir / f"{name}_eval2stage_{tag}.json")
    print(f"\n评估报告：{out.relative_to(utils.ROOT)}")
    print("=" * 74)


if __name__ == "__main__":
    main()
