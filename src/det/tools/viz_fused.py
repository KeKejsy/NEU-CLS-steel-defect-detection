"""成员 C · 融合方案的真值-预测对比图（定性证据）

老交付里有一张 `fused_samples.png`（左=标注真值、右=模型预测），但那份图是用早期
两阶段方案画的。优化版（预训练主干 + GIoU + 128 输入）没有对应的图，本脚本补上，
让「完整跑一遍」的产物与老交付对齐。

每行一张图：左半是 VOC 标注，右半是当前最优配置的预测（阈值/NMS/top_k/max_det 与
正式评估完全一致）。按类别分层抽样，保证 6 类都出现。

用法：
    python src/det/tools/viz_fused.py --split val --per-class 3 --in-size 128 \
        --norm imagenet \
        --verifier-weights results/weights/verifier_pre128_best.pdparams \
        --refiner-weights  results/weights/refiner_giou128_best.pdparams \
        --tag val_fuse_pre128
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import paddle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import data as data_mod  # noqa: E402
from core import utils  # noqa: E402
from core.boxes import batched_nms, clip_boxes_norm  # noqa: E402
from core.refiner import RefinerInfer, RegionRefiner  # noqa: E402
from core.verifier import BG_CLASS, DefectVerifier, VerifierInfer  # noqa: E402
from core.window_detector import make_windows  # noqa: E402

CN = data_mod.CLASS_NAMES
CN_ZH = data_mod.CLASS_NAMES_CN
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def draw(ax, im, boxes, labels, scores, color, title):
    ax.imshow(im)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    W, H = im.size
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        w, h = (x2 - x1) * W, (y2 - y1) * H
        if w < 1 or h < 1:
            continue
        ax.add_patch(plt.Rectangle((x1 * W, y1 * H), w, h, fill=False,
                                   edgecolor=color, linewidth=1.5))
        txt = CN_ZH.get(CN[int(labels[i])], CN[int(labels[i])]) if i < len(labels) else ""
        if scores is not None and i < len(scores):
            txt = f"{txt} {scores[i]:.2f}"
        ax.text(x1 * W, max(y1 * H - 3, 6), txt, fontsize=6, color="white",
                bbox=dict(facecolor=color, alpha=0.65, pad=1, edgecolor="none"))


def main():
    ap = argparse.ArgumentParser(description="C · 融合方案真值-预测对比图")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--per-class", type=int, default=3, help="每类抽几张")
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--in-size", type=int, default=128)
    ap.add_argument("--norm", default="imagenet", choices=["none", "imagenet"])
    ap.add_argument("--score-th", type=float, default=0.6)
    ap.add_argument("--nms-iou", type=float, default=0.4)
    ap.add_argument("--top-k", type=int, default=150)
    ap.add_argument("--max-det", type=int, default=50)
    ap.add_argument("--verifier-weights", default="results/weights/verifier_pre128_best.pdparams")
    ap.add_argument("--refiner-weights", default="results/weights/refiner_giou128_best.pdparams")
    ap.add_argument("--tag", default="val_fuse_pre128")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(2026)

    ds, _ = data_mod.build_loader(utils.ROOT, args.split, 416, 8,
                                  augment=False, shuffle=False)
    # 分层抽样：按图片的主类别取 per_class 张，保证 6 类都出现
    by_cls = {c: [] for c in CN}
    for p, ann_p in ds.samples:
        gt = data_mod.parse_voc_xml(ann_p, CN)
        if len(gt) == 0:
            continue
        c = CN[int(gt[0, 0])]
        if len(by_cls[c]) < args.per_class:
            by_cls[c].append((p, ann_p))
    picked = [x for c in CN for x in by_cls[c]]
    print(f"分层抽样 {len(picked)} 张（每类 {args.per_class}）："
          f"{ {c: len(v) for c, v in by_cls.items()} }")

    refiner = RegionRefiner(num_classes=len(CN) + 1, in_size=args.in_size)
    refiner.set_state_dict(paddle.load(str(utils.ROOT / args.refiner_weights)))
    refiner.eval()
    rinf = RefinerInfer(refiner, in_size=args.in_size, margin=0.15, batch=256, norm=args.norm)
    ver = DefectVerifier(num_classes=len(CN) + 1, in_size=args.in_size)
    ver.set_state_dict(paddle.load(str(utils.ROOT / args.verifier_weights)))
    ver.eval()
    vinf = VerifierInfer(ver, in_size=args.in_size, batch=512, norm=args.norm)

    ncol = 2
    nrow = len(picked)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6, 3.3 * nrow))
    if nrow == 1:
        axes = np.asarray([axes])

    for r, (p, ann_p) in enumerate(picked):
        im = Image.open(p).convert("RGB")
        gt = data_mod.parse_voc_xml(ann_p, CN)
        gtb = np.asarray([[x[1] - x[3] / 2, x[2] - x[4] / 2,
                           x[1] + x[3] / 2, x[2] + x[4] / 2] for x in gt], dtype="float32")
        gtl = gt[:, 0].astype("int64")
        draw(axes[r][0], im, gtb, gtl, None, "red", f"标注真值  {p.name}")

        wins = make_windows(im.size[0], im.size[1], stride=args.stride)
        pr, ref = rinf.run_norm(im, wins)
        pv = vinf.score(im, wins)
        cls_p = np.sqrt(np.clip(pr[:, :len(CN)], 1e-9, None) *
                        np.clip(pv[:, :len(CN)], 1e-9, None))
        def_p = np.sqrt(np.clip(1.0 - pr[:, BG_CLASS], 1e-9, None) *
                        np.clip(1.0 - pv[:, BG_CLASS], 1e-9, None))
        labels = cls_p.argmax(-1)
        keep = def_p >= args.score_th
        if keep.any():
            b, s, l = ref[keep], def_p[keep], labels[keep]
            if len(s) > args.top_k:
                o = np.argsort(-s)[:args.top_k]
                b, s, l = b[o], s[o], l[o]
            tb = paddle.to_tensor(b)
            ts = paddle.to_tensor(s).astype("float32")
            tl = paddle.to_tensor(l).astype("int64")
            good = (tb[:, 2] > tb[:, 0]) & (tb[:, 3] > tb[:, 1])
            gi = paddle.nonzero(good).reshape([-1])
            tb, ts, tl = tb[gi], ts[gi], tl[gi]
            if tb.shape[0] > 0:
                idx = batched_nms(tb, ts, tl, len(CN), iou_threshold=args.nms_iou,
                                  top_k=min(args.max_det, int(tb.shape[0])))
                pb = clip_boxes_norm(tb[idx]).numpy()
                ps = ts[idx].numpy()
                pl = tl[idx].numpy()
            else:
                pb, ps, pl = np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype="int64")
        else:
            pb, ps, pl = np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype="int64")
        draw(axes[r][1], im, pb, pl, ps, "green",
             f"预测  {len(pl)} 框（阈值 {args.score_th}）")

    fig.suptitle(f"成员 C · 滑窗+验证器/精修器融合（优化版：预训练主干 + GIoU + {args.in_size} 输入）"
                 f"　划分 {args.split}　mAP@0.5 见 results/metrics/{args.tag}.json",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = utils.resolve_dir("results/figures") / f"{args.tag}_samples.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"已写出 {out.relative_to(utils.ROOT)}")


if __name__ == "__main__":
    main()
