"""成员 C · 单阶段 vs 两阶段（加区域验证器）对比实验

用法：
    python src/det/tools/two_stage_compare.py --model yolov3 --split val

背景：单阶段检测器的框回归已经可用（GT 所属特征点上的 IoU 约 0.74），
但它的置信度排序不可用（正样本仅占候选位置的 0.3%，obj 分支学不出判别力），
导致 mAP@0.5 恒为 0。本脚本验证「加一个稠密监督的区域验证器重新打分」能否救回来。

做法：
    1. 第一阶段：检测网络产生候选框（取分数最高的前 N 个，做 NMS）；
    2. 第二阶段：对每个候选裁剪图块，送验证器判类（6 缺陷 + 背景）；
       以「1 - 背景概率」作为该候选的缺陷置信度，并取缺陷类别作为最终类别。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import paddle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import data as data_mod  # noqa: E402
from core import utils  # noqa: E402
from core.boxes import batched_nms, clip_boxes_norm  # noqa: E402
from core.det_ops import decode_yolov3, filter_valid_boxes  # noqa: E402
from core.verifier import BG_CLASS, DefectVerifier, VerifierInfer  # noqa: E402


@paddle.no_grad()
def build_candidates(model, batch, im_size, top_n, th):
    """第一阶段：解码 -> 阈值 -> 保留前 top_n -> NMS，返回每图的候选框

    返回 (boxes_norm, scores, labels) 三元组列表，逐图一个。
    """
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
        keep = sc > th
        if int(keep.sum()) == 0:
            out.append((np.zeros((0, 4), dtype="float32"),
                        np.zeros(0, dtype="float32"),
                        np.zeros(0, dtype="int64")))
            continue
        cb, cs, cl = box[bi][keep], sc[keep], cls[bi][keep].argmax(-1)
        # 先按分数取前 top_n，控制第二阶段的开销
        if len(cs) > top_n:
            o = np.argsort(-cs)[:top_n]
            cb, cs, cl = cb[o], cs[o], cl[o]
        b = paddle.to_tensor(cb)
        s = paddle.to_tensor(cs).astype("float32")
        l = paddle.to_tensor(cl).astype("int64")
        b, s, l = filter_valid_boxes(b, s, l)
        if b.shape[0] == 0:
            out.append((np.zeros((0, 4), dtype="float32"), np.zeros(0, dtype="float32"),
                        np.zeros(0, dtype="int64")))
            continue
        out.append((b.numpy(), s.numpy(), l.numpy()))
    return out


@paddle.no_grad()
def evaluate(model, verifier, loader, ds, cfg, top_n, cand_th, bg_th,
             use_verifier, split_name):
    im_size = int(cfg["data"]["im_size"])
    nc = int(cfg["num_classes"])
    preds, gts = [], []
    ii = 0
    for batch in loader:
        cands = build_candidates(model, batch, im_size, top_n, cand_th)
        for bi in range(batch["image"].shape[0]):
            cb, cs, cl = cands[bi]
            boxes = np.zeros((0, 4), dtype="float32")
            scores = np.zeros(0, dtype="float32")
            labels = np.zeros(0, dtype="int64")

            if len(cb):
                if use_verifier:
                    img = Image.open(ds.samples[ii][0]).convert("RGB")
                    prob = verifier.score(img, cb)                  # (N,7)
                    defect_p = 1.0 - prob[:, BG_CLASS]              # 缺陷置信度
                    sel = defect_p > bg_th
                    if sel.any():
                        nb, dp = cb[sel], defect_p[sel]
                        vc = prob[sel][:, :BG_CLASS].argmax(-1)     # 验证器判定的类别
                        tb = paddle.to_tensor(nb)
                        ts = paddle.to_tensor(dp).astype("float32")
                        tl = paddle.to_tensor(vc).astype("int64")
                        idx = batched_nms(tb, ts, tl, nc, iou_threshold=0.5,
                                          top_k=min(100, int(tb.shape[0])))
                        boxes = clip_boxes_norm(tb[idx]).numpy()
                        scores = ts[idx].numpy()
                        labels = tl[idx].numpy()
                else:
                    tb = paddle.to_tensor(cb)
                    ts = paddle.to_tensor(cs).astype("float32")
                    tl = paddle.to_tensor(cl).astype("int64")
                    idx = batched_nms(tb, ts, tl, nc, iou_threshold=0.5,
                                      top_k=min(100, int(tb.shape[0])))
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

    r = utils.evaluate_map(preds, gts, nc, ds.class_names,
                           iou_threshold=float(cfg["eval"]["iou_threshold"]))
    n_boxes = sum(len(p["labels"]) for p in preds) / max(len(preds), 1)
    return r, n_boxes


def main():
    ap = argparse.ArgumentParser(description="单阶段 vs 两阶段对比")
    ap.add_argument("--model", default="yolov3", choices=["yolov3", "ppyoloe_s"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--top-n", type=int, default=60, help="每图送入验证器的候选数")
    ap.add_argument("--cand-th", type=float, default=0.001, help="第一阶段候选阈值")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(2026)
    cfg = utils.load_config(f"src/det/configs/{args.model}.yml")
    name = cfg["name"]

    det = data_mod.build_model(cfg)
    det.set_state_dict(paddle.load(str(utils.ROOT / f"results/weights/{name}_best.pdparams")))
    det.eval()

    ver = DefectVerifier(num_classes=len(data_mod.CLASS_NAMES) + 1, in_size=96)
    vpath = utils.ROOT / "results/weights/verifier_best.pdparams"
    if not vpath.exists():
        raise SystemExit("找不到验证器权重，请先运行 python src/det/train_verifier.py")
    ver.set_state_dict(paddle.load(str(vpath)))
    verifier = VerifierInfer(ver)

    ds, loader = data_mod.build_loader(
        utils.ROOT, args.split, int(cfg["data"]["im_size"]),
        max(int(cfg["train"]["batch_size"]), 8), augment=False, shuffle=False)

    print("=" * 74)
    print(f"C · 单阶段 vs 两阶段对比    {name}    划分={args.split}    样本={len(ds)}")
    print("=" * 74)
    print(f"第一阶段候选数/图 {args.top_n}，候选阈值 {args.cand_th}")

    t = time.time()
    r1, n1 = evaluate(det, verifier, loader, ds, cfg, args.top_n, args.cand_th,
                      0.0, use_verifier=False, split_name=args.split)
    print(f"\n[单阶段] mAP@0.5 = {r1['map']:.4f}   每图框数 {n1:.1f}   "
          f"用时 {time.time()-t:.0f}s")
    print(f"  每类 AP: " + "  ".join(f"{c}={r1['per_class'][c]:.3f}" for c in ds.class_names))

    results = {"single_stage": {"map50": r1["map"], "per_class": r1["per_class"],
                                "boxes_per_img": n1}}
    for bg_th in (0.2, 0.3, 0.5):
        t = time.time()
        r2, n2 = evaluate(det, verifier, loader, ds, cfg, args.top_n, args.cand_th,
                          bg_th, use_verifier=True, split_name=args.split)
        print(f"\n[两阶段 背景阈 {bg_th}] mAP@0.5 = {r2['map']:.4f}   每图框数 {n2:.1f}   "
              f"用时 {time.time()-t:.0f}s")
        print(f"  每类 AP: " + "  ".join(f"{c}={r2['per_class'][c]:.3f}" for c in ds.class_names))
        results[f"two_stage_bg{bg_th}"] = {"map50": r2["map"],
                                           "per_class": r2["per_class"],
                                           "boxes_per_img": n2}

    out = utils.resolve_dir("results/metrics") / f"{name}_two_stage_compare.json"
    utils.dump_json({"model": name, "split": args.split, "top_n": args.top_n,
                     "cand_threshold": args.cand_th, "results": results}, out)
    print(f"\n对比结果已写入 {out.relative_to(utils.ROOT)}")
    print("=" * 74)


if __name__ == "__main__":
    main()
