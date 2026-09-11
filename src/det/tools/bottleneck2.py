"""成员 C · 迭代诊断 2：候选来源与聚合策略对比

用法：
    python src/det/tools/bottleneck2.py --split val --limit 100

[瓶颈诊断 1] 的结论：滑窗 2028 个/图、IoU>=0.5 上限 70.1%、验证器判类 97.35%，
但 GT 最佳候选的**排名中位数是 194** —— 说明瓶颈是「候选太多把正确框淹没」，
不是定位也不是判类。

本脚本对比几种「让正确框浮上来」的策略：

    A. 滑窗原始（基线）
    B. 滑窗先粗后细：粗网格筛出高分区域，再只在附近用细网格
    C. 检测器候选（YOLOv3 / PP-YOLOE-s）用验证器重打分
    D. 滑窗候选做「分数加权的框聚合」（同类高分框互相投票，抑制孤立高分）
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
from core.det_ops import decode_yolov3, filter_valid_boxes  # noqa: E402
from core.verifier import BG_CLASS, DefectVerifier, VerifierInfer  # noqa: E402
from core.window_detector import make_windows  # noqa: E402
from tools.diagnose import iou_matrix  # noqa: E402

CN = data_mod.CLASS_NAMES


def load_verifier():
    ver = DefectVerifier(num_classes=len(CN) + 1, in_size=96)
    ver.set_state_dict(paddle.load(str(utils.ROOT / "results/weights/verifier_best.pdparams")))
    ver.eval()
    return VerifierInfer(ver, batch=512)


def read_gts(ds):
    out = []
    for _img_p, ann_p in ds.samples:
        gt = data_mod.parse_voc_xml(ann_p, CN)
        gtb = np.stack([np.array([x[1] - x[3] / 2, x[2] - x[4] / 2,
                                  x[1] + x[3] / 2, x[2] + x[4] / 2]) for x in gt]) \
            if len(gt) else np.zeros((0, 4), dtype="float32")
        out.append({"boxes": gtb,
                    "labels": gt[:, 0].astype("int64") if len(gt) else np.zeros(0, dtype="int64")})
    return out


def score_to_preds(boxes, scores, labels, num_classes=6, nms_iou=0.45, max_det=50,
                   min_score=0.5):
    preds = []
    for i in range(len(boxes)):
        b, s, l = boxes[i], scores[i], labels[i]
        keep = s >= min_score
        if not keep.any():
            preds.append({"boxes": np.zeros((0, 4), dtype="float32"),
                          "scores": np.zeros(0, dtype="float32"),
                          "labels": np.zeros(0, dtype="int64")})
            continue
        tb = paddle.to_tensor(b[keep])
        ts = paddle.to_tensor(s[keep]).astype("float32")
        tl = paddle.to_tensor(l[keep]).astype("int64")
        tb, ts, tl = filter_valid_boxes(tb, ts, tl)
        if tb.shape[0] == 0:
            preds.append({"boxes": np.zeros((0, 4), dtype="float32"),
                          "scores": np.zeros(0, dtype="float32"),
                          "labels": np.zeros(0, dtype="int64")})
            continue
        idx = batched_nms(tb, ts, tl, num_classes, iou_threshold=nms_iou,
                          top_k=min(max_det, int(tb.shape[0])))
        preds.append({"boxes": clip_boxes_norm(tb[idx]).numpy(),
                      "scores": ts[idx].numpy(), "labels": tl[idx].numpy()})
    return preds


def eval_preds(preds, gts):
    return utils.evaluate_map(preds, gts, len(CN), CN, iou_threshold=0.5)


def agg_boxes(boxes, scores, labels, iou_thr=0.55, top_frac=0.2):
    """分数加权的框聚合

    思路：真实的缺陷会被多个相邻窗口同时命中的（它们重叠度高、分数都高），
    而孤立的误检往往只有单个窗口高分。所以对每个框，统计与它 IoU>thr 的
    其他高分框数量，用「数量 × 平均分」作为新的置信度 —— 抑制孤立高分。
    """
    n = len(boxes)
    if n == 0:
        return boxes, scores, labels, np.zeros(0)
    M = iou_matrix(boxes, boxes)
    np.fill_diagonal(M, 0.0)
    # 只考虑分数在较高区间内的邻居，避免被海量低分框稀释
    order = np.argsort(-scores)
    k = max(int(n * top_frac), 20)
    top_idx = set(order[:k].tolist())
    support = np.zeros(n)
    for i in range(n):
        nb = np.where(M[i] > iou_thr)[0]
        if len(nb) == 0:
            support[i] = 0.0
        else:
            w = [1.0 for j in nb if j in top_idx]
            support[i] = len(w)
    # 新分数 = 自身分数 × log(1+支持数)
    new = scores * np.log1p(support)
    return boxes, new, labels, support


def main():
    ap = argparse.ArgumentParser(description="候选来源与聚合策略对比")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(2026)

    ds, _ = data_mod.build_loader(utils.ROOT, args.split, 416, 8,
                                  augment=False, shuffle=False)
    ds.samples = ds.samples[:args.limit]
    gts = read_gts(ds)
    verifier = load_verifier()

    print("=" * 80)
    print(f"C · 候选来源与聚合策略对比    split={args.split}    {len(ds)} 张图")
    print("=" * 80)

    imgs = [Image.open(p).convert("RGB") for p, _ in ds.samples]

    # ---------- A. 滑窗原始 ----------
    print("\n[A] 滑窗原始（stride 16）")
    wins_all = [make_windows(im.size[0], im.size[1], stride=16) for im in imgs]
    probs = [verifier.score(im, w) for im, w in zip(imgs, wins_all)]
    w_boxes = wins_all
    w_scores = [1.0 - p[:, BG_CLASS] for p in probs]
    w_labels = [p[:, :len(CN)].argmax(-1) for p in probs]
    for th in (0.5, 0.7, 0.9):
        pr = score_to_preds(w_boxes, w_scores, w_labels, min_score=th)
        r = eval_preds(pr, gts)
        npb = sum(len(p["labels"]) for p in pr) / len(pr)
        print(f"    阈值 {th:.1f}: mAP@0.5 = {r['map']:.4f}   框/图 {npb:.1f}")

    # ---------- D. 分数加权聚合 ----------
    print("\n[D] 滑窗 + 分数加权框聚合（抑制孤立高分）")
    for th, iou, topf in ((0.5, 0.55, 0.2), (0.5, 0.5, 0.1), (0.3, 0.55, 0.2)):
        ab, asc, al = [], [], []
        for i in range(len(imgs)):
            b, s, l, sup = agg_boxes(w_boxes[i], w_scores[i], w_labels[i],
                                     iou_thr=iou, top_frac=topf)
            ab.append(b)
            asc.append(s)
            al.append(l)
        pr = score_to_preds(ab, asc, al, min_score=th)
        r = eval_preds(pr, gts)
        npb = sum(len(p["labels"]) for p in pr) / len(pr)
        print(f"    阈值 {th:.1f} 聚合IoU {iou:.2f} top{int(topf*100)}%: "
              f"mAP@0.5 = {r['map']:.4f}   框/图 {npb:.1f}")

    # ---------- B. 先粗后细 ----------
    print("\n[B] 先粗后细：粗网格定位高分区域 -> 局部细网格精修")
    for coarse, fine, rad in ((32, 8, 0.18), (24, 8, 0.15), (32, 12, 0.2)):
        boxes_l, scores_l, labels_l = [], [], []
        n_cand = []
        for i, im in enumerate(imgs):
            W, H = im.size
            cw = make_windows(W, H, stride=coarse)
            cp = verifier.score(im, cw)
            cs = 1.0 - cp[:, BG_CLASS]
            # 取粗网格里分数最高的若干区域
            k = min(8, len(cs))
            top = np.argsort(-cs)[:k]
            parts_b, parts_s, parts_p = [], [], []
            for t in top:
                cx = (cw[t, 0] + cw[t, 2]) / 2 * W
                cy = (cw[t, 1] + cw[t, 3]) / 2 * H
                r = rad * min(W, H)
                # 在局部区域内生成细网格窗口（直接构造，不依赖全局网格对齐）
                loc = []
                for sc in (0.2, 0.3, 0.45, 0.65):
                    for ar in (0.6, 1.0, 1.7):
                        w = sc * min(W, H) * (ar ** 0.5)
                        h = sc * min(W, H) / (ar ** 0.5)
                        for dy in np.arange(-r, r + 1e-6, fine):
                            for dx in np.arange(-r, r + 1e-6, fine):
                                x = cx + dx
                                y = cy + dy
                                loc.append([(x - w / 2) / W, (y - h / 2) / H,
                                            (x + w / 2) / W, (y + h / 2) / H])
                loc = np.asarray(loc, dtype="float32")
                parts_b.append(loc)
            if not parts_b:
                boxes_l.append(np.zeros((0, 4), dtype="float32"))
                scores_l.append(np.zeros(0, dtype="float32"))
                labels_l.append(np.zeros(0, dtype="int64"))
                n_cand.append(0)
                continue
            allb = np.concatenate(parts_b, axis=0)
            allb = np.clip(allb, 0, 1)
            pp = verifier.score(im, allb)
            boxes_l.append(allb)
            scores_l.append(1.0 - pp[:, BG_CLASS])
            labels_l.append(pp[:, :len(CN)].argmax(-1))
            n_cand.append(len(allb))
        for th in (0.5, 0.7):
            pr = score_to_preds(boxes_l, scores_l, labels_l, min_score=th)
            r = eval_preds(pr, gts)
            npb = sum(len(p["labels"]) for p in pr) / len(pr)
            print(f"    粗{coarse}/细{fine}/半径{rad}: 阈值{th:.1f} mAP@0.5 = {r['map']:.4f}  "
                  f"候选 {np.mean(n_cand):.0f}/图  框/图 {npb:.1f}")

    # ---------- C. 检测器候选 + 验证器重打分 ----------
    print("\n[C] 检测器候选 + 验证器重打分")
    for name in ("yolov3", "ppyoloe_s"):
        try:
            cfg = utils.load_config(f"src/det/configs/{name}.yml")
            model = data_mod.build_model(cfg)
            model.set_state_dict(paddle.load(
                str(utils.ROOT / f"results/weights/{name}_best.pdparams")))
            model.eval()
            _, loader = data_mod.build_loader(
                utils.ROOT, args.split, int(cfg["data"]["im_size"]),
                max(int(cfg["train"]["batch_size"]), 8), augment=False, shuffle=False)
            cand_l, n_cand = [], []
            with paddle.no_grad():
                for batch in loader:
                    raw = model(batch["image"])
                    if name == "yolov3":
                        dec = [decode_yolov3([raw[i]], [model.anchors_norm[i]],
                                             [model.strides[i]], model.num_classes,
                                             int(cfg["data"]["im_size"]))
                               for i in range(len(raw))]
                        allp = paddle.concat(dec, axis=1)
                        box = allp[..., 0:4].numpy()
                        obj = allp[..., 4].numpy()
                        cls = allp[..., 5:].numpy()
                        for bi in range(box.shape[0]):
                            sc = obj[bi] * cls[bi].max(-1)
                            o = np.argsort(-sc)[:400]
                            cand_l.append(box[bi][o])
                            n_cand.append(len(o))
                    else:
                        # PP-YOLOE 用自带的 _flatten + Distance2BBox 解码
                        from core.det_ops import Distance2BBox
                        cl, rg, anch, _st = model._flatten(raw)
                        scores = paddle.sigmoid(cl)
                        d2b = Distance2BBox(model.reg_max)
                        anc = paddle.expand(anch.unsqueeze(0), [cl.shape[0], anch.shape[0], 2])
                        boxes = d2b(rg, anc) / float(model.im_size)
                        bn = boxes.numpy()
                        sn = scores.numpy()
                        for bi in range(bn.shape[0]):
                            sc = sn[bi].max(-1)
                            o = np.argsort(-sc)[:400]
                            cand_l.append(bn[bi][o])
                            n_cand.append(len(o))
            # 验证器重打分
            sb, ss, sl = [], [], []
            for i, im in enumerate(imgs):
                cb = cand_l[i] if i < len(cand_l) else np.zeros((0, 4), dtype="float32")
                if len(cb) == 0:
                    sb.append(cb); ss.append(np.zeros(0, dtype="float32"))
                    sl.append(np.zeros(0, dtype="int64")); continue
                pp = verifier.score(im, cb)
                sb.append(cb)
                ss.append(1.0 - pp[:, BG_CLASS])
                sl.append(pp[:, :len(CN)].argmax(-1))
            for th in (0.5, 0.7):
                pr = score_to_preds(sb, ss, sl, min_score=th)
                r = eval_preds(pr, gts)
                npb = sum(len(p["labels"]) for p in pr) / len(pr)
                print(f"    {name:10s} 候选{np.mean(n_cand):.0f}/图 阈值{th:.1f}: "
                      f"mAP@0.5 = {r['map']:.4f}   框/图 {npb:.1f}")
        except Exception as e:
            print(f"    {name} 失败: {str(e)[:90]}")

    print("\n" + "=" * 80)
    print("参考：滑窗 IoU>=0.5 上限 70.1%、验证器判类 97.35%、判缺陷召回 95.1%")
    print("=> 若某策略 mAP 明显提升，说明它成功让正确框浮到了靠前位置")
    print("=" * 80)


if __name__ == "__main__":
    main()
