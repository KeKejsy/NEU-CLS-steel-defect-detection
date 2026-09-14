"""成员 C · 迭代诊断：先量化瓶颈在哪一环，再决定改什么

用法：
    python src/det/tools/bottleneck.py --split val --limit 120

回答四个问题（每个都对应一个可改进的环节）：

    1. **两种候选来源的召回上限**：滑窗 vs YOLOv3 检测器，谁更能覆盖 GT？
       —— 决定第二阶段用谁产生候选。
    2. **候选的定位质量分布**：每个 GT 的最佳候选 IoU 分布。
       —— 若整体偏低，定位是硬瓶颈（要么加框回归、要么加密网格）。
    3. **验证器的两件事拆开**：判「是不是缺陷」（二分类）与判「是哪一类」（6 分类）。
       —— 决定该优化验证器还是优化候选。
    4. **验证器打分后的精度损失**：GT 最佳候选在打分排序里排第几。
       —— 若排名很靠后，说明是排序/阈值问题而非定位问题。
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


def load_detector(name):
    cfg = utils.load_config(f"src/det/configs/{name}.yml")
    m = data_mod.build_model(cfg)
    m.set_state_dict(paddle.load(str(utils.ROOT / f"results/weights/{name}_best.pdparams")))
    m.eval()
    return m, cfg


def load_verifier():
    ver = DefectVerifier(num_classes=len(CN) + 1, in_size=96)
    ver.set_state_dict(paddle.load(str(utils.ROOT / "results/weights/verifier_best.pdparams")))
    ver.eval()
    return VerifierInfer(ver, batch=512)


def read_gts(ds):
    out = []
    for img_p, ann_p in ds.samples:
        gt = data_mod.parse_voc_xml(ann_p, CN)
        gtb = np.stack([np.array([x[1] - x[3] / 2, x[2] - x[4] / 2,
                                  x[1] + x[3] / 2, x[2] + x[4] / 2]) for x in gt]) \
            if len(gt) else np.zeros((0, 4), dtype="float32")
        out.append({"boxes": gtb,
                    "labels": gt[:, 0].astype("int64") if len(gt) else np.zeros(0, dtype="int64")})
    return out


@paddle.no_grad()
def detector_candidates(model, cfg, ds, max_keep=3000):
    """检测器全部候选（不截断），返回逐图 (boxes, score)"""
    _, loader = data_mod.build_loader(
        utils.ROOT, ds.split, int(cfg["data"]["im_size"]),
        max(int(cfg["train"]["batch_size"]), 8), augment=False, shuffle=False)
    out = []
    for batch in loader:
        raw = model(batch["image"])
        decoded = [decode_yolov3([raw[i]], [model.anchors_norm[i]], [model.strides[i]],
                                 model.num_classes, int(cfg["data"]["im_size"]))
                   for i in range(len(raw))]
        allp = paddle.concat(decoded, axis=1)
        box = allp[..., 0:4].numpy()
        obj = allp[..., 4].numpy()
        cls = allp[..., 5:].numpy()
        for bi in range(box.shape[0]):
            sc = obj[bi] * cls[bi].max(-1)
            if len(sc) > max_keep:
                o = np.argsort(-sc)[:max_keep]
                out.append((box[bi][o], sc[o]))
            else:
                out.append((box[bi], sc))
    return out


@paddle.no_grad()
def window_candidates(ds, verifier, stride=16):
    """滑窗候选 + 验证器打分，返回逐图 (wins, defect_prob, cls_prob(6 类), cls_id)"""
    out = []
    for i in range(len(ds)):
        img_p, _ = ds.samples[i]
        im = Image.open(img_p).convert("RGB")
        W, H = im.size
        wins = make_windows(W, H, stride=stride)
        prob = verifier.score(im, wins)
        out.append((wins, 1.0 - prob[:, BG_CLASS], prob[:, :len(CN)],
                    prob[:, :len(CN)].argmax(-1)))
    return out


def per_gt_best_iou(cand_boxes, gts):
    """每个 GT 在候选里的最佳 IoU"""
    vals = []
    for i, g in enumerate(gts):
        gb = np.asarray(g["boxes"])
        if len(gb) == 0:
            continue
        cb = cand_boxes[i]
        M = iou_matrix(gb, cb) if len(cb) else np.zeros((len(gb), 0))
        for j in range(len(gb)):
            vals.append(float(M[j].max()) if M.shape[1] else 0.0)
    return np.asarray(vals)


def main():
    ap = argparse.ArgumentParser(description="检测方案瓶颈诊断")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(2026)

    ds, _ = data_mod.build_loader(utils.ROOT, args.split, 416, 8,
                                  augment=False, shuffle=False)
    ds.samples = ds.samples[:args.limit]
    gts = read_gts(ds)
    n_gt = sum(len(g["labels"]) for g in gts)

    print("=" * 78)
    print(f"C · 瓶颈诊断    split={args.split}    {len(ds)} 张图 / {n_gt} 个 GT")
    print("=" * 78)

    verifier = load_verifier()
    wc = window_candidates(ds, verifier, args.stride)
    w_boxes = [c[0] for c in wc]

    print("\n[1] 候选来源的定位上限（每图候选数 / 各 IoU 阈值的召回上限）")
    print(f"  滑窗（stride {args.stride}）     : {np.mean([len(b) for b in w_boxes]):6.0f} 个/图")
    wbi = per_gt_best_iou(w_boxes, gts)
    for th in (0.3, 0.5, 0.7):
        print(f"      IoU>={th:.1f}: {(wbi >= th).mean()*100:5.1f}%   ", end="")
    print(f" | 最佳 IoU 均值 {wbi.mean():.3f}")

    det_data = {}
    for name in ("yolov3", "ppyoloe_s"):
        try:
            model, cfg = load_detector(name)
            dc = detector_candidates(model, cfg, ds)
            d_boxes = [c[0] for c in dc]
            det_data[name] = dc
            dbi = per_gt_best_iou(d_boxes, gts)
            print(f"  {name:10s}          : {np.mean([len(b) for b in d_boxes]):6.0f} 个/图")
            for th in (0.3, 0.5, 0.7):
                print(f"      IoU>={th:.1f}: {(dbi >= th).mean()*100:5.1f}%   ", end="")
            print(f" | 最佳 IoU 均值 {dbi.mean():.3f}")
        except Exception as e:
            print(f"  {name} 诊断失败: {str(e)[:70]}")

    # ---- 2) 验证器两件事拆开 ----
    print("\n[2] 验证器拆解（在 GT 框处评）")
    gt_boxes_all, gt_lab_all = [], []
    for i, g in enumerate(gts):
        for b, l in zip(g["boxes"], g["labels"]):
            gt_boxes_all.append((i, b))
            gt_lab_all.append(int(l))
    # 逐图批量打分
    bin_correct = []
    cls_correct = []
    for i in range(len(ds)):
        sel = [k for k, (ii, _) in enumerate(gt_boxes_all) if ii == i]
        if not sel:
            continue
        img_p, _ = ds.samples[i]
        im = Image.open(img_p).convert("RGB")
        boxes = np.stack([gt_boxes_all[k][1] for k in sel])
        prob = verifier.score(im, boxes)
        bg = prob[:, BG_CLASS]
        pred_cls = prob[:, :len(CN)].argmax(-1)
        for t, k in enumerate(sel):
            bin_correct.append(bg[t] < 0.5)                      # 判定为缺陷
            cls_correct.append(pred_cls[t] == gt_lab_all[k])     # 类别判对
    bin_correct = np.asarray(bin_correct)
    cls_correct = np.asarray(cls_correct)
    print(f"  判「是不是缺陷」（GT 框被认定为缺陷的比例，即召回）: {bin_correct.mean()*100:.1f}%")
    print(f"  判「是哪一类」   （类别正确率）                    : {cls_correct.mean()*100:.2f}%")
    print(f"  两者都对的联合比例                                : {(bin_correct & cls_correct).mean()*100:.2f}%")

    # ---- 3) 滑窗候选的排序质量 ----
    print("\n[3] 滑窗候选的排序质量（GT 最佳候选在缺陷分数里的排名）")
    ranks = []
    for i, g in enumerate(gts):
        gb = np.asarray(g["boxes"])
        if len(gb) == 0:
            continue
        _, dp, _, _ = wc[i]
        cb = w_boxes[i]
        if len(cb) == 0:
            continue
        M = iou_matrix(gb, cb)
        order = np.argsort(-dp)
        rank_of = np.empty(len(cb), dtype=int)
        rank_of[order] = np.arange(len(cb))
        for j in range(len(gb)):
            k = int(np.argmax(M[j]))
            if M[j][k] >= 0.5:
                ranks.append(int(rank_of[k]))
    if ranks:
        ranks = np.asarray(ranks)
        print(f"  可命中的 GT 数: {len(ranks)}")
        for q in (25, 50, 75, 90):
            print(f"    {q} 分位排名: {np.percentile(ranks, q):.0f}")
        print(f"    排名 <=50 的比例: {(ranks <= 50).mean()*100:.1f}%")
        print(f"    排名 <=200 的比例: {(ranks <= 200).mean()*100:.1f}%")
    else:
        print("  没有 IoU>=0.5 的候选，无法评估排序")

    print("\n[4] 判读建议")
    print("  - 若「判是不是缺陷」召回高但候选整体偏多 -> 是排序/阈值问题，可优化聚合与去重")
    print("  - 若候选的 IoU>=0.5 上限低 -> 是定位问题，需加框回归或更密的网格")
    print("  - 若检测器候选上限明显高于滑窗 -> 该用检测器做第一阶段")
    print("=" * 78)


if __name__ == "__main__":
    main()
