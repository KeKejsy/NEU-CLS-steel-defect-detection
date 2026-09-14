"""成员 C · 用 k-means 统计本数据集的目标框尺寸，生成定制 anchor

为什么需要这个脚本：
    YOLOv3 官方 anchor 是针对 COCO 数据集设计的，而 COCO 的目标普遍较小。
    NEU-DET 的缺陷框平均占全图 17.45%，属于「大框为主」的分布，
    直接套用官方 anchor 会导致：小尺寸 anchor 完全用不上、大框被迫分配给不合适的 anchor，
    宽高回归的目标值一开始就偏得很远。

    实测（训练集 2916 个框）官方 9 个 anchor 的分配占比：
        (10,13) 0.0%   (16,30) 0.0%   (33,23) 0.0%      <- 最小的 3 个完全没被用到
        (30,61) 5.6%   (62,45) 3.6%   (59,119) 23.0%
        (116,90) 17.9% (156,198) 37.8% (373,326) 12.1%
        每个框的最佳 anchor 宽高 IoU 平均只有 0.567

用法：
    python src/det/tools/kmeans_anchors.py                 # 打印统计并生成推荐 anchor
    python src/det/tools/kmeans_anchors.py --k 9 --write   # 把结果写进配置建议文件

产出：
    results/metrics/anchor_kmeans.json   统计结果 + 推荐 anchor（可粘进 yml）
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import utils  # noqa: E402

# YOLOv3 官方 anchor（像素，基准 416），用于对比
YOLOV3_OFFICIAL = [[10, 13], [16, 30], [33, 23],
                   [30, 61], [62, 45], [59, 119],
                   [116, 90], [156, 198], [373, 326]]


def load_wh(split="train"):
    """读某个划分的全部目标框宽高（归一化到 [0,1]）"""
    ann_dir = utils.ROOT / "dataset" / "det" / "Annotations"
    names = [ln.strip() for ln in
             (utils.ROOT / "dataset" / "det" / "ImageSets" / "Main" / f"{split}.txt")
             .read_text(encoding="utf-8").splitlines() if ln.strip()]
    wh = []
    for n in names:
        for obj in ET.parse(ann_dir / f"{n}.xml").getroot().findall("object"):
            bb = obj.find("bndbox")
            w = (float(bb.findtext("xmax")) - float(bb.findtext("xmin"))) / 200.0
            h = (float(bb.findtext("ymax")) - float(bb.findtext("ymin"))) / 200.0
            if w > 0 and h > 0:
                wh.append([w, h])
    return np.asarray(wh, dtype=np.float64)


def kmeans_iou(wh, k=9, iters=100, seed=2026):
    """按「1 - 宽高 IoU」为距离做 k-means（YOLOv3 论文的做法）

    度量用 IoU 而不是欧氏距离：宽高比不同的框（如细长划痕 vs 近方形麻点）
    在欧氏距离下可能被误判为相近，而 IoU 对比例差异更敏感。
    """
    rng = np.random.RandomState(seed)
    n = len(wh)
    centers = wh[rng.choice(n, k, replace=False)].copy()

    for _ in range(iters):
        # (n,k) 的宽高 IoU
        inter = np.minimum(wh[:, None, 0], centers[None, :, 0]) * \
                np.minimum(wh[:, None, 1], centers[None, :, 1])
        union = wh[:, None, 0] * wh[:, None, 1] + \
                centers[None, :, 0] * centers[None, :, 1] - inter
        iou = inter / np.clip(union, 1e-12, None)
        assign = iou.argmax(axis=1)

        new_centers = centers.copy()
        for j in range(k):
            sel = assign == j
            if sel.sum() > 0:
                # 用 IoU 意义下的中位数比均值更稳（对极端长条框不敏感）
                new_centers[j] = np.median(wh[sel], axis=0)
        if np.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers

    inter = np.minimum(wh[:, None, 0], centers[None, :, 0]) * \
            np.minimum(wh[:, None, 1], centers[None, :, 1])
    union = wh[:, None, 0] * wh[:, None, 1] + \
            centers[None, :, 0] * centers[None, :, 1] - inter
    iou = inter / np.clip(union, 1e-12, None)
    assign = iou.argmax(axis=1)
    biou = iou.max(axis=1)

    # 按面积从小到大排序，便于分给 stride 32/16/8 三个尺度
    order = np.argsort(centers[:, 0] * centers[:, 1])
    centers = centers[order]
    remap = {old: new for new, old in enumerate(order)}
    assign = np.array([remap[a] for a in assign])

    return centers, assign, biou


def evaluate_anchors(wh, anchors_norm):
    """给一组 anchor，算每个框的最佳宽高 IoU 与各组分配占比"""
    a = np.asarray(anchors_norm, dtype=np.float64)
    inter = np.minimum(wh[:, None, 0], a[None, :, 0]) * \
            np.minimum(wh[:, None, 1], a[None, :, 1])
    union = wh[:, None, 0] * wh[:, None, 1] + a[None, :, 0] * a[None, :, 1] - inter
    iou = inter / np.clip(union, 1e-12, None)
    best = iou.argmax(axis=1)
    return {
        "mean_best_iou": float(iou.max(axis=1).mean()),
        "assign_ratio": [float((best == j).mean()) for j in range(len(a))],
    }


def main():
    ap = argparse.ArgumentParser(description="k-means 生成定制 anchor")
    ap.add_argument("--k", type=int, default=9, help="anchor 数量（YOLOv3 为 9）")
    ap.add_argument("--split", default="train", help="用哪个划分统计（默认 train）")
    ap.add_argument("--write", action="store_true", help="把推荐 anchor 写入 json")
    args = ap.parse_args()

    wh = load_wh(args.split)
    print("=" * 72)
    print(f"成员 C · 目标框尺寸统计与 k-means 定制 anchor    （{args.split} 划分）")
    print("=" * 72)
    print(f"目标框总数 {len(wh)}")
    print(f"归一化宽: 均值 {wh[:,0].mean():.3f}  中位 {np.median(wh[:,0]):.3f}  "
          f"范围 [{wh[:,0].min():.3f}, {wh[:,0].max():.3f}]")
    print(f"归一化高: 均值 {wh[:,1].mean():.3f}  中位 {np.median(wh[:,1]):.3f}  "
          f"范围 [{wh[:,1].min():.3f}, {wh[:,1].max():.3f}]")
    print(f"宽高比:   均值 {(wh[:,0]/wh[:,1]).mean():.2f}  "
          f"范围 [{(wh[:,0]/wh[:,1]).min():.2f}, {(wh[:,0]/wh[:,1]).max():.2f}]")

    # ---- 官方 anchor 的表现 ----
    official_norm = np.asarray(YOLOV3_OFFICIAL, dtype=np.float64) / 416.0
    off = evaluate_anchors(wh, official_norm)
    print(f"\n[对比基准] YOLOv3 官方 anchor（416 基准）")
    print(f"  每个框的最佳宽高 IoU 平均 = {off['mean_best_iou']:.3f}")
    for j, (aw, ah) in enumerate(YOLOV3_OFFICIAL):
        print(f"    ({aw:4d},{ah:4d})  分配占比 {off['assign_ratio'][j]*100:5.1f}%")

    # ---- k-means 定制 ----
    centers, assign, biou = kmeans_iou(wh, k=args.k)
    print(f"\n[k-means 定制] k={args.k}")
    print(f"  每个框的最佳宽高 IoU 平均 = {biou.mean():.3f}"
          f"   （比官方提升 {biou.mean() - off['mean_best_iou']:+.3f}）")
    for j, c in enumerate(centers):
        ratio = (assign == j).mean()
        print(f"    ({c[0]*416:6.1f},{c[1]*416:6.1f})  分配占比 {ratio*100:5.1f}%")
    print(f"  像素值（基准 416）：{[ [int(round(c[0]*416)), int(round(c[1]*416))] for c in centers ]}")
    print(f"  归一化值：{[ [round(float(c[0]),4), round(float(c[1]),4)] for c in centers ]}")

    # 按 YOLOv3 的三尺度分组：面积升序，每 3 个一组给 stride 8/16/32
    pix = [[int(round(c[0] * 416)), int(round(c[1] * 416))] for c in centers]
    grouped = [pix[0:3], pix[3:6], pix[6:9]] if args.k == 9 else None

    result = {
        "split": args.split,
        "num_boxes": int(len(wh)),
        "box_wh_mean_norm": [float(wh[:, 0].mean()), float(wh[:, 1].mean())],
        "official": {"anchors_px_416": YOLOV3_OFFICIAL,
                     "mean_best_wh_iou": off["mean_best_iou"],
                     "assign_ratio": off["assign_ratio"]},
        "kmeans": {"k": args.k,
                   "anchors_px_416": pix,
                   "anchors_norm": [[float(c[0]), float(c[1])] for c in centers],
                   "mean_best_wh_iou": float(biou.mean()),
                   "assign_ratio": [float((assign == j).mean()) for j in range(args.k)],
                   "grouped_by_stride": grouped},
    }
    out = utils.resolve_dir("results/metrics") / "anchor_kmeans.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入：{out.relative_to(utils.ROOT)}")
    print("（可把 anchors_px_416 填进 configs 的 model.anchors 做对比实验）")
    print("=" * 72)


if __name__ == "__main__":
    main()
