"""成员 C · 诊断脚本：定位 mAP 偏低的真正原因

用途（排查用，不属于交付流程）：
    python src/det/tools/diagnose.py --model yolov3 --split val

它会把「模型到底学到了什么」拆开成可量化的几项，避免对着一个 mAP 数字瞎猜：

    1. 预测框与 GT 的 IoU 分布（不看分数，只看框本身准不准）
    2. 各 IoU 阈值下的 mAP（判断是「大致对但不精确」还是「完全没学会」）
    3. GT 位置与最高分预测位置是否一致（判断是不是位置选错了）
    4. 各类别的召回情况

排查经验（本项目实测）：
    - 若 IoU 分布整体很低（<0.3），说明回归本身没学起来；
    - 若 IoU 分布尚可但 mAP 低，说明是分数排序/正负样本判别的问题；
    - 若训练集 IoU 远高于验证集，说明过拟合，要降容量或加增强。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import paddle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import data as data_mod  # noqa: E402
from core import utils  # noqa: E402

ARCH_TO_CONFIG = {
    "yolov3": "src/det/configs/yolov3.yml",
    "ppyoloe_s": "src/det/configs/ppyoloe_s.yml",
}


def iou_matrix(a, b):
    """a:(N,4) b:(M,4) -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa[:, None] + ab[None, :] - inter, 1e-9, None)


def main():
    ap = argparse.ArgumentParser(description="检测诊断")
    ap.add_argument("--model", required=True, choices=sorted(ARCH_TO_CONFIG))
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    cfg = utils.load_config(args.config or ARCH_TO_CONFIG[args.model])
    utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(2026)
    nc = int(cfg["num_classes"])

    wp = Path(args.weights) if args.weights else utils.ROOT / f"results/weights/{cfg['name']}_best.pdparams"
    if not wp.is_absolute():
        wp = utils.ROOT / wp
    model = data_mod.build_model(cfg)
    model.set_state_dict(paddle.load(str(wp)))
    model.eval()

    ds, loader = data_mod.build_loader(
        utils.ROOT, args.split, int(cfg["data"]["im_size"]),
        max(int(cfg["train"]["batch_size"]), 8), augment=False, shuffle=False)
    preds, gts = data_mod.detect_loader(
        model, loader, nc,
        score_threshold=float(cfg["eval"]["score_threshold"]),
        nms_threshold=float(cfg["eval"]["nms_threshold"]),
        top_k=int(cfg["eval"]["top_k"]))

    print("=" * 74)
    print(f"C · 检测诊断    {cfg['name']}    划分={args.split}    样本={len(ds)}")
    print("=" * 74)

    n_gt = sum(len(g["labels"]) for g in gts)
    n_pd = sum(len(p["labels"]) for p in preds)
    print(f"标注框 {n_gt} 个，预测框 {n_pd} 个（平均每图 {n_pd/max(len(preds),1):.1f}）")

    # ---- 1) 每个 GT 与全部预测框的最大 IoU（不看分数）----
    best_iou, best_cls_ok = [], []
    for i in range(len(preds)):
        gb, gl = np.asarray(gts[i]["boxes"]), np.asarray(gts[i]["labels"]).astype(int)
        pb, pl = np.asarray(preds[i]["boxes"]), np.asarray(preds[i]["labels"]).astype(int)
        if len(gb) == 0:
            continue
        M = iou_matrix(gb, pb) if len(pb) else np.zeros((len(gb), 0))
        for j in range(len(gb)):
            if M.shape[1] == 0:
                best_iou.append(0.0)
                best_cls_ok.append(False)
                continue
            k = int(np.argmax(M[j]))
            best_iou.append(float(M[j, k]))
            best_cls_ok.append(bool(pl[k] == gl[j]))
    best_iou = np.asarray(best_iou)
    best_cls_ok = np.asarray(best_cls_ok)

    print("\n[1] 定位质量（每个 GT 与全部预测的最佳 IoU，与分数无关）")
    if len(best_iou):
        for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75):
            print(f"    GT 能找到 IoU>={th:.2f} 的预测的比例: "
                  f"{(best_iou >= th).mean()*100:5.1f}%")
        print(f"    IoU 均值 {best_iou.mean():.3f}  中位 {np.median(best_iou):.3f}  "
              f"最大 {best_iou.max():.3f}")

    # ---- 2) 不同 IoU 阈值下的 mAP ----
    print("\n[2] 不同 IoU 阈值下的 mAP（判断是「不精确」还是「完全没学会」）")
    maps = {}
    for th in (0.1, 0.2, 0.3, 0.4, 0.5):
        r = utils.evaluate_map(preds, gts, nc, ds.class_names, iou_threshold=th)
        maps[th] = r["map"]
        print(f"    mAP@{th:.1f} = {r['map']:.4f}")

    # ---- 3) 位置是否选错：最高分预测是否落在 GT 附近 ----
    print("\n[3] 位置选择（最高分预测的中心是否落在某个 GT 内）")
    hit = tot = 0
    for i in range(len(preds)):
        gb = np.asarray(gts[i]["boxes"])
        pb, ps = np.asarray(preds[i]["boxes"]), np.asarray(preds[i]["scores"])
        if len(pb) == 0 or len(gb) == 0:
            continue
        k = int(np.argmax(ps))
        cx, cy = (pb[k, 0] + pb[k, 2]) / 2, (pb[k, 1] + pb[k, 3]) / 2
        tot += 1
        inside = ((cx >= gb[:, 0]) & (cx <= gb[:, 2]) &
                  (cy >= gb[:, 1]) & (cy <= gb[:, 3]))
        hit += int(inside.any())
    if tot:
        print(f"    最高分预测落在某 GT 框内的图: {hit}/{tot} = {hit/tot*100:.1f}%")

    # ---- 4) 得分分布 ----
    all_scores = np.concatenate([p["scores"] for p in preds]) if n_pd else np.zeros(0)
    if len(all_scores):
        print("\n[4] 预测分数分布（评估阈值 "
              f"{cfg['eval']['score_threshold']}）")
        for q in (50, 90, 99, 100):
            print(f"    {q} 分位: {np.percentile(all_scores, q):.4f}")
        for th in (0.1, 0.3, 0.5, 0.7):
            print(f"    分数 >= {th}: {(all_scores >= th).sum()} 个")

    # ---- 5) 结论提示 ----
    print("\n[5] 判读")
    if len(best_iou) and best_iou.mean() < 0.25:
        print("    → 定位本身没学起来，应检查回归目标/输入尺度/训练轮数")
    elif maps[0.5] < 0.05 <= maps[0.3]:
        print("    → 框大致对但不够精确，mAP@0.5 对定位精度很敏感（本数据集框大，"
              "小偏差即掉出 0.5）")
    elif maps[0.5] < 0.05:
        print("    → 定位与排序都需改善")
    else:
        print("    → 指标正常")
    print("=" * 74)


if __name__ == "__main__":
    main()
