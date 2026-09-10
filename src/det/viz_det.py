"""成员 C · 检测可视化脚本

用法：
    python src/det/viz_det.py --model yolov3                    # 出全部三张图
    python src/det/viz_det.py --model yolov3 --figs samples     # 只出示例对比图
    python src/det/viz_det.py --model ppyoloe_s --split val

产出（results/figures/）：
    <name>_samples.png    6 类 × 若干张：左=标注真值（红框），右=模型预测（绿框）
    <name>_errors.png     错例分析：漏检（红）、误检（橙）、类别错（紫）
    <name>_size_ap.png    不同目标尺寸下的 mAP（小/中/大三档）

为什么值得出这三张图：
    报告里光有 mAP 数字说不清模型强在哪、弱在哪。示例图能直观看出
    细长缺陷（划痕 Sc）和弥散缺陷（龟裂 Cr）谁更难；错例图能区分是漏检还是误检；
    尺寸-AP 图能回答「小目标是不是瓶颈」，直接影响后续要不要加高分辨率分支。
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import paddle  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import data as data_mod  # noqa: E402
from core import utils  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ARCH_TO_CONFIG = {
    "yolov3": "src/det/configs/yolov3.yml",
    "ppyoloe_s": "src/det/configs/ppyoloe_s.yml",
}

# 文件名前缀 -> 类别代码，用于按类抽样本（与 A 的命名规范一致）
PREFIX_OF = {
    "Cr": "crazing_", "In": "inclusion_", "Pa": "patches_",
    "PS": "pitted_surface_", "RS": "rolled-in_scale_", "Sc": "scratches_",
}


def load_model(cfg, weights):
    wpath = Path(weights) if weights else utils.ROOT / f"results/weights/{cfg['name']}_best.pdparams"
    if not wpath.is_absolute():
        wpath = utils.ROOT / wpath
    if not wpath.exists():
        raise SystemExit(f"找不到权重 {wpath}，请先训练")
    model = data_mod.build_model(cfg)
    model.set_state_dict(paddle.load(str(wpath)))
    model.eval()
    print(f"已加载权重：{wpath.relative_to(utils.ROOT)}")
    return model


def infer_one(model, img_path, cfg):
    """单图推理，返回 (原图 PIL, [(像素 xyxy, 分数, 类别id, 类别名)])"""
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    size = int(cfg["data"]["im_size"])
    arr = np.asarray(im.resize((size, size), Image.BILINEAR)).astype("float32") / 255.0
    x = paddle.to_tensor(np.transpose(arr, (2, 0, 1))).unsqueeze(0)

    res = model.postprocess(model(x), im_shape=None,
                            score_threshold=float(cfg["viz"]["score_threshold"]),
                            nms_threshold=float(cfg["eval"]["nms_threshold"]),
                            top_k=int(cfg["eval"]["top_k"]))[0]
    boxes, scores, labels = res[0].numpy(), res[1].numpy(), res[2].numpy()
    out = []
    for b, s, l in zip(boxes, scores, labels):
        # 归一化坐标 -> 原图像素（数据集里 GT 也是归一化的，所以这里反变换回像素只为了画图）
        out.append(([b[0] * W, b[1] * H, b[2] * W, b[3] * H], float(s), int(l),
                    data_mod.CLASS_NAMES[int(l)]))
    return im, out


def _px_box(b, W, H, min_px=1.0):
    """归一化 xyxy -> 像素 xyxy，并保证画图合法

    为什么需要：后处理里的 clip_boxes_norm 允许最小边长 1e-3（归一化），
    换算到 200 像素的图上不到 1 像素，PIL 的 rectangle 会直接报
    "y1 must be greater than or equal to y0"。所以画图前统一做一次合法化：
    坐标夹进画面内，并保证 x2>=x1+min_px、y2>=y1+min_px。
    """
    x1, y1, x2, y2 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
    x1, x2 = max(0.0, min(x1, W)), max(0.0, min(x2, W))
    y1, y2 = max(0.0, min(y1, H)), max(0.0, min(y2, H))
    if x2 < x1 + min_px:
        x2 = min(W, x1 + min_px)
        x1 = max(0.0, x2 - min_px)
    if y2 < y1 + min_px:
        y2 = min(H, y1 + min_px)
        y1 = max(0.0, y2 - min_px)
    return [x1, y1, x2, y2]


def draw_boxes(im, items, color, width=2):
    im = im.copy()
    dr = ImageDraw.Draw(im)
    W, H = im.size
    for it in items:
        if isinstance(it, tuple) and len(it) == 2:
            box, lab = it
        else:
            box, _s, _l, lab = it
        box = _px_box(box, W, H)
        dr.rectangle(box, outline=color, width=width)
        if lab:
            dr.text((box[0] + 2, max(box[1] - 11, 0)), str(lab), fill=color)
    return im


def gt_boxes_from_xml(xml_path):
    """读原始 VOC 标注 -> [(像素 xyxy, 类别名)]"""
    root = ET.parse(xml_path).getroot()
    out = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        out.append(([float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")],
                    obj.findtext("name")))
    return out


def fig_samples(model, cfg, split, out_path, per_class=2):
    """6 类 × per_class 张：左=真值（红），右=预测（绿）"""
    cls_names = data_mod.CLASS_NAMES
    names = [ln.strip() for ln in
             (utils.ROOT / "dataset" / "det" / "ImageSets" / "Main" / f"{split}.txt")
             .read_text(encoding="utf-8").splitlines() if ln.strip()]

    fig, axes = plt.subplots(len(cls_names), per_class * 2,
                             figsize=(2.0 * per_class * 2, 2.0 * len(cls_names)), dpi=150)
    axes = np.atleast_2d(axes)
    for r, c in enumerate(cls_names):
        picked = [n for n in names if n.startswith(PREFIX_OF[c])][:per_class]
        for j in range(per_class):
            ax_gt, ax_pd = axes[r][2 * j], axes[r][2 * j + 1]
            ax_gt.axis("off")
            ax_pd.axis("off")
            if j >= len(picked):
                continue
            n = picked[j]
            img_p = utils.ROOT / "dataset" / "det" / "JPEGImages" / f"{n}.jpg"
            xml_p = utils.ROOT / "dataset" / "det" / "Annotations" / f"{n}.xml"
            im = Image.open(img_p).convert("RGB")
            ax_gt.imshow(draw_boxes(im, gt_boxes_from_xml(xml_p), (220, 60, 40)))
            _im2, preds = infer_one(model, img_p, cfg)
            ax_pd.imshow(draw_boxes(im, preds, (20, 158, 117)))
            if j == 0:
                ax_gt.set_title(f"{c} 真值", fontsize=9)
                ax_pd.set_title(f"{c} 预测", fontsize=9)
    fig.suptitle(f"检测示例：左=标注真值（红），右=模型预测（绿）  划分={split}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _iou(a, b):
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:], b[2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[0] * wh[1]
    aa = max((a[2] - a[0]) * (a[3] - a[1]), 0)
    ab = max((b[2] - b[0]) * (b[3] - b[1]), 0)
    return inter / max(aa + ab - inter, 1e-9)


def collect_errors(preds, gts, iou_th=0.5):
    """逐图统计错例，返回按错误数降序排列的列表"""
    out = []
    for i in range(len(preds)):
        pb = np.asarray(preds[i]["boxes"])
        pl = np.asarray(preds[i]["labels"]).astype(int)
        gb = np.asarray(gts[i]["boxes"])
        gl = np.asarray(gts[i]["labels"]).astype(int)
        used = np.zeros(len(gb), dtype=bool)
        miss, false_det, wrong_cls = [], [], []
        for b, l in zip(pb, pl):
            best, bi = 0.0, -1
            for j in range(len(gb)):
                if used[j]:
                    continue
                iou = _iou(b, gb[j])
                if iou > best:
                    best, bi = iou, j
            if best >= iou_th:
                used[bi] = True
                if gl[bi] != l:
                    wrong_cls.append((b, l, int(gl[bi])))
            else:
                false_det.append((b, l))
        for j in range(len(gb)):
            if not used[j]:
                miss.append((gb[j], int(gl[j])))
        n_err = len(miss) + len(false_det) + len(wrong_cls)
        if n_err > 0:
            out.append({"idx": i, "n_err": n_err, "miss": miss,
                        "false": false_det, "wrong": wrong_cls})
    out.sort(key=lambda d: -d["n_err"])
    return out


def fig_errors(errors, ds, out_path, k=24):
    """错例图：红=漏检，橙=误检，紫=类别判断错误"""
    picked = errors[:k]
    if not picked:
        return 0
    cols = 6
    rows = (len(picked) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.0 * rows), dpi=150)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")

    for pos, e in enumerate(picked):
        ax = axes[pos // cols][pos % cols]
        img_p, _ann_p = ds.samples[e["idx"]]
        im = Image.open(img_p).convert("RGB")
        W, H = im.size
        dr = ImageDraw.Draw(im)
        for b, _l in e["miss"]:
            dr.rectangle(_px_box(b, W, H), outline=(220, 60, 40), width=2)
        for b, _l in e["false"]:
            dr.rectangle(_px_box(b, W, H), outline=(230, 140, 20), width=2)
        for b, _pl, _gl in e["wrong"]:
            dr.rectangle(_px_box(b, W, H), outline=(120, 80, 200), width=2)
        ax.imshow(im)
        ax.set_title(f"漏{len(e['miss'])} 误{len(e['false'])} 错{len(e['wrong'])}", fontsize=7)

    fig.suptitle("错例分析：红=漏检  橙=误检  紫=类别判断错误", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return len(picked)


def fig_size_ap(preds, gts, class_names, out_path, iou_th=0.5):
    """不同目标尺寸下的 mAP：把 GT 按面积占比分成小/中/大三档分别评估

    分档时**只过滤 GT，不过滤预测**：若把预测也按尺寸筛掉，漏检就统计不出来，
    该档 AP 会被虚高。这样得到的是「该尺寸的目标有多容易被检出」的真实结论。
    """
    bins = [("小 (<5%)", 0.0, 0.05), ("中 (5%~20%)", 0.05, 0.20), ("大 (>20%)", 0.20, 1.01)]
    results = {}
    for label, lo, hi in bins:
        sub_preds, sub_gts = [], []
        for i in range(len(gts)):
            gb = np.asarray(gts[i]["boxes"])
            gl = np.asarray(gts[i]["labels"])
            if len(gb) == 0:
                sub_gts.append({"boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=int)})
            else:
                area = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
                keep = (area >= lo) & (area < hi)
                sub_gts.append({"boxes": gb[keep], "labels": gl[keep].astype(int)})
            sub_preds.append(preds[i])
        if sum(len(g["labels"]) for g in sub_gts) == 0:
            results[label] = 0.0
            continue
        r = utils.evaluate_map(sub_preds, sub_gts, len(class_names), class_names,
                               iou_threshold=iou_th)
        results[label] = r["map"]

    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=150)
    labels = list(results.keys())
    vals = [results[k] for k in labels]
    bars = ax.bar(labels, vals, color=["#D85A30", "#378ADD", "#1D9E75"], width=0.55)
    ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(0, max(max(vals) * 1.3, 0.1))
    ax.set_title("不同目标尺寸下的 mAP@0.5")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return results


def main():
    ap = argparse.ArgumentParser(description="C · 检测可视化")
    ap.add_argument("--model", required=True, choices=sorted(ARCH_TO_CONFIG))
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="默认 val：可视化属于调参行为，按铁律不要拿测试集挑图")
    ap.add_argument("--figs", default="samples,errors,size_ap",
                    help="要出的图，逗号分隔：samples / errors / size_ap")
    ap.add_argument("--score-threshold", type=float, default=0.3,
                    help="可视化用的分数阈值（高于评估阈值，画面更干净）")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    cfg = utils.load_config(args.config or ARCH_TO_CONFIG[args.model])
    cfg["viz"] = {"score_threshold": args.score_threshold}
    utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(2026)

    fig_dir = utils.resolve_dir("results/figures")
    name = cfg["name"]
    want = {s.strip() for s in args.figs.split(",") if s.strip()}

    print("=" * 70)
    print(f"成员 C · 检测可视化    {name}    划分={args.split}")
    print("=" * 70)
    model = load_model(cfg, args.weights)

    if "samples" in want:
        p = fig_dir / f"{name}_samples.png"
        fig_samples(model, cfg, args.split, p)
        print(f"  [1/3] 示例对比图 -> {p.relative_to(utils.ROOT)}")

    ds = preds = gts = None
    if "errors" in want or "size_ap" in want:
        ds, loader = data_mod.build_loader(
            utils.ROOT, args.split, int(cfg["data"]["im_size"]),
            max(int(cfg["train"]["batch_size"]), 8), augment=False, shuffle=False)
        preds, gts = data_mod.detect_loader(
            model, loader, int(cfg["num_classes"]),
            score_threshold=args.score_threshold,
            nms_threshold=float(cfg["eval"]["nms_threshold"]),
            top_k=int(cfg["eval"]["top_k"]))
        print(f"  已在 {len(ds)} 张图上推理完毕")

    if "errors" in want:
        p = fig_dir / f"{name}_errors.png"
        errs = collect_errors(preds, gts, iou_th=float(cfg["eval"]["iou_threshold"]))
        n = fig_errors(errs, ds, p)
        total_miss = sum(len(e["miss"]) for e in errs)
        total_false = sum(len(e["false"]) for e in errs)
        total_wrong = sum(len(e["wrong"]) for e in errs)
        print(f"  [2/3] 错例分析图 -> {p.relative_to(utils.ROOT)}"
              + (f"（{n} 张有错误的图；漏检 {total_miss}、误检 {total_false}、类别错 {total_wrong}）"
                 if n else "（未发现错例）"))

    if "size_ap" in want and preds is not None:
        p = fig_dir / f"{name}_size_ap.png"
        r = fig_size_ap(preds, gts, ds.class_names, p,
                        iou_th=float(cfg["eval"]["iou_threshold"]))
        print(f"  [3/3] 尺寸-AP 图 -> {p.relative_to(utils.ROOT)}")
        for k, v in r.items():
            print(f"        {k}: mAP@0.5 = {v:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
