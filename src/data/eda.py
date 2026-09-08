"""成员 A · 步骤4：数据探索（EDA），出图给报告用

作用：
    1. 每类缺陷在 train / val / test 中的数量柱状图
    2. 6 类缺陷示例图（带检测框）
    3. 整体系灰度直方图 + 每类平均灰度
    4. 目标框宽高分布散点图

输入：dataset/cls/{train,val,test}.txt、dataset/det/JPEGImages/、dataset/det/Annotations/
输出：results/figures/ 下 4 张 png（300dpi）+ results/metrics/data_stats.json

用法：
    python src/data/eda.py
    python src/data/eda.py --samples 4
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"


def read_split(name):
    p = ROOT / "dataset" / "cls" / f"{name}.txt"
    if not p.exists():
        raise SystemExit(f"找不到 {p}，请先运行 split_dataset.py")
    return [ln.strip().rsplit(" ", 1) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def img_path(rel):
    return ROOT / "dataset" / "cls" / "images" / rel


def main():
    ap = argparse.ArgumentParser(description="数据探索与可视化")
    ap.add_argument("--samples", type=int, default=4, help="每类抽几张示例图")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MET_DIR.mkdir(parents=True, exist_ok=True)

    splits = {s: read_split(s) for s in ("train", "val", "test")}

    # ---------- 图1：每类 train/val/test 数量 ----------
    counts = {s: {c: 0 for c in CLASS_NAMES} for s in splits}
    for s, rows in splits.items():
        for rel, _lab in rows:
            counts[s][rel.split("/")[0]] += 1

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    x = np.arange(len(CLASS_NAMES))
    w = 0.26
    colors = ["#378ADD", "#1D9E75", "#D85A30"]
    for i, s in enumerate(("train", "val", "test")):
        vals = [counts[s][c] for c in CLASS_NAMES]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=s, color=colors[i])
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("图片数")
    ax.set_title("各类缺陷在 train / val / test 中的数量（7 : 1.5 : 1.5）")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=300)
    plt.close(fig)
    print("已保存 class_distribution.png")

    # ---------- 图2：6 类示例图（带框） ----------
    ann_dir = ROOT / "dataset" / "det" / "Annotations"
    n = args.samples
    fig, axes = plt.subplots(len(CLASS_NAMES), n, figsize=(1.7 * n, 1.7 * len(CLASS_NAMES)), dpi=150)
    for r, c in enumerate(CLASS_NAMES):
        items = [rel for rel, _ in splits["train"] if rel.startswith(c + "/")][:n]
        for j in range(n):
            ax = axes[r][j] if n > 1 else axes[r]
            ax.axis("off")
            if j >= len(items):
                continue
            p = img_path(items[j])
            im = Image.open(p).convert("RGB")
            xml = ann_dir / (Path(items[j]).stem + ".xml")
            if xml.exists():
                dr = ImageDraw.Draw(im)
                for obj in ET.parse(xml).getroot().findall("object"):
                    bb = obj.find("bndbox")
                    box = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
                    dr.rectangle(box, outline=(220, 60, 40), width=2)
            ax.imshow(im)
            if j == 0:
                ax.set_title(c, fontsize=11)
    fig.suptitle("六类缺陷示例（红框为标注）", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "samples_grid.png", dpi=300)
    plt.close(fig)
    print("已保存 samples_grid.png")

    # ---------- 图3：灰度直方图与每类均值 ----------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    allpix = []
    for c in CLASS_NAMES:
        vals = []
        for rel, _ in splits["train"]:
            if rel.startswith(c + "/"):
                vals.append(np.asarray(Image.open(img_path(rel)).convert("L"), dtype=np.float32))
        arr = np.concatenate([v.ravel() for v in vals])
        allpix.append(arr)
        axes[1].bar(c, arr.mean(), color="#378ADD")
    axes[0].hist(np.concatenate([a[::37] for a in allpix]), bins=64, color="#534AB7")
    axes[0].set_title("整体灰度分布")
    axes[0].set_xlabel("灰度值")
    axes[0].set_ylabel("像素数")
    axes[1].set_title("各类平均灰度")
    axes[1].set_ylabel("平均灰度值")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gray_hist.png", dpi=300)
    plt.close(fig)
    print("已保存 gray_hist.png")

    # ---------- 图4：目标框宽高分布 ----------
    ws, hs, cls_of_box = [], [], []
    if ann_dir.is_dir():
        for xml in ann_dir.glob("*.xml"):
            for obj in ET.parse(xml).getroot().findall("object"):
                bb = obj.find("bndbox")
                xmin = float(bb.findtext("xmin")); ymin = float(bb.findtext("ymin"))
                xmax = float(bb.findtext("xmax")); ymax = float(bb.findtext("ymax"))
                ws.append(xmax - xmin); hs.append(ymax - ymin); cls_of_box.append(obj.findtext("name"))
    if ws:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
        axes[0].scatter(ws, hs, s=6, alpha=0.35, color="#0F6E56")
        axes[0].set_xlabel("框宽 (px)"); axes[0].set_ylabel("框高 (px)")
        axes[0].set_title(f"目标框尺寸分布（共 {len(ws)} 个）")
        axes[0].set_xlim(0, 200); axes[0].set_ylim(0, 200)
        area = np.array(ws) * np.array(hs)
        axes[1].hist(area / (200 * 200) * 100, bins=50, color="#BA7517")
        axes[1].set_xlabel("框面积占全图比例 (%)"); axes[1].set_ylabel("框数")
        axes[1].set_title("目标框面积占比")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "bbox_size_dist.png", dpi=300)
        plt.close(fig)
        print("已保存 bbox_size_dist.png")
    else:
        print("（跳过 bbox_size_dist.png：还没生成 VOC 标注）")

    stats = {
        "counts": {s: counts[s] for s in counts},
        "total": {s: len(v) for s, v in splits.items()},
        "num_boxes": len(ws),
        "box_wh_mean": [float(np.mean(ws)), float(np.mean(hs))] if ws else None,
        "box_area_ratio_mean_pct": float(np.mean(np.array(ws) * np.array(hs)) / 400) if ws else None,
    }
    out = MET_DIR / "data_stats.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n统计结果：{out.relative_to(ROOT)}")

    print("\n数量表（可直接粘进报告）")
    print("| 类别 | 训练 | 验证 | 测试 | 合计 |")
    print("|---|---|---|---|---|")
    for c in CLASS_NAMES:
        a, b, d = counts["train"][c], counts["val"][c], counts["test"][c]
        print(f"| {c} | {a} | {b} | {d} | {a + b + d} |")
    tot = [sum(counts[s].values()) for s in ("train", "val", "test")]
    print(f"| **合计** | **{tot[0]}** | **{tot[1]}** | **{tot[2]}** | **{sum(tot)}** |")


if __name__ == "__main__":
    main()
