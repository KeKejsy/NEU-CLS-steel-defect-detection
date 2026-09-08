"""成员 A · 步骤2：分层随机划分 7 : 1.5 : 1.5

作用：
    对每一类缺陷分别随机打乱后按 70% / 15% / 15% 划分，保证三集中各类比例一致。
    分类任务与检测任务共用同一套划分（同一批图），便于两个任务横向对比。

输入：dataset/cls/images/<类别>/*.jpg
输出：
    dataset/cls/train.txt     每行 "类别/文件名.jpg 标签id"
    dataset/cls/val.txt
    dataset/cls/test.txt      测试集，训练阶段谁都不许碰
    dataset/det/ImageSets/Main/{train,val,test}.txt   每行只有文件名（无后缀）
    results/metrics/split_report.json

用法：
    python src/data/split_dataset.py
    python src/data/split_dataset.py --seed 2026
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
SEED = 2026
RATIO = (0.70, 0.15, 0.15)  # 7 : 1.5 : 1.5


def split_list(items, ratio=RATIO):
    """按比例切成三段，保证总数不丢不重"""
    n = len(items)
    n_train = int(round(n * ratio[0]))
    n_val = int(round(n * ratio[1]))
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def main():
    ap = argparse.ArgumentParser(description="NEU 数据集 7:1.5:1.5 分层划分")
    ap.add_argument("--seed", type=int, default=SEED, help="随机种子，全组统一 2026")
    args = ap.parse_args()

    img_root = ROOT / "dataset" / "cls" / "images"
    if not img_root.is_dir():
        raise SystemExit(f"找不到 {img_root}，请先运行 src/data/download_check.py 整理数据")

    rng = random.Random(args.seed)
    cls_lines = {"train": [], "val": [], "test": []}
    det_lines = {"train": [], "val": [], "test": []}
    stat = {}

    print(f"随机种子：{args.seed}\n")
    print("类别   总数   训练   验证   测试")
    print("-" * 40)

    for idx, c in enumerate(CLASS_NAMES):
        files = sorted(p.name for p in (img_root / c).glob("*.jpg"))
        if not files:
            raise SystemExit(f"类别 {c} 下没有图片，请先运行 download_check.py 整理数据")
        rng.shuffle(files)
        tr, va, te = split_list(files)
        stat[c] = {"total": len(files), "train": len(tr), "val": len(va), "test": len(te)}
        print(f"{c:<6} {len(files):<6} {len(tr):<6} {len(va):<6} {len(te)}")

        for split, names in (("train", tr), ("val", va), ("test", te)):
            for name in names:
                cls_lines[split].append(f"{c}/{name} {idx}")
                det_lines[split].append(Path(name).stem)

    # 写入分类列表（格式：类别/文件名.jpg 标签id）
    cls_out = ROOT / "dataset" / "cls"
    cls_out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (cls_out / f"{split}.txt").write_text("\n".join(cls_lines[split]) + "\n", encoding="utf-8")

    # 写入检测 ImageSets
    det_out = ROOT / "dataset" / "det" / "ImageSets" / "Main"
    det_out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (det_out / f"{split}.txt").write_text("\n".join(det_lines[split]) + "\n", encoding="utf-8")

    total = sum(v["total"] for v in stat.values())
    report = {
        "seed": args.seed,
        "ratio": {"train": RATIO[0], "val": RATIO[1], "test": RATIO[2]},
        "per_class": stat,
        "total": {
            "all": total,
            "train": sum(v["train"] for v in stat.values()),
            "val": sum(v["val"] for v in stat.values()),
            "test": sum(v["test"] for v in stat.values()),
        },
    }
    out = ROOT / "results" / "metrics" / "split_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 40)
    print(f"合计   {total:<6} {report['total']['train']:<6} {report['total']['val']:<6} {report['total']['test']}")
    print(f"\n已写出：dataset/cls/{{train,val,test}}.txt 与 dataset/det/ImageSets/Main/*.txt")
    print(f"划分报告：{out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
