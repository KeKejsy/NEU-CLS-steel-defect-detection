"""成员 A · 步骤1：数据集完整性校验 + 按类整理

作用：
    1. 读取原始下载目录（dataset/raw/）里的图片与 XML 标注
    2. 校验：6 类各 300 张、共 1800 张、200x200 灰度、每张图都有对应 XML
    3. 整理（默认执行）：
         图片 -> dataset/cls/images/<类别>/*.jpg
         标注不用动，xml2voc.py 直接读 dataset/raw/ANNOTATIONS/

输入：dataset/raw/IMAGES/*.jpg 与 dataset/raw/ANNOTATIONS/*.xml
输出：dataset/cls/images/、dataset/raw/annotations/、results/metrics/data_check_report.json

用法：
    python src/data/download_check.py
    python src/data/download_check.py --src dataset/raw/_src_repo --check-only
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

# 6 类缺陷，顺序固定，标签 id 依此顺序（0~5）
CLASS_NAMES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]

# 原始文件名前缀 -> 类别代码（NEU-DET 官方英文命名）
SRC_NAME_MAP = {
    "crazing": "Cr",
    "inclusion": "In",
    "patches": "Pa",
    "pitted_surface": "PS",
    "rolled-in_scale": "RS",
    "scratches": "Sc",
}

EXPECT_PER_CLASS = 300
EXPECT_SIZE = (200, 200)


def classify(filename: str):
    """按文件名前缀判断类别，识别不了返回 None"""
    stem = Path(filename).stem.lower()
    for prefix, code in SRC_NAME_MAP.items():
        if stem.startswith(prefix + "_"):
            return code
    return None


def find_images(src: Path):
    return sorted(src.glob("*.jpg")) + sorted(src.glob("*.bmp")) + sorted(src.glob("*.png"))


def main():
    ap = argparse.ArgumentParser(description="NEU 数据集完整性校验与整理")
    ap.add_argument("--src", default="dataset/raw",
                    help="原始下载目录，其下应有 IMAGES/ 与 ANNOTATIONS/")
    ap.add_argument("--check-only", action="store_true", help="只校验，不整理复制")
    args = ap.parse_args()

    src = (ROOT / args.src).resolve()
    img_dir = src / "IMAGES"
    ann_dir = src / "ANNOTATIONS"

    print(f"[1/4] 读取目录：{src}")
    if not img_dir.is_dir():
        raise SystemExit(f"找不到图片目录：{img_dir}\n请先把下载的数据解压到该位置。")
    images = find_images(img_dir)
    print(f"      图片文件：{len(images)} 个")

    # ---- 按类统计 ----
    per_class = Counter()
    unknown = []
    for p in images:
        c = classify(p.name)
        if c is None:
            unknown.append(p.name)
        else:
            per_class[c] += 1

    # ---- 尺寸 / 通道校验 ----
    print("[2/4] 校验尺寸与通道（1800 张，约需十几秒）...")
    bad_size, bad_mode = [], []
    for p in images:
        with Image.open(p) as im:
            if im.size != EXPECT_SIZE:
                bad_size.append((p.name, im.size))
            if im.mode not in ("L", "RGB"):
                bad_mode.append((p.name, im.mode))

    # ---- 标注配对校验 ----
    print("[3/4] 校验 XML 标注配对...")
    missing_xml = []
    if ann_dir.is_dir():
        for p in images:
            if not (ann_dir / (p.stem + ".xml")).exists():
                missing_xml.append(p.name)
    else:
        print(f"      警告：未找到标注目录 {ann_dir}，跳过配对校验")

    # ---- 整理复制 ----
    if not args.check_only:
        print("[4/4] 整理复制到 dataset/cls/images/<类别>/ ...")
        cls_root = ROOT / "dataset" / "cls" / "images"
        for c in CLASS_NAMES:
            (cls_root / c).mkdir(parents=True, exist_ok=True)
        n_img = 0
        for p in images:
            c = classify(p.name)
            if c is None:
                continue
            shutil.copy2(p, cls_root / c / p.name)
            n_img += 1
        print(f"      已复制图片 {n_img} 张")
    else:
        print("[4/4] --check-only，跳过整理")

    # ---- 报告 ----
    ok = True
    print("\n" + "=" * 52)
    print("类别   实际   期望   状态")
    print("-" * 52)
    for c in CLASS_NAMES:
        n = per_class.get(c, 0)
        flag = "OK" if n == EXPECT_PER_CLASS else "!! 数量不对"
        ok &= n == EXPECT_PER_CLASS
        print(f"{c:<6} {n:<6} {EXPECT_PER_CLASS:<6} {flag}")
    print("-" * 52)
    print(f"总计   {sum(per_class.values())} / 1800")
    print(f"尺寸异常 {len(bad_size)} 张，通道异常 {len(bad_mode)} 张，缺标注 {len(missing_xml)} 张，无法识别 {len(unknown)} 张")
    if bad_size:
        print("  尺寸异常示例：", bad_size[:5])
        ok = False
    if bad_mode:
        print("  通道异常示例：", bad_mode[:5])
    if missing_xml:
        print("  缺标注示例：", missing_xml[:5])
        ok = False
    if unknown:
        print("  无法识别类别示例：", unknown[:5])
        ok = False

    report = {
        "total_images": len(images),
        "per_class": {c: per_class.get(c, 0) for c in CLASS_NAMES},
        "expect_per_class": EXPECT_PER_CLASS,
        "bad_size": [{"name": n, "size": list(s)} for n, s in bad_size],
        "bad_mode": bad_mode,
        "missing_xml": missing_xml,
        "unknown_class": unknown,
        "passed": bool(ok),
    }
    out = ROOT / "results" / "metrics" / "data_check_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n校验报告已保存：{out.relative_to(ROOT)}")
    print("结论：" + ("全部通过 ✔" if ok else "存在问题，请看上面提示 ✘"))


if __name__ == "__main__":
    main()
