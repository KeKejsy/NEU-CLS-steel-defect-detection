"""成员 A · 步骤3：把原始 XML 标注转成 VOC 格式，供 PaddleDetection 使用

作用：
    1. 读取 dataset/raw/annotations/*.xml
    2. 统一类别名为 Cr / In / Pa / PS / RS / Sc（原始英文命名各不相同）
    3. 校验边框：坐标越界、宽高小于等于 0、类别名不认识，都会报错并列出
    4. 输出标准 VOC 结构：
         dataset/det/JPEGImages/*.jpg
         dataset/det/Annotations/*.xml
         dataset/det/label_list.txt

输入：dataset/raw/ANNOTATIONS/*.xml、dataset/cls/images/<类别>/*.jpg
输出：dataset/det/ 下的 VOC 结构 + results/metrics/voc_convert_report.json

用法：
    python src/data/xml2voc.py
"""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]

# 标注里可能出现的各种写法 -> 统一类别代码
NAME_ALIAS = {
    "crazing": "Cr", "cr": "Cr", "crack": "Cr",
    "inclusion": "In", "in": "In",
    "patches": "Pa", "pa": "Pa", "patch": "Pa",
    "pitted_surface": "PS", "pittedsurface": "PS", "ps": "PS", "pitted": "PS",
    "rolled-in_scale": "RS", "rolled_in_scale": "RS", "rolledinscale": "RS",
    "rolled-in-scale": "RS", "rs": "RS", "scale": "RS",
    "scratches": "Sc", "sc": "Sc", "scratch": "Sc",
}


def norm_name(raw: str):
    """把标注里的类名归一化，识别不了返回 None"""
    key = raw.strip().lower().replace(" ", "_")
    return NAME_ALIAS.get(key)


def build_voc(filename, width, height, objects):
    """生成标准 VOC 格式的 XML 文本"""
    lines = [
        "<annotation>",
        "  <folder>JPEGImages</folder>",
        f"  <filename>{filename}</filename>",
        "  <source>",
        "    <database>NEU-DET</database>",
        "  </source>",
        "  <size>",
        f"    <width>{width}</width>",
        f"    <height>{height}</height>",
        "    <depth>1</depth>",
        "  </size>",
        "  <segmented>0</segmented>",
    ]
    for name, (xmin, ymin, xmax, ymax) in objects:
        lines += [
            "  <object>",
            f"    <name>{name}</name>",
            "    <pose>Unspecified</pose>",
            "    <truncated>0</truncated>",
            "    <difficult>0</difficult>",
            "    <bndbox>",
            f"      <xmin>{xmin}</xmin>",
            f"      <ymin>{ymin}</ymin>",
            f"      <xmax>{xmax}</xmax>",
            f"      <ymax>{ymax}</ymax>",
            "    </bndbox>",
            "  </object>",
        ]
    lines.append("</annotation>")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="XML 标注转 VOC 格式")
    ap.add_argument("--ann", default="dataset/raw/ANNOTATIONS", help="原始标注目录")
    args = ap.parse_args()

    ann_in = (ROOT / args.ann).resolve()
    if not ann_in.is_dir():
        raise SystemExit(f"找不到标注目录：{ann_in}，请先运行 download_check.py")

    jpeg_out = ROOT / "dataset" / "det" / "JPEGImages"
    ann_out = ROOT / "dataset" / "det" / "Annotations"
    jpeg_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    # 图片来源（已按类整理好的）
    img_index = {}
    for c in CLASS_NAMES:
        for p in (ROOT / "dataset" / "cls" / "images" / c).glob("*.jpg"):
            img_index[p.stem] = p

    xmls = sorted(ann_in.glob("*.xml"))
    print(f"待转换标注：{len(xmls)} 个")

    n_obj_total = 0
    per_class = Counter()
    problems = []
    converted = 0

    for x in xmls:
        stem = x.stem
        try:
            root = ET.parse(x).getroot()
        except ET.ParseError as e:
            problems.append({"file": x.name, "reason": f"XML 解析失败：{e}"})
            continue

        size_node = root.find("size")
        w = int(float(size_node.findtext("width", "200"))) if size_node is not None else 200
        h = int(float(size_node.findtext("height", "200"))) if size_node is not None else 200

        objects = []
        for obj in root.findall("object"):
            raw_name = obj.findtext("name", "").strip()
            code = norm_name(raw_name)
            if code is None:
                problems.append({"file": x.name, "reason": f"类别名无法识别：{raw_name}"})
                continue
            bb = obj.find("bndbox")
            if bb is None:
                problems.append({"file": x.name, "reason": "缺少 bndbox"})
                continue
            try:
                xmin = int(float(bb.findtext("xmin")))
                ymin = int(float(bb.findtext("ymin")))
                xmax = int(float(bb.findtext("xmax")))
                ymax = int(float(bb.findtext("ymax")))
            except (TypeError, ValueError):
                problems.append({"file": x.name, "reason": "bndbox 坐标不是数字"})
                continue

            # 越界与非法框检查
            if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
                problems.append({"file": x.name,
                                 "reason": f"坐标越界 ({xmin},{ymin},{xmax},{ymax}) 超出 {w}x{h}"})
                xmin = max(0, min(xmin, w)); ymin = max(0, min(ymin, h))
                xmax = max(0, min(xmax, w)); ymax = max(0, min(ymax, h))
            if xmax <= xmin or ymax <= ymin:
                problems.append({"file": x.name,
                                 "reason": f"宽高非正 ({xmin},{ymin},{xmax},{ymax})"})
                continue

            objects.append((code, (xmin, ymin, xmax, ymax)))
            per_class[code] += 1
            n_obj_total += 1

        if not objects:
            problems.append({"file": x.name, "reason": "没有任何有效目标"})
            continue

        (ann_out / f"{stem}.xml").write_text(
            build_voc(f"{stem}.jpg", w, h, objects), encoding="utf-8")

        src_img = img_index.get(stem)
        if src_img is not None and not (jpeg_out / f"{stem}.jpg").exists():
            shutil.copy2(src_img, jpeg_out / f"{stem}.jpg")
        converted += 1

    # label_list.txt
    (ROOT / "dataset" / "det" / "label_list.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

    print(f"成功转换：{converted} 个，目标框总数：{n_obj_total}")
    print("\n类别   目标框数")
    print("-" * 22)
    for c in CLASS_NAMES:
        print(f"{c:<6} {per_class.get(c, 0)}")

    if problems:
        print(f"\n发现 {len(problems)} 处问题（前 10 条）：")
        for p in problems[:10]:
            print(f"  {p['file']}: {p['reason']}")
    else:
        print("\n未发现越界 / 非法框 / 未知类别 ✔")

    report = {
        "converted_files": converted,
        "total_objects": n_obj_total,
        "objects_per_class": {c: per_class.get(c, 0) for c in CLASS_NAMES},
        "problems": problems,
        "passed": len(problems) == 0,
    }
    out = ROOT / "results" / "metrics" / "voc_convert_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n转换报告：{out.relative_to(ROOT)}")
    print(f"VOC 数据：dataset/det/ （JPEGImages {len(list(jpeg_out.glob('*.jpg')))} 张、"
          f"Annotations {len(list(ann_out.glob('*.xml')))} 个）")


if __name__ == "__main__":
    main()
