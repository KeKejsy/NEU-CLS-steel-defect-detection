"""成员 A · 步骤5：交付自检（重点检查训练/测试集是否泄漏）

作用：
    1. 检查 dataset 目录结构是否齐全
    2. 检查 train / val / test 三个集合之间有没有重复文件（数据泄漏检查）
    3. 检查划分比例是否接近 7 : 1.5 : 1.5
    4. 检查每张图都存在、每张图都有对应 VOC 标注、label_list.txt 正确
    5. 输出 PASS / FAIL 清单

输入：dataset/ 下的全部内容
输出：results/metrics/submit_check.json

用法：
    python src/data/check_submit.py
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
SPLITS = ("train", "val", "test")


def check(cond, ok_msg, bad_msg, results):
    results.append({"item": ok_msg if cond else bad_msg, "passed": bool(cond)})
    print(("  [PASS] " if cond else "  [FAIL] ") + (ok_msg if cond else bad_msg))
    return bool(cond)


def main():
    ap = argparse.ArgumentParser(description="交付前自检")
    args = ap.parse_args()

    results = []
    print("=" * 56)
    print("NEU 数据集交付自检")
    print("=" * 56)

    # ---- 1. 目录结构 ----
    print("\n[1] 目录结构")
    required = [
        "dataset/cls/images", "dataset/cls/train.txt", "dataset/cls/val.txt", "dataset/cls/test.txt",
        "dataset/det/JPEGImages", "dataset/det/Annotations", "dataset/det/label_list.txt",
        "dataset/det/ImageSets/Main/train.txt", "dataset/det/ImageSets/Main/val.txt",
        "dataset/det/ImageSets/Main/test.txt",
    ]
    for rel in required:
        check((ROOT / rel).exists(), f"存在 {rel}", f"缺失 {rel}", results)

    # ---- 2. 分类列表与泄漏检查 ----
    print("\n[2] 分类数据集")
    cls_sets = {}
    for s in SPLITS:
        p = ROOT / "dataset" / "cls" / f"{s}.txt"
        if p.exists():
            cls_sets[s] = {ln.strip().rsplit(" ", 1)[0]
                           for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}
    if len(cls_sets) == 3:
        check(len(cls_sets["train"] & cls_sets["test"]) == 0,
              "训练集与测试集无重复文件", f"训练集与测试集重复 {len(cls_sets['train'] & cls_sets['test'])} 个！", results)
        check(len(cls_sets["train"] & cls_sets["val"]) == 0,
              "训练集与验证集无重复文件", f"训练集与验证集重复 {len(cls_sets['train'] & cls_sets['val'])} 个！", results)
        check(len(cls_sets["val"] & cls_sets["test"]) == 0,
              "验证集与测试集无重复文件", f"验证集与测试集重复 {len(cls_sets['val'] & cls_sets['test'])} 个！", results)
        total = sum(len(v) for v in cls_sets.values())
        r_train = len(cls_sets["train"]) / total
        r_val = len(cls_sets["val"]) / total
        r_test = len(cls_sets["test"]) / total
        check(abs(r_train - 0.70) < 0.02 and abs(r_val - 0.15) < 0.02 and abs(r_test - 0.15) < 0.02,
              f"比例符合要求（{r_train:.1%} / {r_val:.1%} / {r_test:.1%}）",
              f"比例偏离 7:1.5:1.5（{r_train:.1%} / {r_val:.1%} / {r_test:.1%}）", results)
        check(total == 1800, f"总图片数 1800（实际 {total}）", f"总图片数不是 1800（实际 {total}）", results)

        missing = [rel for s in SPLITS for rel in cls_sets[s]
                   if not (ROOT / "dataset" / "cls" / "images" / rel).exists()]
        check(not missing, "列表中的图片文件都存在", f"列表中有 {len(missing)} 个图片文件不存在", results)

        # 每类数量
        per = {c: {s: 0 for s in SPLITS} for c in CLASS_NAMES}
        for s in SPLITS:
            for rel in cls_sets[s]:
                per[rel.split("/")[0]][s] += 1
        bad = [c for c in CLASS_NAMES if sum(per[c].values()) != 300]
        check(not bad, "6 类各 300 张", f"类别数量不对：{bad}", results)

    # ---- 3. 检测数据集 ----
    print("\n[3] 检测数据集（VOC）")
    det_sets = {}
    for s in SPLITS:
        p = ROOT / "dataset" / "det" / "ImageSets" / "Main" / f"{s}.txt"
        if p.exists():
            det_sets[s] = {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}
    if len(det_sets) == 3:
        check(len(det_sets["train"] & det_sets["test"]) == 0,
              "训练集与测试集无重复文件", f"训练集与测试集重复 {len(det_sets['train'] & det_sets['test'])} 个！", results)
        check(len(det_sets["train"] & det_sets["val"]) == 0,
              "训练集与验证集无重复文件", f"训练集与验证集重复 {len(det_sets['train'] & det_sets['val'])} 个！", results)

        jpegs = {p.stem for p in (ROOT / "dataset" / "det" / "JPEGImages").glob("*.jpg")}
        anns = {p.stem for p in (ROOT / "dataset" / "det" / "Annotations").glob("*.xml")}
        check(len(jpegs) == 1800, f"JPEGImages 有 1800 张（实际 {len(jpegs)}）",
              f"JPEGImages 不是 1800 张（实际 {len(jpegs)}）", results)
        check(jpegs == anns, "每张图都有对应标注",
              f"图片与标注不匹配：缺标注 {len(jpegs - anns)}、多标注 {len(anns - jpegs)}", results)

        all_ids = set().union(*det_sets.values())
        check(all_ids == jpegs, "ImageSets 覆盖了全部图片",
              f"ImageSets 与图片不一致：列表 {len(all_ids)} 张 / 图片 {len(jpegs)} 张", results)

    ll = ROOT / "dataset" / "det" / "label_list.txt"
    if ll.exists():
        names = [x.strip() for x in ll.read_text(encoding="utf-8").splitlines() if x.strip()]
        check(names == CLASS_NAMES, f"label_list.txt 正确（{','.join(names)}）",
              f"label_list.txt 内容不对：{names}", results)

    # ---- 汇总 ----
    passed = sum(1 for r in results if r["passed"])
    print("\n" + "=" * 56)
    print(f"通过 {passed} / {len(results)} 项")
    if passed == len(results):
        print("全部通过 ✔ 可以交给 B 和 C 了")
    else:
        print("存在未通过项 ✘ 请按上面 [FAIL] 提示修复")
    print("=" * 56)

    out = ROOT / "results" / "metrics" / "submit_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"passed": passed, "total": len(results),
                               "all_passed": passed == len(results),
                               "details": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"自检报告：{out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
