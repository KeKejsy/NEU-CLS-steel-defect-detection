"""成员 C · 汇总全部检测结果，生成可直接写进报告的对比表

用法：
    python src/det/tools/summarize.py

读取 results/metrics/ 下的各份评估 JSON 与训练摘要，汇总成：
    results/metrics/det_summary.json    机器可读汇总
    results/metrics/det_summary.md      Markdown 表格（可直接粘进报告）

为什么不手工整理：这些数字来自多次实验（单阶段、两阶段、滑窗、不同阈值），
手工抄容易抄错，而且改一次配置就得重抄一遍。统一由脚本汇总，保证「报告里的数字」
与「results/ 里的产物」永远一致。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import utils  # noqa: E402

CN = {"Cr": "龟裂", "In": "夹杂", "Pa": "斑块", "PS": "麻点",
      "RS": "氧化铁皮压入", "Sc": "划痕"}


def load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="汇总 results/metrics/ 下的全部评估结果，生成报告用对比表")
    ap.add_argument("--out", default="results/metrics",
                    help="输出目录（默认 results/metrics）")
    args = ap.parse_args()

    metrics = utils.resolve_dir(args.out)
    print("=" * 78)
    print("成员 C · 检测结果汇总")
    print("=" * 78)

    train_sums, evals = {}, {}
    for name in ("yolov3", "ppyoloe_s"):
        s = load_json(metrics / f"{name}_train_summary.json")
        if s:
            train_sums[name] = s
        for split in ("val", "test"):
            for mode in ("detect", "window"):
                e = load_json(metrics / f"{name}_eval2stage_{mode}_{split}.json")
                if e:
                    evals[f"{name}|{mode}|{split}"] = e
    verifier = load_json(metrics / "verifier_train_summary.json")
    anchor_km = load_json(metrics / "anchor_kmeans.json")

    # ---- 1. 训练摘要 ----
    print("\n【1】单阶段检测网络训练情况")
    print(f"{'网络':<12}{'输入':>6}{'batch':>7}{'epoch':>7}{'参数量(万)':>12}"
          f"{'最佳验证mAP@0.5':>16}{'秒/epoch':>10}{'总耗时(分)':>12}")
    print("-" * 90)
    for n, s in train_sums.items():
        print(f"{n:<12}{s['im_size']:>6}{s['batch_size']:>7}{s['epochs']:>7}"
              f"{s['params_wan']:>12.2f}{s['best_val_map50']:>16.4f}"
              f"{s['sec_per_epoch']:>10.1f}{s['total_sec']/60:>12.1f}")
    if verifier:
        print(f"\n【1b】区域验证器（第二阶段）")
        print(f"  训练图块 {verifier['train_tiles']} / 验证图块 {verifier['val_tiles']}"
              f"  参数量 {verifier['params_wan']} 万")
        print(f"  最佳图块分类准确率 {verifier['best_val_acc']:.4f}")

    # ---- 2. 检测评估 ----
    print("\n【2】检测评估结果（mAP@0.5，测试集为最终口径）")
    print(f"{'配置':<34}{'划分':>6}{'mAP@0.5':>10}{'精确率':>9}{'召回率':>9}"
          f"{'每图框数':>10}")
    print("-" * 78)
    rows = []
    for key, e in sorted(evals.items()):
        name, mode, split = key.split("|")
        label = f"{name} · {'滑窗+验证器' if mode=='window' else '检测器+验证器'}"
        o = e["overall"]
        npb = e["num_pred_boxes"] / max(e["num_images"], 1)
        print(f"{label:<34}{split:>6}{e['map50']:>10.4f}{o['precision']:>9.4f}"
              f"{o['recall']:>9.4f}{npb:>10.1f}")
        rows.append({"label": label, "split": split, "map50": e["map50"],
                     "precision": o["precision"], "recall": o["recall"],
                     "boxes_per_image": round(npb, 1), "per_class": e["per_class"]})

    # ---- 3. 每类 AP ----
    print("\n【3】每类 AP@0.5 明细")
    for key, e in sorted(evals.items()):
        name, mode, split = key.split("|")
        if split != "test" and mode != "window":
            continue
        pc = e["per_class"]
        line = "  ".join(f"{c}={pc[c]['ap50']:.3f}" for c in CN if c in pc)
        print(f"  {name} · {mode} · {split}: {line}")

    # ---- 4. anchor 分析 ----
    if anchor_km:
        print("\n【4】anchor 与数据集的匹配度（k-means 分析）")
        print(f"  目标框总数 {anchor_km['num_boxes']}")
        print(f"  官方 YOLOv3 anchor 的最佳宽高 IoU 均值: "
              f"{anchor_km['official']['mean_best_wh_iou']:.3f}")
        print(f"  k-means 定制 anchor:              "
              f"{anchor_km['kmeans']['mean_best_wh_iou']:.3f}")
        print(f"  提升 {anchor_km['kmeans']['mean_best_wh_iou'] - anchor_km['official']['mean_best_wh_iou']:+.3f}")

    # ---- 5. 写 Markdown ----
    md = ["# 成员 C · 检测结果汇总", "",
          "> 本文件由 `src/det/tools/summarize.py` 自动生成，"
          "数字与 `results/metrics/` 下的产物一一对应。", "",
          "## 1. 单阶段检测网络训练情况", "",
          "| 网络 | 输入 | batch | epoch | 参数量(万) | 最佳验证 mAP@0.5 | 秒/epoch | 总耗时(分钟) |",
          "|---|---|---|---|---|---|---|---|"]
    for n, s in train_sums.items():
        md.append(f"| {n} | {s['im_size']} | {s['batch_size']} | {s['epochs']} | "
                  f"{s['params_wan']:.2f} | {s['best_val_map50']:.4f} | "
                  f"{s['sec_per_epoch']:.1f} | {s['total_sec']/60:.1f} |")
    if verifier:
        md += ["", "**区域验证器（两阶段第二阶段）**", "",
               f"- 训练/验证图块：{verifier['train_tiles']} / {verifier['val_tiles']}",
               f"- 参数量：{verifier['params_wan']} 万",
               f"- 图块分类准确率：**{verifier['best_val_acc']:.4f}**"]

    md += ["", "## 2. 检测评估结果（mAP@0.5）", "",
           "| 配置 | 划分 | mAP@0.5 | 精确率 | 召回率 | 每图框数 |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['label']} | {r['split']} | {r['map50']:.4f} | "
                  f"{r['precision']:.4f} | {r['recall']:.4f} | {r['boxes_per_image']} |")

    md += ["", "## 3. 每类 AP@0.5", ""]
    for key, e in sorted(evals.items()):
        name, mode, split = key.split("|")
        if split != "test" and mode != "window":
            continue
        pc = e["per_class"]
        md.append(f"**{name} · {mode} · {split}**（mAP@0.5 = {e['map50']:.4f}）")
        md += ["", "| 类别 | 中文 | AP@0.5 | 精确率 | 召回率 | F1 |",
               "|---|---|---|---|---|---|"]
        for c in CN:
            if c in pc:
                d = pc[c]
                md.append(f"| {c} | {CN[c]} | {d['ap50']:.4f} | {d['precision']:.4f} | "
                          f"{d['recall']:.4f} | {d['f1']:.4f} |")
        md.append("")

    if anchor_km:
        md += ["## 4. anchor 与数据集匹配度", "",
               f"- 目标框总数：{anchor_km['num_boxes']}",
               f"- 官方 YOLOv3 anchor 最佳宽高 IoU：{anchor_km['official']['mean_best_wh_iou']:.3f}",
               f"- k-means 定制 anchor：{anchor_km['kmeans']['mean_best_wh_iou']:.3f}", ""]

    (metrics / "det_summary.md").write_text("\n".join(md), encoding="utf-8")
    utils.dump_json({"train_summaries": train_sums, "verifier": verifier,
                     "anchor_kmeans": anchor_km, "evaluations": rows},
                    metrics / "det_summary.json")
    print(f"\n汇总已写入：{(metrics / 'det_summary.md').relative_to(utils.ROOT)}")
    print(f"            {(metrics / 'det_summary.json').relative_to(utils.ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
