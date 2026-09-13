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
    refiner = load_json(metrics / "refiner_train_summary.json")
    anchor_km = load_json(metrics / "anchor_kmeans.json")

    # ---- 迭代版流程：滑窗 + 验证器/精修器融合 ----
    fused = {}
    for tag, label in (("val_verifier", "滑窗+验证器(初版)"),
                       ("val_refiner", "滑窗+精修器"),
                       ("val_fuse", "滑窗+融合(第二轮基础参数)"),
                       ("val_fuse_best", "滑窗+融合(第二轮最优参数)"),
                       ("val_fuse_v2", "滑窗+融合(第三轮:窄宽高比)"),
                       ("test_fuse_best", "滑窗+融合(第二轮,测试集)"),
                       ("val_fuse_final", "★完整流水线 融合(验证集)"),
                       ("test_fuse_final", "★完整流水线 融合(测试集)")):
        d = load_json(metrics / f"{tag}.json")
        if d:
            fused[tag] = {"label": label, "data": d}
    ab_aspects = load_json(metrics / "ab_aspects.json")
    refiner_ab = load_json(metrics / "refiner_ab.json")

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
    if refiner:
        print(f"\n【1c】区域精修器（第二阶段升级：判类 + 框回归）")
        print(f"  训练样本 {refiner['train_samples']} / 验证样本 {refiner['val_samples']}"
              f"  参数量 {refiner['params_wan']} 万")
        print(f"  最佳验证准确率 {refiner['best_val_acc']:.4f}"
              f"  精修后平均 IoU {refiner['final_mean_iou_after_refine']:.4f}")

    # ---- 迭代版流程结果（本方案最终采用）----
    if fused:
        print("\n【1d】迭代版流程：滑窗 + 验证器/精修器融合（**最终采用方案**）")
        print(f"{'流程':<26}{'划分':>6}{'mAP@0.5':>10}{'精确率':>9}{'召回率':>9}"
              f"{'框/图':>8}{'ms/图':>8}")
        print("-" * 84)
        for tag, item in fused.items():
            d = item["data"]
            o = d["overall"]
            print(f"{item['label']:<26}{d['split']:>6}{d['map50']:>10.4f}"
                  f"{o['precision']:>9.4f}{o['recall']:>9.4f}"
                  f"{d['outputs_per_image']:>8.1f}{d.get('ms_per_image', 0):>8.0f}")

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
    if refiner:
        md += ["", "**区域精修器（第二阶段升级：判类 + 框回归）**", "",
               f"- 训练/验证样本：{refiner['train_samples']} / {refiner['val_samples']}",
               f"- 参数量：{refiner['params_wan']} 万",
               f"- 验证准确率：{refiner['best_val_acc']:.4f}",
               f"- 精修后平均 IoU：{refiner['final_mean_iou_after_refine']:.4f}"]

    if fused:
        md += ["", "## 1b. 迭代版流程：滑窗 + 验证器/精修器融合（**最终采用**）", "",
               "| 流程 | 划分 | mAP@0.5 | 精确率 | 召回率 | 每图框数 | ms/图 |",
               "|---|---|---|---|---|---|---|"]
        for tag, item in fused.items():
            d = item["data"]
            o = d["overall"]
            md.append(f"| {item['label']} | {d['split']} | {d['map50']:.4f} | "
                      f"{o['precision']:.4f} | {o['recall']:.4f} | "
                      f"{d['outputs_per_image']:.1f} | {d.get('ms_per_image', 0):.0f} |")
        if refiner_ab:
            md += ["", "**精修器抖动范围 A/B**（证明「合成指标会误导」）：", "",
                   "| 版本 | 合成 IoU | 推理精修增益(IoU) | 命中率增益 | mAP@0.5 |",
                   "|---|---|---|---|---|"]
            for k, v in refiner_ab["results"].items():
                md.append(f"| {k} | — | {v['iou_after_mean']-v['iou_before_mean']:+.4f} | "
                          f"{v['hit_after_pct']-v['hit_before_pct']:+.1f}pp | {v['map50']:.4f} |")
        if ab_aspects:
            md += ["", "**滑窗宽高比 A/B 确认**（分层抽样 "
                       f"{ab_aspects['per_class_sample']}/类 = {ab_aspects['num_images']} 张）：", "",
                   "| 配置 | mAP@0.5 | 说明 |", "|---|---|---|"]
            for k, v in ab_aspects["results"].items():
                md.append(f"| {k} | {v['map50']:.4f} | 候选 {v['candidates_per_image']:.0f}/图 |")
        md += ["", "每类 AP@0.5（融合方案，测试集）：", "",
               "| 类别 | 中文 | AP@0.5 | 精确率 | 召回率 | TP | FP | FN |",
               "|---|---|---|---|---|---|---|---|"]
        best_d = fused.get("test_fuse_final", fused.get("test_fuse_best", {}))
        if best_d:
            for c in CN:
                v = best_d["data"]["per_class"].get(c)
                if v:
                    md.append(f"| {c} | {CN[c]} | {v['ap50']:.4f} | {v['precision']:.4f} | "
                              f"{v['recall']:.4f} | {v['tp']} | {v['fp']} | {v['fn']} |")

    md += ["", "## 2. 检测评估结果（mAP@0.5，早期方案对比）", "",
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
