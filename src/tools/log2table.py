# src/tools/log2table.py
import json
import csv
import os

# 找到项目根目录（也就是有大门的那个文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/tools
project_root = os.path.dirname(os.path.dirname(current_dir))  # 项目根目录
metrics_dir = os.path.join(project_root, "results", "metrics")
output_file = os.path.join(project_root, "results", "summary_table.csv")

# 准备一张空表格
table = []
table.append(["任务", "网络/方案", "参数量", "关键指标", "单张推理", "负责人", "状态"])

# -------- 1. 抄 B 的分类账本 --------
cls_files = {
    "ResNet50_vd": "cls_train_result_resnet50_vd.json",
    "MobileNetV3-small": "cls_train_result_mobilenet_v3_small.json"
}
for model_name, filename in cls_files.items():
    path = os.path.join(metrics_dir, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        params = data.get("num_params", "未知")
        acc = data.get("best_val_acc", "未知")
        acc_txt = f"准确率 {float(acc):.4f}" if isinstance(acc, (int, float)) else f"准确率 {acc}"
        table.append(["分类", model_name, params, acc_txt, "见权重说明", "B", "✅"])

# -------- 2. 抄 C 的检测账本 --------
# 单阶段两个网络都是失败基线（conf 分支学不出排序），验证集 mAP 约 0，如实标注。
det_files = {
    "YOLOv3": "yolov3_train_summary.json",
    "PP-YOLOE-s": "ppyoloe_s_train_summary.json"
}
for model_name, filename in det_files.items():
    path = os.path.join(metrics_dir, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        params = data.get("params_wan", "未知")
        map50 = data.get("best_val_map50", "未知")
        # 注意：这个值极小（如 1.65e-06），直接打印会变成科学计数法，格式化到 4 位更可读
        map_txt = f"mAP@0.5 {float(map50):.4f}" if isinstance(map50, (int, float)) else f"mAP@0.5 {map50}"
        table.append(["检测", model_name, params, map_txt, "—", "C", "❌ 失败"])

# -------- 3. 抄 C 的融合方案账本 --------
# 取数优先级 = 「最新 + 最完整」：优化版（预训练主干 + GIoU + 128 输入）优于旧流水线。
# 2026-09-16 更新：优化版测试集/验证集已产出，原先只读 test_fuse_final.json 会让
# 汇总表停留在旧数字（0.1552），与实际交付不符。
fuse_candidates = [
    ("test_fuse_pre128.json", "滑窗+验证器/精修器融合（优化版）", "测试集"),
    ("val_fuse_pre128.json", "滑窗+验证器/精修器融合（优化版）", "验证集"),
    ("test_fuse_final.json", "滑窗+验证器/精修器融合（旧流水线）", "测试集"),
]
for filename, label, split in fuse_candidates:
    path = os.path.join(metrics_dir, filename)
    if not os.path.exists(path):
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    map50 = data.get("map50")
    ms = data.get("ms_per_image")
    map_txt = f"mAP@0.5 {float(map50):.4f}（{split}）" if isinstance(map50, (int, float)) else f"mAP@0.5 {map50}"
    ms_txt = f"{float(ms):.0f} ms" if isinstance(ms, (int, float)) else f"{ms} ms"
    table.append(["检测", label, "259万 ×2", map_txt, ms_txt, "C", "⚠️ 部分达标"])

# -------- 4. 把表格写到 results 文件夹 --------
with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(table)

print("大表已经做好！请去 results 文件夹里看 summary_table.csv")