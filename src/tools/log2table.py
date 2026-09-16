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
        table.append(["分类", model_name, params, f"准确率 {acc}", "见权重说明", "B", "✅"])

# -------- 2. 抄 C 的检测账本 --------
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
        table.append(["检测", model_name, params, f"mAP@0.5 {map50}", "—", "C", "❌ 失败"])

# -------- 3. 抄 C 的融合方案账本 --------
fuse_path = os.path.join(metrics_dir, "test_fuse_final.json")
if not os.path.exists(fuse_path):
    fuse_path = os.path.join(metrics_dir, "val_fuse.json")
if os.path.exists(fuse_path):
    with open(fuse_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    map50 = data.get("map50", "未知")
    ms = data.get("ms_per_image", "未知")
    table.append(["检测", "滑窗+验证器/精修器融合", "259万 ×2", f"mAP@0.5 {map50}", f"{ms} ms", "C", "⚠️ 部分达标"])

# -------- 4. 把表格写到 results 文件夹 --------
with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(table)

print("大表已经做好！请去 results 文件夹里看 summary_table.csv")