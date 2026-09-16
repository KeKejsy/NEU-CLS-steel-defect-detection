# src/tools/plot_summary.py
import json
import os
import matplotlib.pyplot as plt

# 让电脑认识中文（Windows常用字体）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 找到项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/tools
project_root = os.path.dirname(os.path.dirname(current_dir))  # 项目根目录
metrics_dir = os.path.join(project_root, "results", "metrics")
fig_dir = os.path.join(project_root, "results", "figures")
os.makedirs(fig_dir, exist_ok=True)

# 准备两个空盒子，一个装名字，一个装分数
models = []
scores = []

# 1. 读 B 的分类账本
cls_data = {
    "ResNet50": "cls_train_result_resnet50_vd.json",
    "MobileNetV3": "cls_train_result_mobilenet_v3_small.json"
}
for name, fn in cls_data.items():
    p = os.path.join(metrics_dir, fn)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        models.append(name)
        scores.append(d.get("best_val_acc", 0))

# 2. 读 C 的检测账本
# 2026-09-16 更新：融合方案原先读 val_fuse.json（0.1572，第二轮基础参数），
# 与实际交付不符；现改为读优化版 val_fuse_pre128.json（0.2473），并保留旧流水线作对照。
det_data = {
    "YOLOv3": ("yolov3_train_summary.json", "best_val_map50"),
    "PP-YOLOE-s": ("ppyoloe_s_train_summary.json", "best_val_map50"),
    "融合(旧流水线)": ("val_fuse_final.json", "map50"),
    "融合(优化版)": ("val_fuse_pre128.json", "map50")
}
for name, (fn, key) in det_data.items():
    p = os.path.join(metrics_dir, fn)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        models.append(name)
        scores.append(d.get(key, 0))

# 3. 画图
plt.figure(figsize=(10, 6))
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
plt.bar(models, scores, color=colors[:len(models)])
plt.title("模型关键指标对比（分类准确率 / 检测验证集 mAP@0.5）")
plt.ylabel("分数")
plt.ylim(0, 1.1)  # 分数最高是1.0，留点空间写数字

# 在柱子上写数字
for i, v in enumerate(scores):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')

# 4. 保存到 results/figures 文件夹
save_path = os.path.join(fig_dir, "summary_compare.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"对比图已经画好！请去 results/figures 文件夹里看 summary_compare.png")