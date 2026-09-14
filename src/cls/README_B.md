# B 部分（任务1：钢材表面缺陷分类）使用说明

> 负责人：B　｜　网络：ResNet50 / MobileNetV3-small　｜　框架：PaddlePaddle 3.3.1（CPU 版）

## 一、文件清单

| 文件 | 作用 |
|---|---|
| `src/cls/_common.py` | 公共模块：类别定义、模型构建、数据读取、数据增强、指标计算（供下面 4 个脚本共用） |
| `src/cls/configs/resnet50_vd.yml` | ResNet50 的超参配置 |
| `src/cls/configs/mobilenet_v3_small.yml` | MobileNetV3-small 的超参配置 |
| `src/cls/train.py` | 训练（`--model` 切换网络），每个 epoch 存一次权重，自动存验证集最好权重 |
| `src/cls/eval_cls.py` | 评估：准确率 / 精确率 / 召回率 / F1 + 混淆矩阵 |
| `src/cls/cam.py` | Grad-CAM 热力图 |
| `src/cls/export.py` | 导出推理模型 + 测参数量与 FPS |
| `src/cls/train_curves.py` | 训练曲线对比图（附加，纯读结果 json） |

## 二、运行命令

```bash
conda activate paddle_env
cd D:\project1_steel

# 训练（先轻后重，两个网络串行跑）
python src/cls/train.py --model mobilenet_v3_small     # 约 26 分钟
python src/cls/train.py --model resnet50_vd            # 约 1 小时 55 分

# 评估（调参用 --split val；测试集只在最后跑一次）
python src/cls/eval_cls.py --model mobilenet_v3_small
python src/cls/eval_cls.py --model resnet50_vd

# 热力图、导出测速、训练曲线
python src/cls/cam.py --model mobilenet_v3_small
python src/cls/export.py --model mobilenet_v3_small
python src/cls/train_curves.py
```

常用可选参数：`--epochs`、`--batch_size`、`--lr`、`--limit`（冒烟测试只取 N 张）、
`--no-pretrained`（不要 ImageNet 预训练权重）、`--split val|test`（评估用）、`--config`（换配置）。

## 三、产出文件

| 路径 | 内容 |
|---|---|
| `results/weights/cls_<模型>_best.pdparams` | 验证集最好的权重 |
| `results/weights/cls_<模型>_last.pdparams` | 最后一轮的权重（每轮覆盖，防中途崩溃） |
| `results/weights/cls_<模型>_infer.*` | 导出的推理模型（`.json` 计算图 + `.pdiparams` 权重） |
| `results/logs/cls_<模型>_<时间>.log / .csv` | 训练日志与逐轮曲线 |
| `results/metrics/cls_train_result_<模型>.json` | 超参 + 每轮记录 + 验证集完整指标 |
| `results/metrics/cls_eval_result.json` | 两个网络在测试集上的指标（D 汇总用） |
| `results/metrics/cls_summary.csv` | 超参 + 结果汇总表（一张表看全） |
| `results/metrics/cls_export_result.json` | 参数量、模型体积、FPS |
| `results/figures/cls_confusion_<模型>.png` | 混淆矩阵 |
| `results/figures/cls_cam_<模型>.png` | Grad-CAM 热力图 |
| `results/figures/cls_train_curves.png` | 训练曲线对比 |
| `results/logs/cls_test_usage.csv` | 测试集使用记录（证明只用过一次） |

## 四、与 D 的接口

1. `eval_cls.py` 会先尝试 `import src.tools.metrics`，若 D 的公共指标库提供了
   `classification_metrics / cls_metrics / evaluate_classification / compute_cls_metrics / cls_report`
   之一，就把它的结果一并写进 `cls_eval_result.json` 的 `tools_metrics` 字段，方便核对两边口径。
   当前该文件还是空的，所以用的是 `_common.py` 里的本地实现。
2. D 画对比图可以直接读 `results/metrics/cls_summary.csv`
   （列：model, num_params_M, epochs, batch_size, lr, best_val_acc, test_acc,
   macro_precision, macro_recall, macro_f1, fps_batch1, fps_batch8, train_time）。

## 五、注意事项

- 测试集只在最后用一次，`results/logs/cls_test_usage.csv` 会留痕；调参一律 `--split val`。
- 随机种子统一 2026（`_common.py` 里 `set_seed` 同时设了 random / numpy / paddle）。
- 配置和代码里全部用相对路径，项目放在纯英文路径下。
- 输入图像是 200×200、内容为灰度但按 RGB 三通道存储，按 3 通道送入网络与预训练权重对齐。
- 训练时 `label_smoothing=0.1`，交叉熵的理论下限约 0.42，所以日志里训练 loss 停在 0.45 左右是正常的。
