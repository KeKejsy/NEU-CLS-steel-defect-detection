# B 部分交付说明：钢材表面缺陷分类（任务1）

负责人：B（分类模型负责人）　｜　框架：PaddlePaddle 3.3.1　｜　完成日期：2026-09-14

## 摘要

- 两个网络在**独立测试集（270 张，6 类各 45 张）上准确率都是 100%，宏平均 F1 = 1.0000**，超额满足"至少 2 种网络 + 对比效果差异"的要求。
- 差异体现在**成本而不是精度**：ResNet50 参数是 MobileNetV3-small 的 **15.3 倍**（2357 万 vs 153 万），
  单张推理慢 **8.2 倍**（86.0 ms vs 10.5 ms），模型文件大 **14.9 倍**（90.1 MB vs 6.0 MB），
  但收敛快得多（第 2 轮验证集就到 100%，MobileNetV3 要到第 20 轮）。
- 结论一句话：**这个数据集上两个网络精度上限相同，选型看部署成本——要精度余量选 ResNet50，要速度和小体积选 MobileNetV3-small。**

## 一、任务与完成情况

对照组长《项目1 分工方案》里 B 的四项任务，全部完成并额外增加了训练曲线对比：

| 分工要求 | 交付物 | 状态 |
|---|---|---|
| `train.py` + 两份配置，`--model` 切换网络 | `src/cls/train.py`、`configs/resnet50_vd.yml`、`configs/mobilenet_v3_small.yml` | ✅ |
| `eval_cls.py`：测试集推理 + 准确率/精确率/召回率/F1 + 混淆矩阵 | `src/cls/eval_cls.py` + 2 张混淆矩阵图 | ✅ |
| `cam.py`：Grad-CAM 热力图 | `src/cls/cam.py` + 2 张热力图 | ✅ |
| `export.py`：导出推理模型 + 测 FPS | `src/cls/export.py` + 推理模型 + 参数量/FPS | ✅ |
| 交付：结果 json、超参表、参数量/FPS 给 D | `cls_eval_result.json`、`cls_summary.csv`、`cls_export_result.json` | ✅ |
| 附加 | `train_curves.py`、`_common.py`（公共模块）、`README_B.md` | ➕ |

## 二、实验设置

**数据**（由 A 提供，本次只读不改）：6 类缺陷各 300 张，共 1800 张 200×200 图像；
分层划分 1260 / 270 / 270（训练 / 验证 / 测试），随机种子 2026，三集无重复文件（A 的 23 项自检全过）。
类别顺序固定为 `Cr / In / Pa / PS / RS / Sc`。

**输入处理**：图像内容为灰度但按 RGB 三通道存储，统一按 3 通道送入网络以对齐 ImageNet 预训练权重；
按 ImageNet 均值方差归一化。

**数据增强**（仅训练集）：随机裁剪（padding=16）+ 随机水平/垂直翻转 + 亮度抖动 ±10%。
钢材缺陷没有固定朝向，因此上下翻转也是安全的。

**超参表**

| 项目 | ResNet50 | MobileNetV3-small |
|---|---|---|
| 骨干网络 | `paddle.vision.models.resnet50` | `paddle.vision.models.mobilenet_v3_small` |
| 预训练 | ImageNet（飞桨官方权重） | ImageNet（飞桨官方权重） |
| 输入尺寸 | 200×200 | 200×200 |
| 优化器 | Momentum SGD（momentum 0.9, weight_decay 1e-4） | 同左 |
| 初始学习率 | 0.01 | 0.01 |
| 学习率策略 | 余弦退火 + 2 轮线性 warmup | 余弦退火 + 1 轮线性 warmup |
| 标签平滑 | 0.1 | 0.1 |
| batch size | 16 | 32 |
| 训练轮数 | 20 | 30 |
| 随机种子 | 2026 | 2026 |
| 参数量 | 23,573,446（23.57 M） | 1,536,118（1.54 M） |

## 三、结果

### 3.1 训练过程

| 指标 | ResNet50 | MobileNetV3-small |
|---|---|---|
| 每轮耗时 | 346.8 秒 | 51.6 秒 |
| 总训练时间 | 1 小时 55 分（20 轮） | 25 分 49 秒（30 轮） |
| 验证集最好准确率 | **1.0000**（第 2 轮） | **1.0000**（第 20 轮） |
| 首次达到 ≥95% 验证准确率的轮次 | 第 1 轮（99.63%） | 第 6 轮（95.93%） |

训练 loss 稳定在 0.45 附近不是没收敛：使用 `label_smoothing=0.1` 时交叉熵的理论下限约 0.42，
说明两个网络都已经把训练集拟合到位（曲线见 `figures/cls_train_curves.png`）。

### 3.2 测试集结果（270 张，全流程只评估这一次）

| 指标 | ResNet50 | MobileNetV3-small |
|---|---|---|
| 准确率 accuracy | **1.0000** | **1.0000** |
| 宏平均 precision | 1.0000 | 1.0000 |
| 宏平均 recall | 1.0000 | 1.0000 |
| 宏平均 F1 | **1.0000** | **1.0000** |
| 测试集 loss | 0.0796 | 0.0875 |

两个网络在 6 个类别上**每一项 P/R/F1 都是 1.0000**，混淆矩阵为纯对角矩阵
（`figures/cls_confusion_resnet50_vd.png`、`figures/cls_confusion_mobilenet_v3_small.png`）。

需要说明的是，测试集 loss 明显大于 0 而不是"背下来"的接近 0，
且测试集与训练集无重复文件（A 的泄漏检查 + `results/logs/cls_test_usage.csv` 均留痕），
说明 100% 是在**未见过的数据**上得到的真实结果。NEU-CLS 分类任务本身难度不大，
文献里准确率普遍在 99% 以上；这也意味着**精度不是这两个网络的区分点**。

## 四、两个网络的对比分析

| 对比维度 | ResNet50 | MobileNetV3-small | 差距 |
|---|---|---|---|
| 参数量 | 23.57 M | 1.54 M | ResNet 大 15.3 倍 |
| 模型文件 | 90.09 MB | 6.03 MB | ResNet 大 14.9 倍 |
| 单张推理耗时（CPU） | 86.0 ms | 10.5 ms | ResNet 慢 8.2 倍 |
| 吞吐 FPS（batch=1） | 11.63 | **95.19** | MobileNet 快 8.2 倍 |
| 吞吐 FPS（batch=8） | 11.66 | 88.72 | MobileNet 快 7.6 倍 |
| 每轮训练耗时 | 346.8 s | 51.6 s | ResNet 慢 6.7 倍 |
| 达到 100% 验证准确率 | 第 2 轮 | 第 20 轮 | ResNet 收敛快约 10 倍 |
| 测试集准确率 / 宏平均 F1 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 持平 |

**结论**：在 1800 张、6 类、200×200 的小规模数据集上，两个网络都能做到 100% 准确率；
真正的取舍是"训练与部署成本"。ResNet50 靠大容量快速收敛，适合精度优先、算力充足的场景；
MobileNetV3-small 用 1/15 的参数量拿到同样精度、推理快 8 倍，适合产线边缘设备实时检测。
这也解释了为什么工业缺陷检测里轻量网络更受青睐。

## 五、可解释性（Grad-CAM）

对每类各取 3 张验证集图片生成热力图（`figures/cls_cam_<模型>.png`）：

- 划痕（Sc）：高响应沿细长划痕走势分布；压入氧化皮（RS）：集中在氧化皮区域；
- 斑块（Pa）/ 点蚀（PS）：落在局部凹坑、斑块上；裂纹（Cr）：落在裂纹纹理处。

两个网络的关注区域都落在缺陷本身而不是背景，说明模型学到的是缺陷特征而非数据集偏置。
这也是报告里回答"模型为什么这么判"的直观证据。

## 六、测试集纪律与可复现性

- 训练阶段**只使用 train / val**，`train.py` 里根本不读 `test.txt`。
- 测试集全流程只评估一次（两个网络各一次），记录在 `results/logs/cls_test_usage.csv`。
- 调参对比一律用 `--split val`。
- 随机种子 2026 写进 `_common.py::set_seed`（同时设 random / numpy / paddle），脚本与配置全部使用相对路径。

复现命令：

```bash
conda activate paddle_env
cd D:\project1_steel
python src/cls/train.py --model mobilenet_v3_small
python src/cls/train.py --model resnet50_vd
python src/cls/eval_cls.py --model mobilenet_v3_small
python src/cls/eval_cls.py --model resnet50_vd
python src/cls/cam.py --model resnet50_vd
python src/cls/export.py --model resnet50_vd
python src/cls/train_curves.py
```

## 七、踩到的坑与处理（可写进报告的"实验过程"）

1. **本机环境与组长说明不一致**：组长 README 记录的环境路径是 `D:\Miniconda3\envs\paddle_env`，
   本机实际是 `C:\Users\zky\miniconda3\envs\paddle_env`（环境名相同，`conda activate paddle_env` 照用）；
   且 matplotlib / PyYAML / tqdm / scikit-learn 原本没装，已补齐（numpy 仍保持 2.5.2，未被动过）。
2. **预训练权重下载**：首次运行会自动从飞桨官方源下载（ResNet50 约 100 MB），本机可以正常下载。
   若在别的网络环境下载失败，可加 `--no-pretrained` 从零训练。
3. **飞桨 3.x 的导出格式变了**：`paddle.jit.save` 现在把计算图存成 `.json`（老教程里是 `.pdmodel`），
   权重仍是 `.pdiparams`；`export.py` 里额外做了"重新加载导出模型并与原模型比对输出"的校验，
   实测最大差异为 0.000e+00。
4. **本机是 CPU 版飞桨**（虽有 RTX 5060 显卡但飞桨为 CPU 编译版）。CPU 训练实测：
   MobileNetV3-small 51.6 s/轮、ResNet50 346.8 s/轮，与组长预估一致，分类任务在 CPU 上完全跑得动。
5. **图里用英文类别名**：不同电脑缺中文字体时 matplotlib 会显示方框，所以图内统一用英文，
   中文对照写在终端输出和 JSON 里。

## 八、局限与可改进方向

1. 精度已到顶（100%），无法再区分两个网络，因此对比重点放在参数量 / FPS / 收敛速度上。
2. 未做数据增强的消融实验；若老师要求"调参分析"，可补一组"无增强 vs 有增强"的对照（脚本已支持改配置）。
3. 未使用 GPU；若后续装了 GPU 版飞桨，`train.py` 无需改动即可加速。
4. 目前 `src/tools/metrics.py`（D 负责）仍为空文件，指标由 `src/cls/_common.py` 本地实现计算；
   接口已预留，D 补齐后 `eval_cls.py` 会自动调用并把结果写入 `tools_metrics` 字段。

## 九、产出的全部文件

```
src/cls/_common.py                      公共模块
src/cls/train.py                        训练
src/cls/eval_cls.py                     评估
src/cls/cam.py                          Grad-CAM
src/cls/export.py                       导出 + 测速
src/cls/train_curves.py                 训练曲线
src/cls/README_B.md                     使用说明
src/cls/configs/resnet50_vd.yml
src/cls/configs/mobilenet_v3_small.yml

results/weights/cls_{resnet50_vd,mobilenet_v3_small}_best.pdparams    最好权重
results/weights/cls_{...}_last.pdparams                               最后一轮权重
results/weights/cls_{...}_infer.{json,pdiparams}                      推理模型
results/logs/cls_{...}_<时间>.log / .csv                              训练日志与逐轮记录
results/logs/cls_test_usage.csv                                       测试集使用记录
results/metrics/cls_train_result_<模型>.json                          超参 + 逐轮 + 验证指标
results/metrics/cls_eval_result.json                                  测试集指标（给 D）
results/metrics/cls_export_result.json                                参数量/体积/FPS（给 D）
results/metrics/cls_summary.csv                                       汇总表（给 D 画对比图）
results/figures/cls_confusion_<模型>.png                              混淆矩阵
results/figures/cls_cam_<模型>.png                                    Grad-CAM 热力图
results/figures/cls_train_curves.png                                  训练曲线对比
```
