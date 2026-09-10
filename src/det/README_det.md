# 成员 C · 检测任务说明（src/det/）

本文件说明检测部分的实现思路、目录结构、使用方法与实测结论。
面向的读者是：接手复现的同学、写报告/PPT 的同学、以及答辩时被问到的评审。

---

## 一、总体设计：为什么不用 PaddleDetection

原计划按 README 用 PaddleDetection（PP-YOLOE-s + YOLOv3）。动手前实测了它的可行性，
结论是**不能用**，理由有实测依据：

| 检查项 | 实测结果 |
|---|---|
| `PaddleDetection/requirements.txt` 第 1 行 | `numpy < 2.0`，而本机 numpy 是 **2.5.2** |
| `git clone` release/2.9 后 `import ppdet` | 逐层失败：缺 `pycocotools` → `requests` → `imgaug.augmentables` → … |
| 需要的额外依赖 | 约 15 个，且必须把 numpy 降到 <2 |

把 numpy 降级会**连带搞崩成员 A 已经跑通的数据脚本**，这是项目铁律明确要避免的事。
更重要的是：**根本没必要**。实测 `paddle.vision.ops` 已经自带检测所需的全部算子：

```
deform_conv2d   nms（原生支持 category_idxs 分类批量 NMS）   yolo_box
matrix_nms      roi_align      prior_box      yolo_loss
generate_proposals      distribute_fpn_proposals      psroi_pool
```

所以最终方案是：**用原生 Paddle 手写两个检测网络，零新增依赖、不编译自定义 OP。**

这一选择也符合本项目既有风格 —— README 给成员 B 的建议原话是
「就用 `paddle.vision.models` 自己写训练循环，别去折腾 PaddleClas 源码……
手写训练循环不到 100 行，可控、好 debug、出错好查」。检测部分同理，且更甚。

---

## 二、目录结构

```
src/det/
├── train.py               训练脚本（--config 切换网络）
├── eval_det.py            测试集评估：mAP@0.5 / 每类 AP / P / R / F1 / PR 曲线 / 混淆矩阵
├── viz_det.py             可视化：真值-预测对比图、错例分析图、尺寸-AP 图
├── demo.py                单图/批量推理 Demo，可测 FPS
├── configs/
│   ├── yolov3.yml         YOLOv3 配置
│   └── ppyoloe_s.yml      PP-YOLOE-s 配置
├── core/                  两个网络共用的基础模块
│   ├── data.py            数据集、VOC 解析、Mosaic/缩放/翻转增强、推理封装
│   ├── boxes.py           坐标换算、IoU、NMS、坐标夹紧
│   ├── det_ops.py         IoU/DFL 损失、解码、正样本分配辅助、退化框过滤
│   ├── layers.py          ConvBN、Bottleneck、CSPLayer、RepConv、SPPF、ESE
│   └── utils.py           配置读取、随机种子、CSV/JSON 日志、**mAP 计算**
└── nets/
    ├── darknet.py         Darknet53 主干（YOLOv3 用）
    ├── yolov3.py          YOLOv3 检测器
    ├── cspresnet.py       CSPResNet 主干（PP-YOLOE-s 用）
    └── ppyoloe.py         PP-YOLOE-s 检测器
```

**共用与分工**：两个网络共用 `core/` 里的数据管线、坐标约定、NMS 后处理与评估代码，
差异**只体现在网络结构与回归方式**上。这样两者的 mAP 对比才能归因到
「网络设计差异」，而不是「数据口径不同导致的假差异」。

---

## 三、两个网络的设计对比

| | YOLOv3 | PP-YOLOE-s |
|---|---|---|
| 主干 | Darknet53（残差瓶颈 ×26） | CSPResNet（CSP + RepConv + ESE 注意力） |
| 颈部 | FPN 自顶向下三层融合 | CSPPAN 双向融合四层 |
| 检测方式 | **anchor-based**，9 个预设 anchor | **anchor-free**，每个特征点直接回归 |
| 框回归 | 直接回归 + GIoU 损失 | **DFL 分布式回归**（学「边落在哪个区间」的分布取期望）+ GIoU |
| 置信度 | obj × cls（两个分支） | 类别分数即置信度（无 obj 分支） |
| 正样本分配 | 宽高 IoU 最大的 anchor + 所在网格 | 按框等效边长选层 + 中心 2.5 格内锚点 |
| 参数量 | 6440.10 万 | 2375.17 万 |
| 输入尺寸 | 416 | 640 |

**为什么这个对比有意义**：本数据集的框宽高比跨度极大 ——
从细长划痕（Sc）到近方形麻点（PS）。anchor-based 方法需要预设 9 个宽高比，
很难覆盖全部形态；而 anchor-free + DFL 不依赖先验宽高，理论上在这种数据上更有优势。
实验设计上就是为了检验这一点，而不是随便挑两个网络凑数。

---

## 四、使用方法

```bash
conda activate paddle_env      # 本机实际路径见下方「环境」一节

# 1) 训练（两个网络串行跑，不要同时开，避免抢显存）
python src/det/train.py --config src/det/configs/yolov3.yml
python src/det/train.py --config src/det/configs/ppyoloe_s.yml

# 快速联调（3 个 epoch 看流程是否通，不追求精度）
python src/det/train.py --config src/det/configs/yolov3.yml --epochs 3 --batch-size 4

# 2) 评估（测试集只在这一步用一次）
python src/det/eval_det.py --model yolov3
python src/det/eval_det.py --model ppyoloe_s
python src/det/eval_det.py --model yolov3 --split val      # 调参请看验证集

# 3) 可视化
python src/det/viz_det.py --model yolov3
python src/det/viz_det.py --model ppyoloe_s --figs samples,errors,size_ap

# 4) 单图推理
python src/det/demo.py --model yolov3 --image dataset/det/JPEGImages/crazing_27.jpg
python src/det/demo.py --model ppyoloe_s --benchmark        # 顺带测 FPS
```

### 产出文件

| 路径 | 内容 |
|---|---|
| `results/weights/<name>_best.pdparams` | 验证集 mAP 最高的权重 |
| `results/weights/<name>_last.pdparams` | 最后 epoch 的权重 |
| `results/logs/<name>_train.csv` | 逐 epoch 记录（可交给 D 的 `log2table.py` 解析） |
| `results/metrics/<name>_train_summary.json` | 训练摘要：超参、最佳 mAP、耗时、逐 epoch 历史 |
| `results/metrics/<name>_eval_result.json` | 评估结果：mAP、每类 AP/P/R/F1、PR 曲线、混淆矩阵 |
| `results/figures/<name>_pr_curve.png` | 每类 PR 曲线 |
| `results/figures/<name>_per_class_ap.png` | 每类 AP 柱状图 |
| `results/figures/<name>_confusion.png` | 混淆矩阵 |
| `results/figures/<name>_samples.png` | 6 类真值 vs 预测对比图 |
| `results/figures/<name>_errors.png` | 错例分析（漏检/误检/类别错） |
| `results/figures/<name>_size_ap.png` | 不同目标尺寸下的 mAP |

---

## 五、环境（本机实测，与 README 记载不同）

README 的「开发环境」一节是按**另一台机器**写的，在本机不成立，以实测为准：

| 项 | README 记载 | 本机实测 |
|---|---|---|
| 环境路径 | `D:\Miniconda3\envs\paddle_env` | **`C:\Users\htt22\miniconda3\envs\paddle_env`** |
| Paddle 版本 | 3.3.1（CPU 版） | **paddlepaddle-gpu 3.2.0** |
| GPU | 无，CPU 训练 | **RTX 5070 Ti Laptop 12GB（sm_120）** |
| 本次补装的包 | — | `PyYAML 6.0.3`、`tqdm 4.70.0`、`opencv-python 5.0.0.93` |

补装用 `pip install --upgrade-strategy only-if-needed`，并先跑 `pip --dry-run` 预演，
确认 **numpy 2.5.2 不会被降级**（这是 README 反复强调的红线）。

> README 里「ResNet50 要跑 3 小时、检测网络在 CPU 上非常吃力、建议申请 AI Studio」
> 这一整套规划在本机不适用：有 GPU 之后训练快约 30 倍，检测任务完全跑得动。

---

## 六、实测性能

RTX 5070 Ti Laptop（12GB），1260 张训练集：

| 网络 | 输入 | batch | 单步耗时 | 每 epoch | 峰值显存 | 跑满训练耗时 |
|---|---|---|---|---|---|---|
| YOLOv3 | 416 | 16 | 181 ms | 约 19 s | 3.17 GB | 120 epoch / 38.0 分钟 |
| PP-YOLOE-s | 640 | 8 | 249 ms | 约 50 s | 9.7 GB | 100 epoch / 81.9 分钟 |
| PP-YOLOE-s | 640 | 16 | 51984 ms | 不可用 | 12.18 GB（顶满） | — |

**⚠️ 显存红线**：PP-YOLOE-s 在 640 输入下 **batch 必须 ≤ 8**。
bs=16 时峰值显存 12.18GB 正好顶满 12GB 的卡，触发显存重试，
单步从 249ms 暴涨到 **52 秒（慢 200 倍）**。这不是报错，是静默变慢，很容易被忽略。

**同类陷阱（实测踩过）**：**两个训练进程并发跑同样会触发显存重试**。
PP-YOLOE-s 在另一进程同时训练时，单 epoch 从 51 秒暴涨到 **7224 秒（慢 134 倍）**。
所以两个网络必须**串行训练** —— 与 README「CPU 训练实操建议」里
「两个网络串行训，不要同时开」是同一个道理，只是原因从 CPU 核数变成了显存。
本次交付前的跑满训练即按串行执行，全程显存稳定在 9.7GB、GPU 利用率 79~99%。

---

## 七、评测口径

- **坐标**：归一化 xyxy，取值 [0,1]，贯穿数据、网络、后处理、评估全流程。
  因此评估阶段**不需要任何坐标反变换**，预测与标注天然同尺度可比。
- **mAP**：VOC 口径 —— IoU 阈值 0.5、每类按分数降序匹配、101 点插值算 AP、
  mAP = 6 类 AP 的算术平均。与同类复现结果可直接对比。
- **评估输入**：统一缩放到固定正方形（`resize_mode='square'`）；
  训练用等比缩放 + padding。之所以评估不保留 padding：不同图的黑边比例不同，
  会让输入分布不确定、结果不可复现。
- **正确性验证**：mAP 实现经过 **6 项已知答案单测**：
  完美预测 = 1.0；框微抖（IoU 仍 >0.5）= 1.0；完全错位 = 0.0668；
  类别全错 = 0.0；只命中一半 = 0.5944；PR 曲线 recall 单调不减。全部符合预期。

### 指标实现说明（与成员 D 的接口）

`mAP / 每类 AP / PR 曲线` 目前实现在 `src/det/core/utils.py` 的 `evaluate_map()`。
原因：成员 D 的 `src/tools/metrics.py` 尚未交付（当时为 0 字节），C 的评估不能停工等它。
**接口按可替换设计**：训练与评估脚本只通过 `utils.evaluate_map()` 这一个入口调用指标，
将来 D 的指标库就绪后，把该函数内部改为 `from tools.metrics import evaluate_map`
即可整体切换，调用方一行都不用改。

---

## 八、工程细节：几个踩过的坑（都已修复，记下来避免重犯）

这一节对写报告「遇到的问题与解决」很有用，按代价从大到小排。

### 1. `masked_select` 的反向慢 1600 倍
最初用 `masked_select` 取出正样本再算回归损失：前向只要 1ms，
但**反向要 1.6 秒/步**（8500×68 中间张量展开 + 散射梯度）。
改成「全量算损失 + 掩码加权归约」后，**单步从 3927ms 降到 249ms，快 16 倍**。

### 2. IoU 损失误用两两矩阵导致爆显存
`IoULoss` 原实现调用 `box_iou` 构造 (N,N) 矩阵。输入是 (B,N,4) 时 N 可达 68000，
矩阵元素上亿，实测申请 **34.45GB**（显存只有 12GB）直接 OOM。
实际上 IoU 损失是**一一对应**关系（第 i 个预测框 vs 第 i 个目标框），
根本不需要两两矩阵，改为逐元素直算即可。

### 3. 退化框污染 mAP
`clip_boxes_norm` 原实现把四个坐标各自 clip 到 [0,1]，
会把部分在画面外的框压成零宽/零高（x1、x2 同时被压到 1）。
零面积框在 NMS 里被当成互不重叠而全部留下 —— 实测 **27000 个框里有 4082 个是退化框**。
改成「先夹中心点与宽高，再由中心重构四角」后，退化框 **0 个**。
（附带发现：`paddle.clip` 的 min/max 只接受标量，逐元素边界要用 `maximum`/`minimum`。）

### 4. 非二次幂输入尺寸让 FPN 相加报错
原图 200×200 时三尺度是 25/13/7，而 7 的两倍是 14、目标层是 13，
硬写 `scale_factor=2` 直接 Broadcast 报错。
改为按目标特征图尺寸 `interpolate(size=[h,w])` 自适应缩放，对任意输入边长都成立。

### 5. 损失归一化口径写错会让训练直接发散
YOLOv3 的 obj 分支原按「正样本数」归一化，但一张图有上万候选位置、
正样本只有 2~3 个，损失被放大到 **44636**、梯度范数 **2.4e6**。
正确做法是**按全部元素取平均**，正负不均衡靠 `pos_weight` 调节，绝不改分母。

### 6. 学习率 warmup 形同虚设
自写 `LambdaDecay` 闭包时，optimizer 内部还会对 lr 再做一次处理，
实测学习率只升到预期的 1/30（3.1e-5 而非 9.5e-4），训练 4 个 epoch 几乎没学到东西。
改用原生 `LinearWarmup + CosineAnnealingDecay` 组合并逐点验证后才正常。
另外 `warmup_epochs` 不能超过总 epoch 的 1/5，否则短程试验会全程处于 warmup。

### 7. 主干结构连写错三次
Darknet53 的 stem 下采样次数、各阶段通道数、YOLOv3 到底取哪三个阶段的输出，
这三处各错了一次，症状都是「通道数不匹配」或「特征图尺寸不对」。
最后靠**逐层打印实际形状**（而不是靠推演）才定位。
教训：像 Darknet53 这种有固定官方层序的结构，应当逐层写死并与官方表对照，
不要用「循环 + 通道表」去推导 —— 容易把 /2 与 /4 的位置搞反。

---

## 九、结果

> 全部数字由 `python src/det/tools/summarize.py` 自动汇总到
> `results/metrics/det_summary.md`，与 `results/metrics/*.json` 一一对应，不存在手抄误差。

### 9.1 区域验证器（本方案实际有效的核心）

| 指标 | 数值 |
|---|---|
| 图块分类准确率（验证集） | **0.8872** |
| GT 框上的区域分类准确率（验证集） | **95.31%** |
| GT 框上的区域分类准确率（**测试集**） | **97.00%** |
| 参数量 | 258.67 万 |

测试集 634 个 GT 框上，各类召回：Cr 0.981 / In 0.974 / Pa 0.954 / PS 0.943 / RS 0.978 / Sc 0.988。
训练样本为「GT 框裁剪（6 类缺陷）+ 随机背景区域」，共 4142 个图块。

### 9.2 检测性能（mAP@0.5）

| 方案 | 划分 | mAP@0.5 | 精确率 | 召回率 | 每图框数 |
|---|---|---|---|---|---|
| YOLOv3 单阶段（120 epoch 跑满） | 验证集 | 0.0007 | — | — | — |
| PP-YOLOE-s 单阶段（100 epoch 跑满） | 验证集 | 0.0000 | — | — | — |
| YOLOv3 检测器 + 验证器 | 验证集 | 0.0000 | 0.0000 | 0.0000 | 46.5 |
| YOLOv3 检测器 + 验证器 | 测试集 | 0.0000 | 0.0002 | 0.0032 | 47.3 |
| **滑窗 + 验证器** | 验证集 | **0.0709** | 0.0262 | **0.5383** | 48.7 |
| **滑窗 + 验证器** | **测试集** | **0.0548** | 0.0248 | **0.5142** | 48.8 |

> 两个单阶段网络均已**按配置跑满**（YOLOv3 120 epoch / 38.0 分钟，
> PP-YOLOE-s 100 epoch / 81.9 分钟）。跑满后最佳验证 mAP 分别为 0.0007 与 0.0000，
> 与调试期只跑 40/17 epoch 时相比**没有实质改善**（0.0000 → 0.0007）。
> 这条实验证据很重要：它排除了「训练不充分」这一解释，
> 确认问题出在置信度分支的机制本身（见 9.3 的根因分析）。

每类 AP@0.5（滑窗 + 验证器）：

| 类别 | 中文 | 验证集 AP | 测试集 AP |
|---|---|---|---|
| Cr | 龟裂 | 0.090 | 0.092 |
| In | 夹杂 | 0.013 | 0.006 |
| Pa | 斑块 | 0.120 | 0.112 |
| PS | 麻点 | 0.086 | 0.044 |
| RS | 氧化铁皮压入 | 0.116 | 0.074 |
| Sc | 划痕 | 0.000 | 0.001 |

### 9.3 结论（这部分建议直接写进报告）

**1) 拿到了什么**

- 一个可靠的**缺陷区域分类器**：在测试集 GT 框上 97.00% 准确率，每类召回均 ≥0.94；
- 一个可跑的**端到端检测流程**：滑窗 + 验证器，测试集 mAP@0.5 = 0.0548、召回 0.514；
- **两套检测网络**（YOLOv3 / PP-YOLOE-s）的完整实现、训练、评估与可视化脚本，
  全部基于原生 Paddle，零新增依赖、不编译自定义 OP。

**2) 没拿到什么，以及为什么（这是本次工作最有价值的部分）**

单阶段检测器的**框定位是好的，但「哪个框可信」学不出来**：

| 检查项 | 实测 | 说明 |
|---|---|---|
| GT 所属特征点上的框 IoU | **0.742 / 0.757** | 定位没问题 |
| 同一框的 obj 分数 | 0.056 | 在 307 个候选中排**第 135 位** |
| 结果 | mAP@0.5 = 0 | 被 top_k 截断淘汰 |
| 根因 | 正样本 2~3 个/图 vs 候选位置 10^4 量级（0.3%） | 置信度分支的类别极度不平衡 |

已尝试并**排除**的方案（都有实测数据支撑）：

| 尝试 | 结果 |
|---|---|
| 改置信度排序方式：obj×cls / 纯 cls / 纯 obj / obj×cls² | mAP 全部为 0 |
| 提高 top_k：100 → 300 → 1000 | 0.0002 → 0.0025 → 0.0025（触顶） |
| 用边缘能量（梯度幅值）当分数 | 0（该数据集背景本身纹理就重，边缘不是有效线索） |
| obj 分支偏置初始化 -4.5 / -2.0 / 0 | 无实质改善 |
| obj 损失改为正负样本各自归一化 | 无实质改善 |
| k-means 定制 anchor（IoU 0.567→0.692） | 定位略好，mAP 仍为 0 |
| 统一训练/验证输入尺度（原来随机缩放导致不一致） | 修正了一个真 bug，但 mAP 仍为 0 |
| 双阶段：检测器候选 + 验证器重打分 | 仍为 0，因为正确框排在 135 位、被候选数上限截断 |
| 按目标尺寸分档评估（`viz_det.py --figs size_ap`） | 大/中/小框的 mAP **全部为 0** —— 排除「小目标难检」这一解释 |
| **把训练跑满**（YOLOv3 40→120 epoch、PP-YOLOE-s 17→100 epoch） | 0.0000 → **0.0007**（YOLOv3）、0.0000（PP-YOLOE-s）—— 训练轮数翻三倍几乎没有变化，**排除「训练不充分」** |

**3) 根因分析**

单阶段检测器的置信度分支本质上是「在 10^4 量级的候选里做二分类、正样本只有 2~3 个」。
本数据集只有 1800 张图、平均每图 2.3 个框，正样本的绝对数量太少，
网络学不出「这个位置的框好不好」这个判断 —— 它能把框回归准（因为那是稠密监督），
但学不出可靠的置信度（那是极端不平衡的稀疏监督）。

这也解释了为什么**滑窗 + 验证器能工作**：它把同一件事（判断一块区域是不是缺陷）
变成了**稠密监督的分类任务**，每个训练样本都有明确标签，不存在正负样本悬殊的问题。

**4) 下一步改进建议**

1. 给置信度分支加**真正有效的重采样/损失**：Focal Loss 或 OHEM（只回传损失最大的
   若干负样本），而不是简单地调 pos_weight；
2. 换用**有预训练权重**的主干（本项目受限于 PaddleDetection 的 numpy<2 依赖无法引入，
   若后续能单独拿到 backbone 权重，对小数据集帮助会很大）；
3. 把验证器的定位也做起来：在滑窗基础上加一个**框回归头**（类似 RPN 的 refine），
   把 IoU 从目前的 0.507 提到 0.6+ 应能显著改善 mAP@0.5；
4. 数据层面：把 200×200 的原图做重叠裁块扩充训练样本数，
   可缓解小数据集下正样本过少的问题。

### 9.4 关于测试集

按项目铁律，测试集只在最终评估时使用。本任务中测试集在下列脚本中被读取过：
`eval_2stage.py --split test`（滑窗与检测器两种模式）、
以及验证器在 GT 框上的分类准确率统计。
训练阶段（`train.py` / `train_verifier.py`）**始终只加载 train 与 val**，
测试集未参与任何调参决策。

