# 成员 C · 检测任务工作总结

| 项目 | 内容 |
|---|---|
| 项目 | 基于东北大学 NEU-CLS 数据集的钢材表面缺陷检测 |
| 负责范围 | 成员 C —— 任务2「缺陷检测与定位」 |
| 工作目录 | `src/det/`（代码）、`results/`（产物） |
| 完成日期 | 2026-09-10 |
| 技术文档 | `src/det/README_det.md`（实现细节与踩坑记录） |
| 结果总表 | `results/metrics/det_summary.md`（脚本自动生成） |

---

## 一、任务要求与完成情况

作业对检测部分的硬性要求：

1. 训练集 / 验证集 / 测试集 = **7 : 1.5 : 1.5**，测试集与训练集完全独立；
2. 至少搭建 **2 种网络**并对比效果差异。

| 要求 | 完成情况 |
|---|---|
| 数据划分 7:1.5:1.5（1260/270/270） | ✅ 直接使用成员 A 的划分，训练全程只加载 train/val，测试集仅在最终评估使用 |
| 至少 2 种网络对比 | ✅ 实现了 **2 个检测网络**（YOLOv3、PP-YOLOE-s）+ **1 个区域验证器**，共 3 个网络 |
| 缺陷检测与定位 | ✅ 交付可运行的端到端检测流程，并产出 mAP@0.5、每类 AP、PR 曲线、混淆矩阵、检测效果图 |

**超额部分**：README 要求「至少 2 种」，本任务实际训练并评估了 3 个网络
（外加 2 种检测流程：单阶段、两阶段/滑窗）。

---

## 二、最重要的一个方案调整：没有使用 PaddleDetection

原计划按 README 用 PaddleDetection（PP-YOLOE-s + YOLOv3）。**动手前实测发现不能用**：

| 检查项 | 实测结果 |
|---|---|
| `PaddleDetection/requirements.txt` 第 1 行 | `numpy < 2.0`，而本机是 **numpy 2.5.2** |
| `git clone` release/2.9 后 `import ppdet` | 逐层失败：缺 `pycocotools` → `requests` → `imgaug.augmentables` → … |
| 需要的额外依赖 | 约 15 个，且**必须把 numpy 降到 <2** |

降 numpy 会连带破坏成员 A 已完成的数据脚本，这是项目铁律明确要避免的事。
更关键的是**没必要**——实测 `paddle.vision.ops` 原生自带检测所需的全部算子：

```
deform_conv2d   nms（原生支持 category_idxs 分类批量 NMS）   yolo_box
matrix_nms      roi_align      prior_box      yolo_loss
generate_proposals      distribute_fpn_proposals      psroi_pool
```

**最终方案：用原生 Paddle 手写两个检测网络，零新增依赖、不编译自定义 OP。**
这也符合项目既有风格（README 给成员 B 的建议同样是「自己写训练循环，别折腾框架源码」）。

---

## 三、交付物清单

### 3.1 代码（`src/det/`，24 个 Python 文件，3708 行）

| 模块 | 行数 | 作用 |
|---|---|---|
| `train.py` | 199 | 训练脚本，`--config` 切换网络 |
| `train_verifier.py` | 124 | 区域验证器训练 |
| `eval_det.py` | 247 | 单阶段评估：mAP@0.5 / 每类 AP / P / R / F1 / PR / 混淆矩阵 |
| `eval_2stage.py` | 241 | 两阶段评估（检测器或滑窗 + 验证器） |
| `viz_det.py` | 294 | 真值-预测对比图、错例分析、尺寸-AP 图 |
| `demo.py` | 196 | 单图/批量推理 + FPS 实测 |
| `core/data.py` | 313 | 数据集、VOC 解析、Mosaic/缩放/翻转增强、推理封装 |
| `core/det_ops.py` | 241 | IoU/DFL 损失、编解码、退化框过滤 |
| `core/utils.py` | 191 | 配置、随机种子、日志、**mAP 计算** |
| `core/verifier.py` | 157 | 区域验证器（两阶段第二阶段） |
| `core/layers.py` | 140 | ConvBN / Bottleneck / CSPLayer / RepConv / SPPF / ESE |
| `core/boxes.py` | 97 | 坐标换算、IoU、NMS、坐标夹紧 |
| `core/window_detector.py` | 80 | 多尺度多宽高比滑窗检测器 |
| `nets/yolov3.py` + `nets/darknet.py` | 240 + 63 | YOLOv3 与 Darknet53 主干 |
| `nets/ppyoloe.py` + `nets/cspresnet.py` | 244 + 57 | PP-YOLOE-s 与 CSPResNet 主干 |
| `tools/kmeans_anchors.py` | 144 | k-means 统计目标尺寸、生成定制 anchor |
| `tools/diagnose.py` | 135 | 诊断脚本（定位质量、各 IoU 阈值 mAP 等） |
| `tools/two_stage_compare.py` | 157 | 单阶段 vs 两阶段对比实验 |
| `tools/summarize.py` | 130 | 汇总全部结果生成报告用表格 |

配置：`configs/yolov3.yml`、`configs/ppyoloe_s.yml`（共 105 行，含超参与设计理由注释）

### 3.2 结果产物（`results/`）

| 类型 | 内容 |
|---|---|
| 权重 | `yolov3_best/last.pdparams`、`ppyoloe_s_best.pdparams`、`verifier_best.pdparams` |
| 评估报告 | 5 份 `*_eval2stage_*.json`（含每类 AP、PR 曲线、混淆矩阵）+ `det_summary.json/md` |
| 训练日志 | `*_train.csv`（逐 epoch）、`*_train_summary.json`（超参+最佳结果+耗时） |
| 图表 | 16 张：PR 曲线、每类 AP、混淆矩阵、检测示例、错例分析、尺寸-AP |
| 分析 | `anchor_kmeans.json`（anchor 匹配度分析） |

### 3.3 文档

- `src/det/README_det.md`（197 行）：设计思路、使用方法、环境、评测口径、**踩坑记录**
- `results/metrics/det_summary.md`：结果总表（由 `tools/summarize.py` 自动生成）
- 本文件：工作总结

---

## 四、环境情况（与原 README 记载不同）

**本机与 README 下方「开发环境」一节记载的不是同一台机器**，实测如下：

| 项 | README 记载 | 本机实测 |
|---|---|---|
| 环境路径 | `D:\Miniconda3\envs\paddle_env` | **`C:\Users\htt22\miniconda3\envs\paddle_env`** |
| Paddle 版本 | 3.3.1（CPU 版） | **paddlepaddle-gpu 3.2.0** |
| GPU | 无，CPU 训练 | **RTX 5070 Ti Laptop 12GB（sm_120）** |
| numpy | 2.5.2 | 2.5.2（未变动） |

本次为检测任务补装了 3 个包（用 `--upgrade-strategy only-if-needed`，
并先跑 `pip --dry-run` 预演确认不会降级 numpy）：

```
PyYAML 6.0.3        tqdm 4.70.0        opencv-python 5.0.0.93
```

**实测训练速度**（有 GPU，比 README 记载的 CPU 数据快约 30 倍）：

| 网络 | 输入 | batch | 单步 | 每 epoch | 峰值显存 |
|---|---|---|---|---|---|
| YOLOv3 | 416 | 16 | 181 ms | 约 19 s | 3.17 GB |
| PP-YOLOE-s | 640 | 8 | 249 ms | 约 50 s | 9.7 GB |
| PP-YOLOE-s | 640 | 16 | **51984 ms** | 不可用 | 12.18 GB（顶满） |

> ⚠️ **显存红线**：PP-YOLOE-s 在 640 输入下 batch **必须 ≤ 8**。
> bs=16 时峰值显存顶满 12GB，触发显存重试，单步从 249ms 暴涨到 52 秒（慢 200 倍）。
> 这不是报错，是静默变慢，很容易被误判成"模型太大跑不动"。
>
> 另一个同类陷阱：**两个训练进程并发跑也会触发显存重试**。
> 实测 PP-YOLOE-s 在另一进程同时训练时，单 epoch 从 54 秒暴涨到 7224 秒（慢 134 倍）。
> 所以两个网络必须**串行训练**——这也是 README「CPU 训练实操建议」里
> 「两个网络串行训，不要同时开」的同一个道理，只是原因从 CPU 核数变成了显存。

**两个网络的跑满训练结果**（均已按配置跑满，非调试期的截断训练）：

| 网络 | epoch | 参数量(万) | 最佳验证 mAP@0.5 | 训练耗时 |
|---|---|---|---|---|
| YOLOv3 | **120（跑满）** | 6440.10 | **0.0007** | 38.0 分钟（19.0 s/epoch） |
| PP-YOLOE-s | **100（跑满）** | 2375.17 | **0.0000** | 81.9 分钟（49.2 s/epoch） |

> 此前调试期只跑了 40 / 17 epoch，本次已重跑至配置上限。**结论没有改变**：
> YOLOv3 从 0.0000 升到 0.0007，PP-YOLOE-s 仍为 0.0000。
> 这从实验上确认了 §8.2 的判断 —— **mAP 偏低的原因不是训练轮数不够**，
> 而是置信度分支在正样本占比 0.3% 的极端不平衡下学不出判别力。
> 训练轮数翻三倍只换来 0.0007 的提升，这条线索可以排除了。

**这个对比的设计意图**：本数据集框宽高比跨度极大（细长划痕 Sc 到近方形麻点 PS），
anchor-based 方法需要预设宽高比，而 anchor-free + DFL 不依赖先验宽高，
理论上更适合。实验就是为了检验这一点。

---

## 五、两个检测网络的设计对比

| | YOLOv3 | PP-YOLOE-s |
|---|---|---|
| 主干 | Darknet53（残差瓶颈） | CSPResNet（CSP + RepConv + ESE 注意力） |
| 颈部 | FPN 自顶向下三层融合 | CSPPAN 双向融合四层 |
| 检测方式 | anchor-based（9 个预设 anchor） | **anchor-free**（每点直接回归） |
| 框回归 | 直接回归 + GIoU | **DFL 分布式回归** + GIoU |
| 置信度 | obj × cls | 类别分数 |
| 参数量 | 6440.10 万 | 2375.17 万 |
| 输入尺寸 | 416 | 640 |
| 训练情况 | **120 epoch（跑满），38.0 分钟**，最佳验证 mAP 0.0007 | **100 epoch（跑满），81.9 分钟**，最佳验证 mAP 0.0000 |

**这个对比的设计意图**：本数据集框宽高比跨度极大（细长划痕 Sc 到近方形麻点 PS），
anchor-based 方法需要预设宽高比，而 anchor-free + DFL 不依赖先验宽高，
理论上更适合。实验就是为了检验这一点。

---

## 六、结果

> 全部数字由 `python src/det/tools/summarize.py` 汇总，与 `results/metrics/*.json` 一一对应。

### 6.1 区域验证器（本方案实际有效的核心）

| 指标 | 数值 |
|---|---|
| 图块分类准确率（验证集） | **0.8872** |
| **GT 框上的区域分类准确率（测试集）** | **97.00%** |
| GT 框上的区域分类准确率（验证集） | 95.31% |
| 参数量 | 258.67 万 |
| 训练样本 | 4142 个图块（GT 框裁剪 + 随机背景），30 epoch / 2.3 分钟 |

测试集 634 个 GT 框上各类召回：

| 类别 | Cr 龟裂 | In 夹杂 | Pa 斑块 | PS 麻点 | RS 氧化铁皮压入 | Sc 划痕 |
|---|---|---|---|---|---|---|
| 召回 | 0.981 | 0.974 | 0.954 | 0.943 | 0.978 | 0.988 |

### 6.2 检测性能（mAP@0.5）

| 方案 | 划分 | mAP@0.5 | 精确率 | 召回率 |
|---|---|---|---|---|
| YOLOv3 单阶段 | 验证集 | 0.0000 | 0.0000 | 0.0000 |
| YOLOv3 检测器 + 验证器 | 测试集 | 0.0000 | 0.0002 | 0.0032 |
| **滑窗 + 验证器** | 验证集 | **0.0709** | 0.0262 | **0.5383** |
| **滑窗 + 验证器** | **测试集** | **0.0548** | 0.0248 | **0.5142** |

每类 AP@0.5（滑窗 + 验证器，测试集）：

| 类别 | Cr | In | Pa | PS | RS | Sc |
|---|---|---|---|---|---|---|
| AP | 0.092 | 0.006 | 0.112 | 0.044 | 0.074 | 0.001 |

### 6.3 anchor 与数据集匹配度分析

| 项 | 最佳宽高 IoU 均值 |
|---|---|
| YOLOv3 官方 anchor | 0.567 |
| k-means 定制 anchor | **0.692**（+0.124） |

官方 anchor 是为 COCO（目标普遍很小）设计的，而本数据集缺陷框平均占全图 17.45%。
实测官方 9 个 anchor 中**最小的 3 个分配占比为 0%**，完全没被用到。

---

## 七、遇到的问题（按代价排序）

详细技术记录见 `src/det/README_det.md` 第八节。下面按"花了多少时间"排序：

| # | 问题 | 根因 | 修复 |
|---|---|---|---|
| 1 | **单阶段检测器 mAP 恒为 0** | 置信度分支正样本仅占候选位置 0.3%，学不出判别力 | 未根治；改用「滑窗 + 验证器」绕过（详见第七节） |
| 2 | `masked_select` 反向慢 1600 倍 | 8500×68 中间张量展开 + 散射梯度 | 改为「全量算损失 + 掩码加权归约」，提速 **16 倍** |
| 3 | IoU 损失爆显存（申请 34GB） | 误用两两矩阵，而 IoU 损失是一一对应关系 | 改为逐元素直算 |
| 4 | 退化框污染 mAP（4082/27000） | 四个坐标各自 clip 到 [0,1] 把框压成零宽高 | 改为「先夹中心点与宽高，再由中心重构」，退化框归 0 |
| 5 | 学习率 warmup 形同虚设 | 自写 `LambdaDecay` 被 optimizer 二次处理，只升到预期 1/30 | 改用原生 `LinearWarmup + CosineAnnealingDecay` |
| 6 | obj 损失 44636、梯度 2.4e6 | 按正样本数归一化，而负样本上万 | 改为按全部元素取平均 |
| 7 | 训练/验证输入尺度不一致 | 随机缩放后每 batch 尺寸不同，特征图 7×7 vs 13×13 | 统一 padding 到固定边长 |
| 8 | Darknet53 结构连写错 3 次 | 用循环+通道表推导时把 /2 与 /4 位置搞反 | 改为逐层写死并与官方表对照，靠逐层打印形状定位 |
| 9 | FPN 相加报 Broadcast 错 | 200 输入时尺度是 25/13/7，硬写 `scale_factor=2` | 改为按目标尺寸 `interpolate(size=)` |
| 10 | `nms` 报 top_k 断言失败 | 训练早期一张图只有几个框，少于 top_k | 把 top_k 夹到实际框数 |
| 11 | `viz_det.py` PIL 报错 | 极小框转像素后不足 1px | 画图前统一做框合法化 |

---

## 八、结论与后续建议

### 8.1 拿到了什么

1. **一个可靠的缺陷区域分类器**：测试集 GT 框上 97.00% 准确率，每类召回均 ≥0.94；
2. **一个可跑的端到端检测流程**：滑窗 + 验证器，测试集 mAP@0.5 = 0.0548、召回 0.514；
3. **两套检测网络的完整实现**（YOLOv3 / PP-YOLOE-s）+ 训练/评估/可视化/Demo 全套脚本，
   全部基于原生 Paddle，零新增依赖、不编译自定义 OP；
4. **一套可复现的评测工具链**：mAP 实现经 6 项已知答案单测
   （完美预测=1.0、框微抖=1.0、完全错位=0.0668、类别全错=0.0、只命中一半=0.5944、
   PR 曲线 recall 单调不减）。

### 8.2 没拿到什么，以及为什么

单阶段检测器的**框定位是好的，但「哪个框可信」学不出来**：

| 检查项 | 实测 |
|---|---|
| GT 所属特征点上的框 IoU | **0.742 / 0.757**（定位没问题） |
| 同一框的 obj 分数 | 0.056，在 307 个候选中排**第 135 位** |
| 结果 | 被 top_k 截断淘汰，mAP@0.5 = 0 |
| 根因 | 正样本 2~3 个/图 vs 候选位置 10⁴ 量级（0.3%） |

已尝试并**逐项排除**的方案（均有实测数据）：

改排序方式（obj×cls / 纯 cls / 纯 obj / obj×cls²）· top_k 100→1000 ·
边缘能量当分数 · obj 偏置初始化（-4.5 / -2.0 / 0）· obj 损失正负各自归一化 ·
k-means 定制 anchor · 统一输入尺度 · 检测器候选 + 验证器重打分 ·
按尺寸分档评估（大中小框 mAP **全部为 0**，排除"小目标难检"这一解释）

**核心判断**：单阶段检测器的置信度分支本质是「在上万候选里做二分类、正样本只有 2~3 个」。
本数据集仅 1800 张图、平均每图 2.3 个框，正样本绝对数量太少，学不出可靠置信度
（框回归能学准，因为它是稠密监督；置信度学不准，因为它是极端不平衡的稀疏监督）。

这也正好解释了**为什么滑窗 + 验证器能工作**：它把同一件事（判断一块区域是不是缺陷）
变成了**稠密监督的分类任务**，不存在正负样本悬殊的问题。

### 8.3 下一步改进建议

1. **给置信度分支加真正有效的机制**：Focal Loss 或 OHEM（只回传损失最大的若干负样本），
   而不是简单调 pos_weight；
2. **换用有预训练权重的主干**：本项目受限于 PaddleDetection 的 numpy<2 依赖无法引入，
   若能单独拿到 backbone 权重，对小数据集帮助会很大；
3. **给验证器加框回归头**：在滑窗基础上做类似 RPN 的 refine，
   把 IoU 从目前 0.507 提到 0.6+ 应能显著改善 mAP@0.5；
4. **数据层面扩充**：把 200×200 原图做重叠裁块，缓解小数据集下正样本过少的问题。

---

## 九、复现方式

```bash
conda activate paddle_env      # 本机实际路径见第四节

# 1) 训练两个检测网络（串行跑，避免抢显存）
python src/det/train.py --config src/det/configs/yolov3.yml
python src/det/train.py --config src/det/configs/ppyoloe_s.yml

# 2) 训练区域验证器（两阶段第二阶段，约 2 分钟）
python src/det/train_verifier.py

# 3) 评估（测试集只在这一步用一次）
python src/det/eval_det.py --model yolov3                        # 单阶段
python src/det/eval_2stage.py --mode window --split test         # 滑窗+验证器

# 4) 可视化与 Demo
python src/det/viz_det.py --model yolov3 --split val
python src/det/demo.py --model yolov3 --benchmark

# 5) 汇总结果（生成报告用表格）
python src/det/tools/summarize.py

# 辅助分析工具
python src/det/tools/kmeans_anchors.py      # anchor 匹配度分析
python src/det/tools/diagnose.py --model yolov3 --split val   # 诊断 mAP 偏低的环节

# 快速联调（3 个 epoch 验证流程是否通）
python src/det/train.py --config src/det/configs/yolov3.yml --epochs 3 --batch-size 8
```

### 关于测试集

按项目铁律，训练阶段（`train.py` / `train_verifier.py`）**始终只加载 train 与 val**。
测试集仅在最终评估时被读取：`eval_2stage.py --split test`（滑窗与检测器两种模式），
以及验证器在 GT 框上的分类准确率统计。**测试集未参与任何调参决策。**

---

## 十、遗留事项与风险

| 项 | 状态 | 说明 |
|---|---|---|
| 两个网络均已跑满 | ✅ 已完成 | YOLOv3 120 epoch / 38.0 分钟（验证 mAP 0.0007）、PP-YOLOE-s 100 epoch / 81.9 分钟（验证 mAP 0.0000）。均按配置上限训练，无截断 |
| 单阶段检测器 mAP 仍接近 0 | ⚠️ 已知限制 | 跑满训练后并未改善（0.0000→0.0007），已确认根因不在训练轮数（详见 §8.2）。最终可用的是「滑窗 + 验证器」流程 |
| 模型权重未纳入版本库 | ℹ️ 正常 | `results/weights/` 被 `.gitignore` 排除（清理中间快照后约 683MB），符合项目约定，需重新训练或单独传输 |
| 训练中间快照占空间 | ℹ️ 已处理 | `save_every: 20` 会为每个网络生成 5~6 个快照（单网络可达 1.5GB）。交付前已清理，仅保留 `best` 与 `last` |
| `src/det/__init__.py` 为空 | ℹ️ 正常 | 脚本以 `sys.path` 注入方式导入 `core`/`nets`，不需要包级导出 |
| mAP 实现暂在 `core/utils.py` | ℹ️ 待替换 | 成员 D 的 `src/tools/metrics.py` 尚未交付。接口已按可替换设计：训练/评估只通过 `utils.evaluate_map()` 调用，将来把该函数内部改为 `from tools.metrics import evaluate_map` 即可整体切换，调用方一行不用改 |

---

## 附：一句话向组员说明

> C 的检测部分做完了：两个网络（YOLOv3、PP-YOLOE-s）都是原生 Paddle 手写的，
> 没用 PaddleDetection（它要求 numpy<2，会把环境搞崩）。
> 两个网络都已按配置跑满（120 / 100 epoch），但实测下来单阶段检测器的框定位虽然准（IoU 0.74），
> 「哪个框可信」这个判断学不出来——因为正样本只占候选位置的 0.3%，太少；
> 跑满训练也验证了这一点（翻三倍轮数只换来 0.0007 的提升）。
> 所以我又训练了一个缺陷区域分类器（测试集 GT 框上 **97% 准确率**），
> 用「滑窗 + 验证器」跑通了端到端检测，测试集 **mAP@0.5 = 0.0548、召回 0.514**。
> 详细原因分析和 11 个踩过的坑都写在 `src/det/README_det.md` 里了。
