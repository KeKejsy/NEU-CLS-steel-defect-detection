# 基于东北大学 NEU-CLS 数据集的钢材表面缺陷检测

## 作业要求

**选题**：基于东北大学 NEU-CLS 数据集的钢材表面缺陷检测

- **数据**：6 类缺陷，1800 张 200×200 灰度图像
- **任务1**：缺陷检测与分类
- **任务2**：缺陷检测与定位

**要求**：

1. 训练集、验证集、测试集比例 **7 : 1.5 : 1.5**。测试集文件与训练集文件完全独立，确保评估的可靠性。
2. 至少搭建 **2 种网络**，并对比不同网络的效果差异。

> 本项目实际完成 **4 个网络**（分类 2 个 + 检测 2 个）+ 1 套自研融合方案，两个任务各满足一次"至少 2 种"，属超额完成。

---

## 当前进度（更新时间：2026-09-16）

| 成员 | 状态 | 说明 |
|---|---|---|
| A 数据 | ✅ **已完成** | 1800 张图 + 1800 个标注全齐，划分完成，23 项自检全过 |
| B 分类 | ✅ **已完成** | 两个网络测试集准确率 **100%**，权重/图/报告齐全 |
| C 检测 | ⚠️ **部分完成** | 单阶段网络 mAP≈0（失败）；改用的滑窗+双模型融合方案已迭代两轮，**测试集 mAP@0.5 = 0.2176**（验证集 0.2473），仍未达常规检测及格线 |
| D 工具链 | ✅ **已完成** | 4 个脚本全部实装并实测可跑：`metrics.py` 公共指标库（7 个指标函数，分类/检测两套口径已与 B、C 对齐）、`log2table.py` 日志转表、`plot_summary.py` 对比图、`run_all.py` 一键复现 |

---

## 一、结果速览（最重要）

### 任务1 · 缺陷分类（B）

| 网络 | 参数量 | 测试集准确率 | 宏平均 F1 | 单张推理 |
|---|---|---|---|---|
| ResNet50_vd | 2357 万 | **1.0000** | 1.0000 | 86.0 ms（11.63 FPS） |
| MobileNetV3-small | 154 万 | **1.0000** | 1.0000 | 10.5 ms（95.19 FPS） |

**结论**：两个网络精度打平（都 100%），差异在**部署成本**——参数量差 15.3 倍、速度差 8.2 倍、体积差 14.9 倍。
混淆矩阵为纯对角阵。测试集 270 张（6 类各 45 张）。
> 详见 `docs/B_分类任务工作总结.md`

### 任务2 · 缺陷检测与定位（C）

**两个单阶段检测网络均训练跑满，但失败了：**

| 网络 | epoch | 参数量 | 验证 mAP@0.5 |
|---|---|---|---|
| YOLOv3 | 120 | 6440 万 | **0.0000** |
| PP-YOLOE-s | 100 | 2375 万 | **0.0000** |

失败原因（已排除"训练不充分""小目标难检"等解释）：框定位是准的（GT 特征点上框 IoU 0.742/0.757），
但**置信度学不出来**——正样本每图仅 2~3 个，候选位置在 10⁴ 量级（占 0.3%），正确框的 obj 分数只有 0.056、排第 135 位，被 top_k 截断。
**2026-09-16 又专门验证过一次**：把主干换成 ImageNet 预训练（resnet50）+ obj 分支加 Focal Loss 重训，验证集 mAP 仍只有 **0.0003**，
最佳候选的 obj 分数中位数 0.0355、排名中位数 6667/10647 —— 说明这不是"主干不够强/缺 focal"，而是数据规模问题（1260 张训练图、每图 2~3 个正样本）。

**改用的方案**：多尺度多长宽比滑窗 → 区域验证器（判类）+ 区域精修器（判类+框回归）→ 概率几何平均融合 → NMS

| 划分 | 旧流水线 mAP@0.5 | **优化版 mAP@0.5** | 精确率 | 召回率 | 每图框数 |
|---|---|---|---|---|---|
| 验证集 270 张 | 0.1681 | **0.2473** | 0.1529 | 0.6072 | 9.4 |
| **测试集 270 张** | 0.1552 | **0.2176** | **0.1363** | **0.5521** | 9.5 |

> 优化版 = 判别模型主干换 ImageNet 预训练 + 输入 ImageNet 归一化 → 精修器框回归损失换 GIoU → 输入 96 提到 128，
> 三步各自实测：+0.0358 / +0.0132 / +0.0302（验证集）。后处理沿用原参数（0.6/0.4/150/50），未在验证集上挑参。

**每类 AP@0.5（优化版，测试集）**：PS 麻点 **0.536** ｜ Cr 龟裂 0.219 ｜ Pa 斑块 0.159 ｜ RS 0.142 ｜ In 夹杂 0.134 ｜ Sc 划痕 0.117

> 旧流水线的测试集数字（0.1552，每类 PS 0.353 / Pa 0.201 / Cr 0.136 / RS 0.112 / Sc 0.070 / In 0.059）
> 来自一次完整流水线重跑（`python src/det/tools/full_pipeline.py`，186.2 分钟，12 个子任务 11 项通过）。
> 相比最初版本（测试集 0.0548），优化版累计提升 **+297%**。
> 区域验证器图块分类准确率从 0.8816 提到 **0.9564**，精修器从 0.7795 提到 0.8384、精修后 IoU 从 0.4442 提到 0.4919。

**⚠️ 如实说明**：召回率 0.55、精确率 0.136（每图输出 9.5 个框，真值平均 2.4 个），
mAP@0.5 = 0.2176，**仍远低于常规检测任务的及格线（0.5）**。任务2 属于**部分达标**，报告中请勿美化。
优化轮试过并否决的方向也一并记录（逐类阈值不叠加、步长 8 收益噪声级、单阶段换预训练主干无效、学习式重排序 0.2540→0.1949）。
> 详见 `docs/C_检测任务工作总结.md`、`src/det/README_det.md`

### 对比汇总

| 任务 | 网络数 | 最佳指标 | 达标情况 |
|---|---|---|---|
| 任务1 分类 | 2 | 准确率 1.0000 | ✅ 完全达标 |
| 任务2 定位 | 2 + 2（附加） | 测试集 mAP@0.5 0.2176 | ⚠️ 部分达标 |

---

## 二、开发环境（实测确认）

### 本项目涉及两台不同的机器

| 项 | A / B 的机器 | C 的机器 |
|---|---|---|
| 环境路径 | `D:\Miniconda3\envs\paddle_env` | `C:\Users\htt22\miniconda3\envs\paddle_env` |
| Paddle | **3.3.1（CPU 版，无 CUDA）** | **paddlepaddle-gpu 3.2.0** |
| 硬件 | 纯 CPU（24 逻辑核） | RTX 5070 Ti Laptop 12GB |
| 本机补装 | — | PyYAML、tqdm、opencv-python |

**好消息**：B 和 C 的代码都只依赖 `numpy / paddle / PIL / matplotlib / yaml`，**两台机器上都能跑**。
整合后已在 CPU 环境做过完整验证：59 个 `.py` 文件语法全通过，B 的 6 个模块 + C 的 12 个核心模块全部导入成功。

**代价**：C 的训练（单阶段 40+86 分钟、融合评估 3.4 秒/图）在纯 CPU 上不现实，**复现需要 GPU**。

### CPU 训练速度实测（A/B 机器，200×200 输入、batch=16）

| 网络 | 单步耗时 | 训练 1260 张 1 个 epoch | 跑 30 epoch |
|---|---|---|---|
| ResNet50 | 4.61 s | 约 6 分钟 | 约 3 小时 |
| MobileNetV3_small | 0.67 s | 约 0.9 分钟 | 约 27 分钟 |

结论：**分类任务 CPU 完全跑得动**（实测 ResNet50 约 116 分钟、MobileNetV3 约 26 分钟；
即 ResNet50 建议晚上挂着跑，MobileNetV3 白天随便跑）。
**检测任务（PP-YOLOE-s / YOLOv3）在 CPU 上非常吃力**，必须在 GPU 上跑。

### 依赖清单

| 包 | 状态 |
|---|---|
| Python 3.12.13 / numpy 2.5.2 / paddle 3.3.1 / Pillow 12.3.0 / matplotlib 3.11.1 | ✅ 已装 |
| PyYAML 6.0.3 / tqdm 4.70.0 | ✅ 已装（B、C 都需要） |
| pandas 3.0.5 / scikit-learn 1.9.0 / scipy 1.18.1 | ✅ 已装 |
| opencv (cv2) | ❌ 未装（B/C 的代码都没用到，不影响运行） |
| visualdl | ❌ 未装（想看训练曲线再装） |
| **paddledet** | 🚫 **绝对不要装**，见下 |

### 🚫 红线：不要 `pip install paddledet`

PyPI 上 `paddledet` 最新只到 2.6.0，要求 `numpy < 1.24`，而本机是 **numpy 2.5.2**。
装了会强制降级 numpy，**A 已跑通的数据脚本和 B/C 的代码会一起崩**。

C 实测过 PaddleDetection（release/2.9）：`import ppdet` 会逐层失败（缺 pycocotools → requests → imgaug…），
且其 `requirements.txt` 首行就是 `numpy<2.0`。**最终 C 改用原生 Paddle 手写检测网络，零新增依赖。**

**环境名注意**：`env.paddle` 是 VSCode 里的显示名，conda 环境名实际是 **`paddle_env`**。

```bash
conda activate paddle_env
python -c "import paddle; paddle.utils.run_check()"
```

### 网络为什么都用原生 Paddle 手写

- **B 的分类**：直接用 `paddle.vision.models`（实测可用）搭网络 + 手写训练循环：

  ```python
  from paddle.vision.models import resnet50, mobilenet_v3_small
  resnet50(num_classes=6)            # 2357 万参数
  mobilenet_v3_small(num_classes=6)  # 153 万参数
  ```

  只有 6 分类、1800 张图，手写训练循环不到 100 行，可控好 debug；
  PaddleClas 是给几百类大工程用的，配置文件套娃，不值得。
- **C 的检测**：原生 Paddle 手写 YOLOv3 与 PP-YOLOE-s 两套网络（原因见上一节红线）。

---

## 三、数据状态（成员 A 已完成 ✅）

### 数据集来源

Google Drive / 百度网盘国内不通，改用 GitHub 社区镜像 `siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection`（127MB）。
该镜像把数据拆成 `IMAGES`(1770) + `Validation_Images`(30)，`fetch_dataset.py` 已自动合并，合并后 **6 类各 300 张，共 1800 张**。

### 已确认的事实

- **图像实际是 RGB 三通道保存的**（内容确实是灰度）。报告里写"200×200 灰度图像按 RGB 编码存储"即可。
- **类别名统一**：`Cr` / `In` / `Pa` / `PS` / `RS` / `Sc`（顺序固定）
- **划分**：1260 / 270 / 270，随机种子 2026，三集**无任何重复文件** ✔
- **分类与检测共用同一批图的同一划分**，两个任务可直接横向对比
  （证据：`results/metrics/split_report.json` 与 C 流水线产出的同名文件 **md5 完全一致**）
- **标注**：1800 个标准 PASCAL VOC XML，共 **4189 个目标框**（训练集 2916 个），无坐标越界

### A 产出的文件

| 路径 | 内容 |
|---|---|
| `dataset/raw/IMAGES/` `dataset/raw/ANNOTATIONS/` | 1800 张原始图 + 1800 个原始 XML |
| `dataset/cls/images/<6类>/` | 按类分文件夹的图片 |
| `dataset/cls/{train,val,test}.txt` | 每行「相对路径 标签id」 |
| `dataset/det/JPEGImages/` `Annotations/` | 检测用图与 VOC 标注 |
| `dataset/det/ImageSets/Main/*.txt` | 只放文件名，不带后缀 |
| `dataset/det/label_list.txt` | 6 行，一行一个类名 |
| `results/metrics/*.json` | 5 份报告：data_check / data_stats / split / voc_convert / submit_check |
| `results/figures/*.png` | 4 张图：samples_grid（六类示例+红框）、class_distribution、gray_hist、bbox_size_dist |

自检：`python src/data/check_submit.py` → **23 / 23 全部通过** ✔

### B / C 读数据的接口

```
B 分类：dataset/cls/images/  +  dataset/cls/{train,val,test}.txt   （每行「Cr/xxx.jpg 0」）
C 检测：dataset/det/JPEGImages/  +  dataset/det/Annotations/  +  dataset/det/ImageSets/Main/*.txt
       类别顺序：dataset/det/label_list.txt → Cr / In / Pa / PS / RS / Sc
```

---

## 四、目录结构

```
基于东北大学NEU-CLS数据集的钢材表面缺陷检测/
├── README.md                 ← 本文件
├── requirements.txt          ← 依赖清单（含实际版本 + 禁止安装项）
├── .gitignore                ← 数据与权重不进仓库，但放行 results/figures/*.png
├── .gitattributes            ← 统一换行符，防止 Windows/Linux 之间产生假冲突
├── dataset/                  ← A 的地盘（已完成，其他人只读）
│   ├── raw/                  ← IMAGES(1800) + ANNOTATIONS(1800)
│   ├── cls/                  ← 分类用数据（images/ 6 类 + train/val/test.txt）
│   └── det/                  ← 检测用数据（VOC 格式 + ImageSets/Main + label_list.txt）
├── src/
│   ├── data/    (6 个脚本)   ← A：数据下载、校验、划分、VOC 转换、EDA、自检
│   ├── cls/     (10 个文件)  ← B：分类训练/评估/CAM/导出 + 公共模块
│   ├── det/     (46 个文件)  ← C：检测训练/评估/可视化/两阶段方案 + core/nets/tools
│   └── tools/   (4 个脚本)   ← D：公共指标库、日志转表、对比图、一键复现
├── results/
│   ├── metrics/   ← 55 份（数据自检报告、分类评估、检测评估、汇总表）
│   ├── figures/   ← 62 张（数据分布图、分类混淆矩阵/CAM、检测 PR/AP/样例图）
│   ├── logs/      ← 7 份（B 的训练日志、超参记录与测试集使用记录）
│   └── weights/   ← 16 个权重文件（981 MB，不进 git）
├── demo/                     ← C：交互式演示程序（任选数据集图片查看检测结果）
└── docs/
    ├── B_分类任务工作总结.md
    └── C_检测任务工作总结.md
```

## 五、代码归属

### A · 数据（`src/data/`，6 个脚本）✅

| 文件 | 做什么 |
|---|---|
| `fetch_dataset.py` | 下载并解压合并数据集（已下载会跳过） |
| `download_check.py` | 完整性校验（6 类各 300、尺寸、通道）+ 按类整理 |
| `split_dataset.py` | 分层随机划分 7:1.5:1.5，种子 2026 |
| `xml2voc.py` | XML → 标准 VOC，统一类别名，坐标越界检查 |
| `eda.py` | 六类示例图、类别分布、灰度直方图、框尺寸分布 |
| `check_submit.py` | 交付自检（23 项），含训练/测试集泄漏检查 |

### B · 分类（`src/cls/`，10 个文件）✅

| 文件 | 做什么 |
|---|---|
| `train.py` | 训练，`--model` 切换两个网络 |
| `configs/resnet50_vd.yml`、`configs/mobilenet_v3_small.yml` | 超参配置 |
| `eval_cls.py` | 测试集推理 + 准确率/精确率/召回率/F1 + 混淆矩阵 |
| `cam.py` | Grad-CAM 热力图 |
| `export.py` | 导出推理模型 + 测参数量/FPS |
| `_common.py` | 公共模块（建网络、数据管线、指标），`export.py` 等依赖它 |
| `train_curves.py` | 训练曲线对比图（附加） |
| `README_B.md` | B 的使用说明 |

### C · 检测（`src/det/`，46 个文件）⚠️

**主脚本（9 个）**

| 文件 | 做什么 |
|---|---|
| `train.py` | 单阶段检测网络训练，`--config` 切网络 |
| `eval_det.py` | 单阶段评估：mAP@0.5 / 每类 AP / PR 曲线 / 混淆矩阵 |
| `viz_det.py` | 可视化：真值-预测对比、错例分析、尺寸-AP |
| `demo.py` | 单图/批量推理 Demo，可测 FPS |
| `eval_2stage.py` | 两阶段评估（检测器或滑窗 + 验证器） |
| `eval_refine.py` | 滑窗 + 精修器评估（含框回归消融） |
| **`eval_fused.py`** | **最终方案入口**：滑窗 + 验证器/精修器融合 |
| `train_verifier.py` | 区域验证器训练（判类） |
| `train_refiner.py` | 区域精修器训练（判类 + 框回归） |

**`core/`（9 个文件 = 8 个模块 + `__init__.py`）**：`data` / `boxes` / `det_ops` / `layers` / `utils`(含 mAP 实现) / `verifier` / `refiner` / `window_detector`

**`nets/`（5 个文件 = 4 个网络 + `__init__.py`）**：`darknet` + `yolov3`、`cspresnet` + `ppyoloe`

**`tools/`（21 个）**：诊断与调优工具。`full_pipeline.py` 是一键复现入口，`summarize.py` 生成结果汇总，
`check_docs.py` 校验文档与产物一致性，`viz_fused.py` 出真值-预测对比图，其余为迭代过程中的消融/调参/诊断脚本（C 保留作为工作量证据）。

> **运行方式注意**：C 的脚本都在开头 `sys.path.insert(0, 脚本所在目录)`，
> 所以必须**从项目根目录用 `python src/det/xxx.py` 运行**，不能直接 `import src.det.xxx`。

### D · 工具链（`src/tools/`）✅ 已完成

| 文件 | 计划做什么 | 现状 |
|---|---|---|
| `metrics.py` | 公共指标库，B/C 直接 import | ✅ 已实现：`accuracy` / `precision_recall_f1` / `confusion_matrix` / `compute_ap` / `compute_map` / `pr_curve` / `count_params` 全部落地，另提供 `evaluate_map` 以及 B 探测用的 `classification_metrics` 等别名 |
| `log2table.py` | 解析训练日志 → 结果表 | ✅ 已实现并实测通过（产出 `results/summary_table.csv`） |
| `plot_summary.py` | 网络对比柱状图 | ✅ 已实现并实测通过（产出 `results/figures/summary_compare.png`） |
| `run_all.py` | 一键复现全流程 | ✅ 已实现（数据自检 → 汇总表 → 对比图） |

> **口径已经统一**：`metrics.py` 的分类口径与 B 的 `_common.py:metrics_from_cm` 一致，
> 检测口径与 C 的 `core/utils.py:evaluate_map` 一致（逐类按分数全局排序匹配 GT，用 VOC 2010+ 的 101 点插值法求 AP）。
> 已用 B/C 入库的产物反向验证：`compute_ap` 能**逐位复现** C 记录的六类 AP 与 mAP@0.5（偏差 0），
> 分类指标与 B 记录的 `accuracy` / `macro` / `num_samples` 也完全一致。
>
> 接入方式：
> - **B**：`_common.py` 的 `external_cls_metrics()` 会按名字探测，现已命中
>   `src/tools/metrics.py::classification_metrics`，结果额外记入 `tools_metrics` 字段做交叉核对
>   （主口径仍是 B 自己的实现，已有结果不会变）。
> - **C**：把 `core/utils.py` 里 `evaluate_map` 的内部实现换成 `from tools.metrics import evaluate_map`
>   即可，入参顺序与返回结构完全一致（本模块额外多返回一个 `overall` 字段）。

## 六、四个网络的结果对比

| 任务 | 网络 | 参数量 | 关键指标 | 负责人 | 状态 |
|---|---|---|---|---|---|
| 分类 | ResNet50_vd | 2357 万 | 准确率 1.0000 | B | ✅ |
| 分类 | MobileNetV3-small | 154 万 | 准确率 1.0000 | B | ✅ |
| 检测 | YOLOv3 | 6440 万 | mAP@0.5 **0.0000** | C | ❌ 失败 |
| 检测 | PP-YOLOE-s | 2375 万 | mAP@0.5 **0.0000** | C | ❌ 失败 |
| 检测 | 滑窗 + 验证器/精修器融合（自研，优化版） | 259 万 ×2 | mAP@0.5 **0.2176** | C | ⚠️ 部分达标 |
| 检测 | 滑窗 + 验证器/精修器融合（旧流水线） | 259 万 ×2 | mAP@0.5 0.1552 | C | ⚠️ 部分达标 |

## 七、成员署名区

| 代号 | 姓名 | 角色 | 负责目录 |
|---|---|---|---|
| A | | 数据负责人 | `src/data/`、`dataset/` |
| B | | 分类模型负责人 | `src/cls/` |
| C | | 检测模型负责人 | `src/det/` |
| D | | 评测工具链负责人 | `src/tools/` |

## 八、脚本用法

> **第一次拿到这个项目？先看 [`docs/使用说明.md`](docs/使用说明.md)** —— 15 分钟上手：环境、一键运行、
> 结果在哪、常见问题、分模块手动运行命令，都在那一份里。
> 想知道**六个网络各自的结构**（主干/颈部/检测头、损失、参数量）看
> [`docs/模型结构说明.md`](docs/模型结构说明.md)。
>
> **推荐入口：`python run_project.py`**（仓库根目录，全项目一键流水线，CPU / GPU 通用）。
> 它把 A/B/C/D 四人的入口脚本串成一条流水线，**默认重跑每个阶段**（想跳过已完成的加 `--resume`）：
>
> | 命令 | 做什么 | 耗时 |
> |---|---|---|
> | `python run_project.py --profile eval` | 只跑自检 + 汇总 + 校验，**不训练** | 约 5 秒（不需 GPU） |
> | `python run_project.py --profile standard` | 训练最终检测方案（验证器+精修器）并评估验证集与测试集 | GPU 约 1 小时 |
> | `python run_project.py --profile full` | 再加单阶段 YOLOv3 / PP-YOLOE-s 基线 | GPU 约 3 小时 |
> | `python run_project.py --quick` | 冒烟：每步都跑但极小规模（1 epoch / 20 张图） | 约 5 分钟 |
> | `python run_project.py --dry-run` | 只打印将执行的命令，零写入 | 秒级 |
> | `python run_project.py --device cpu` | 强制 CPU，自动降级为 96 输入 / 小 epoch | — |
>
> 其他参数：`--resume` 跳过已完成阶段（默认重跑）、`--retrain` / `--force` 强制重训、
> `--list` 列阶段、`--only a,b` / `--skip a,b` 选阶段、`--in-size` / `--epochs-scale` 调规模、
> `--no-pretrained` 复现旧流水线口径、`--python <解释器>` 指定解释器。运行记录写
> `results/metrics/run_project_summary.json`，每阶段日志在 `results/logs/run_project/`。
>
> **不用先 activate 环境**：脚本会自检当前解释器有没有 paddle，没有就自动找到装了 paddle 的
> 解释器（本项目为 `miniconda3\envs\paddle_env\python.exe`）并用它重新运行；加 `--no-auto-python` 可关闭该行为。
>
> **交互式演示（答辩/验收推荐）**：双击 `demo/run_gui.bat` 打开图形界面，可从 1800 张里任选一张
> 查看融合流程的检测结果（真值/预测叠加、逐框分数与判定、候选漏斗，可导出 HTML 报告与三联图）。
> 说明见 [`demo/README.md`](demo/README.md)。

下面是**逐条命令**的原始用法（想单独跑某一步时看这里）：

```bash
conda activate paddle_env      # 环境名是 paddle_env，不是 env.paddle

# ---------- A：数据（已全部跑完，需重跑时按序执行）----------
python src/data/fetch_dataset.py
python src/data/download_check.py
python src/data/split_dataset.py
python src/data/xml2voc.py
python src/data/eda.py
python src/data/check_submit.py

# ---------- B：分类（已训练完成，权重在 results/weights/）----------
python src/cls/train.py --model mobilenet_v3_small      # 约 26 分钟（CPU）
python src/cls/train.py --model resnet50_vd             # 约 116 分钟（CPU）
python src/cls/eval_cls.py --model resnet50_vd
python src/cls/eval_cls.py --model mobilenet_v3_small
python src/cls/cam.py --model resnet50_vd
python src/cls/export.py --model resnet50_vd
python src/cls/train_curves.py

# ---------- C：检测 ----------
# 最终方案（优化版：预训练主干 + GIoU + 128 输入）——先训两个区域判别模型
python src/det/train_verifier.py --name verifier_pre128 --pretrained --norm imagenet --in-size 128
python src/det/train_refiner.py --name refiner_giou128 --pretrained --norm imagenet \
    --in-size 128 --reg-loss giou --init-verifier results/weights/verifier_pre128_best.pdparams

# 最终方案评估（--norm / --in-size 必须与训练一致）
python src/det/eval_fused.py --split test --mode fuse \
    --stride 12 --score-th 0.6 --nms-iou 0.4 --top-k 150 --max-det 50 \
    --norm imagenet --in-size 128 \
    --verifier-weights results/weights/verifier_pre128_best.pdparams \
    --refiner-weights  results/weights/refiner_giou128_best.pdparams \
    --tag test_fuse_pre128

# 旧流水线（随机初始化主干 + L2 + 96 输入）等价于 --norm none --in-size 96
python src/det/eval_fused.py --split test --mode fuse \
    --verifier-weights results/weights/verifier_best.pdparams \
    --refiner-weights  results/weights/refiner_best.pdparams

# 单阶段检测器训练/评估（失败基线，完整复现约 40 + 86 分钟，需 GPU）
python src/det/train.py --config src/det/configs/yolov3.yml
python src/det/train.py --config src/det/configs/ppyoloe_s.yml
python src/det/eval_det.py --model yolov3 --weights results/weights/yolov3_best.pdparams

# 从数据自检到报告汇总一键复现（旧流水线，约 186 分钟，需 GPU）
python src/det/tools/full_pipeline.py

# ---------- D：工具链（4 个脚本全部可用）----------
python src/tools/run_all.py        # 一键：数据自检 → 汇总表 → 对比图
python src/tools/log2table.py      # 只生成 results/summary_table.csv
python src/tools/plot_summary.py   # 只生成 results/figures/summary_compare.png
```

## 九、权重文件（`results/weights/`，共 981 MB）

**⚠️ 该目录已被 `.gitignore` 排除，不会上传到 GitHub。** 需要权重时从本地拷贝。

| 文件 | 大小 | 说明 |
|---|---|---|
| `cls_resnet50_vd_best.pdparams` / `_last` | 94.3 MB ×2 | B：ResNet50 训练权重 |
| `cls_mobilenet_v3_small_best.pdparams` / `_last` | 6.2 MB ×2 | B：MobileNetV3 训练权重 |
| `cls_*_infer.json` + `_infer.pdiparams` | — | B：导出好的推理模型，部署直接用这两个 |
| `verifier_best.pdparams` | 10.4 MB | C：旧流水线（随机主干 + 96 输入）验证器 |
| `refiner_best.pdparams` | 10.4 MB | C：旧流水线精修器 |
| `verifier_pre128_best.pdparams` | 10.4 MB | **C：优化版必需**（ImageNet 预训练主干 + 128 输入）验证器 |
| `refiner_giou128_best.pdparams` | 10.4 MB | **C：优化版必需**（GIoU 回归 + 128 输入）精修器 |
| `verifier_pre_best.pdparams` / `refiner_pre_best.pdparams` / `refiner_giou_best.pdparams` | 10.4 MB ×3 | C：优化轮中间版本（96 输入 / L2 对照），仅作 A/B 记录 |
| `yolov3_best.pdparams` / `_last` | 257.6 MB ×2 | C：单阶段失败基线（保留作对照） |
| `ppyoloe_s_best.pdparams` / `_last` | 95.1 MB ×2 | C：单阶段失败基线（保留作对照） |
| `权重说明.txt` | — | B 的权重使用说明（含部署代码示例） |
| `MANIFEST.txt` | — | C 的权重包说明（含复现命令） |

> **想复现 C 的最终指标，只需要 `verifier_pre128_best` + `refiner_giou128_best`（19.8 MB）**；
> 其余是单阶段失败实验与优化轮中间版本的记录。

## 十、四条铁律

1. **测试集只许最后用一次**，调参一律用验证集。`check_submit.py` 会检查 train/test 有没有混。
2. **随机种子统一 2026**，写进脚本；配置里一律用相对路径；**项目路径必须纯英文无空格**。
3. **只在自己的目录改文件**；要改 `src/tools/` 或别人的目录，先在群里说。
4. **D 的 `metrics.py` 先给函数签名**（返回假值的空壳也行），B/C 才能照着写评估脚本。（已落实，函数体也已补齐）

## 十一、训练实操建议

1. **两个网络串行训，不要同时开**。CPU 上是因为核数有限；GPU 上是因为**并发会触发显存重试**——
   C 实测 PP-YOLOE-s 并发时单 epoch 从 51 秒涨到 7224 秒（慢 134 倍）。
2. **先跑轻的**：先训 MobileNetV3_small（26 分钟）跑通全流程、确认评估脚本没问题，再上 ResNet50（116 分钟）。
3. **显存红线**：PP-YOLOE-s 在 640 输入下 batch 必须 ≤ 8。bs=16 时峰值显存 12.18GB 顶满 12GB 显卡，
   单步从 249ms 暴涨到 52 秒（**静默变慢，不报错，很容易被忽略**）。
4. **每个 epoch 都存一次权重**到 `results/weights/`，训练崩了不丢进度。

## 十二、已知问题与后续改进

1. **任务2 仍未达标**（最需要解决）：优化后测试集 mAP@0.5 = 0.2176、精确率 0.136，离 0.5 的及格线还远。
   优化轮已经把「换预训练主干」这条建议做完（+0.0358），也**实测否决**了三条：
   给单阶段检测器加 Focal Loss + 预训练主干（val 0.0003，obj 排名中位数 6667/10647，属数据规模问题）、
   逐类分数阈值（增益不叠加，+0.0008~0.0013）、学习式重排序替代几何平均（0.2540 → 0.1949）。
   **剩下还没做的**：把 200×200 原图做**重叠裁块**扩充训练样本；类别专属滑窗尺度集合；
   若允许换技术栈，用 COCO 预训练的检测器微调（迁移强度远高于 ImageNet 分类特征）。
2. **D 的工具链已全部落地**：4 个脚本均实装并实测可跑，`metrics.py` 的分类 / 检测两套口径
   已分别与 B（`src/cls/_common.py`）、C（`src/det/core/utils.py`）对齐，并用两边已入库的产物反向验证一致。
   目前是「本地实现为主 + D 的模块做交叉核对」的双轨状态；若要真正切成单一出口，
   把 B 的 `evaluate_metrics()` 主口径与 C 的 `evaluate_map()` 内部行替换即可，两边都只需改一行。
3. **C 的复现门槛高**：单阶段训练与融合评估均需 GPU（CPU 上耗时不可接受）；
   优化版比旧流水线更慢（128 输入，测试集单图 4986 ms）。现有可视化图与指标已入库，无需重跑即可查证。
4. **项目文档与产物的一致性**：C 提供了 `src/det/tools/check_docs.py` 可自动校验，改动数字后建议跑一次。
5. **测试集使用次数**：第一轮交付与优化轮各评估过一次（共两次），后续再动测试集需要团队确认。
