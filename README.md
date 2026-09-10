# 基于东北大学 NEU-CLS 数据集的钢材表面缺陷检测

## 作业要求

**选题**：基于东北大学 NEU-CLS 数据集的钢材表面缺陷检测

- **数据**：6 类缺陷，1800 张 200×200 灰度图像
- **任务1**：缺陷检测与分类
- **任务2**：缺陷检测与定位

**要求**：

1. 训练集、验证集、测试集比例 **7 : 1.5 : 1.5**。测试集文件与训练集文件完全独立，确保评估的可靠性。
2. 至少搭建 **2 种网络**，并对比不同网络的效果差异。

> 本方案实际要跑 **4 个网络**（分类 2 个 + 检测 2 个），两个任务各满足一次"至少 2 种"，报告里可以直接写超额完成。

---

## 当前进度（更新时间：2026-09-10）

> 📄 **成员 C 的检测任务已完成**，工作总结见 `docs/C_检测任务工作总结.md`，
> 技术实现与踩坑记录见 `src/det/README_det.md`，结果总表见 `results/metrics/det_summary.md`。

| 成员 | 状态 | 说明 |
|---|---|---|
| A 数据 | ✅ **已完成** | 1800 张图 + 1800 个标注全齐，划分完成，23 项自检全过 |
| D 工具链 | ⬜ 未开始 | 可以开始，数据分布已确认，够定 `metrics.py` 的函数签名了 |
| B 分类 | ⬜ 未开始 | **数据已就绪，可以直接开工** |
| C 检测 | ✅ **已完成** | 两套检测网络 + 区域验证器 + 完整评估，详见 `src/det/README_det.md` |

**下一步**：B 可以开工了（M 类问题的详细结论见 `src/det/README_det.md` 第九节，
其中「为什么单阶段检测器的置信度分支在本数据集上学不出来」的分析对 B 也有参考价值）。

### C 的实际结论（2026-09-10，本机实测）

方案与原计划有一处重要调整：**没用 PaddleDetection**。原因是它 `requirements.txt`
第一行就是 `numpy < 2.0`，与本机 numpy 2.5.2 冲突，会破坏 A 已跑通的数据脚本；
而 `paddle.vision.ops` 原生自带全部所需检测算子，不需要这个框架。
最终用原生 Paddle 手写了 YOLOv3 与 PP-YOLOE-s 两套网络。

| 结果 | 数值 |
|---|---|
| 区域验证器图块分类准确率（验证集） | 0.8872 |
| 区域验证器在 GT 框上的分类准确率（**测试集**） | **97.00%** |
| 滑窗 + 验证器 检测 mAP@0.5（测试集） | 0.0548（召回 0.514） |
| 单阶段检测器 mAP@0.5 | 0.0000（框定位其实可用，IoU 0.74/0.757，但置信度排序学不出来） |

完整结果表由 `python src/det/tools/summarize.py` 生成到
`results/metrics/det_summary.md`，含每类 AP、各类尝试的实测对比与根因分析。

> ⚠️ 环境修正：本机与 README 下方「开发环境」一节记载的**不是同一台机器**。
> 本机实测为 `C:\Users\htt22\miniconda3\envs\paddle_env`、
> **paddlepaddle-gpu 3.2.0**、**RTX 5070 Ti Laptop 12GB**（有 GPU，非 CPU 训练）。
> 训练速度比 README 记载的 CPU 数据快约 30 倍。

---

## ⚠️ 三条必读（动手前先看完）

### 1. 环境是 **CPU 版 Paddle，没有 GPU**

实测 `paddle.device.is_compiled_with_cuda()` = `False`，`cuda.device_count()` = `0`。

这不是装错了，是当初装的就是 CPU 版。训练速度实测（200×200 输入、batch=16）：

| 网络 | 单步耗时 | 训练 1260 张 1 个 epoch | 跑 30 epoch |
|---|---|---|---|
| ResNet50 | 4.61 s | **约 6 分钟** | 约 3 小时 |
| MobileNetV3_small | 0.67 s | **约 0.9 分钟** | 约 27 分钟 |

结论：**分类任务 CPU 完全跑得动**，就是慢。ResNet50 建议晚上挂着跑，MobileNetV3 白天随便跑。
**检测任务（PP-YOLOE-s / YOLOv3）在 CPU 上会非常吃力**，C 必须提前规划：输入尺寸压到 320、epoch 控制在 30 以内、优先跑 PP-YOLOE-s（比 YOLOv3 轻），或者全组申请 AI Studio 的免费 GPU。

### 2. **绝对不要 `pip install paddledet`**

PyPI 上 `paddledet` 最新只到 **2.6.0**，它要求 `numpy < 1.24`，而当前环境是 **numpy 2.5.2**。装它会自动把 numpy 降级，整个环境直接崩，A 已经跑通的数据脚本也会跟着挂。

C 的正确做法：**`git clone` PaddleDetection 源码用，不要 pip 装**。

```bash
# 在项目外找个目录（别放 dataset/ 或 src/ 里）
git clone -b release/2.9 https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
pip install -r requirements.txt   # 注意：先看它会不会动 numpy，会动就手动装缺的
python setup.py install
```

如果 `setup.py install` 因版本冲突失败，**退回方案**：不装，直接在 PaddleDetection 源码目录下 `python tools/train.py -c <配置>` 跑，只要能 import 通就行。

### 3. 环境名是 `paddle_env`，不是 `env.paddle`

之前说的 `env.paddle` 是 VSCode 里的显示名，实际路径和 conda 名是：

```
D:\Miniconda3\envs\paddle_env
```

激活命令用 `conda activate paddle_env`。VSCode 里 `Ctrl+Shift+P` → `Python: Select Interpreter` 选这个目录。

---

## 开发环境（已实测确认）

| 项目 | 实际值 |
|---|---|
| 环境路径 | `D:\Miniconda3\envs\paddle_env` |
| conda 环境名 | `paddle_env`（VSCode 里显示成 `env.paddle`） |
| Python | 3.12.13 |
| PaddlePaddle | **3.3.1（CPU 版，无 CUDA）** |
| numpy | 2.5.2 |
| Pillow | 12.3.0 |
| matplotlib | 3.11.1 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| scipy | 1.18.1 |
| **opencv (cv2)** | ❌ 未装，C 做可视化前要装 |
| **visualdl** | ❌ 未装，想看训练曲线再装 |
| **paddleclas / paddledet** | ❌ 未装（正常，这两个本来就该用源码，不是 pip 装） |

**验证环境是否正常**（每人开工前跑一遍）：

```bash
conda activate paddle_env
python -c "import paddle; paddle.utils.run_check()"
```

看到 `PaddlePaddle is installed successfully!` 就对了。

**`paddle.vision.models` 已经实测可用**，B 可以直接拿现成网络：

```python
from paddle.vision.models import resnet50, mobilenet_v3_small
resnet50(num_classes=6)          # 2357 万参数
mobilenet_v3_small(num_classes=6)  # 153 万参数
```

> 建议 B **就用 `paddle.vision.models` 自己写训练循环**，别去折腾 PaddleClas 源码。理由：只有 6 分类、1800 张图，手写训练循环不到 100 行，可控、好 debug、出错好查；PaddleClas 是给几百类大工程用的，配置文件套娃，大二同学容易陷进去。

---

## 数据状态（成员 A 已完成 ✅）

### 数据集来源

Google Drive / 百度网盘国内不通，改用 GitHub 社区镜像 `siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection`（127MB 压缩包）。
该镜像把数据拆成 `IMAGES`（1770 张）+ `Validation_Images`（30 张），`fetch_dataset.py` 已自动合并，合并后 **6 类各 300 张，共 1800 张**。

### 已确认的事实

- **图像实际是 RGB 三通道保存的**（内容确实是灰度）。报告里写"200×200 灰度图像按 RGB 编码存储"即可，不影响训练，预训练模型也习惯 3 通道输入。
- **类别名统一为**：`Cr` / `In` / `Pa` / `PS` / `RS` / `Sc`（顺序固定，`label_list.txt` 就是这个顺序）
- **划分**：1260 / 270 / 270，随机种子 2026，训练/验证/测试**无任何重复文件** ✔
- **分类和检测用同一批图的同一划分**，两个任务可以直接横向对比
- **标注**：1800 个标准 PASCAL VOC XML，共 **4189 个目标框**，无坐标越界

### A 产出的文件

| 文件 | 内容 |
|---|---|
| `dataset/raw/IMAGES/` | 1800 张原始图 |
| `dataset/raw/ANNOTATIONS/` | 1800 个原始 XML |
| `dataset/cls/images/<6类>/` | 按类分文件夹的图片 |
| `dataset/cls/{train,val,test}.txt` | 每行「相对路径 标签id」 |
| `dataset/det/JPEGImages/` | 检测用图片 |
| `dataset/det/Annotations/` | VOC 格式 XML |
| `dataset/det/ImageSets/Main/{train,val,test}.txt` | 只放文件名，不带后缀 |
| `dataset/det/label_list.txt` | 6 行，一行一个类名 |
| `results/metrics/*.json` | 5 份报告：data_check / data_stats / split / voc_convert / submit_check |
| `results/figures/*.png` | 4 张图：samples_grid（六类示例+红框）、class_distribution、gray_hist、bbox_size_dist |

自检：`python src/data/check_submit.py` → **23 / 23 全部通过** ✔

---

## B / C 的数据接口（照这个路径读，别自己找）

**B 分类**：

```
图片根目录：dataset/cls/images/
列表文件：  dataset/cls/train.txt   （1260 行）
            dataset/cls/val.txt     （270 行）
            dataset/cls/test.txt    （270 行，最后才许用）
每行格式：  Cr/crazing_1.jpg 0
```

**C 检测**：

```
VOC 根目录：dataset/det/
图片：      dataset/det/JPEGImages/*.jpg
标注：      dataset/det/Annotations/*.xml
划分：      dataset/det/ImageSets/Main/{train,val,test}.txt  （只存文件名，无后缀）
类别：      dataset/det/label_list.txt   →  Cr / In / Pa / PS / RS / Sc
```

---

## 一、目录结构

```
基于东北大学NEU-CLS数据集的钢材表面缺陷检测/
├── README.md                 ← 本文件，D 维护
├── requirements.txt          ← 依赖清单（已填实际版本 + 禁止安装项）
├── .gitignore
├── dataset/                  ← A 的地盘（已完成，其他人只读）
│   ├── raw/                  ← IMAGES(1800) + ANNOTATIONS(1800)
│   ├── cls/                  ← 分类用数据
│   │   ├── images/           ← 6 个子文件夹：Cr / In / Pa / PS / RS / Sc
│   │   ├── train.txt         ← 每行「相对路径 标签id」
│   │   ├── val.txt
│   │   └── test.txt          ← 测试集，训练阶段谁都不许碰
│   └── det/                  ← 检测用数据（VOC 格式）
│       ├── JPEGImages/       ← 图片
│       ├── Annotations/      ← XML 标注
│       ├── ImageSets/Main/   ← train.txt / val.txt / test.txt（只放文件名，不带后缀）
│       └── label_list.txt    ← 6 行，一行一个类名，顺序固定
├── src/
│   ├── data/                 ← A 的地盘（已完工，改前打招呼）
│   ├── cls/                  ← B 的地盘
│   ├── det/                  ← C 的地盘
│   └── tools/                ← D 的地盘（公共模块，改前先在群里说）
├── results/                  ← B/C 写，D 读
│   ├── metrics/              ← 各类 *_eval_result.json（A 的 5 份报告也在这）
│   ├── figures/              ← 混淆矩阵、PR 曲线、检测效果图、对比柱状图
│   ├── logs/                 ← 训练日志
│   └── weights/              ← 各网络最佳权重
└── docs/                     ← 报告与 PPT
```

## 二、代码归属

| 文件 | 负责人 | 状态 | 做什么 |
|---|---|---|---|
| `src/data/fetch_dataset.py` | A | ✅ | 数据集下载并解压合并（已下载会跳过） |
| `src/data/download_check.py` | A | ✅ | 完整性校验（6 类各 300 张、尺寸与通道）+ 按类整理 |
| `src/data/split_dataset.py` | A | ✅ | 分层随机划分 7:1.5:1.5，种子 2026 |
| `src/data/xml2voc.py` | A | ✅ | XML → VOC 标注，含坐标越界检查 |
| `src/data/eda.py` | A | ✅ | 六类示例图、类别分布、灰度直方图、框尺寸分布 |
| `src/data/check_submit.py` | A | ✅ | 交付自检（23 项），含训练/测试集泄漏检查 |
| `src/cls/train.py` + `configs/resnet50_vd.yml`、`configs/mobilenet_v3_small.yml` | B | ⬜ | 分类训练，`--model` 切换网络 |
| `src/cls/eval_cls.py` | B | ⬜ | 测试集推理 + 准确率/精确率/召回率/F1 + 混淆矩阵 |
| `src/cls/cam.py` | B | ⬜ | Grad-CAM 热力图 |
| `src/cls/export.py` | B | ⬜ | 导出推理模型 + 测 FPS |
| `src/det/train.py` + `configs/ppyoloe_s.yml`、`configs/yolov3.yml` | C | ✅ | 检测训练，`--model` 切换网络（原生 Paddle 实现，不用 PaddleDetection） |
| `src/det/eval_det.py` | C | ✅ | mAP@0.5、每类 AP、PR 曲线、混淆矩阵 |
| `src/det/viz_det.py` | C | ✅ | 检测框效果图、错例对比图、尺寸-AP 图 |
| `src/det/demo.py` | C | ✅ | 单图/批量推理 Demo，可测 FPS |
| `src/det/train_verifier.py` + `src/det/core/verifier.py` | C | ✅ | 区域验证器（两阶段第二阶段），GT 框分类准确率 97% |
| `src/det/eval_2stage.py` + `core/window_detector.py` | C | ✅ | 两阶段检测评估（检测器/滑窗 + 验证器） |
| `src/det/tools/` | C | ✅ | k-means anchor、诊断、两阶段对比、结果汇总 |
| `src/tools/metrics.py` | D | ⬜ | 公共指标库，B/C 直接 import |
| `src/tools/log2table.py` | D | ⬜ | 解析训练日志 → 超参/结果表（CSV + Markdown） |
| `src/tools/plot_summary.py` | D | ⬜ | 网络对比柱状图、自动拼图 |
| `src/tools/run_all.py` | D | ⬜ | 一键复现：划分 → 训练 → 评估 → 出图 |

## 三、四个网络

| 任务 | 网络 | 参数量 | 负责人 | 状态 |
|---|---|---|---|---|
| 分类 | ResNet50_vd | 2357 万 | B | ⬜ |
| 分类 | MobileNetV3_small | 153 万 | B | ⬜ |
| 检测 | PP-YOLOE-s | — | C | ⬜ |
| 检测 | YOLOv3 | — | C | ⬜ |

（两个分类网络的参数量是实测量，用 `paddle.vision.models` 建的）

## 四、成员署名区（填姓名）

| 代号 | 姓名 | 角色 | 负责目录 |
|---|---|---|---|
| A | | 数据负责人 | `src/data/`、`dataset/` |
| B | | 分类模型负责人 | `src/cls/` |
| C | | 检测模型负责人 | `src/det/` |
| D | | 评测工具链负责人 | `src/tools/` |

## 五、脚本用法

```bash
conda activate paddle_env      # 注意：是 paddle_env，不是 env.paddle

# A：数据（已全部跑完，需要重跑时按顺序执行）
python src/data/fetch_dataset.py
python src/data/download_check.py
python src/data/split_dataset.py
python src/data/xml2voc.py
python src/data/eda.py
python src/data/check_submit.py

# B：分类
python src/cls/train.py --model resnet50_vd
python src/cls/train.py --model mobilenet_v3_small
python src/cls/eval_cls.py --model resnet50_vd
python src/cls/cam.py
python src/cls/export.py

# C：检测
python src/det/train.py --model ppyoloe_s
python src/det/train.py --model yolov3
python src/det/eval_det.py --model ppyoloe_s
python src/det/viz_det.py
python src/det/demo.py

# D：汇总与复现
python src/tools/log2table.py
python src/tools/plot_summary.py
python src/tools/run_all.py
```

## 六、四条铁律

1. **测试集只许最后用一次**，调参一律用验证集。`check_submit.py` 会检查 train/test 有没有混。
2. **随机种子统一 2026**，写进脚本；配置里一律用相对路径；**项目路径必须纯英文无空格**。
3. **只在自己的目录改文件**；要改 `src/tools/` 或别人的目录，先在群里说。
4. **D 的 `metrics.py` 先给函数签名**（返回假值的空壳也行），B/C 才能照着写评估脚本。

## 七、CPU 训练的三条实操建议

1. **两个网络串行训，不要同时开**。CPU 核数有限，并行只会都变慢，日志还会混在一起没法对比。
2. **先跑轻的**：先训 MobileNetV3_small（27 分钟）跑通全流程、确认评估脚本没问题，再上 ResNet50（3 小时）。检测同理，先 PP-YOLOE-s。
3. **每个 epoch 都存一次权重**到 `results/weights/`。CPU 训练中途崩了不丢进度，别等训完再存。
