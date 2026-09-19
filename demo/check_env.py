"""成员 C · 演示程序环境自检（换电脑后先跑这个）

逐个检查「能不能跑起来」需要的条件，并给出结论：

    python demo/check_env.py

检查项：
    1. Python 版本（需要 3.9~3.12）与解释器路径
    2. 依赖：paddle / numpy / PIL / matplotlib / yaml
    3. 计算设备：是否有可用的 CUDA GPU（没有也能跑，但会很慢）
    4. 数据：dataset/det/JPEGImages(1800) + Annotations(1800) + ImageSets + label_list.txt
    5. 权重：最终交付需要的 verifier_pre128_best / refiner_giou128_best
    6. 代码：能不能 import 到 demo_fused 依赖的模块（core / eval_det / eval_fused）
最后给出「可以直接演示 / 只能 CPU 跑（慢）/ 缺少什么」的明确结论。
"""

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # 项目根目录
OK, BAD, WARN = [], [], []
GPU = False


def sec(t):
    print("\n" + t)


def chk(cond, ok_msg, bad_msg, warn=False):
    if cond:
        OK.append(ok_msg)
        print(f"  [OK]   {ok_msg}")
    elif warn:
        WARN.append(bad_msg)
        print(f"  [警告] {bad_msg}")
    else:
        BAD.append(bad_msg)
        print(f"  [缺失] {bad_msg}")


print("=" * 78)
print("NEU-CLS 钢材缺陷检测 · 演示程序环境自检")
print(f"项目根目录：{ROOT}")
print(f"解释器    ：{sys.executable}")
print("=" * 78)

sec("【1】Python 版本")
v = sys.version_info
chk((3, 9) <= (v.major, v.minor) <= (3, 12),
    f"Python {v.major}.{v.minor}.{v.micro}（本项目在 3.12.13 上验证）",
    f"Python {v.major}.{v.minor}.{v.micro} —— 建议用 3.10~3.12（3.13+ 缺少 paddle 轮子）")

sec("【2】依赖包")
for mod, name, need in (("paddle", "paddlepaddle（核心）", True),
                        ("numpy", "numpy", True),
                        ("PIL", "pillow（画框/查看器）", True),
                        ("matplotlib", "matplotlib（导出三联图）", True),
                        ("yaml", "pyyaml（读配置）", True),
                        ("tkinter", "tkinter（图形界面，Python 自带）", True)):
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "") or getattr(m, "TkVersion", "")
        print(f"  [OK]   {name:<28} {ver}")
        OK.append(name)
    except Exception as e:
        chk(False, "", f"{name} 未安装（{type(e).__name__}）")

sec("【3】计算设备")
try:
    import paddle
    cuda_compiled = paddle.device.is_compiled_with_cuda()
    n = paddle.device.cuda.device_count() if cuda_compiled else 0
    GPU = bool(cuda_compiled and n > 0)
    if GPU:
        print(f"  [OK]   CUDA 可用：{paddle.device.cuda.get_device_name(0)}（{n} 张卡）")
        print("         预期速度：每张图约 4~6 秒（GPU）")
    else:
        print(f"  [警告] 没有可用 GPU（paddle 是否带 CUDA：{cuda_compiled}）")
        print("         仍可运行（程序会自动降级到 CPU），但每张图会慢很多；")
        print("         建议加 --stride 16 提速，或只在 CPU 上抽查少量图片")
        WARN.append("无 GPU：只能 CPU 运行（慢）")
except Exception as e:
    print(f"  [缺失] 无法检测设备：{type(e).__name__}: {e}")

sec("【4】数据集")
img_dir = ROOT / "dataset" / "det" / "JPEGImages"
ann_dir = ROOT / "dataset" / "det" / "Annotations"
sets_dir = ROOT / "dataset" / "det" / "ImageSets" / "Main"
n_img = len(list(img_dir.glob("*.jpg"))) if img_dir.is_dir() else 0
n_ann = len(list(ann_dir.glob("*.xml"))) if ann_dir.is_dir() else 0
chk(n_img >= 1800, f"图片 {n_img} 张（{img_dir}）", f"图片只有 {n_img} 张，期望 1800：{img_dir}")
chk(n_ann >= 1800, f"标注 {n_ann} 个（{ann_dir}）", f"标注只有 {n_ann} 个，期望 1800：{ann_dir}")
for sp in ("train", "val", "test"):
    f = sets_dir / f"{sp}.txt"
    chk(f.exists(), f"划分文件 {sp}.txt（{len(f.read_text(encoding='utf-8').splitlines()) if f.exists() else 0} 行）",
        f"缺少划分文件 {f}")
chk((ROOT / "dataset" / "det" / "label_list.txt").exists(),
    "类别清单 label_list.txt", "缺少 dataset/det/label_list.txt")

sec("【5】权重（最终交付只需要这两个）")
for w in ("verifier_pre128_best.pdparams", "refiner_giou128_best.pdparams"):
    p = ROOT / "results" / "weights" / w
    chk(p.exists(), f"{w}（{p.stat().st_size / 1048576:.1f} MiB）" if p.exists()
        else "", f"缺少权重 {p}")

sec("【6】代码模块（demo 依赖 src/det 下的公共实现）")
sys.path.insert(0, str(ROOT / "src" / "det"))
for mod in ("core.utils", "core.data", "core.boxes", "core.verifier", "core.refiner",
            "core.window_detector", "eval_det", "eval_fused"):
    try:
        importlib.import_module(mod)
        print(f"  [OK]   import {mod}")
        OK.append(mod)
    except Exception as e:
        chk(False, "", f"import {mod} 失败：{type(e).__name__}: {e}")

sec("结论")
if BAD:
    print(f"  ✘ 不能直接运行：缺 {len(BAD)} 项")
    for b in BAD:
        print("     -", b)
    print("  建议：确认拿到的是完整程序包（含 dataset/ 与 results/weights/），或先装好依赖")
    code = 1
elif WARN:
    print("  △ 可以运行，但有注意项：")
    for w in WARN:
        print("     -", w)
    print("  双击 demo/run_gui.bat 即可（程序会自动用 CPU）；想快请在有 NVIDIA GPU 的机器上跑")
    code = 0
else:
    print("  ✔ 环境完整，可以直接演示：双击 demo/run_gui.bat（或 python demo/demo_fused.py --gui）")
    code = 0
print("=" * 78)
sys.exit(code)
