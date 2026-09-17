"""全项目一键运行入口（CPU / GPU 通用）

## 这个脚本解决什么

项目里有 A/B/C/D 四个成员的工作，各自有若干入口脚本，散在 `src/data`、`src/cls`、
`src/det`、`src/tools` 四个目录，运行顺序与参数（尤其 C 的 `--norm/--in-size`）容易搞错。
本脚本把它们串成一条**可断点续跑**的流水线：

    [0] 环境与数据自检        （不需要 GPU）
    [1] A 数据自检
    [2] B 分类训练 + 评估      （CPU 也能跑，只是慢）
    [3] C 检测：区域判别模型训练（最终方案的核心）
    [4] C 检测：融合方案正式评估（验证集 + 测试集）
    [5] C 检测：真值-预测对比图
    [6] C 检测：单阶段失败基线训练 + 评估（仅 full 档，很慢）
    [7] D 工具链：汇总表 + 对比图
    [8] 汇总与一致性校验（summarize / check_docs / check_submit）
    [9] 运行报告

## 三档 profile

| profile | 做什么 | 大致耗时（GPU） | 用途 |
|---|---|---|---|
| `eval` | 只跑自检/汇总/校验，**不训练** | 约 1~2 分钟 | 验收、查结果 |
| `standard` | 训练最终检测方案并评估 val+test | 约 1 小时 | 复现交付指标 |
| `full` | 再加单阶段 YOLOv3 / PP-YOLOE-s | 约 3 小时 | 完整复现所有实验 |

## CPU 与 GPU

- `--device auto`（默认）：有 CUDA 的 Paddle 且能用 GPU 时走 GPU；否则自动降到 CPU。
- `--device cpu`：强制 CPU。为避免"CPU 上跑 40 个 epoch 要一夜"，会自动把 C 的训练
  降级为 **96 输入 + 5 epoch + batch 32**（`--epochs-scale` 可再调）；评估也用 96 输入。
  `full` 档的单阶段训练在 CPU 上会被跳过（实测 CPU 上不可用，见 README）。
- `--device gpu`：强制 GPU，最终交付口径（预训练主干 + GIoU + **128 输入**）。

## 重跑策略

**默认是重跑**：每个阶段都会执行，看到上次的 `.done` 标记只会提示一句、继续跑
（这样"跑完什么也没做"的误会就不会再出现）。
想利用断点续跑、跳过已完成的阶段，加 `--resume`。
`--retrain` 会在权重已存在时也重新训练（默认复用已有权重，避免白跑几小时）；
`--only a,b` / `--skip b` 用来挑选阶段。

## 常用命令

    python run_project.py --dry-run                 # 只打印将要执行的命令（零写入）
    python run_project.py --profile eval            # 验收：自检 + 汇总 + 校验
    python run_project.py --quick                   # 冒烟：每步都跑，但只 1 个 epoch / 20 张图
    python run_project.py --profile standard        # 复现最终指标（GPU）
    python run_project.py --device cpu --profile standard
    python run_project.py --list                    # 列出所有阶段

## 注意

- 必须用能 import paddle 的解释器运行本脚本（子进程复用同一个解释器 `sys.executable`）。
- 任务只会**串行**执行：本项目实测 GPU 上并发训练会触发显存重试（慢 100 倍以上）。
- 测试集：`standard`/`full` 会评估测试集。按项目铁律，测试集每次评估都要记录用途，
  本脚本把它计入运行报告，避免"悄悄多用几次"。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent          # 仓库根目录（本脚本放在根目录）
PY = sys.executable                              # 复用同一个解释器（保证能 import paddle）
RUN_DIR = BASE / "results" / "logs" / "run_project"

# 解释器探测结果缓存（探测一次要几秒）
_PADDLE_CACHE: dict[str, str | None] = {}


# --------------------------------------------------------------------------- #
# 解释器探测：确保用的是装了 paddle 的那个 python
# --------------------------------------------------------------------------- #
def _paddle_in(py: str, timeout: int = 180):
    """某个解释器能否 import paddle；能就返回 "版本|paddle 版本"，否则 None

    结果按解释器路径缓存（探测一次要几秒，避免重复开销）。
    """
    if py in _PADDLE_CACHE:
        return _PADDLE_CACHE[py]
    info = None
    try:
        r = subprocess.run([py, "-c", "import paddle,sys;"
                            "print(sys.version.split()[0] + '|' + paddle.__version__)"],
                           capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="ignore")
        if r.returncode == 0 and "|" in (r.stdout or ""):
            info = r.stdout.strip()
    except Exception:                                     # noqa: BLE001
        info = None
    _PADDLE_CACHE[py] = info
    if os.environ.get("RUN_PROJECT_DEBUG"):
        print(f"   (探测 {py} -> {info})")
    return info


def candidate_pythons() -> list[str]:
    """列出机器上值得一试的解释器（去重，保持优先级）

    优先 conda 环境：本项目就是在 paddle_env 里跑出来的；其次是 PATH 上的 python。
    """
    import glob
    import shutil as _sh
    home = Path.home()
    out = [sys.executable]
    if os.environ.get("CONDA_PREFIX"):
        out.append(str(Path(os.environ["CONDA_PREFIX"]) / "python.exe"))
    for root in (home / "miniconda3", home / "anaconda3", home / "miniforge3",
                 Path("C:/ProgramData/miniconda3"), Path("C:/ProgramData/anaconda3")):
        out.append(str(root / "python.exe"))
        for env_py in sorted(glob.glob(str(root / "envs" / "*" / "python.exe"))):
            out.append(env_py)
    for name in ("python3", "py"):
        w = _sh.which(name)
        if w:
            out.append(w)
    seen, uniq = set(), []
    for p in out:
        key = p.lower()
        if key not in seen and Path(p).exists():
            seen.add(key)
            uniq.append(p)
    return uniq


def switch_to_paddle_python(force: str | None) -> str | None:
    """找一个能 import paddle 的解释器并返回；当前解释器就可用时返回 None

    返回值不为空表示"需要用它重新运行本脚本"。
    """
    if _paddle_in(sys.executable) is not None:
        return None                                        # 当前就可用，什么都不用做
    if force:
        if _paddle_in(force) is None:
            print(f"✘ --python 指定的解释器不可用（import paddle 失败）：{force}")
            return "FAIL"
        return force
    if os.environ.get("RUN_PROJECT_NO_AUTO") == "1":
        return None
    print("⚠️ 当前解释器没有 paddle，正在查找机器上可用的解释器…")
    for py in candidate_pythons():
        if Path(py) == Path(sys.executable):
            continue
        info = _paddle_in(py)
        if info:
            ver, pad = info.split("|", 1)
            print(f"   ✔ 找到：{py}")
            print(f"     Python {ver} / paddle {pad}")
            return py
        print(f"   ✘ {py}")
    return None

# --------------------------------------------------------------------------- #
# 阶段定义
# --------------------------------------------------------------------------- #
# 每个阶段：
#   id       阶段标识（--only/--skip 用它，也用作 .done 标记名）
#   title    中文标题
#   profiles 属于哪些档（eval 档也会跑「不需要训练」的阶段）
#   cmds     ctx -> list[list[str]]：要依次执行的命令
#   skip     ctx -> str | None：返回跳过原因（None 表示不跳过）
#   eta      预计耗时文字（仅用于打印）
#
# ctx 里可用的键：device, quick, force, in_size, norm, v_epochs, r_epochs, batch, limit


def _py(script: str) -> list[str]:
    return [PY, "-u", str(BASE / script)]


def has_flag(script: str, flag: str, cache: dict) -> bool:
    """判断某个脚本是否支持某个命令行参数（跑一次 --help 探测，结果缓存）

    为什么不写死名单：脚本会演进，写死容易在参数改名后静默失效 ——
    这里探测失败就当作"不支持"，宁可不传，也不传一个会报错的参数。
    """
    key = (script, flag)
    if key in cache:
        return cache[key]
    ok = False
    try:
        r = subprocess.run([PY, "-u", str(BASE / script), "--help"],
                           capture_output=True, text=True, timeout=60,
                           cwd=str(BASE), encoding="utf-8", errors="ignore")
        ok = flag in (r.stdout or "")
    except Exception:                                     # noqa: BLE001
        ok = False
    cache[key] = ok
    return ok


def dev(cmd: list[str], script: str, ctx: dict) -> list[str]:
    """按需给命令补上 --device（脚本支持才加）"""
    if has_flag(script, "--device", ctx["flag_cache"]):
        cmd += ["--device", ctx["device"]]
    return cmd


# --------------------------------------------------------------------------- #
# 各阶段的命令构造
# --------------------------------------------------------------------------- #
def cmd_data_check(ctx):
    return [_py("src/data/check_submit.py")]


def cmd_cls_train(ctx):
    out = []
    for model in ("mobilenet_v3_small", "resnet50_vd"):
        c = _py("src/cls/train.py") + ["--model", model]
        if ctx["quick"]:
            c += ["--epochs", "1", "--limit", "200"]
        out.append(c)
    return out


def cmd_cls_eval(ctx):
    out = []
    for model in ("mobilenet_v3_small", "resnet50_vd"):
        out.append(_py("src/cls/eval_cls.py") + ["--model", model, "--split", "test"])
    return out


def cmd_cls_export(ctx):
    return [_py("src/cls/export.py") + ["--model", "resnet50_vd"]]


def _verifier_name(ctx):
    """权重命名与已交付口径对齐：verifier_pre128 = 预训练主干 + 128 输入

    对齐之后有两个好处：① `--profile standard` 在权重已存在时直接复用，不必重训；
    ② 评估命令里引用的权重名与文档/交付说明里写的一致，避免"跑出来却对不上"。
    """
    if ctx["quick"]:
        return "verifier_smoke"
    return (f"verifier_pre{ctx['in_size']}" if ctx["pretrained"]
            else f"verifier_scratch{ctx['in_size']}")


def _refiner_name(ctx):
    """精修器统一用 GIoU 回归损失，命名对齐已交付的 refiner_giou128"""
    if ctx["quick"]:
        return "refiner_smoke"
    return (f"refiner_giou{ctx['in_size']}" if ctx["pretrained"]
            else f"refiner_giou_scratch{ctx['in_size']}")


def cmd_region_train(ctx):
    vname, rname = _verifier_name(ctx), _refiner_name(ctx)
    v = _py("src/det/train_verifier.py") + ["--name", vname]
    r = _py("src/det/train_refiner.py") + ["--name", rname]
    if ctx["pretrained"]:
        v += ["--pretrained"]
        r += ["--pretrained"]
    for c in (v, r):
        c += ["--norm", ctx["norm"], "--in-size", str(ctx["in_size"]),
              "--epochs", str(ctx["v_epochs"] if c is v else ctx["r_epochs"]),
              "--batch-size", str(ctx["batch"])]
    r += ["--reg-loss", "giou", "--init-verifier",
          f"results/weights/{vname}_best.pdparams"]
    return [dev(v, "src/det/train_verifier.py", ctx), dev(r, "src/det/train_refiner.py", ctx)]


def cmd_region_skip(ctx):
    v = BASE / f"results/weights/{_verifier_name(ctx)}_best.pdparams"
    r = BASE / f"results/weights/{_refiner_name(ctx)}_best.pdparams"
    if v.exists() and r.exists():
        return f"权重已存在（{v.name} / {r.name}）"
    return None


def cmd_fused_eval(ctx):
    v = f"results/weights/{_verifier_name(ctx)}_best.pdparams"
    r = f"results/weights/{_refiner_name(ctx)}_best.pdparams"
    out = []
    for split, tag in (("val", f"val_fuse_{ctx['tag']}"), ("test", f"test_fuse_{ctx['tag']}")):
        c = _py("src/det/eval_fused.py") + [
            "--split", split, "--mode", "fuse", "--stride", "12", "--score-th", "0.6",
            "--nms-iou", "0.4", "--top-k", "150", "--max-det", "50",
            "--norm", ctx["norm"], "--in-size", str(ctx["in_size"]),
            "--verifier-weights", v, "--refiner-weights", r, "--tag", tag]
        if ctx["quick"]:
            c += ["--limit", str(ctx["limit"])]
        out.append(dev(c, "src/det/eval_fused.py", ctx))
    return out


def cmd_viz(ctx):
    v = f"results/weights/{_verifier_name(ctx)}_best.pdparams"
    r = f"results/weights/{_refiner_name(ctx)}_best.pdparams"
    c = _py("src/det/tools/viz_fused.py") + [
        "--split", "val", "--per-class", "1" if ctx["quick"] else "2",
        "--norm", ctx["norm"], "--in-size", str(ctx["in_size"]),
        "--verifier-weights", v, "--refiner-weights", r, "--tag", f"val_fuse_{ctx['tag']}"]
    return [dev(c, "src/det/tools/viz_fused.py", ctx)]


def cmd_baseline_train(ctx):
    out = []
    for cfg in ("src/det/configs/yolov3.yml", "src/det/configs/ppyoloe_s.yml"):
        c = _py("src/det/train.py") + ["--config", cfg]
        if ctx["quick"]:
            c += ["--epochs", "1", "--batch-size", "4"]
        out.append(dev(c, "src/det/train.py", ctx))
    return out


def cmd_baseline_eval(ctx):
    out = []
    for model in ("yolov3", "ppyoloe_s"):
        c = _py("src/det/eval_det.py") + ["--model", model, "--split", "test"]
        if ctx["quick"]:
            c += ["--limit", str(ctx["limit"])]
        out.append(dev(c, "src/det/eval_det.py", ctx))
    return out


def cmd_tools(ctx):
    return [_py("src/tools/log2table.py"), _py("src/tools/plot_summary.py")]


def cmd_summarize(ctx):
    return [_py("src/det/tools/summarize.py")]


def cmd_verify(ctx):
    return [_py("src/det/tools/check_docs.py")]


STAGES = [
    dict(id="data", title="A · 数据自检（23 项）", profiles={"eval", "standard", "full"},
         cmds=cmd_data_check, skip=None, eta="几秒"),

    dict(id="cls_train", title="B · 分类训练（MobileNetV3 + ResNet50）",
         profiles={"standard", "full"}, cmds=cmd_cls_train,
         skip=lambda ctx: "权重已存在" if (BASE / "results/weights/cls_resnet50_vd_best.pdparams").exists()
         and (BASE / "results/weights/cls_mobilenet_v3_small_best.pdparams").exists() else None,
         eta="CPU 约 140 分钟 / GPU 快得多"),

    dict(id="cls_eval", title="B · 分类评估（测试集）", profiles={"standard", "full"},
         cmds=cmd_cls_eval, skip=None, eta="约 2 分钟"),

    dict(id="cls_export", title="B · 导出推理模型", profiles={"full"},
         cmds=cmd_cls_export, skip=None, eta="约 1 分钟"),

    dict(id="region_train", title="C · 区域判别模型训练（验证器 + 精修器，最终方案核心）",
         profiles={"standard", "full"}, cmds=cmd_region_train, skip=cmd_region_skip,
         eta="GPU 约 22 分钟（128 输入）/ CPU 约 30~90 分钟（96 输入、5 epoch）"),

    dict(id="fused_eval", title="C · 融合方案正式评估（验证集 + 测试集）",
         profiles={"standard", "full"}, cmds=cmd_fused_eval, skip=None,
         eta="GPU 约 45 分钟（两个划分，128 输入）/ CPU 更慢"),

    dict(id="viz", title="C · 真值-预测对比图", profiles={"standard", "full"},
         cmds=cmd_viz, skip=None, eta="约 1~3 分钟"),

    dict(id="baseline_train", title="C · 单阶段失败基线训练（YOLOv3 + PP-YOLOE-s）",
         profiles={"full"}, cmds=cmd_baseline_train, skip=None, eta="GPU 约 130 分钟"),

    dict(id="baseline_eval", title="C · 单阶段失败基线评估", profiles={"full"},
         cmds=cmd_baseline_eval, skip=None, eta="约 5 分钟"),

    dict(id="tools", title="D · 工具链（汇总表 + 对比图）",
         profiles={"eval", "standard", "full"}, cmds=cmd_tools, skip=None, eta="几秒"),

    dict(id="summarize", title="汇总：生成 det_summary.md/json",
         profiles={"eval", "standard", "full"}, cmds=cmd_summarize, skip=None, eta="几秒"),

    dict(id="verify", title="校验：文档与产物一致性",
         profiles={"eval", "standard", "full"}, cmds=cmd_verify, skip=None, eta="几秒"),
]


# --------------------------------------------------------------------------- #
# 环境探测
# --------------------------------------------------------------------------- #
def detect_device(prefer: str) -> tuple[str, str]:
    """返回 (实际设备, 说明)"""
    try:
        import paddle                                    # noqa: PLC0415
    except Exception as exc:                             # noqa: BLE001
        return "cpu", f"无法 import paddle（{type(exc).__name__}），按 CPU 处理"

    if prefer == "cpu":
        return "cpu", "按参数强制 CPU"
    try:
        if paddle.is_compiled_with_cuda():
            import numpy as np                            # noqa: PLC0415
            x = paddle.to_tensor(np.ones([8, 8], dtype="float32"))
            _ = (x @ x).sum().item()                      # 真正跑一次，确认驱动可用
            return "gpu", f"CUDA 可用（paddle {paddle.__version__}）"
    except Exception as exc:                             # noqa: BLE001
        if prefer == "gpu":
            print(f"  ⚠️ 强制要求 GPU，但探测失败：{type(exc).__name__}: {exc}")
        return "cpu", f"CUDA 不可用（{type(exc).__name__}），自动降级 CPU"
    return "cpu", "当前 Paddle 是 CPU 版，按 CPU 处理"


def preflight(ctx) -> bool:
    """环境与数据自检，返回是否继续"""
    print("=" * 84)
    print("成员 C · 全项目一键运行")
    print("=" * 84)
    ok = True

    print(f"\n[环境]")
    print(f"  仓库根目录 : {BASE}")
    print(f"  解释器     : {PY}")
    try:
        import paddle                                     # noqa: PLC0415
        import numpy as np                                # noqa: PLC0415
        print(f"  paddle     : {paddle.__version__}"
              f"（CUDA 编译={'是' if paddle.is_compiled_with_cuda() else '否'}）")
        print(f"  numpy      : {np.__version__}")
    except Exception as exc:                              # noqa: BLE001
        print(f"  ✘ 依赖缺失：{type(exc).__name__}: {exc}")
        ok = False
    print(f"  设备       : {ctx['device']}（{ctx['device_reason']}）")

    need = ["dataset/det/ImageSets/Main/train.txt", "dataset/det/ImageSets/Main/val.txt",
            "dataset/det/ImageSets/Main/test.txt", "dataset/det/label_list.txt"]
    miss = [n for n in need if not (BASE / n).exists()]
    n_img = len(list((BASE / "dataset/det/JPEGImages").glob("*.jpg")))
    n_xml = len(list((BASE / "dataset/det/Annotations").glob("*.xml")))
    print(f"\n[数据]")
    print(f"  检测图/标注 : {n_img} / {n_xml}")
    print(f"  划分文件    : {'齐' if not miss else '缺 ' + ', '.join(miss)}")
    if miss or n_img == 0:
        print("  ✘ 数据不完整：请先让 A 跑数据脚本，或确认 dataset/ 已就位")
        ok = False
    if " " in str(BASE):
        print("\n[提示] 项目路径里有空格：项目铁律要求纯英文无空格的路径。"
              "\n       目前实测能跑通，但遇到古怪报错时可以先怀疑这一点。")
    free_gb = shutil.disk_usage(str(BASE)).free / 1024 ** 3
    print(f"  磁盘剩余    : {free_gb:.1f} GB")
    if free_gb < 3:
        print("  ⚠️ 剩余空间偏少（权重与缓存需要数 GB）")
    return ok


# --------------------------------------------------------------------------- #
# 执行
# --------------------------------------------------------------------------- #
def run_cmd(cmd: list[str], log_path: Path, ctx) -> tuple[bool, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(BASE), stdout=f, stderr=subprocess.STDOUT, env=env)
    sec = time.time() - t0
    if p.returncode != 0:
        print(f"      ✘ 返回码 {p.returncode}，日志末尾：")
        try:
            tail = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-25:]
            for line in tail:
                print("        " + line)
        except Exception:                                # noqa: BLE001
            pass
    return p.returncode == 0, sec


def stage_done_path(sid: str) -> Path:
    return RUN_DIR / f"{sid}.done"


def main():
    ap = argparse.ArgumentParser(
        description="全项目一键运行（CPU / GPU 通用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例：python run_project.py --profile eval\n"
               "    python run_project.py --device cpu --profile standard\n"
               "    python run_project.py --quick --dry-run")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"],
                    help="auto=有 GPU 就用 GPU，否则 CPU")
    ap.add_argument("--profile", default="eval", choices=["eval", "standard", "full"],
                    help="eval=只自检/汇总（不训练）；standard=训练最终检测方案；full=再加单阶段基线")
    ap.add_argument("--quick", action="store_true",
                    help="冒烟模式：每步都跑但极小规模（1 epoch / 20 张图 / 1 类各 1 张）")
    ap.add_argument("--epochs-scale", type=float, default=None,
                    help="训练 epoch 倍率（默认 GPU=1.0，CPU=0.17，即 30→5、40→5）")
    ap.add_argument("--in-size", type=int, default=None,
                    help="判别模型输入尺寸（默认 GPU=128、CPU=96；必须与训练一致）")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="不用 ImageNet 预训练主干（复现旧流水线口径时用；首次需联网下载则必须不用）")
    ap.add_argument("--only", default=None, help="只跑这些阶段，逗号分隔（见 --list）")
    ap.add_argument("--skip", default=None, help="跳过这些阶段，逗号分隔")
    ap.add_argument("--resume", action="store_true",
                    help="跳过已完成的阶段（默认是**重跑**：即使上次跑过也再跑一遍）")
    ap.add_argument("--retrain", action="store_true",
                    help="即使权重已存在也重新训练（默认复用已有权重，避免白跑几小时）")
    ap.add_argument("--force", action="store_true",
                    help="等价于 --retrain（保留兼容：现在的默认行为已经是重跑）")
    ap.add_argument("--keep-going", action="store_true", help="某个阶段失败也继续后面的")
    ap.add_argument("--dry-run", action="store_true", help="只打印将执行的命令，不真正运行")
    ap.add_argument("--list", action="store_true", help="列出所有阶段后退出")
    ap.add_argument("--python", default=None,
                    help="用指定的解释器运行（默认自动选：优先找装了 paddle 的那个）")
    ap.add_argument("--no-auto-python", action="store_true",
                    help="不要自动切换解释器（当前 python 没有 paddle 时直接停下并给出提示）")
    args = ap.parse_args()

    if args.list:
        print(f"{'id':<16}{'档位':<22}标题")
        print("-" * 90)
        for s in STAGES:
            print(f"{s['id']:<16}{'/'.join(sorted(s['profiles'])):<22}{s['title']}")
        print("\n提示：--only / --skip 用上面的 id，逗号分隔")
        return 0

    # 解释器自检：没激活 conda 环境时，直接 `python run_project.py` 会是系统 Python（无 paddle）。
    # 这里自动找到装了 paddle 的解释器并用它**重新运行本脚本**，整个流程（含训练/评估子进程）
    # 都会用同一个解释器，避免"到处找不到 paddle"。
    if args.list is False:
        pick = None if args.no_auto_python else switch_to_paddle_python(args.python)
        if pick == "FAIL":
            print("请用正确的解释器运行，例如：")
            print(f'  "{Path.home() / "miniconda3/envs/paddle_env/python.exe"}" run_project.py')
            return 2
        if pick:
            cmd = [pick, str(BASE / "run_project.py"), *sys.argv[1:], "--no-auto-python"]
            print(f"   ↻ 改用该解释器重新运行：{Path(pick).name} run_project.py "
                  f"{' '.join(sys.argv[1:])}\n")
            env = dict(os.environ, RUN_PROJECT_NO_AUTO="1", PYTHONIOENCODING="utf-8")
            return subprocess.run(cmd, cwd=str(BASE), env=env).returncode
        if _paddle_in(sys.executable) is None:
            if args.no_auto_python:
                print("\n✘ 当前解释器没有 paddle（已按 --no-auto-python 跳过自动查找）。请：")
            else:
                print("\n✘ 当前解释器没有 paddle，机器上也没找到装有 paddle 的解释器。请：")
            print("    conda activate paddle_env            # 激活本项目使用的环境")
            print(f'    "{Path.home() / "miniconda3/envs/paddle_env/python.exe"}" run_project.py   # 或直接用该解释器')
            print("  （环境已装好但不在常见位置时，用 --python <解释器路径> 手动指定）")
            return 2

    device, reason = detect_device(args.device)
    if args.quick:
        device = "cpu" if device == "cpu" else "gpu"      # 冒烟模式同样尊重设备探测

    if args.in_size is not None:
        in_size = args.in_size
    else:
        in_size = 128 if device == "gpu" else 96
    if args.epochs_scale is not None:
        scale = args.epochs_scale
    elif args.quick:
        scale = 0.0                                      # 由 quick 单独指定 epoch
    else:
        scale = 1.0 if device == "gpu" else 0.17
    v_epochs = 1 if args.quick else max(1, round(30 * scale))
    r_epochs = 1 if args.quick else max(1, round(40 * scale))
    norm = "none" if args.no_pretrained else "imagenet"
    pretrained = not args.no_pretrained

    ctx = dict(
        device=device, device_reason=reason, quick=args.quick,
        resume=args.resume, retrain=(args.retrain or args.force),
        in_size=in_size, norm=norm, pretrained=pretrained,
        v_epochs=v_epochs, r_epochs=r_epochs,
        batch=16 if args.quick else (64 if device == "gpu" else 32),
        limit=20, flag_cache={},
        # tag 同时编码「是否预训练」与输入尺寸，避免不同配置互相覆盖产物；
        # quick 冒烟单独用 smoke，绝不写进正式报告文件名（否则会覆盖交付产物）
        tag="smoke" if args.quick else f"{'pre' if pretrained else 'scratch'}{in_size}",
    )

    if not preflight(ctx):
        print("\n✘ 环境/数据自检未通过，已停止。")
        return 2

    # 阶段筛选
    only = {s.strip() for s in args.only.split(",")} if args.only else None
    skip = {s.strip() for s in args.skip.split(",")} if args.skip else set()
    valid_ids = {s["id"] for s in STAGES}
    if only:
        bad = sorted(only - valid_ids)
        if bad:
            print(f"\n✘ --only 里有不存在的阶段：{', '.join(bad)}")
            print(f"  可用阶段：{', '.join(s['id'] for s in STAGES)}")
            return 2
    bad_skip = sorted(skip - valid_ids)
    if bad_skip:
        print(f"⚠️ --skip 里有不存在的阶段（已忽略）：{', '.join(bad_skip)}")

    todo = []
    for s in STAGES:
        # --only 的优先级高于 profile：显式点了名的阶段就一定要跑，
        # 否则会出现「--only region_train --profile eval」这种组合下什么都没跑的情况。
        if only is not None:
            if s["id"] not in only:
                continue
        elif args.profile not in s["profiles"]:
            continue
        if s["id"] in skip:
            continue
        todo.append(s)
    if not todo:
        print("\n⚠️ 没有任何阶段需要执行（检查 --profile / --only / --skip 的组合）")
        return 0

    print(f"\n[计划] profile={args.profile}  设备={device}  输入={in_size}  归一化={norm}"
          f"  预训练主干={'是' if pretrained else '否'}"
          f"  训练 epoch=验证器 {v_epochs} / 精修器 {r_epochs}"
          + ("  （quick 冒烟）" if args.quick else ""))
    print(f"       共 {len(todo)} 个阶段：{', '.join(s['id'] for s in todo)}")
    print(f"       重跑策略：{'跳过已完成（--resume）' if args.resume else '默认重跑全部阶段'}"
          f"{'；即使权重已存在也重训（--retrain）' if ctx['retrain'] else '；训练阶段若权重已存在则复用'}")
    if device == "cpu" and args.profile != "eval":
        print("       ⚠️ CPU 训练很慢（实测 CPU 上单阶段训练不可用）；已自动降级为 96 输入 / 小 epoch。")

    if args.dry_run:
        print("\n[DRY-RUN] 将执行以下命令：")
        for s in todo:
            skip_reason = s["skip"](ctx) if s["skip"] else None
            flag = f"  （跳过：{skip_reason}）" if (skip_reason and not args.force) else ""
            print(f"\n  ▸ {s['id']}  {s['title']}   预计 {s['eta']}{flag}")
            if not skip_reason or args.force:
                for c in s["cmds"](ctx):
                    print("      $ " + " ".join(c))
        print("\n（dry-run 结束，未执行任何命令，也未写入任何文件）")
        return 0

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    results, t_start = [], time.time()

    for s in todo:
        sid = s["id"]
        done = stage_done_path(sid)
        print(f"\n{'─' * 84}\n▸ [{sid}] {s['title']}   预计 {s['eta']}")
        if done.exists():
            info = json.loads(done.read_text(encoding="utf-8"))
            if args.resume:
                print(f"  ⏭ 已完成，按 --resume 跳过（{info.get('sec', 0):.0f}s，{info['time']}）")
                results.append({**{k: s[k] for k in ("id", "title")},
                                "status": "skipped(done)", "sec": 0,
                                "extra": f"上次 {info['time']}"})
                continue
            print(f"  ↻ 上次已完成于 {info['time']}，默认重跑（要跳过加 --resume）")
        if s["skip"]:
            reason = s["skip"](ctx)
            if reason and not ctx["retrain"]:
                print(f"  ⏭ 跳过：{reason}")
                results.append({**{k: s[k] for k in ("id", "title")},
                                "status": "skipped(reason)", "sec": 0, "extra": reason})
                done.write_text(json.dumps({"time": time.strftime("%F %T"), "sec": 0,
                                            "status": "skipped", "extra": reason},
                                           ensure_ascii=False, indent=2), encoding="utf-8")
                continue

        stage_ok, stage_sec = True, 0.0
        for i, cmd in enumerate(s["cmds"](ctx), 1):
            print(f"  ▸ 命令 {i}/{len(s['cmds'](ctx))}: {' '.join(cmd[:6])} …")
            log = RUN_DIR / f"{sid}_{i}.log"
            ok, sec = run_cmd(cmd, log, ctx)
            stage_sec += sec
            print(f"      {'✔' if ok else '✘'} {sec/60:.1f} 分钟   日志 results/logs/run_project/{log.name}")
            if not ok:
                stage_ok = False
                break
        status = "ok" if stage_ok else "failed"
        results.append({**{k: s[k] for k in ("id", "title")},
                        "status": status, "sec": stage_sec, "extra": ""})
        if stage_ok:
            done.write_text(json.dumps({"time": time.strftime("%F %T"), "sec": stage_sec,
                                        "status": "ok"}, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        elif not args.keep_going:
            print("\n✘ 阶段失败，已停止（--keep-going 可继续后续阶段）")
            break

    # ---------------- 运行报告 ----------------
    total = time.time() - t_start
    rep = {"time": time.strftime("%F %T"), "profile": args.profile, "device": device,
           "device_reason": reason, "in_size": in_size, "norm": norm,
           "pretrained": pretrained, "quick": args.quick,
           "verifier_epochs": v_epochs, "refiner_epochs": r_epochs,
           "total_sec": round(total, 1), "stages": results,
           "python": PY, "cwd": str(BASE)}
    (BASE / "results/metrics").mkdir(parents=True, exist_ok=True)
    jp = BASE / "results/metrics/run_project_summary.json"
    jp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 84}\n运行报告（共 {total/60:.1f} 分钟）\n{'=' * 84}")
    print(f"{'阶段':<16}{'状态':<18}{'耗时':>10}  说明")
    print("-" * 84)
    for r in results:
        t = f"{r['sec']/60:.1f} 分" if r.get("sec") else "—"
        print(f"{r['id']:<16}{r['status']:<18}{t:>10}  {r.get('extra', '')}")

    ok_n = sum(1 for r in results if r["status"] == "ok")
    failed = [r["id"] for r in results if r["status"] == "failed"]
    print(f"\n成功 {ok_n} 个 / 失败 {len(failed)} 个" + (f"：{', '.join(failed)}" if failed else ""))

    if not args.quick:
        print("\n[结果在哪看]")
        print("  检测结果总表        results/metrics/det_summary.md")
        print("  融合方案正式报告    results/metrics/val_fuse_*.json、test_fuse_*.json")
        print("  图（PR/AP/混淆/对照）results/figures/*_fuse_*_{pr,ap,conf,samples}.png")
        print("  全项目汇总表        results/summary_table.csv")
        print("  文档                README.md、docs/C_检测任务工作总结.md、src/det/README_det.md")
        print(f"  本次运行记录        results/metrics/run_project_summary.json")
    else:
        print("\n（quick 冒烟模式：产物只用于验证流程可跑通，数字不代表真实指标）")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
