"""成员 C · 演示程序启动器（跨机器通用）

作用：不需要手动指定 conda 路径。它会自动找一个**装了 paddle 的 Python**，
再用它去启动 demo_fused.py。换到别的电脑时，只要这台机器上有 Python 且装了
paddle，双击 run_gui.bat 就能跑。

查找顺序：
    1) 环境变量 DEMO_PY 指定的解释器
    2) 当前解释器自己（如果它已经能 import paddle）
    3) py -3 / python / python3（PATH 上能找到的）
    4) 常见的 conda 安装位置（miniconda/anaconda 下的 envs/*/python.exe）
    5) demo_fused.py 里记录的默认解释器路径（本机 paddle_env）

用法：
    python demo/launch.py                # 图形界面
    python demo/launch.py --list         # 其它参数原样传给 demo_fused.py
    python demo/launch.py --check        # 只做环境自检（等价于 check_env.py）
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "demo_fused.py"
# 本机默认解释器（换机器时若前几项都找不到，会用它兜底并给出提示）
FALLBACK = [r"C:\Users\htt22\miniconda3\envs\paddle_env\python.exe"]


def has_paddle(py):
    """该解释器能否 import paddle（不加载模型，1~2 秒）"""
    try:
        r = subprocess.run([py, "-c", "import paddle;print(paddle.__version__)"],
                           capture_output=True, timeout=120, text=True)
        return r.returncode == 0, (r.stdout or "").strip()
    except Exception:
        return False, ""


def candidates():
    out = []
    if os.environ.get("DEMO_PY"):
        out.append(os.environ["DEMO_PY"])
    out.append(sys.executable)
    for name in ("py", "python", "python3"):
        out.append(name)
    # 常见 conda 位置
    for base in (Path.home() / "miniconda3", Path.home() / "anaconda3",
                 Path(r"C:\ProgramData\miniconda3"), Path(r"C:\ProgramData\anaconda3"),
                 Path(r"C:\Users\htt22\miniconda3"), Path(r"C:\Users\htt22\anaconda3")):
        if base.is_dir():
            out.append(str(base / "python.exe"))
            for env in sorted(base.glob("envs/*/python.exe")):
                out.append(str(env))
    out += FALLBACK
    seen, uniq = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def main():
    if "--check" in sys.argv:
        return subprocess.call([sys.executable, str(HERE / "check_env.py")])
    print("正在查找装有 paddle 的 Python 解释器 …", flush=True)
    found = None
    for c in candidates():
        tag = c if c not in ("py", "python", "python3") else f"{c} (PATH)"
        ok, ver = has_paddle(c)
        print(f"  {'✔' if ok else '·'} {tag}" + (f"  paddle {ver}" if ok else ""), flush=True)
        if ok:
            found = c
            break
    if not found:
        print("\n[错误] 没找到装有 paddle 的 Python。请先安装环境：")
        print("  1) 安装 Python 3.10~3.12")
        print("  2) pip install paddlepaddle-gpu==3.2.0   （或 CPU 版 paddlepaddle==3.2.0）")
        print("  3) 再装依赖：pip install numpy pillow matplotlib pyyaml")
        print("  也可以先设环境变量 DEMO_PY 指向你的解释器，或运行 python demo/check_env.py 看详情")
        return 2
    args = [a for a in sys.argv[1:]]
    cmd = [found, str(TARGET)] + (args if args else ["--gui"])
    print(f"\n使用解释器：{found}\n命令：{' '.join(cmd)}\n")
    return subprocess.call(cmd, cwd=str(HERE.parent))


if __name__ == "__main__":
    sys.exit(main())
