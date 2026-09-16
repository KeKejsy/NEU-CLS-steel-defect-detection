# src/tools/run_all.py
import os
import sys
import subprocess

# 找到项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

print("=" * 50)
print("欢迎使用一键复现总开关！")
print("=" * 50)

# 获取当前 Python 解释器的准确路径
python_exe = sys.executable

# 第一步：检查数据
data_check = os.path.join(project_root, "src", "data", "check_submit.py")
if os.path.exists(data_check):
    print("\n>>> 正在检查数据...")
    subprocess.run([python_exe, data_check], check=False)
else:
    print("\n>>> 跳过数据检查（未找到 A 的脚本）")

# 第二步：汇总结果（对账本）
print("\n>>> 正在汇总结果（对账本）...")
log2table = os.path.join(current_dir, "log2table.py")
subprocess.run([python_exe, log2table], check=False)

# 第三步：画对比图
print("\n>>> 正在画对比图...")
plot_summary = os.path.join(current_dir, "plot_summary.py")
subprocess.run([python_exe, plot_summary], check=False)

print("\n" + "=" * 50)
print("全部完成！请去 results 文件夹里看结果。")
print("=" * 50)