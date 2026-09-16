import sys
import os

# 把 src 加入搜索路径
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

try:
    from tools import metrics
    print("导入成功！")
    names = [n for n in dir(metrics) if not n.startswith("_")]
    print("里面有的函数：", names)
except Exception as e:
    print("导入失败，错误是：", e)