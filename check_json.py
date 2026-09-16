import json
import glob
import os

# 找到 results/metrics 下所有 json
files = sorted(glob.glob("results/metrics/*.json"))
print("一共找到", len(files), "个 json 文件")

for p in files:
    print("=" * 60)
    print(p)
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            print("里面的项目名（keys）：", list(obj.keys()))
        else:
            print("类型：", type(obj).__name__, "长度：", len(obj))
    except Exception as e:
        print("读取出错：", e)