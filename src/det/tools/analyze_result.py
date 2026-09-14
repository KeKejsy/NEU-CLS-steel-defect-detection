"""成员 C · 结果分析：从混淆矩阵定位短板类别

用法：
    python src/det/tools/analyze_result.py --json results/metrics/test_fuse_best.json
    python src/det/tools/analyze_result.py            # 默认读融合方案的验证集结果

它把混淆矩阵拆成三个视角，用来判断「该优化哪个环节」：

    1. 行视角（预测侧）：某一类被预测出来时，有多少是对的 —— 低则说明**误检多**；
    2. 列视角（真实侧）：某一类的 GT 被检出的比例 —— 低则说明**漏检多**；
    3. 错分去向：漏检/错分具体跑到哪一类去了 —— 指向「该加什么特征或尺度」。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import utils  # noqa: E402

CN = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
CN_CN = {"Cr": "龟裂", "In": "夹杂", "Pa": "斑块", "PS": "麻点",
         "RS": "氧化铁皮压入", "Sc": "划痕"}


def main():
    ap = argparse.ArgumentParser(description="检测结果分析（混淆矩阵拆解）")
    ap.add_argument("--json", default="results/metrics/test_fuse_best.json")
    args = ap.parse_args()

    p = Path(args.json)
    if not p.is_absolute():
        p = utils.ROOT / p
    if not p.exists():
        raise SystemExit(f"找不到 {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    labels = d["confusion_matrix"]["labels"]
    M = d["confusion_matrix"]["matrix"]

    print("=" * 84)
    print(f"C · 结果分析    {p.relative_to(utils.ROOT)}")
    print(f"  流程={d.get('pipeline')} 模式={d.get('mode','-')} 划分={d['split']}  "
          f"mAP@0.5={d['map50']:.4f}")
    print("=" * 84)

    print("\n[1] 混淆矩阵（行=预测类别，列=真实类别，末行/列=背景）")
    w = 12
    print(" " * 10 + "".join(f"{l:>{w}}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:>8}  " + "".join(f"{v:>{w}}" for v in M[i]))

    print("\n[2] 行视角：该类被预测出来时有多少是对的（衡量误检）")
    print(f"{'类别':<6}{'中文':<14}{'预测总数':>10}{'正确':>8}{'误检(无GT)':>12}"
          f"{'错分到别类':>12}{'行精度':>10}")
    for i, c in enumerate(CN):
        row = M[i]
        tot = sum(row)
        tp = row[i]
        fp = row[6]
        other = sum(row[:6]) - tp
        prec = tp / tot if tot else 0.0
        print(f"{c:<6}{CN_CN[c]:<14}{tot:>10}{tp:>8}{fp:>12}{other:>12}{prec:>10.4f}")

    print("\n[3] 列视角：该类 GT 被检出的情况（衡量漏检）")
    print(f"{'类别':<6}{'中文':<14}{'GT总数':>10}{'正确检出':>10}{'漏检':>8}{'列召回':>10}")
    for j, c in enumerate(CN):
        col = [M[i][j] for i in range(len(labels))]
        tp = M[j][j]
        tot = sum(col)
        miss = M[6][j]
        rec = tp / tot if tot else 0.0
        print(f"{c:<6}{CN_CN[c]:<14}{tot:>10}{tp:>10}{miss:>8}{rec:>10.4f}")

    print("\n[4] 错分去向（漏检以外，预测成了哪些别的类）")
    for j, c in enumerate(CN):
        confused = [(CN[i], M[i][j]) for i in range(len(CN)) if i != j and M[i][j] > 0]
        confused.sort(key=lambda t: -t[1])
        miss = M[6][j]
        top = ", ".join(f"{a}×{b}" for a, b in confused[:3]) or "无"
        print(f"  {c:<4} 漏检 {miss:>5}  错分到: {top}")

    print("\n[5] 判读")
    rows = [(CN[i], M[i][i] / max(sum(M[i]), 1)) for i in range(len(CN))]
    worst_row = min(rows, key=lambda t: t[1])
    best_row = max(rows, key=lambda t: t[1])
    print(f"  行精度最高: {best_row[0]} {best_row[1]:.4f}   最低: {worst_row[0]} {worst_row[1]:.4f}")
    neg = [i for i in range(len(CN)) if M[i][6] > 0.8 * max(sum(M[i]), 1)]
    if neg:
        print(f"  误检占比超 80% 的类别: {', '.join(CN[i] for i in neg)}"
              f" —— 这些类的分数阈值需要提高，或需要更强的负样本")
    print("=" * 84)


if __name__ == "__main__":
    main()
