"""成员 C · 文档一致性校验

用法：
    python src/det/tools/check_docs.py

用途：确认三份文档（工作总结 / 技术文档 / 结果总表）里的关键数字与
`results/metrics/*.json` 产物一致，避免改了实验却忘了同步文档
（本项目就出现过「文档写着 mAP 0.0548、实际已是 0.1435」的风险）。

校验方式：从 JSON 产物读出真值，检查每份文档是否都包含这些数字的字符串形式。
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import utils  # noqa: E402

DOCS = ["docs/C_检测任务工作总结.md",
        "src/det/README_det.md",
        "results/metrics/det_summary.md"]


def jload(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def variants(x):
    """同一个数值的多种合法书写方式（文档里 2~5 位小数都可能出现）

    例如 0.15747 在文档里可能写成 0.1575（4 位）或 0.157（3 位四舍五入），
    也可能写成 0.15747（原始值）。只认一种精度会误报"缺失"。
    """
    try:
        f = float(x)
    except (TypeError, ValueError):
        return {str(x)}
    out = {f"{f:.5f}", f"{f:.4f}", f"{f:.3f}", f"{f:.2f}", f"{f:g}", str(f)}
    if abs(f - round(f)) < 1e-9:
        out.add(str(int(round(f))))
    return out


def contains_value(text, value):
    """判断文档里是否出现了该数值

    不能用简单的 `f"{value:.3f}" in text` —— 那会误命中：
    例如 value=0.88715 的三位写法 "0.887" 会匹配到文档里的 "0.8872"，
    而那是另一个不同的数值。所以这里改成**解析文档中所有数字再比较**，
    允许文档写法与真值之间有不超过半个末位的舍入差。
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value) in text
    if abs(f - round(f)) < 1e-9 and str(int(round(f))) in text:
        return True
    for tok in re.findall(r"\d+\.\d+", text):
        try:
            dv = float(tok)
        except ValueError:
            continue
        # 按文档里写了几位小数来定容差：写 3 位就允许 ±0.0005
        dec = len(tok.split(".")[1])
        tol = 0.5 * (10 ** -dec) + 1e-12
        if abs(dv - f) <= tol:
            return True
    return False


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="校验三份文档的关键数字与 results/metrics/*.json 产物是否一致")
    ap.add_argument("--docs", default=",".join(DOCS),
                    help="要校验的文档，逗号分隔")
    args = ap.parse_args()
    docs = [s.strip() for s in args.docs.split(",") if s.strip()]

    m = utils.ROOT / "results" / "metrics"
    facts = {}

    for name in ("yolov3", "ppyoloe_s"):
        d = jload(m / f"{name}_train_summary.json")
        if d:
            facts[f"{name} epoch"] = d["epochs"]
            facts[f"{name} 最佳验证 mAP"] = d["best_val_map50"]

    v = jload(m / "verifier_train_summary.json")
    if v:
        facts["验证器图块准确率"] = v["best_val_acc"]
    r = jload(m / "refiner_train_summary.json")
    if r:
        facts["精修器验证准确率"] = r["best_val_acc"]
        facts["精修后平均 IoU"] = r["final_mean_iou_after_refine"]

    for tag, label in (("val_fuse_best", "融合-验证集 mAP"),
                       ("test_fuse_best", "融合-测试集 mAP"),
                       ("val_verifier", "仅验证器-验证集 mAP"),
                       ("val_refiner", "仅精修器-验证集 mAP"),
                       ("yolov3_eval2stage_window_test", "初版-测试集 mAP"),
                       ("yolov3_eval2stage_window_val", "初版-验证集 mAP")):
        d = jload(m / f"{tag}.json")
        if d:
            facts[label] = d["map50"]

    d = jload(m / "test_fuse_best.json")
    if d:
        for c, val in d["per_class"].items():
            facts[f"测试集 {c} AP"] = val["ap50"]

    print("=" * 74)
    print("成员 C · 文档一致性校验")
    print("=" * 74)
    print(f"\n从产物读到的真值（{len(facts)} 项）：")
    for k, val in facts.items():
        v = val if not isinstance(val, float) else f"{val:.4f}"
        print(f"  {k:22s} = {v}")

    print("\n各文档是否包含上述数字（接受 2/3/4 位小数等合法写法）：")
    all_ok = True
    for doc in docs:
        p = utils.ROOT / doc
        if not p.exists():
            print(f"  {doc:40s} [文件不存在]")
            all_ok = False
            continue
        t = p.read_text(encoding="utf-8")
        miss = [f"{k}={val}" for k, val in facts.items()
                if not contains_value(t, val)]
        if miss:
            all_ok = False
            print(f"  {doc:40s} 缺 {len(miss)} 项")
            for x in miss[:8]:
                print(f"      - {x}")
        else:
            print(f"  {doc:40s} ✔ 全部命中")

    # 反向检查：初版数据现在只应用作「对比基线」出现，不能作为最终结论
    print("\n初版数据的出现次数（应仅作为对比基线出现在表格里，不作为结论）：")
    stale_keys = ["初版-测试集 mAP", "初版-验证集 mAP"]
    for doc in docs:
        p = utils.ROOT / doc
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        cnt = {}
        for k in stale_keys:
            val = facts.get(k)
            if val is None:
                continue
            n = sum(t.count(vs) for vs in variants(val))
            cnt[k] = n
        print(f"  {doc:40s} " + "  ".join(f"{k}×{c}" for k, c in cnt.items()))

    print("\n" + ("✔ 校验通过：文档与产物一致" if all_ok else "✘ 存在不一致，请同步文档"))
    print("=" * 74)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
