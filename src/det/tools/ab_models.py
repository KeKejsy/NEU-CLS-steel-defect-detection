"""成员 C · 模型版本 A/B 对比（在分层抽样验证集上）

用法：
    python src/det/tools/ab_models.py --model verifier --per-class 20
    python src/det/tools/ab_models.py --model refiner --per-class 20

## 为什么单独写

`eval_fused.py` 固定读 `dataset/det/ImageSets/Main/val.txt`。做版本 A/B 时若直接
在验证集前 N 张上比较，会踩「划分按类别前缀排序」的坑（前 80 张只有 Cr/In 两类）。
本脚本的做法是：**临时把分层抽样列表写入一个临时划分文件，评估完立刻还原**，
既保证抽样无偏，又不改动项目数据。

## 输出

对每个版本给出：mAP@0.5、每类 AP、精确率/召回率、每图框数，
便于判断「新版本到底好在哪一类」。
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import utils  # noqa: E402
from tools.sweep_refine import stratified_sample  # noqa: E402

CN = utils.ROOT / "dataset" / "det" / "ImageSets" / "Main"


def main():
    ap = argparse.ArgumentParser(description="模型版本 A/B（分层抽样）")
    ap.add_argument("--model", default="verifier", choices=["verifier", "refiner"],
                    help="要对比的模型类型")
    ap.add_argument("--variants", default=None,
                    help="逗号分隔的 name:label；缺省按模型类型给默认值")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--score-th", type=float, default=0.6)
    ap.add_argument("--nms-iou", type=float, default=0.4)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    if args.variants is None:
        if args.model == "verifier":
            args.variants = "verifier:v1(原),verifier_v2:v2(加难负样本)"
        else:
            args.variants = "refiner:v1(匹配分布),refiner_v2:v2(放宽抖动)"

    # ---- 生成分层抽样列表并临时替换 val.txt ----
    names = [l.strip() for l in (CN / "val.txt").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    pick = stratified_sample(names, args.per_class)
    backup = CN / "val_backup_ab.txt"
    shutil.copy2(CN / "val.txt", backup)
    try:
        (CN / "val.txt").write_text("\n".join(sorted(pick)) + "\n", encoding="utf-8")
        print("=" * 84)
        print(f"C · 模型版本 A/B    {args.model}    分层抽样 {args.per_class}/类 = "
              f"{len(pick)} 张")
        print("=" * 84)

        results = {}
        for spec in args.variants.split(","):
            name, label = spec.split(":")
            wp = utils.ROOT / f"results/weights/{name}_best.pdparams"
            if not wp.exists():
                print(f"\n[跳过] {label}: 找不到 {wp.name}")
                continue
            cmd = [sys.executable, str(utils.ROOT / "src/det/eval_fused.py"),
                   "--split", "val", "--mode", "fuse",
                   "--stride", str(args.stride), "--score-th", str(args.score_th),
                   "--nms-iou", str(args.nms_iou), "--top-k", "150", "--max-det", "50",
                   "--device", args.device,
                   "--tag", f"ab_{name}"]
            if args.model == "verifier":
                cmd += ["--verifier-weights", str(wp)]
            else:
                cmd += ["--refiner-weights", str(wp)]
            print(f"\n>>> 评估 {label} ({wp.name}) ...")
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                print(f"    失败: {(r.stderr or '')[-300:]}")
                continue
            jp = utils.ROOT / f"results/metrics/ab_{name}.json"
            if not jp.exists():
                print("    未生成结果 JSON")
                continue
            d = json.loads(jp.read_text(encoding="utf-8"))
            results[label] = d
            o = d["overall"]
            print(f"    mAP@0.5 = {d['map50']:.4f}   P = {o['precision']:.4f}   "
                  f"R = {o['recall']:.4f}   框/图 {d['outputs_per_image']:.1f}")
            print("    每类 AP: " + "  ".join(
                f"{c}={d['per_class'][c]['ap50']:.3f}" for c in CN_ORDER if c in d["per_class"]))

        if len(results) >= 2:
            ks = list(results.keys())
            print("\n" + "=" * 84)
            print("裁决（同一批图、同一套滑窗与后处理）：")
            base = results[ks[0]]
            for k in ks:
                d = results[k]
                print(f"  {k:<24} mAP={d['map50']:.4f}")
            for k in ks[1:]:
                d = results[k]
                delta = d["map50"] - base["map50"]
                print(f"\n  {k} 相对 {ks[0]}: mAP {delta:+.4f}")
                for c in CN_ORDER:
                    if c in d["per_class"] and c in base["per_class"]:
                        dv = d["per_class"][c]["ap50"] - base["per_class"][c]["ap50"]
                        if abs(dv) > 0.005:
                            print(f"      {c}: {base['per_class'][c]['ap50']:.3f} -> "
                                  f"{d['per_class'][c]['ap50']:.3f}  {dv:+.3f}")
            best = max(ks, key=lambda k: results[k]["map50"])
            print(f"\n  => 更优: {best}")
        utils.dump_json({"model": args.model, "per_class_sample": args.per_class,
                         "num_images": len(pick), "results": results},
                        utils.ROOT / f"results/metrics/ab_{args.model}_summary.json")
    finally:
        # 无论成功失败都还原 val.txt，避免污染项目数据
        shutil.move(str(backup), str(CN / "val.txt"))
        print("\n已还原 dataset/det/ImageSets/Main/val.txt")
    print("=" * 84)


CN_ORDER = ["Cr", "In", "Pa", "PS", "RS", "Sc"]

if __name__ == "__main__":
    main()
