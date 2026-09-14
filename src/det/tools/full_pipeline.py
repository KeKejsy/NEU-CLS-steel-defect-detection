"""成员 C · 完整流水线（一键复现，支持断点续跑）

用法：
    python src/det/tools/full_pipeline.py                 # 从数据自检跑到结果汇总
    python src/det/tools/full_pipeline.py --skip-train    # 跳过训练（只重跑评估与出图）
    python src/det/tools/full_pipeline.py --only detect   # 只跑某一段
    python src/det/tools/full_pipeline.py --from eval     # 从某一段开始

## 为什么要写成脚本

本项目此前是「手工按顺序敲命令」，容易出现：漏跑某一步、参数与文档不一致、
中途中断后不知道跑到哪。本脚本把**最终采用的方案**固化下来：

    阶段 1  data    数据自检（A 的交付）+ 重新生成划分无关的统计
    阶段 2  detect  训练两个检测网络（YOLOv3 / PP-YOLOE-s）
    阶段 3  region  训练区域验证器 + 区域精修器
    阶段 4  eval    验证集 + 测试集评估（融合方案，最优参数）
    阶段 5  viz     可视化出图
    阶段 6  report  汇总结果表 + 文档一致性校验

**断点续跑**：每阶段完成会写一个 `.done` 标记到 results/logs/_pipeline/，
重跑时跳过已完成的阶段（用 --force 可强制重跑）。

## 关键参数（实测最优，改这里就等于改最终方案）

    滑窗：步长 12、宽高比 (0.35, 0.6, 1.0, 1.7)、尺度 (0.2, 0.3, 0.45, 0.65)
    融合：验证器 + 精修器概率的几何平均
    后处理：分数阈值 0.6、NMS IoU 0.4、先截 top_k 150、最终最多 50 框/图
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import utils  # noqa: E402

# 最终方案的评估参数（唯一权威来源，脚本与文档都以此为准）
EVAL_ARGS = ["--stride", "12", "--score-th", "0.6", "--nms-iou", "0.4",
             "--top-k", "150", "--max-det", "50"]

STAGES = ["data", "detect", "region", "eval", "viz", "report"]


class Runner:
    def __init__(self, mark_dir: Path, force=False, dry=False):
        self.mark = mark_dir
        self.mark.mkdir(parents=True, exist_ok=True)
        self.force = force
        self.dry = dry
        self.log = []

    def done_marker(self, stage):
        return self.mark / f"{stage}.done"

    def is_done(self, stage):
        return self.done_marker(stage).exists() and not self.force

    def run(self, stage, name, cmd, timeout=None):
        tag = f"[{stage}] {name}"
        print(f"\n{'=' * 78}\n{tag}\n{'=' * 78}", flush=True)
        print("  $ " + " ".join(str(c) for c in cmd), flush=True)
        if self.dry:
            print("  （--dry-run，未执行）", flush=True)
            return True
        t0 = time.time()
        r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        dt = time.time() - t0
        ok = r.returncode == 0
        # 把子进程输出落盘，便于事后查错
        out_file = self.mark / f"{stage}_{name.replace(' ', '_').replace('/', '_')}.log"
        out_file.write_text((r.stdout or "") + "\n--- STDERR ---\n" + (r.stderr or ""),
                            encoding="utf-8")
        tail = (r.stdout or "").strip().splitlines()[-6:]
        for line in tail:
            print("  | " + line, flush=True)
        print(f"  -> {'成功' if ok else '失败'}  用时 {dt:.0f}s  "
              f"（完整输出见 {out_file.name}）", flush=True)
        self.log.append({"stage": stage, "name": name, "ok": ok,
                         "sec": round(dt, 1), "log": out_file.name})
        return ok

    def complete(self, stage):
        self.done_marker(stage).write_text(
            time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def py(*args):
    """构造 python 调用命令（用当前解释器，保证环境一致）"""
    return [sys.executable] + [str(a) for a in args]


def main():
    ap = argparse.ArgumentParser(description="成员 C · 完整流水线")
    ap.add_argument("--only", default=None, choices=STAGES, help="只跑指定阶段")
    ap.add_argument("--from", dest="start", default=None, choices=STAGES,
                    help="从指定阶段开始")
    ap.add_argument("--skip-train", action="store_true", help="跳过两个检测网络的训练")
    ap.add_argument("--force", action="store_true", help="忽略 .done 标记强制重跑")
    ap.add_argument("--dry-run", dest="dry", action="store_true",
                    help="只打印将执行的命令")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    R = utils.ROOT
    det = R / "src/det"
    mark = R / "results/logs/_pipeline"
    runner = Runner(mark, force=args.force, dry=args.dry)

    stages = STAGES
    if args.only:
        stages = [args.only]
    elif args.start:
        stages = STAGES[STAGES.index(args.start):]

    print("=" * 78)
    print("成员 C · 完整流水线")
    print("=" * 78)
    print(f"项目根目录 : {R}")
    print(f"将执行阶段 : {', '.join(stages)}")
    print(f"评估参数   : {' '.join(EVAL_ARGS)}")
    print(f"断点标记   : {mark.relative_to(R)}")

    t_all = time.time()

    # ---------------- 阶段 1：数据自检 ----------------
    if "data" in stages:
        if runner.is_done("data"):
            print("\n[data] 已完成，跳过（--force 可强制重跑）")
        else:
            ok = runner.run("data", "数据交付自检", py(det.parent / "data/check_submit.py"))
            if ok:
                runner.complete("data")

    # ---------------- 阶段 2：训练两个检测网络 ----------------
    if "detect" in stages:
        if runner.is_done("detect"):
            print("\n[detect] 已完成，跳过")
        elif args.skip_train:
            print("\n[detect] --skip-train，跳过")
        else:
            ok = True
            for cfg in ("yolov3", "ppyoloe_s"):
                ok &= runner.run("detect", f"训练 {cfg}",
                                 py(det / "train.py", "--config", det / f"configs/{cfg}.yml",
                                    "--device", args.device))
            if ok:
                runner.complete("detect")

    # ---------------- 阶段 3：训练区域模型 ----------------
    if "region" in stages:
        if runner.is_done("region"):
            print("\n[region] 已完成，跳过")
        elif args.skip_train:
            print("\n[region] --skip-train，跳过")
        else:
            ok = runner.run("region", "训练区域验证器",
                            py(det / "train_verifier.py", "--device", args.device))
            ok &= runner.run("region", "训练区域精修器",
                             py(det / "train_refiner.py", "--epochs", "40",
                                "--lr", "0.003", "--device", args.device))
            if ok:
                runner.complete("region")

    # ---------------- 阶段 4：评估 ----------------
    if "eval" in stages:
        if runner.is_done("eval"):
            print("\n[eval] 已完成，跳过")
        else:
            ok = True
            # 单阶段网络评估（对比基线）
            for cfg in ("yolov3", "ppyoloe_s"):
                ok &= runner.run("eval", f"{cfg} 两阶段评估(val)",
                                 py(det / "eval_2stage.py", "--mode", "window",
                                    "--model", cfg, "--split", "val", "--device", args.device))
            # 融合方案：验证集 + 测试集
            for split in ("val", "test"):
                ok &= runner.run("eval", f"融合方案({split})",
                                 py(det / "eval_fused.py", "--split", split, "--mode", "fuse",
                                    "--tag", f"{split}_fuse_final", "--device", args.device,
                                    *EVAL_ARGS))
            if ok:
                runner.complete("eval")

    # ---------------- 阶段 5：可视化 ----------------
    if "viz" in stages:
        if runner.is_done("viz"):
            print("\n[viz] 已完成，跳过")
        else:
            ok = runner.run("viz", "单阶段可视化",
                            py(det / "viz_det.py", "--model", "yolov3", "--split", "val",
                               "--device", args.device))
            if ok:
                runner.complete("viz")

    # ---------------- 阶段 6：汇总与校验 ----------------
    if "report" in stages:
        if runner.is_done("report") and not args.force:
            print("\n[report] 已完成，跳过")
        else:
            ok = runner.run("report", "汇总结果表", py(det / "tools/summarize.py"))
            ok &= runner.run("report", "文档一致性校验", py(det / "tools/check_docs.py"))
            if ok:
                runner.complete("report")

    # ---------------- 总结 ----------------
    total = time.time() - t_all
    print("\n" + "=" * 78)
    print(f"流水线结束  总用时 {total/60:.1f} 分钟")
    n_ok = sum(1 for x in runner.log if x["ok"])
    print(f"执行 {len(runner.log)} 个子任务，成功 {n_ok}，失败 {len(runner.log)-n_ok}")
    for x in runner.log:
        print(f"  [{'OK ' if x['ok'] else 'FAIL'}] {x['stage']:<7} {x['name']:<28} {x['sec']:>7.0f}s")
    if not args.dry:
        utils.dump_json({"total_sec": round(total, 1), "tasks": runner.log},
                        R / "results/metrics/_pipeline_run.json")
    print("=" * 78)
    return 0 if all(x["ok"] for x in runner.log) else 1


if __name__ == "__main__":
    sys.exit(main())
