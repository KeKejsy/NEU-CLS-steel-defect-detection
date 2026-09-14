"""成员 B · 任务1：导出推理模型 + 测参数量与 FPS

用法：
    python src/cls/export.py --model mobilenet_v3_small
    python src/cls/export.py --model resnet50_vd --iters 50

输出：
    results/weights/cls_<模型>_infer.pdmodel / .pdiparams    飞桨推理模型（部署/Demo 用）
    results/metrics/cls_export_result.json                   参数量、模型体积、FPS（给 D 画对比图）
    results/metrics/cls_summary.csv                          汇总表（顺便刷新一次）
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import paddle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    MODEL_DESC,
    MODEL_NAMES,
    ROOT,
    count_params,
    dump_json,
    human_time,
    load_hparams,
    load_json,
    load_trained_model,
    rebuild_summary_csv,
    rel_to_root,
    set_seed,
)


def measure_fps(model, size, batch: int, iters: int, warmup: int = 5) -> float:
    """测吞吐：预热若干次后统计平均 FPS（张/秒）"""
    x = paddle.randn([batch, 3, int(size[0]), int(size[1])])
    with paddle.no_grad():
        for _ in range(warmup):
            model(x)
        start = time.time()
        for _ in range(iters):
            model(x)
        elapsed = time.time() - start
    return float(iters * batch / elapsed) if elapsed > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="导出推理模型并测速（B）")
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--batch", type=int, default=1, help="测速用的 batch 大小（下面 1 和 8 都会测一遍）")
    ap.add_argument("--iters", type=int, default=30, help="测速迭代次数")
    args = ap.parse_args()

    cfg = load_hparams(args.model)
    set_seed(cfg["seed"])
    size = tuple(cfg["size"])

    print("=" * 66)
    print(f"导出推理模型 + 测速  |  {args.model}")
    print("=" * 66)
    print(f"网络：{MODEL_DESC[args.model]}")

    model, weights = load_trained_model(args.model, args.weights)
    n_params = count_params(model)
    print(f"权重：{rel_to_root(weights)}")
    print(f"参数量：{n_params:,}（{n_params / 1e6:.2f} M）")

    # ---- 导出推理模型 ----
    infer_dir = ROOT / "results" / "weights"
    infer_dir.mkdir(parents=True, exist_ok=True)
    infer_base = infer_dir / f"cls_{args.model}_infer"
    input_spec = [paddle.static.InputSpec([None, 3, int(size[0]), int(size[1])], "float32", name="x")]
    paddle.jit.save(model, str(infer_base), input_spec=input_spec)
    model_files = sorted(infer_dir.glob(f"cls_{args.model}_infer.*"))
    size_mb = sum(f.stat().st_size for f in model_files) / 1024 / 1024
    # 飞桨 3.x 的新版 IR 会把计算图存成 .json（老版本是 .pdmodel），所以这里按实际文件报告
    program_files = [f for f in model_files if f.suffix in (".pdmodel", ".json")]
    exported_model_rel = rel_to_root(program_files[0]) if program_files else rel_to_root(infer_base) + ".pdmodel"
    print(f"\n推理模型（飞桨 3.x：计算图 + 权重）：{rel_to_root(infer_base)}.*  合计 {size_mb:.2f} MB")
    for f in model_files:
        print(f"  - {f.name}  {f.stat().st_size / 1024 / 1024:.2f} MB")

    # ---- 校验导出的模型能加载、且和训练时的模型输出一致 ----
    verify = {}
    x_check = paddle.randn([2, 3, int(size[0]), int(size[1])])
    try:
        loaded = paddle.jit.load(str(infer_base))
        loaded.eval()
        with paddle.no_grad():
            diff = float(paddle.max(paddle.abs(model(x_check) - loaded(x_check))))
        verify = {"loaded": True, "max_abs_diff": diff}
        print(f"导出模型校验：加载成功，与原模型输出最大差异 {diff:.3e}（越接近 0 越好）")
    except Exception as exc:                                   # noqa: BLE001
        verify = {"loaded": False, "error": f"{type(exc).__name__}: {exc}"}
        print(f"导出模型校验：加载失败（{verify['error']}），但权重文件和上面的 .json/.pdiparams 已经生成")

    # ---- 测速 ----
    print(f"\n测速（CPU，输入 {size[0]}×{size[1]}，各 {args.iters} 次取平均）...")
    fps1 = measure_fps(model, size, batch=1, iters=max(args.iters, 10))
    fps8 = measure_fps(model, size, batch=8, iters=max(args.iters // 2, 5))
    print(f"  batch=1 : {fps1:.2f} FPS  （单张 {1000 / fps1:.1f} 毫秒）")
    print(f"  batch=8 : {fps8:.2f} FPS")

    # ---- 写结果 ----
    result_path = ROOT / "results" / "metrics" / "cls_export_result.json"
    data = load_json(result_path, default={})
    data.setdefault("task", "classification")
    data.setdefault("owner", "B")
    data["device"] = "CPU" if not paddle.device.is_compiled_with_cuda() else "GPU"
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data.setdefault("runs", {})
    data["runs"][args.model] = {
        "model": args.model,
        "num_params": n_params,
        "params_M": round(n_params / 1e6, 3),
        "input_size": [int(size[0]), int(size[1])],
        "exported_model": exported_model_rel,
        "exported_files": [rel_to_root(f) for f in model_files],
        "inference_check": verify,
        "exported_size_MB": round(size_mb, 3),
        "fps_batch1": round(fps1, 2),
        "fps_batch8": round(fps8, 2),
        "latency_ms_batch1": round(1000 / fps1, 2),
        "weights": rel_to_root(weights),
        "measured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    dump_json(data, result_path)
    summary = rebuild_summary_csv()

    print(f"\n结果 JSON：{rel_to_root(result_path)}")
    print(f"汇总表  ：{rel_to_root(summary)}")

    train_json = load_json(ROOT / "results" / "metrics" / f"cls_train_result_{args.model}.json")
    if train_json:
        print(f"该网络训练耗时：{train_json.get('train_time_human', '-')}"
              f"（{train_json.get('seconds_per_epoch', '-')} 秒/轮）")


if __name__ == "__main__":
    main()
