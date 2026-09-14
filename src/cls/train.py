"""成员 B · 任务1：钢材表面缺陷分类训练（ResNet50 / MobileNetV3-small）

用法：
    conda activate paddle_env
    python src/cls/train.py --model mobilenet_v3_small     # 先跑轻的，半小时跑完全流程
    python src/cls/train.py --model resnet50_vd           # 重的，晚上挂机跑

常用可选参数：
    --epochs 30          覆盖配置里的轮数
    --batch_size 32      覆盖 batch 大小
    --limit 64           只用前 64 张（冒烟测试，验证流程通不通）
    --no-pretrained      不用 ImageNet 预训练权重（权重下载失败时的备选方案）
    --config <路径>      指定别的 yml 配置

输出（全部写在约定位置，D 的脚本直接读）：
    results/weights/cls_<模型>_best.pdparams          验证集最好的权重
    results/weights/cls_<模型>_last.pdparams          每个 epoch 都覆盖保存一次，CPU 训练中途崩了不丢
    results/logs/cls_<模型>_<时间戳>.log              人看的日志
    results/logs/cls_<模型>_<时间戳>.csv              逐轮曲线，D 的 log2table.py 可直接读
    results/metrics/cls_train_result_<模型>.json      超参 + 每轮结果 + 最好成绩
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
    NUM_CLASSES,
    ROOT,
    SEED,
    append_csv,
    build_loader,
    build_model,
    count_params,
    dump_json,
    evaluate_metrics,
    human_time,
    load_hparams,
    rel_to_root,
    run_epoch,
    set_seed,
    weights_path,
)


class Tee:
    """同时往终端和日志文件写，日志里留着过程方便老师核验"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def build_optimizer(model, cfg: dict, steps_per_epoch: int):
    """动量 SGD + 余弦退火（可选线性 warmup）"""
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=float(cfg["lr"]), T_max=int(cfg["epochs"]), eta_min=float(cfg["lr"]) * 0.01
    )
    warmup = int(cfg.get("warmup_epochs") or 0)
    if warmup > 0:
        scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=scheduler,
            warmup_steps=warmup * max(steps_per_epoch, 1),
            start_lr=float(cfg["lr"]) * 0.1,
            end_lr=float(cfg["lr"]),
        )
    optimizer = paddle.optimizer.Momentum(
        learning_rate=scheduler,
        momentum=float(cfg["momentum"]),
        parameters=model.parameters(),
        weight_decay=float(cfg["weight_decay"]),
    )
    return optimizer, scheduler


def main() -> None:
    ap = argparse.ArgumentParser(description="NEU 钢材缺陷分类训练（B）")
    ap.add_argument("--model", required=True, choices=MODEL_NAMES, help="要训练的网络")
    ap.add_argument("--config", default=None, help="配置文件路径，默认 configs/<model>.yml")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖配置里的轮数")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 张（冒烟测试）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-pretrained", action="store_true", help="不用 ImageNet 预训练权重")
    args = ap.parse_args()

    cfg = load_hparams(args.model, args.config)
    for key in ("epochs", "batch_size", "lr"):           # 命令行优先
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.no_pretrained:
        cfg["pretrained"] = False

    set_seed(args.seed)
    cfg["seed"] = args.seed

    # ---- 日志文件 ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"cls_{args.model}_{stamp}.log"
    csv_path = log_dir / f"cls_{args.model}_{stamp}.csv"
    log_file = log_path.open("w", encoding="utf-8")
    tee = Tee(sys.stdout, log_file)
    sys.stdout = tee

    print("=" * 66)
    print(f"NEU 钢材缺陷分类训练  |  {args.model}  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 66)
    print(f"网络      : {MODEL_DESC[args.model]}")
    print(f"配置文件  : {cfg['config_path']}")
    print(f"数据      : dataset/cls/{{train,val}}.txt（测试集本轮不碰）")
    print(f"超参      : epochs={cfg['epochs']} batch={cfg['batch_size']} lr={cfg['lr']} "
          f"wd={cfg['weight_decay']} warmup={cfg['warmup_epochs']} seed={cfg['seed']}")
    print(f"预训练    : {'是' if cfg['pretrained'] else '否'}   输入尺寸: {tuple(cfg['size'])}")
    if args.limit:
        print(f"⚠️  冒烟测试模式：每个集合只用前 {args.limit} 张")
    print("-" * 66)

    # ---- 数据 ----
    train_ds, train_loader = build_loader(
        "train", cfg["size"], cfg["batch_size"], train=True, limit=args.limit, num_workers=cfg["num_workers"]
    )
    val_ds, val_loader = build_loader(
        "val", cfg["size"], cfg["batch_size"], train=False, limit=args.limit, num_workers=cfg["num_workers"]
    )
    steps_per_epoch = len(train_loader)
    print(f"训练集 {len(train_ds)} 张 / 验证集 {len(val_ds)} 张，每轮 {steps_per_epoch} 步")

    # ---- 模型 ----
    model = build_model(args.model, num_classes=int(cfg["num_classes"]), pretrained=bool(cfg["pretrained"]))
    n_params = count_params(model)
    print(f"模型参数量: {n_params:,} ({n_params / 1e6:.2f} M)")

    criterion = paddle.nn.CrossEntropyLoss(label_smoothing=float(cfg["label_smoothing"]))
    optimizer, scheduler = build_optimizer(model, cfg, steps_per_epoch)

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    train_start = time.time()

    print("-" * 66)
    print(f"{'轮次':>4}{'lr':>10}{'训练loss':>11}{'训练acc':>10}{'验证loss':>11}{'验证acc':>10}{'耗时':>10}")
    print("-" * 66)

    for epoch in range(1, int(cfg["epochs"]) + 1):
        epoch_start = time.time()
        model.train()
        losses, correct, seen = [], 0, 0

        for step, (x, y) in enumerate(train_loader, start=1):
            y = paddle.reshape(y, [-1]).astype("int64")
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            losses.append(float(loss))
            acc = paddle.metric.accuracy(logits, y.unsqueeze(1))
            correct += float(acc) * int(y.shape[0])
            seen += int(y.shape[0])

            if cfg["log_every"] and step % int(cfg["log_every"]) == 0:
                print(f"  epoch {epoch}  step {step}/{steps_per_epoch}  "
                      f"loss={np.mean(losses[-int(cfg['log_every']):]):.4f}  已用 {human_time(time.time() - epoch_start)}")

        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_acc = correct / max(seen, 1)
        val_loss, y_true, y_pred = run_epoch(model, val_loader, criterion)
        val_acc = float((y_true == y_pred).mean())
        lr_now = float(optimizer.get_lr())
        scheduler.step()
        epoch_sec = time.time() - epoch_start

        print(f"{epoch:>4}{lr_now:>10.5f}{train_loss:>11.4f}{train_acc:>10.4f}"
              f"{val_loss:>11.4f}{val_acc:>10.4f}{human_time(epoch_sec):>10}")

        history.append({
            "epoch": epoch, "lr": lr_now,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "epoch_seconds": round(epoch_sec, 2),
        })
        append_csv(
            csv_path,
            ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_seconds"],
            [epoch, f"{lr_now:.6f}", f"{train_loss:.6f}", f"{train_acc:.6f}",
             f"{val_loss:.6f}" if val_loss is not None else "", f"{val_acc:.6f}", f"{epoch_sec:.2f}"],
        )

        # 每个 epoch 都存一次最新权重（CPU 训练崩了不丢进度），验证集变好时另存 best
        paddle.save(model.state_dict(), str(weights_path(args.model, "last")))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            paddle.save(model.state_dict(), str(weights_path(args.model, "best")))

    total_sec = time.time() - train_start
    print("-" * 66)
    print(f"训练结束：共 {len(history)} 轮，用时 {human_time(total_sec)}")
    print(f"验证集最好成绩：acc = {best_val_acc:.4f}（第 {best_epoch} 轮）")

    # ---- 用最好的权重在验证集上算一遍完整指标（含每类 P/R/F1） ----
    model.set_state_dict(paddle.load(str(weights_path(args.model, "best"))))
    _, y_true, y_pred = run_epoch(model, val_loader, criterion)
    val_metrics = evaluate_metrics(y_true, y_pred)

    result = {
        "model": args.model,
        "task": "classification",
        "owner": "B",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_params": n_params,
        "params_M": round(n_params / 1e6, 3),
        "hyperparams": {
            "epochs": int(cfg["epochs"]),
            "batch_size": int(cfg["batch_size"]),
            "lr": float(cfg["lr"]),
            "momentum": float(cfg["momentum"]),
            "weight_decay": float(cfg["weight_decay"]),
            "label_smoothing": float(cfg["label_smoothing"]),
            "lr_scheduler": cfg["lr_scheduler"],
            "warmup_epochs": int(cfg["warmup_epochs"]),
            "seed": int(cfg["seed"]),
            "size": list(cfg["size"]),
            "pretrained": bool(cfg["pretrained"]),
            "augment": "随机裁剪(pad=16)+随机水平/垂直翻转+亮度抖动±10%",
            "optimizer": "Momentum SGD",
        },
        "train_dataset": {"split": "train", "num_images": len(train_ds), "steps_per_epoch": steps_per_epoch},
        "val_dataset": {"split": "val", "num_images": len(val_ds)},
        "best_val_acc": round(best_val_acc, 6),
        "best_epoch": best_epoch,
        "train_seconds": round(total_sec, 1),
        "train_time_human": human_time(total_sec),
        "seconds_per_epoch": round(total_sec / max(len(history), 1), 2),
        "val_metrics": val_metrics,
        "history": history,
        "artifacts": {
            "best_weights": rel_to_root(weights_path(args.model, "best")),
            "last_weights": rel_to_root(weights_path(args.model, "last")),
            "log": rel_to_root(log_path),
            "csv": rel_to_root(csv_path),
        },
    }
    out_path = ROOT / "results" / "metrics" / f"cls_train_result_{args.model}.json"
    dump_json(result, out_path)

    print("\n验证集完整指标：")
    from _common import print_metrics_table

    print_metrics_table(val_metrics, f"[{args.model}] 验证集")
    print(f"\n权重：{rel_to_root(weights_path(args.model, 'best'))}")
    print(f"日志：{rel_to_root(log_path)}")
    print(f"结果：{rel_to_root(out_path)}")
    print("\n下一步：python src/cls/eval_cls.py --model " + args.model + "   （在测试集上正式评估一次）")

    sys.stdout = tee.streams[0]
    log_file.close()


if __name__ == "__main__":
    main()
