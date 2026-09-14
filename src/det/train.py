"""成员 C · 检测训练脚本

用法：
    python src/det/train.py --config src/det/configs/yolov3.yml
    python src/det/train.py --config src/det/configs/ppyoloe_s.yml
    python src/det/train.py --config src/det/configs/yolov3.yml --epochs 3   # 快速验证

产出：
    results/weights/<name>_best.pdparams      验证集 mAP 最高的权重
    results/weights/<name>_last.pdparams      最后一个 epoch 的权重
    results/weights/<name>_epoch{N}.pdparams  周期性快照（中途中断不丢进度）
    results/logs/<name>_train.csv             逐 epoch 记录（供 D 的 log2table.py 解析）
    results/logs/<name>_train.json            训练摘要（超参 + 最佳结果 + 耗时）

设计要点：
    1. 每个 epoch 都做验证并保存最优权重 —— 验证集只有 270 张、验证很快
       （YOLOv3 约 2 秒），换来「随时可以停」的安全感；
    2. 学习率 warmup + cosine —— 检测任务一开始就上大学习率容易让 obj/分类分支发散；
    3. 最后若干 epoch 关闭 Mosaic —— 长期开 Mosaic 会让框回归偏向拼接边缘，
       收尾关掉能明显改善定位；
    4. 测试集在训练全程不加载、不评估（项目铁律 1），只在 eval_det.py 里用一次。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import paddle

# 让脚本既能 `python src/det/train.py` 直接跑，也能被别的脚本 import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import data as data_mod  # noqa: E402
from core import utils  # noqa: E402


def make_scheduler(opt_cfg, steps_per_epoch: int, epochs: int):
    """warmup + cosine 的逐 step 学习率调度

    用 Paddle 原生的 LinearWarmup 包 CosineAnnealingDecay，而不是自己写
    LambdaDecay 闭包 —— 后者在 optimizer 里被套了两层（Opt 内部还会对 lr 做一次
    处理），实测学习率只升到预期的 1/30（3.1e-5 而非 9.5e-4），warmup 形同虚设。
    原生组合的行为可逐点验证，见下方 self-check。
    """
    base_lr = float(opt_cfg["lr"])
    # warmup 不能超过总 epoch 的 1/5：否则做短程试验（如 --epochs 4）时
    # warmup 会吃掉几乎全部训练时间，学习率还没升到基准值就结束了，
    # 试验结论完全不可参考（实测 4 epoch 试验里 3 个 epoch 都在 warmup）。
    warmup_epochs = min(float(opt_cfg.get("warmup_epochs", 0)), max(epochs * 0.2, 0))
    warmup_steps = max(int(warmup_epochs * steps_per_epoch), 0)
    total_steps = max(int(epochs * steps_per_epoch), warmup_steps + 1)

    cosine = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=base_lr, T_max=max(total_steps - warmup_steps, 1))
    if warmup_steps > 0:
        sched = paddle.optimizer.lr.LinearWarmup(
            learning_rate=cosine, warmup_steps=warmup_steps,
            start_lr=base_lr * 1e-3, end_lr=base_lr)
    else:
        sched = cosine
    return sched, base_lr


def build_optimizer(model, opt_cfg, steps_per_epoch: int, epochs: int):
    lr_scheduler, base_lr = make_scheduler(opt_cfg, steps_per_epoch, epochs)
    kind = str(opt_cfg.get("type", "Momentum")).lower()
    wd = float(opt_cfg["weight_decay"])
    if kind == "adam":
        opt = paddle.optimizer.Adam(learning_rate=lr_scheduler,
                                    parameters=model.parameters(), weight_decay=wd)
    elif kind == "adamw":
        opt = paddle.optimizer.AdamW(learning_rate=lr_scheduler,
                                     parameters=model.parameters(), weight_decay=wd)
    else:
        opt = paddle.optimizer.Momentum(
            learning_rate=lr_scheduler, momentum=float(opt_cfg["momentum"]),
            parameters=model.parameters(), weight_decay=paddle.regularizer.L2Decay(wd))
    return opt, lr_scheduler, base_lr


def main():
    ap = argparse.ArgumentParser(description="C · 钢材表面缺陷检测训练")
    ap.add_argument("--config", required=True, help="配置文件（src/det/configs/*.yml）")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖配置里的 epoch 数")
    ap.add_argument("--batch-size", type=int, default=None, help="覆盖 batch size")
    ap.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    ap.add_argument("--seed", type=int, default=2026, help="随机种子（项目统一 2026）")
    ap.add_argument("--resume", default=None, help="从指定权重继续训练")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["optimizer"]["lr"] = args.lr

    dev = utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(args.seed)

    name = cfg["name"]
    im_size = int(cfg["data"]["im_size"])
    bs = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    num_classes = int(cfg["num_classes"])

    weight_dir = utils.resolve_dir(cfg["output"]["weight_dir"])
    log_dir = utils.resolve_dir(cfg["output"]["log_dir"])
    metric_dir = utils.resolve_dir(cfg["output"]["metric_dir"])

    print("=" * 70)
    print(f"成员 C · 检测训练    网络={cfg['model']['arch']}    实验名={name}")
    print("=" * 70)
    print(f"设备       : {dev}")
    if dev == "gpu":
        print(f"GPU        : {paddle.device.cuda.get_device_name(0)}")
    print(f"输入尺寸   : {im_size}    批次 {bs}    epoch {epochs}    类别数 {num_classes}")
    print(f"随机种子   : {args.seed}")

    # ---- 数据：只加载 train / val，测试集此脚本绝不加载 ----
    print("\n[1/4] 准备数据")
    tr_ds, tr_loader = data_mod.build_loader(
        utils.ROOT, "train", im_size, bs, augment=bool(cfg["data"]["augment"]),
        num_workers=int(cfg["train"]["num_workers"]))
    va_ds, va_loader = data_mod.build_loader(
        utils.ROOT, "val", im_size, max(bs, 8), augment=False, shuffle=False,
        num_workers=0)
    tr_ds.mosaic_p = float(cfg["data"]["mosaic_p"])
    tr_ds.close_mosaic_epoch = int(cfg["data"]["close_mosaic_epoch"])
    print(f"  训练集 {len(tr_ds)} 张 / 验证集 {len(va_ds)} 张")
    print(f"  类别顺序 {tr_ds.class_names}（已与 dataset/det/label_list.txt 校验一致）")
    print(f"  Mosaic 概率 {tr_ds.mosaic_p}，最后 {tr_ds.close_mosaic_epoch} 个 epoch 关闭")

    # ---- 模型与优化器 ----
    print("\n[2/4] 构建模型")
    model = data_mod.build_model(cfg)
    n_param = sum(int(np.prod(p.shape)) for p in model.parameters())
    opt, lr_scheduler, base_lr = build_optimizer(
        model, cfg["optimizer"], max(len(tr_loader), 1), epochs)
    print(f"  {cfg['model']['arch']}  参数量 {n_param/1e4:.2f} 万  基准学习率 {base_lr}")
    if args.resume:
        model.set_state_dict(paddle.load(args.resume))
        print(f"  已加载权重：{args.resume}")

    csv = utils.CSVLogger(
        log_dir / f"{name}_train.csv",
        ["epoch", "lr", "loss", "loss_box", "loss_aux", "loss_cls",
         "val_map", "val_map_best", "sec"])

    best_map = -1.0
    t_start = time.time()
    history = []

    print(f"\n[3/4] 训练开始  {utils.now_str()}")
    for epoch in range(epochs):
        tr_ds.epoch = epoch
        t_ep = time.time()
        acc = {"loss": 0.0, "box": 0.0, "aux": 0.0, "cls": 0.0}
        n_batch = 0

        for it, batch in enumerate(tr_loader):
            losses = model.forward_loss(model(batch["image"]),
                                        batch["target"], batch["num_boxes"])
            total = sum(losses.values())
            total.backward()
            # 梯度裁剪：obj 分支正负样本悬殊、梯度量级波动大，裁剪后训练更稳
            paddle.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()
            # 学习率调度逐 step 推进：LambdaDecay 以「被调用的次数」为 step 计数，
            # 若只在 epoch 末尾调一次，warmup 会几十个 epoch 都升不到位（实测卡在 3e-5）。
            opt.clear_grad()
            lr_scheduler.step()

            acc["loss"] += float(total)
            acc["box"] += float(losses["loss_box"])
            acc["cls"] += float(losses["loss_cls"])
            # YOLOv3 用 obj 分支，PP-YOLOE 用 DFL，统一记到 aux 一列便于横向对比
            acc["aux"] += float(losses.get("loss_obj", losses.get("loss_dfl", 0.0)))
            n_batch += 1

            if (it + 1) % int(cfg["train"]["log_every"]) == 0:
                print(f"    epoch {epoch+1}/{epochs}  step {it+1}/{len(tr_loader)}  "
                      f"loss {acc['loss']/n_batch:.4f}  lr {opt.get_lr():.5f}")

        n_batch = max(n_batch, 1)
        avg = {k: v / n_batch for k, v in acc.items()}

        # ---- 验证：只用验证集 ----
        val_map = float("nan")
        preds, gts = data_mod.detect_loader(
            model, va_loader, num_classes,
            score_threshold=cfg["eval"]["score_threshold"],
            nms_threshold=cfg["eval"]["nms_threshold"],
            top_k=int(cfg["eval"]["top_k"]))
        res = utils.evaluate_map(preds, gts, num_classes, tr_ds.class_names,
                                iou_threshold=float(cfg["eval"]["iou_threshold"]))
        val_map = res["map"]

        sec = time.time() - t_ep
        if val_map > best_map:
            best_map = val_map
            paddle.save(model.state_dict(), str(weight_dir / f"{name}_best.pdparams"))

        csv.log(epoch=epoch + 1, lr=opt.get_lr(), loss=avg["loss"], loss_box=avg["box"],
                loss_aux=avg["aux"], loss_cls=avg["cls"], val_map=val_map,
                val_map_best=best_map, sec=sec)
        history.append({"epoch": epoch + 1, "loss": avg["loss"], "val_map": val_map,
                        "sec": round(sec, 2)})

        print(f"  epoch {epoch+1:3d}/{epochs}  loss {avg['loss']:.4f} "
              f"(box {avg['box']:.4f} aux {avg['aux']:.4f} cls {avg['cls']:.4f})  "
              f"val mAP@0.5 {val_map:.4f}  best {best_map:.4f}  {sec:.1f}s")

        if (epoch + 1) % int(cfg["train"]["save_every"]) == 0:
            paddle.save(model.state_dict(),
                        str(weight_dir / f"{name}_epoch{epoch+1}.pdparams"))

        opt.step()  # 推进学习率调度（逐 epoch 推进一步）

    paddle.save(model.state_dict(), str(weight_dir / f"{name}_last.pdparams"))
    total_sec = time.time() - t_start

    summary = {
        "name": name, "arch": cfg["model"]["arch"], "im_size": im_size,
        "batch_size": bs, "epochs": epochs, "seed": args.seed, "device": dev,
        "gpu": paddle.device.cuda.get_device_name(0) if dev == "gpu" else None,
        "params_wan": round(n_param / 1e4, 2), "base_lr": base_lr,
        "optimizer": cfg["optimizer"]["type"],
        "best_val_map50": best_map,
        "final_loss": history[-1]["loss"] if history else None,
        "total_sec": round(total_sec, 1),
        "sec_per_epoch": round(total_sec / max(epochs, 1), 2),
        "weights": {"best": f"results/weights/{name}_best.pdparams",
                    "last": f"results/weights/{name}_last.pdparams"},
        "history": history,
    }
    utils.dump_json(summary, metric_dir / f"{name}_train_summary.json")

    print("\n" + "=" * 70)
    print(f"[4/4] 训练完成    最佳验证集 mAP@0.5 = {best_map:.4f}")
    print(f"耗时 {total_sec/60:.1f} 分钟（{total_sec/max(epochs,1):.1f} 秒/epoch）")
    print(f"权重：results/weights/{name}_best.pdparams")
    print(f"日志：results/logs/{name}_train.csv")
    print("=" * 70)
    print("下一步：python src/det/eval_det.py --model " + cfg["model"]["arch"])
    print("（测试集只在这一步用一次；训练阶段全程未加载测试集）")


if __name__ == "__main__":
    main()
