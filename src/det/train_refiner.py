"""成员 C · 训练区域精修器（两阶段检测第二阶段升级版）

用法：
    python src/det/train_refiner.py                      # 默认 40 epoch
    python src/det/train_refiner.py --epochs 60 --lr 0.002

产出：
    results/weights/refiner_best.pdparams
    results/logs/refiner_train.csv
    results/metrics/refiner_train_summary.json

任务：对候选框同时做两件事 —— 判类（7 类）+ 回归框的修正量。
主干从已训练好的验证器权重迁移，训练更快也更稳。

为什么要有这个升级：见 core/refiner.py 顶部。简短版：验证器能判对「是不是缺陷」
（95.1%）和「是哪一类」（97.35%），但它没见过推理时的密集重叠窗口，
导致几乎所有窗口都判成缺陷（阈值 0.5->0.9 时每图框数只从 49.8 降到 48.6），
正确框被淹没（排名中位数 194）。精修框的位置能让正确框浮上来。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import paddle

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import data as data_mod        # noqa: E402
from core import utils                    # noqa: E402
from core.det_ops import IoULoss          # noqa: E402
from core.refiner import (RegionRefiner, RefineDataset,  # noqa: E402
                          decode_refine_batch)
from core.verifier import BG_CLASS        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="C · 区域精修器训练")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--in-size", type=int, default=96)
    ap.add_argument("--per-gt", type=int, default=6, help="每个 GT 生成几个模拟候选")
    ap.add_argument("--neg-per-img", type=int, default=3)
    ap.add_argument("--reg-weight", type=float, default=1.0, help="框回归损失权重")
    ap.add_argument("--reg-loss", default="l2", choices=["l2", "giou", "diou", "ciou"],
                    help="框回归损失：l2=对归一化偏移做平方误差（历史默认）；"
                         "giou/diou/ciou=在**解码后的框**上直接优化 IoU 系列指标。"
                         "AP@0.5 的命中门槛就是 IoU 0.5，后者更对症")
    ap.add_argument("--name", default="refiner",
                    help="权重与日志的文件名前缀，便于 A/B 对比不同版本")
    ap.add_argument("--pretrained", action="store_true",
                    help="主干用 ImageNet 预训练权重（默认关，保持历史结果可复现）")
    ap.add_argument("--norm", default="none", choices=["none", "imagenet"],
                    help="输入归一化：配 --pretrained 时用 imagenet（训练与推理必须同值）")
    ap.add_argument("--init-verifier", default="results/weights/verifier_best.pdparams",
                    help="从哪个验证器权重迁移主干（A/B 时指向新验证器）")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    dev = utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(args.seed)
    num_classes = len(data_mod.CLASS_NAMES) + 1

    print("=" * 72)
    print("成员 C · 区域精修器训练（判类 + 框精修）")
    print("=" * 72)
    print(f"设备 {dev} | 图块 {args.in_size} | 类别 {num_classes} | "
          f"每 GT {args.per_gt} 个模拟候选 | 回归权重 {args.reg_weight}")
    print(f"主干 {'ImageNet 预训练' if args.pretrained else '随机初始化'}"
          f" | 输入归一化 {args.norm} | 回归损失 {args.reg_loss}")

    tr = RefineDataset(utils.ROOT, "train", args.in_size, args.per_gt,
                       args.neg_per_img, seed=args.seed, norm=args.norm)
    va = RefineDataset(utils.ROOT, "val", args.in_size, max(args.per_gt // 2, 2),
                       args.neg_per_img, seed=args.seed, norm=args.norm)
    tr_ld = paddle.io.DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                                 drop_last=True, return_list=True)
    va_ld = paddle.io.DataLoader(va, batch_size=args.batch_size, shuffle=False,
                                 return_list=True)
    n_bg = sum(1 for s in tr.samples if s[2] == BG_CLASS)
    print(f"训练样本 {len(tr)}（背景 {n_bg} / 前景 {len(tr)-n_bg}）  验证样本 {len(va)}")

    model = RegionRefiner(num_classes=num_classes, in_size=args.in_size,
                          pretrained=args.pretrained)
    vpath = Path(args.init_verifier)
    if not vpath.is_absolute():
        vpath = utils.ROOT / vpath
    if vpath.exists():
        loaded, skipped = model.load_from_verifier(vpath)
        print(f"从验证器迁移主干（{vpath.name}）：加载 {len(loaded)} 个参数，跳过 {len(skipped)} 个")
    else:
        print(f"未找到验证器权重 {vpath}，主干保持{'预训练' if args.pretrained else '随机'}初始化")
    n_param = sum(int(np.prod(p.shape)) for p in model.parameters())
    print(f"参数量 {n_param/1e4:.2f} 万")

    lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=args.lr,
                                                  T_max=max(args.epochs, 1))
    opt = paddle.optimizer.Momentum(learning_rate=lr, momentum=0.9,
                                    parameters=model.parameters(),
                                    weight_decay=paddle.regularizer.L2Decay(5e-4))
    cls_lossf = paddle.nn.CrossEntropyLoss()
    # 框回归损失：l2 = 对归一化偏移的平方误差（历史默认）；
    # 其余三种在解码后的框上直接算 IoU 系损失（reduction='none'，再按有 GT 的样本加权）
    iou_lossf = (IoULoss(args.reg_loss, reduction="none")
                 if args.reg_loss != "l2" else None)

    weight_dir = utils.resolve_dir("results/weights")
    log_dir = utils.resolve_dir("results/logs")
    metric_dir = utils.resolve_dir("results/metrics")
    csv = utils.CSVLogger(log_dir / f"{args.name}_train.csv",
                          ["epoch", "lr", "loss_cls", "loss_reg", "train_acc",
                           "val_acc", "val_cls_acc", "mean_iou_after", "sec"])

    best = -1.0
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        te = time.time()
        s_cls = s_reg = 0.0
        correct = seen = 0
        for b in tr_ld:
            x = paddle.to_tensor(b["image"])
            y = b["cls"]
            delta = paddle.to_tensor(b["delta"])
            has_gt = paddle.to_tensor(b["has_gt"])
            cand = paddle.to_tensor(b["cand"])
            gt_box = paddle.to_tensor(b["gt"])
            logit, pred_d = model(x)
            loss_cls = cls_lossf(logit, y)
            # 只对前景样本算回归损失（背景没有可回归的目标框）
            w = has_gt.unsqueeze(-1)
            n_pos = paddle.clip(w.sum(), min=1.0)
            if iou_lossf is None:
                loss_reg = (((pred_d - delta) ** 2) * w).sum() / n_pos / 4.0
            else:
                # IoU 系损失：先把预测偏移解码成真实像素框，再和 GT 框逐对算 IoU
                pred_box = decode_refine_batch(cand, pred_d)
                per_box = iou_lossf(pred_box, gt_box)
                loss_reg = (per_box * has_gt).sum() / n_pos
            loss = loss_cls + args.reg_weight * loss_reg
            loss.backward()
            paddle.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            opt.clear_grad()
            s_cls += float(loss_cls)
            s_reg += float(loss_reg)
            correct += int((logit.argmax(-1).numpy() == y.numpy()).sum())
            seen += len(y)
        tr_acc = correct / max(seen, 1)
        lr.step()

        # ---- 验证：判类准确率 + 精修后 IoU ----
        # 精修后 IoU 是本模块最关键的指标（直接衡量框有没有被拉近真值），
        # 它需要原始候选框，而 DataLoader 的 batch 里没有，所以单独用数据集直接跑一遍。
        ious = _eval_refine_iou(model, va, args.in_size)
        model.eval()
        with paddle.no_grad():
            vc = vseen = clsc = clsseen = 0
            for b in va_ld:
                x = paddle.to_tensor(b["image"])
                y = b["cls"].numpy()
                pred = model(x)[0].argmax(-1).numpy()
                ok = pred == y
                vc += int(ok.sum())
                vseen += len(y)
                m = y != BG_CLASS
                clsc += int(ok[m].sum())
                clsseen += int(m.sum())
        va_acc = vc / max(vseen, 1)
        va_cls = clsc / max(clsseen, 1)

        if va_acc > best:
            best = va_acc
            paddle.save(model.state_dict(), str(weight_dir / f"{args.name}_best.pdparams"))
        sec = time.time() - te
        csv.log(epoch=ep + 1, lr=lr.get_lr(), loss_cls=s_cls / max(len(tr_ld), 1),
                loss_reg=s_reg / max(len(tr_ld), 1), train_acc=tr_acc, val_acc=va_acc,
                val_cls_acc=va_cls, mean_iou_after=ious, sec=sec)
        print(f"epoch {ep+1:3d}/{args.epochs}  cls {s_cls/max(len(tr_ld),1):.4f}  "
              f"reg {s_reg/max(len(tr_ld),1):.4f}  train {tr_acc:.4f}  "
              f"val {va_acc:.4f}  前景类 {va_cls:.4f}  "
              f"精修后IoU {ious:.4f}  {sec:.1f}s")

    summary = {
        "epochs": args.epochs, "in_size": args.in_size, "batch_size": args.batch_size,
        "lr": args.lr, "per_gt": args.per_gt, "neg_per_img": args.neg_per_img,
        "reg_weight": args.reg_weight, "seed": args.seed,
        "pretrained": bool(args.pretrained), "norm": args.norm,
        "reg_loss": args.reg_loss,
        "init_verifier": str(vpath.name),
        "params_wan": round(n_param / 1e4, 2),
        "train_samples": len(tr), "val_samples": len(va),
        "best_val_acc": best, "final_mean_iou_after_refine": ious,
        "total_sec": round(time.time() - t0, 1),
        "weights": f"results/weights/{args.name}_best.pdparams",
    }
    utils.dump_json(summary, metric_dir / f"{args.name}_train_summary.json")
    print("=" * 72)
    print(f"完成  最佳验证准确率 {best:.4f}  精修后 IoU {ious:.4f}")
    print(f"权重：results/weights/{args.name}_best.pdparams")
    print("=" * 72)


def _eval_refine_iou(model, ds, in_size):
    """评估「精修后候选框与最近 GT 的平均 IoU」

    这是本模块最关键的指标：它直接衡量精修有没有把框拉近真值。
    对比训练前的候选 IoU 均值即可看出收益。
    """
    from core.refiner import decode_refine
    from PIL import Image
    model.eval()
    ious = []
    with paddle.no_grad():
        for img_p, cand, cls_id, gt in ds.samples[::20][:200]:
            if gt is None:
                continue
            im = Image.open(img_p).convert("RGB")
            from core.verifier import crop_with_margin, prep_crop
            c = crop_with_margin(im, cand, ds.margin)
            if c is None:
                continue
            arr = prep_crop(c, in_size, getattr(ds, "norm", "none"))
            x = paddle.to_tensor(arr).unsqueeze(0)
            delta = model(x)[1].numpy()[0]
            ref = decode_refine(cand, delta, ds.img_size)
            ious.append(_iou(ref, gt))
    return float(np.mean(ious)) if ious else 0.0


def _iou(a, b):
    lt = [max(a[0], b[0]), max(a[1], b[1])]
    rb = [min(a[2], b[2]), min(a[3], b[3])]
    iw, ih = max(0.0, rb[0] - lt[0]), max(0.0, rb[1] - lt[1])
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-9)


if __name__ == "__main__":
    main()
