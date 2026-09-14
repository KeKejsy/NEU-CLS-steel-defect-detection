"""成员 C · 训练区域验证器（两阶段检测的第二阶段）

用法：
    python src/det/train_verifier.py                      # 默认 30 epoch
    python src/det/train_verifier.py --epochs 50 --lr 0.002

产出：
    results/weights/verifier_best.pdparams    验证集准确率最高的权重
    results/logs/verifier_train.csv           逐 epoch 记录
    results/metrics/verifier_train_summary.json

任务：对候选框裁剪出的 96x96 图块判类（6 类缺陷 + 背景），共 7 类。
数据：训练集 GT 框作正样本 + 每图随机 1 个与 GT 不重叠的区域作背景负样本。

为什么需要它：见 core/verifier.py 顶部的说明 —— 单阶段检测器的置信度分支在本数据集上
学不出可靠排序（正样本仅占候选位置的 0.3%），而「图块分类」是稠密监督任务，能学得很稳。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import paddle

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import data as data_mod          # noqa: E402
from core import utils                     # noqa: E402
from core.verifier import DefectVerifier, VerifierDataset, BG_CLASS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="C · 区域验证器训练")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--in-size", type=int, default=96)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--neg-per-img", type=int, default=1, help="每图背景样本数")
    ap.add_argument("--dilate-per-gt", type=int, default=0,
                    help="每个 GT 生成几个「放大框」难负样本（0=关闭，见 VerifierDataset 说明）")
    ap.add_argument("--pos-jitter", type=float, default=0.0,
                    help="正样本抖动幅度（0=直接用 GT 框）")
    ap.add_argument("--name", default="verifier", help="权重与日志前缀，便于 A/B")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    dev = utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(args.seed)
    num_classes = len(data_mod.CLASS_NAMES) + 1

    print("=" * 70)
    print("成员 C · 区域验证器训练（两阶段检测第二阶段）")
    print("=" * 70)
    print(f"设备 {dev} | 图块尺寸 {args.in_size} | 类别数 {num_classes}"
          f"（{len(data_mod.CLASS_NAMES)} 缺陷 + 背景）")

    tr_ds = VerifierDataset(utils.ROOT, "train", args.in_size, args.neg_per_img,
                            args.margin, args.seed, pos_jitter=args.pos_jitter,
                            dilate_per_gt=args.dilate_per_gt)
    # 验证集用与训练相同的样本构成，指标才可比（但固定种子保证不随机变化）
    va_ds = VerifierDataset(utils.ROOT, "val", args.in_size, args.neg_per_img,
                            args.margin, args.seed, pos_jitter=args.pos_jitter,
                            dilate_per_gt=args.dilate_per_gt)
    tr_ld = paddle.io.DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                                 drop_last=True, return_list=True)
    va_ld = paddle.io.DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                                 return_list=True)
    print(f"训练图块 {len(tr_ds)} 个 / 验证图块 {len(va_ds)} 个")
    n_bg = sum(1 for s in tr_ds.samples if s[2] == BG_CLASS)
    print(f"  其中背景 {n_bg} 个、缺陷 {len(tr_ds)-n_bg} 个")

    model = DefectVerifier(num_classes=num_classes, in_size=args.in_size)
    n_param = sum(int(np.prod(p.shape)) for p in model.parameters())
    print(f"参数量 {n_param/1e4:.2f} 万")

    lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=args.lr,
                                                  T_max=max(args.epochs, 1))
    opt = paddle.optimizer.Momentum(learning_rate=lr, momentum=0.9,
                                    parameters=model.parameters(),
                                    weight_decay=paddle.regularizer.L2Decay(5e-4))
    lossf = paddle.nn.CrossEntropyLoss()

    weight_dir = utils.resolve_dir("results/weights")
    log_dir = utils.resolve_dir("results/logs")
    metric_dir = utils.resolve_dir("results/metrics")
    csv = utils.CSVLogger(log_dir / f"{args.name}_train.csv",
                          ["epoch", "lr", "loss", "train_acc", "val_acc", "val_defect_acc", "sec"])
    best = -1.0
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        te = time.time()
        tot = 0.0
        correct = 0
        seen = 0
        for b in tr_ld:
            x = paddle.to_tensor(b["image"])
            y = b["label"]
            logit = model(x)
            loss = lossf(logit, y)
            loss.backward()
            paddle.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            opt.clear_grad()
            tot += float(loss)
            correct += int((logit.argmax(-1).numpy() == y.numpy()).sum())
            seen += len(y)
        tr_acc = correct / max(seen, 1)

        model.eval()
        vc = vd = vseen = dseen = 0
        with paddle.no_grad():
            for b in va_ld:
                x = paddle.to_tensor(b["image"])
                y = b["label"].numpy()
                pred = model(x).argmax(-1).numpy()
                ok = pred == y
                vc += int(ok.sum())
                vseen += len(y)
                dm = y != BG_CLASS
                vd += int(ok[dm].sum())
                dseen += int(dm.sum())
        va_acc = vc / max(vseen, 1)
        va_def = vd / max(dseen, 1)

        if va_acc > best:
            best = va_acc
            paddle.save(model.state_dict(), str(weight_dir / f"{args.name}_best.pdparams"))
        lr.step()
        sec = time.time() - te
        csv.log(epoch=ep + 1, lr=lr.get_lr(), loss=tot / max(len(tr_ld), 1),
                train_acc=tr_acc, val_acc=va_acc, val_defect_acc=va_def, sec=sec)
        print(f"epoch {ep+1:3d}/{args.epochs}  loss {tot/max(len(tr_ld),1):.4f}  "
              f"train_acc {tr_acc:.4f}  val_acc {va_acc:.4f}  "
              f"val_缺陷类准确率 {va_def:.4f}  {sec:.1f}s")

    summary = {
        "epochs": args.epochs, "in_size": args.in_size, "margin": args.margin,
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "params_wan": round(n_param / 1e4, 2),
        "train_tiles": len(tr_ds), "val_tiles": len(va_ds),
        "best_val_acc": best, "total_sec": round(time.time() - t0, 1),
        "weights": f"results/weights/{args.name}_best.pdparams",
    }
    utils.dump_json(summary, metric_dir / f"{args.name}_train_summary.json")
    print("=" * 70)
    print(f"完成  最佳验证准确率 {best:.4f}")
    print(f"权重：results/weights/{args.name}_best.pdparams")
    print("=" * 70)


if __name__ == "__main__":
    main()
