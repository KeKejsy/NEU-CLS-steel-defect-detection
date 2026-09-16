"""成员 C · C 方案（学习式重排序）第 2 步：训练重排序器

输入：`tools/rerank_cache.py` 在**训练集**上导出的候选特征缓存
输出：`results/weights/rerank_mlp.pdparams` + 标准化统计（`rerank_mlp.json`）

模型刻意做到最小：26 -> 64 -> 6 的 MLP（约 2 千参数），因为
- 候选特征只有 20 多维，特征本身已经是两个 CNN 的输出，不需要再堆容量；
- 参数量与样本量（几十万条）相比极小，几乎不会过拟合，也便于解释。

划分纪律：按**图片**切训练/验证（不是按候选切），避免同一张图的窗口同时出现在两边。
验证集只用于挑 epoch 与报告，真正的指标在 `rerank_eval.py` 里用未参与训练的 val 划分算。

用法：
    python src/det/tools/rerank_train.py \
        --cache results/logs/rerank_cache_train180_i128.npz \
        --epochs 12 --batch-size 8192
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import paddle
import paddle.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import data as data_mod  # noqa: E402
from core import utils  # noqa: E402
from tools.rerank_common import (FEATURE_NAMES, FEATURE_DIM, N_CLS,  # noqa: E402
                                 build_features, labels_for_image, load_cache)

CN = data_mod.CLASS_NAMES


class ReRankMLP(nn.Layer):
    """26 维特征 -> 6 类分数（sigmoid）的小型重排序器"""

    def __init__(self, in_dim=FEATURE_DIM, hidden=64, out_dim=N_CLS, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser(description="C · 重排序器训练")
    ap.add_argument("--cache", required=True, help="训练集特征缓存 npz")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--val-ratio", type=float, default=0.15, help="按图片切出的验证比例")
    ap.add_argument("--iou-th", type=float, default=0.5)
    ap.add_argument("--neg-subsample", type=float, default=0.35,
                    help="负样本抽样比例：正样本极稀疏，全量负样本会让训练很慢且无必要")
    ap.add_argument("--name", default="rerank_mlp")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(args.seed)

    refs, pvs, prs, stems = load_cache(args.cache)
    print("=" * 84)
    print(f"C · 重排序器训练    {len(refs)} 张训练图 | 特征 {FEATURE_DIM} 维")
    print("=" * 84)

    # GT：按 stem 对齐（缓存的图片顺序与划分文件一致）
    ds, _ = data_mod.build_loader(utils.ROOT, "train", 416, 8,
                                  augment=False, shuffle=False)
    gt_by_stem = {}
    for (p, ann_p) in ds.samples:
        gt_by_stem[p.stem] = ann_p

    def gts_of(stem):
        gp = gt_by_stem[stem]
        gt = data_mod.parse_voc_xml(gp, CN)
        boxes = (np.stack([np.array([x[1] - x[3] / 2, x[2] - x[4] / 2,
                                     x[1] + x[3] / 2, x[2] + x[4] / 2]) for x in gt])
                 if len(gt) else np.zeros((0, 4), dtype="float32"))
        labels = gt[:, 0].astype("int64") if len(gt) else np.zeros(0, dtype="int64")
        return {"boxes": boxes, "labels": labels}

    # 按图片切分训练/验证
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(refs))
    n_val = max(1, int(len(refs) * args.val_ratio))
    val_idx, tr_idx = set(order[:n_val].tolist()), order[n_val:].tolist()

    def assemble(idxs, train_mode):
        Xs, Ys = [], []
        for i in idxs:
            f = build_features(refs[i], pvs[i], prs[i])
            y = labels_for_image(refs[i], gts_of(stems[i]), args.iou_th)
            if train_mode and args.neg_subsample < 1.0:
                keep = (y.max(-1) > 0) | (rng.rand(len(f)) < args.neg_subsample)
                f, y = f[keep], y[keep]
            Xs.append(f)
            Ys.append(y)
        return np.concatenate(Xs), np.concatenate(Ys)

    t0 = time.time()
    Xtr, Ytr = assemble(tr_idx, True)
    Xva, Yva = assemble(list(val_idx), False)
    print(f"样本：训练 {Xtr.shape[0]} 条（正类占比 {Ytr.mean()*100:.4f}%）"
          f"  验证 {Xva.shape[0]} 条   组装耗时 {time.time()-t0:.0f}s")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd

    model = ReRankMLP(hidden=args.hidden)
    n_param = sum(int(p.numel()) for p in model.parameters())
    print(f"模型参数量 {n_param}（{n_param/1e4:.2f} 万）")
    opt = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())
    lossf = paddle.nn.BCEWithLogitsLoss(reduction="mean")

    xt = paddle.to_tensor(Xtr)
    yt = paddle.to_tensor(Ytr)
    xv = paddle.to_tensor(Xva)
    yv = paddle.to_tensor(Yva)
    n = xt.shape[0]
    best = {"loss": 1e9, "epoch": -1, "weights": None}

    weight_dir = utils.resolve_dir("results/weights")
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = paddle.randperm(n).numpy()
        tot = 0.0
        nb = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = xt[paddle.to_tensor(idx)]
            yb = yt[paddle.to_tensor(idx)]
            logit = model(xb)
            loss = lossf(logit, yb)
            loss.backward()
            opt.step()
            opt.clear_grad()
            tot += float(loss)
            nb += 1
        model.eval()
        with paddle.no_grad():
            vl = float(lossf(model(xv), yv))
        print(f"epoch {ep:2d}/{args.epochs}  train_bce {tot/max(nb,1):.5f}  val_bce {vl:.5f}")
        if vl < best["loss"]:
            best = {"loss": vl, "epoch": ep,
                    "weights": {k: v.numpy().copy() for k, v in model.state_dict().items()}}

    model.set_state_dict({k: paddle.to_tensor(v) for k, v in best["weights"].items()})
    paddle.save(model.state_dict(), str(weight_dir / f"{args.name}.pdparams"))

    meta = {
        "name": args.name, "feature_names": FEATURE_NAMES, "feature_dim": FEATURE_DIM,
        "hidden": args.hidden, "classes": CN, "iou_th": args.iou_th,
        "normalize": {"mean": mu.tolist(), "std": sd.tolist()},
        "train_images": len(tr_idx), "val_images": len(val_idx),
        "train_samples": int(Xtr.shape[0]), "val_samples": int(Xva.shape[0]),
        "best_epoch": best["epoch"], "best_val_bce": best["loss"],
        "params": n_param, "cache": Path(args.cache).name,
        "note": "特征来自训练集缓存；验证按图片切分。评估用 rerank_eval.py 在 val 上做。",
    }
    meta_path = utils.resolve_dir("results/metrics") / f"{args.name}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n最佳 epoch {best['epoch']}（val_bce {best['loss']:.5f}）")
    print(f"权重：results/weights/{args.name}.pdparams")
    print(f"元信息：results/metrics/{args.name}.json")
    print("=" * 84)


if __name__ == "__main__":
    main()
