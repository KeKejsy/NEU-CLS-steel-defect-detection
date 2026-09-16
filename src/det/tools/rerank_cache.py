"""成员 C · C 方案（学习式重排序）第 1 步：导出候选特征缓存

## 为什么需要它

`tools/tune_fused.py` 的缓存只存了「融合后的 defect 分数 + 预测类别 + 精修框」，
足以调后处理参数，但**不足以训练重排序模型** —— 后者需要两个区域判别模型的
**原始概率**（验证器 pv、精修器 pr），它们是互补误差的来源，也是重排序器最主要的输入。

本脚本额外存下：
    ref (N,4)  精修后的框（归一化 xyxy）
    pv  (N,7)  验证器对每个窗口的 7 类概率
    pr  (N,7)  精修器对每个窗口的 7 类概率
    stems       每张图对应的样本名（用于和 GT 对齐）

GT 不进缓存（体积大且随划分变），训练/评估时用 `tune_fused.get_gts()` 现取。

## 用法

    # 训练用（分层抽样，控制时间）
    python src/det/tools/rerank_cache.py --split train --per-class 30 --in-size 128 \
        --norm imagenet --verifier-weights results/weights/verifier_pre128_best.pdparams \
        --refiner-weights results/weights/refiner_giou128_best.pdparams \
        --out results/logs/rerank_cache_train180_i128.npz

    # 评估用（全量 val）
    python src/det/tools/rerank_cache.py --split val --all --in-size 128 \
        --norm imagenet ... --out results/logs/rerank_cache_val270_i128.npz
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import paddle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import data as data_mod  # noqa: E402
from core import utils  # noqa: E402
from core.refiner import RefinerInfer, RegionRefiner  # noqa: E402
from core.verifier import DefectVerifier, VerifierInfer  # noqa: E402
from core.window_detector import make_windows  # noqa: E402
from tools.sweep_refine import stratified_sample  # noqa: E402

CN = data_mod.CLASS_NAMES


def main():
    ap = argparse.ArgumentParser(description="C · 重排序特征缓存导出")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--per-class", type=int, default=30, help="分层抽样每类图片数")
    ap.add_argument("--all", action="store_true", help="用整个划分（不做分层抽样）")
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--in-size", type=int, default=96)
    ap.add_argument("--norm", default="none", choices=["none", "imagenet"])
    ap.add_argument("--verifier-weights", default="results/weights/verifier_pre128_best.pdparams")
    ap.add_argument("--refiner-weights", default="results/weights/refiner_giou128_best.pdparams")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(2026)

    ds, _ = data_mod.build_loader(utils.ROOT, args.split, 416, 8,
                                  augment=False, shuffle=False)
    if not args.all:
        names = stratified_sample([p.stem for p, _ in ds.samples], args.per_class)
        keep = set(names)
        ds.samples = [s for s in ds.samples if s[0].stem in keep]
    print("=" * 84)
    print(f"C · 重排序特征缓存    {args.split} {len(ds)} 张 | 步长 {args.stride} | "
          f"输入 {args.in_size} | 归一化 {args.norm}")
    print("=" * 84)

    refiner = RegionRefiner(num_classes=len(CN) + 1, in_size=args.in_size)
    refiner.set_state_dict(paddle.load(str(utils.ROOT / args.refiner_weights)))
    refiner.eval()
    rinf = RefinerInfer(refiner, in_size=args.in_size, margin=0.15, batch=256, norm=args.norm)
    ver = DefectVerifier(num_classes=len(CN) + 1, in_size=args.in_size)
    ver.set_state_dict(paddle.load(str(utils.ROOT / args.verifier_weights)))
    ver.eval()
    vinf = VerifierInfer(ver, in_size=args.in_size, batch=512, norm=args.norm)

    refs, pvs, prs, stems = [], [], [], []
    t0 = time.time()
    for k in range(len(ds)):
        p, _ = ds.samples[k]
        im = Image.open(p).convert("RGB")
        wins = make_windows(im.size[0], im.size[1], stride=args.stride)
        pr, ref = rinf.run_norm(im, wins)
        pv = vinf.score(im, wins)
        refs.append(np.asarray(ref, dtype="float32"))
        pvs.append(np.asarray(pv, dtype="float32"))
        prs.append(np.asarray(pr, dtype="float32"))
        stems.append(p.stem)
        if (k + 1) % 20 == 0 or k + 1 == len(ds):
            done = k + 1
            el = time.time() - t0
            print(f"  {done}/{len(ds)} 张   已用 {el/60:.1f} 分钟   "
                  f"预计剩余 {el/done*(len(ds)-done)/60:.1f} 分钟", flush=True)

    out = Path(args.out)
    if not out.is_absolute():
        out = utils.ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, n=len(ds), stems=np.asarray(stems),
                        **{f"r{i}": v for i, v in enumerate(refs)},
                        **{f"v{i}": v for i, v in enumerate(pvs)},
                        **{f"p{i}": v for i, v in enumerate(prs)})
    print(f"\n已写出 {out.relative_to(utils.ROOT)}  "
          f"（{out.stat().st_size/1024/1024:.1f} MB，{len(ds)} 张，"
          f"平均 {np.mean([len(x) for x in refs]):.0f} 个窗口/图）")
    print(f"总耗时 {(time.time()-t0)/60:.1f} 分钟")
    print("=" * 84)


if __name__ == "__main__":
    main()
