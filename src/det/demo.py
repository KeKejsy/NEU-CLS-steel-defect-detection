"""成员 C · 单图推理 Demo

用法：
    # 单张图片
    python src/det/demo.py --model yolov3 --image dataset/det/JPEGImages/crazing_27.jpg

    # 整个目录（批量）
    python src/det/demo.py --model ppyoloe_s --image <某目录> --out results/figures/demo

    # 顺带测推理速度（报告里的 FPS 数据来源）
    python src/det/demo.py --model yolov3 --image <某图> --benchmark

    # 不带 --image：随机抽验证集里的图演示
    python src/det/demo.py --model yolov3

产出：
    <out>/<原文件名>_pred.jpg    画好检测框的结果图
    <out>/<原文件名>_pred.json   结构化结果（类别 / 分数 / 归一化框 / 像素框）

说明：本脚本只做推理，不参与训练与评估。默认用验证集图片演示，
不会用测试集图去挑样例（项目铁律 1）。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import paddle
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import data as data_mod  # noqa: E402
from core import utils  # noqa: E402

ARCH_TO_CONFIG = {
    "yolov3": "src/det/configs/yolov3.yml",
    "ppyoloe_s": "src/det/configs/ppyoloe_s.yml",
}

# 每类固定颜色，同一类在不同图上颜色一致，便于横向比较
CLASS_COLORS = {
    0: (55, 138, 221), 1: (29, 158, 117), 2: (216, 90, 48),
    3: (83, 74, 183), 4: (186, 117, 23), 5: (15, 110, 86),
}


def load_model(cfg, weights):
    wpath = Path(weights) if weights else utils.ROOT / f"results/weights/{cfg['name']}_best.pdparams"
    if not wpath.is_absolute():
        wpath = utils.ROOT / wpath
    if not wpath.exists():
        raise SystemExit(f"找不到权重 {wpath}，请先训练：\n"
                         f"  python src/det/train.py --config src/det/configs/{cfg['name']}.yml")
    model = data_mod.build_model(cfg)
    model.set_state_dict(paddle.load(str(wpath)))
    model.eval()
    return model, wpath


def preprocess(img: Image.Image, im_size: int):
    """原图 -> 网络输入张量（缩放到固定正方形、归一化、NCHW）"""
    arr = np.asarray(img.resize((im_size, im_size), Image.BILINEAR)).astype("float32") / 255.0
    return paddle.to_tensor(np.transpose(arr, (2, 0, 1))).unsqueeze(0)


def predict(model, img: Image.Image, im_size: int, score_threshold: float,
            nms_threshold: float, top_k: int):
    """返回 [(类别id, 类别名, 中文名, 分数, 归一化框, 像素框)]"""
    W, H = img.size
    res = model.postprocess(model(preprocess(img, im_size)), im_shape=None,
                            score_threshold=score_threshold,
                            nms_threshold=nms_threshold, top_k=top_k)[0]
    boxes, scores, labels = res[0].numpy(), res[1].numpy(), res[2].numpy()
    out = []
    for b, s, l in zip(boxes, scores, labels):
        name = data_mod.CLASS_NAMES[int(l)]
        out.append({
            "class_id": int(l),
            "class_name": name,
            "class_cn": data_mod.CLASS_NAMES_CN.get(name, ""),
            "score": round(float(s), 4),
            "box_norm": [round(float(v), 4) for v in b],
            "box_pixel": [round(float(b[0] * W), 1), round(float(b[1] * H), 1),
                          round(float(b[2] * W), 1), round(float(b[3] * H), 1)],
        })
    return out


def draw(img: Image.Image, dets):
    """画框 + 类别标签，标签底色用该类的固定颜色"""
    im = img.copy()
    dr = ImageDraw.Draw(im)
    for d in dets:
        box = d["box_pixel"]
        color = CLASS_COLORS.get(d["class_id"], (255, 0, 0))
        dr.rectangle(box, outline=color, width=2)
        text = f"{d['class_name']} {d['score']:.2f}"
        # 标签放在框上方；顶部空间不够则放到框内
        tx, ty = box[0] + 2, box[1] - 12
        if ty < 0:
            ty = box[1] + 2
        try:
            tb = dr.textbbox((tx, ty), text)
            dr.rectangle(tb, fill=color)
        except Exception:
            pass
        dr.text((tx, ty), text, fill=(255, 255, 255))
    return im


def collect_images(image_arg, cfg):
    """确定要推理的图片列表"""
    if image_arg:
        p = Path(image_arg)
        if not p.is_absolute():
            p = utils.ROOT / p
        if p.is_dir():
            files = sorted([f for f in p.iterdir()
                            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
            if not files:
                raise SystemExit(f"目录里没有图片：{p}")
            return files
        if not p.exists():
            raise SystemExit(f"找不到图片：{p}")
        return [p]

    # 未指定：随机抽验证集图（不用测试集，遵守铁律 1）
    names = [ln.strip() for ln in
             (utils.ROOT / "dataset" / "det" / "ImageSets" / "Main" / "val.txt")
             .read_text(encoding="utf-8").splitlines() if ln.strip()]
    rng = np.random.RandomState(2026)
    pick = rng.choice(len(names), size=min(4, len(names)), replace=False)
    print(f"未指定 --image，随机抽取验证集 {len(pick)} 张演示（种子 2026）")
    return [utils.ROOT / "dataset" / "det" / "JPEGImages" / f"{names[i]}.jpg" for i in pick]


def benchmark(model, cfg, n=50):
    """测单图推理耗时与 FPS（含前后处理，报告里的速度数据）"""
    im_size = int(cfg["data"]["im_size"])
    x = paddle.randn([1, 3, im_size, im_size])
    for _ in range(5):  # 预热：首次会有算子编译开销，不能计入
        model.postprocess(model(x), im_shape=None, score_threshold=0.3,
                          nms_threshold=0.5, top_k=100)
    paddle.device.synchronize()
    ts = []
    for _ in range(n):
        t = time.time()
        model.postprocess(model(x), im_shape=None, score_threshold=0.3,
                          nms_threshold=0.5, top_k=100)
        paddle.device.synchronize()
        ts.append(time.time() - t)
    ms = float(np.median(ts)) * 1000
    return ms, 1000.0 / ms


def main():
    ap = argparse.ArgumentParser(description="C · 单图推理 Demo")
    ap.add_argument("--model", required=True, choices=sorted(ARCH_TO_CONFIG))
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--image", default=None, help="图片路径或目录；缺省则抽验证集演示")
    ap.add_argument("--out", default="results/figures/demo", help="结果输出目录")
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--nms-threshold", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--no-save", action="store_true", help="只在终端打印，不写文件")
    ap.add_argument("--benchmark", action="store_true", help="额外测推理速度")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    cfg = utils.load_config(args.config or ARCH_TO_CONFIG[args.model])
    im_size = int(cfg["data"]["im_size"])
    dev = utils.pick_device(prefer_gpu=(args.device == "gpu"))
    utils.seed_everything(2026)

    print("=" * 70)
    print(f"成员 C · 单图推理 Demo    {cfg['name']}    设备={dev}")
    print("=" * 70)

    model, wpath = load_model(cfg, args.weights)
    print(f"权重       : {wpath.relative_to(utils.ROOT)}")
    print(f"输入尺寸   : {im_size}   分数阈值: {args.score_threshold}   NMS: {args.nms_threshold}")

    if args.benchmark:
        ms, fps = benchmark(model, cfg)
        print(f"\n[速度] 单图 {ms:.1f} ms  ->  {fps:.1f} FPS"
              f"（batch=1，含前向+解码+NMS；已预热 5 次，取 {50} 次中位数）")

    images = collect_images(args.image, cfg)
    out_dir = utils.ROOT / args.out
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[推理] 共 {len(images)} 张")
    all_results = []
    for p in images:
        img = Image.open(p).convert("RGB")
        t = time.time()
        dets = predict(model, img, im_size, args.score_threshold,
                       args.nms_threshold, args.top_k)
        cost = (time.time() - t) * 1000

        print(f"\n  {p.name}  ({img.size[0]}x{img.size[1]})  用时 {cost:.0f} ms")
        if not dets:
            print("    （无检出；可尝试降低 --score-threshold）")
        for d in dets[:10]:
            print(f"    {d['class_name']:<3} {d['class_cn']:<8} 分数 {d['score']:.4f}  "
                  f"框(像素) {d['box_pixel']}")
        if len(dets) > 10:
            print(f"    ... 另有 {len(dets)-10} 个框")

        if not args.no_save:
            vis = draw(img, dets)
            img_out = out_dir / f"{p.stem}_pred.jpg"
            vis.save(img_out, quality=95)
            json_out = out_dir / f"{p.stem}_pred.json"
            json_out.write_text(json.dumps({
                "image": str(p.relative_to(utils.ROOT)) if utils.ROOT in p.parents else str(p),
                "model": cfg["name"],
                "weights": str(wpath.relative_to(utils.ROOT)),
                "im_size": im_size,
                "score_threshold": args.score_threshold,
                "nms_threshold": args.nms_threshold,
                "num_detections": len(dets),
                "infer_ms": round(cost, 1),
                "detections": dets,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            all_results.append({"image": p.name, "n": len(dets),
                                "figure": str(img_out.relative_to(utils.ROOT))})

    if not args.no_save and all_results:
        print(f"\n结果图与 JSON 已写入：{out_dir.relative_to(utils.ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
