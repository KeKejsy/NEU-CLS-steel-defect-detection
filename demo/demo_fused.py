"""成员 C · 缺陷检测融合流程 · 交互式演示程序

任选数据集里的一张图片，直接看到最终交付流程（滑窗 + 验证器 + 精修器融合）的结果：
图像上叠加真值框与预测框，旁边列出逐框分数/匹配情况与全库指标对照。

本程序放在 <项目根>/demo/ 下，与其它代码一起；双击 run_gui.bat 即可启动。

用法：
    python demo/demo_fused.py                    # 打开图形界面（默认）
    python demo/demo_fused.py --list             # 命令行列出全部 1800 张（可筛选）
    python demo/demo_fused.py --image crazing_10 # 命令行看单张（名称即可，也可给路径）
    python demo/demo_fused.py --image scratches_10 --export-dir demo/export --open
    python demo/demo_fused.py --index 33         # 按 --list 的序号选图（从 1 开始，数值序）
    python demo/demo_fused.py --selftest         # 无界面自检（构建界面+跑一张+导出 PNG）

说明：
    · 用的是最终交付权重 verifier_pre128_best + refiner_giou128_best（必须配 norm=imagenet、
      in_size=128）；参数与交付评估一致：stride 12 / score_th 0.6 / NMS 0.4 / top-k 150 / max-det 50。
    · 每张图首次计算约 4~6 秒（GPU），之后走本地缓存，切换是瞬时的。
    · 缓存默认放在系统临时目录（%TEMP%/neu_det_demo_cache），不污染仓库；
      导出默认写到 demo/export/。
"""

import argparse
import base64
import hashlib
import json
import queue
import re
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np

# 本文件位于 <项目根>/demo/ ，而公共代码在 <项目根>/src/det/ 下
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "src" / "det"))

from core import data as data_mod  # noqa: E402
from core import utils  # noqa: E402
from core.refiner import RefinerInfer, RegionRefiner  # noqa: E402
from core.verifier import DefectVerifier, VerifierInfer  # noqa: E402
from core.window_detector import make_windows  # noqa: E402
from eval_det import compute_prf, match_stats  # noqa: E402
from eval_fused import combine  # noqa: E402

ROOT = utils.ROOT
CN = data_mod.CLASS_NAMES
CN_ZH = data_mod.CLASS_NAMES_CN
PREFIX_TO_CN = {"crazing": "Cr", "inclusion": "In", "patches": "Pa",
                "pitted_surface": "PS", "rolled-in_scale": "RS", "scratches": "Sc"}
CODE_TO_PREFIX = {v: k for k, v in PREFIX_TO_CN.items()}
# 列表顺序：先按类别（Cr→In→Pa→PS→RS→Sc，即 NEU-DET 的顺序），类别内按编号的数值大小
CLASS_ORDER = {CODE_TO_PREFIX[c]: i for i, c in enumerate(CN)}
SPLITS = ("train", "val", "test")


def sort_key(name):
    """crazing_2 排在 crazing_10 前面（数值序），而不是文件名字典序"""
    m = re.match(r"^(.*?)_(\d+)$", name)
    prefix, num = (m.group(1), int(m.group(2))) if m else (name, -1)
    return (CLASS_ORDER.get(prefix, len(CLASS_ORDER)), prefix, num, name)

# 交付配置（与 results/logs/eval_test_fuse_pre128.log 一致）
DEFAULT = dict(stride=12, score_th=0.6, nms_iou=0.4, top_k=150, max_det=50,
               in_size=128, norm="imagenet", mode="fuse", device="gpu")
V_WEIGHTS = "results/weights/verifier_pre128_best.pdparams"
R_WEIGHTS = "results/weights/refiner_giou128_best.pdparams"

# 参考指标（交付结果，用于图上对照）
REFERENCE = {"test_map": 0.2176, "test_p": 0.1363, "test_r": 0.5521,
             "val_map": 0.2473, "cr_ap": 0.2186}

COLOR_GT = "#00c853"      # 真值
COLOR_TP = "#ff1744"      # 命中
COLOR_FP = "#ff9100"      # 误检
COLOR_BG = "#11151c"


# --------------------------------------------------------------------------
# 数据集索引
# --------------------------------------------------------------------------
def index_dataset():
    """扫描全部 1800 张图：名称 / 类别 / 所属划分 / GT 框数"""
    img_dir = ROOT / "dataset" / "det" / "JPEGImages"
    ann_dir = ROOT / "dataset" / "det" / "Annotations"
    split_of = {}
    for sp in SPLITS:
        f = ROOT / "dataset" / "det" / "ImageSets" / "Main" / f"{sp}.txt"
        if f.exists():
            for ln in f.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    split_of[ln.strip()] = sp
    items = []
    for p in sorted((q.stem for q in img_dir.glob("*.jpg")), key=sort_key):
        name = p
        prefix = name.rsplit("_", 1)[0]
        code = PREFIX_TO_CN.get(prefix, "?")
        ann = ann_dir / f"{name}.xml"
        nb = 0
        if ann.exists():
            nb = len(data_mod.parse_voc_xml(ann, CN))
        items.append({"name": name, "img": img_dir / f"{name}.jpg", "ann": ann, "code": code,
                      "class_cn": CN_ZH.get(code, code), "split": split_of.get(name, "?"),
                      "n_gt": nb})
    return items


def read_gt(ann_p):
    gt = data_mod.parse_voc_xml(ann_p, CN)
    if len(gt) == 0:
        return np.zeros((0, 4), dtype="float32"), np.zeros(0, dtype="int64")
    boxes = np.stack([gt[:, 1] - gt[:, 3] / 2, gt[:, 2] - gt[:, 4] / 2,
                      gt[:, 1] + gt[:, 3] / 2, gt[:, 2] + gt[:, 4] / 2], axis=1)
    return boxes.astype("float32"), gt[:, 0].astype("int64")


# --------------------------------------------------------------------------
# 推理管线（模型常驻 + 结果缓存）
# --------------------------------------------------------------------------
class Pipeline:
    def __init__(self, cfg, cache_dir=None, verbose=True):
        self.cfg = dict(cfg)
        utils.pick_device(prefer_gpu=(cfg["device"] == "gpu"))
        self.device = "gpu" if cfg["device"] == "gpu" else "cpu"
        if verbose:
            print(f"[演示] 设备 {self.device}；正在加载权重 …", flush=True)
        t0 = time.time()
        vw, rw = ROOT / V_WEIGHTS, ROOT / R_WEIGHTS
        for w in (vw, rw):
            if not w.exists():
                raise SystemExit(f"找不到权重：{w}\n（请确认已训练/拷贝最终交付权重）")
        self.refiner = RegionRefiner(num_classes=len(CN) + 1, in_size=cfg["in_size"])
        self.refiner.set_state_dict(__import__("paddle").load(str(rw)))
        self.refiner.eval()
        self.rinf = RefinerInfer(self.refiner, in_size=cfg["in_size"], margin=0.15,
                                 batch=256, norm=cfg["norm"])
        self.verifier = DefectVerifier(num_classes=len(CN) + 1, in_size=cfg["in_size"])
        self.verifier.set_state_dict(__import__("paddle").load(str(vw)))
        self.verifier.eval()
        self.vinf = VerifierInfer(self.verifier, in_size=cfg["in_size"], margin=0.2,
                                  batch=512, norm=cfg["norm"])
        self.load_sec = time.time() - t0
        tag = hashlib.md5(json.dumps({
            "cfg": {k: self.cfg[k] for k in ("stride", "score_th", "nms_iou", "top_k",
                                             "max_det", "in_size", "norm", "mode")},
            "device": self.device,
            "v": [vw.stat().st_mtime_ns, vw.stat().st_size],
            "r": [rw.stat().st_mtime_ns, rw.stat().st_size],
        }, sort_keys=True).encode()).hexdigest()[:10]
        self.cache_dir = Path(cache_dir) if cache_dir else (
            Path(tempfile.gettempdir()) / "neu_det_demo_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        if verbose:
            print(f"[演示] 权重加载完成 {self.load_sec:.1f}s；缓存目录 {self.cache_dir}", flush=True)

    def cache_path(self, name):
        return self.cache_dir / f"{name}__{self.tag}.json"

    def infer(self, item, use_cache=True):
        """对一张图跑完整融合流程，返回结果字典（含 GT / 预测 / 匹配 / 计时）"""
        cp = self.cache_path(item["name"])
        if use_cache and cp.exists():
            r = json.loads(cp.read_text(encoding="utf-8"))
            r["from_cache"] = True
            return r
        from PIL import Image
        cfg = self.cfg
        t0 = time.time()
        im = Image.open(item["img"]).convert("RGB")
        W, H = im.size
        gt_boxes, gt_labels = read_gt(item["ann"])

        wins = make_windows(W, H, stride=cfg["stride"])
        t_win = time.time()
        pr, ref = self.rinf.run_norm(im, wins)          # 精修器：类别 + 精修框
        pv = self.vinf.score(im, wins)                  # 验证器：类别
        t_infer = time.time() - t_win

        cls_p, def_p = combine(pr, pv, cfg["mode"])
        labels = cls_p.argmax(-1)
        keep = def_p >= cfg["score_th"]
        n_cand = int(keep.sum())
        boxes_out = np.zeros((0, 4), dtype="float32")
        scores_out = np.zeros(0, dtype="float32")
        labels_out = np.zeros(0, dtype="int64")
        n_kept = 0
        if keep.any():
            import paddle
            kb, ks, kl = ref[keep], def_p[keep], labels[keep]
            if len(ks) > cfg["top_k"]:
                o = np.argsort(-ks)[:cfg["top_k"]]
                kb, ks, kl = kb[o], ks[o], kl[o]
            n_kept = int(len(ks))
            tb = paddle.to_tensor(kb)
            ts = paddle.to_tensor(ks).astype("float32")
            tl = paddle.to_tensor(kl).astype("int64")
            good = (tb[:, 2] > tb[:, 0]) & (tb[:, 3] > tb[:, 1])
            gi = paddle.nonzero(good).reshape([-1])
            tb, ts, tl = tb[gi], ts[gi], tl[gi]
            if tb.shape[0] > 0:
                from core.boxes import batched_nms, clip_boxes_norm
                idx = batched_nms(tb, ts, tl, len(CN), iou_threshold=cfg["nms_iou"],
                                  top_k=min(cfg["max_det"], int(tb.shape[0])))
                boxes_out = clip_boxes_norm(tb[idx]).numpy()
                scores_out = ts[idx].numpy()
                labels_out = tl[idx].numpy()

        pred = {"boxes": boxes_out.astype("float32"), "scores": scores_out.astype("float32"),
                "labels": labels_out.astype("int64")}
        gt_d = {"boxes": gt_boxes, "labels": gt_labels}
        tp, fp, fn, conf = match_stats([pred], [gt_d], len(CN), iou_threshold=0.5)

        def iou_xyxy(a, b):
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            it = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - it
            return it / u if u > 1e-9 else 0.0

        gt_list = [{"class": CN[int(l)], "class_cn": CN_ZH.get(CN[int(l)], ""),
                    "box_norm": [round(float(v), 4) for v in b],
                    "box_px": [round(float(b[0] * W), 1), round(float(b[1] * H), 1),
                               round(float(b[2] * W), 1), round(float(b[3] * H), 1)]}
                   for b, l in zip(gt_boxes, gt_labels)]
        preds = []
        for i in range(len(scores_out)):
            b, s, l = boxes_out[i], scores_out[i], int(labels_out[i])
            bv, bi = 0.0, -1
            for gi2, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if int(gl) == l:
                    v = iou_xyxy(b, gb)
                    if v > bv:
                        bv, bi = v, gi2
            preds.append({
                "rank": i + 1, "class": CN[l], "class_cn": CN_ZH.get(CN[l], ""),
                "label": l,
                "score": round(float(s), 4),
                "box_norm": [round(float(v), 4) for v in b],
                "box_px": [round(float(b[0] * W), 1), round(float(b[1] * H), 1),
                           round(float(b[2] * W), 1), round(float(b[3] * H), 1)],
                # 最近的真值（仅作定位参考，判定另算）
                "best_gt": (f"GT{bi + 1}" if bi >= 0 else None),
                "best_gt_iou": round(float(bv), 4),
                "verdict": None, "matched_gt": None, "dup": False})

        # 判定必须与 mAP 的匹配口径完全一致：按分数降序、同类别、IoU≥0.5、**真值只能被占用一次**。
        # 否则会出现「表格里两个框都标 TP，但统计只有 1 个 TP」的自相矛盾。
        used = [False] * len(gt_boxes)
        n_tp = n_fp = 0
        for p in sorted(preds, key=lambda x: -x["score"]):
            best, bj = 0.0, -1
            for gi2, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if used[gi2] or int(gl) != p["label"]:
                    continue
                v = iou_xyxy(p["box_norm"], gb)
                if v > best:
                    best, bj = v, gi2
            if bj >= 0 and best >= 0.5:
                used[bj] = True
                p["verdict"] = "TP"
                p["matched_gt"] = f"GT{bj + 1}"
                n_tp += 1
            else:
                p["verdict"] = "FP"
                n_fp += 1
                # IoU 够但真值已被分数更高的框占走 → 标注为「重复框」，便于解释 FP 的来源
                p["dup"] = bool(p["best_gt_iou"] >= 0.5)

        # 自检：逐框判定必须与项目评估用的 match_stats 完全一致（同口径，不该有偏差）
        if (n_tp, n_fp) != (int(tp.sum()), int(fp.sum())):
            print(f"[警告] {item['name']}：逐框判定 TP/FP = {n_tp}/{n_fp} 与 "
                  f"match_stats 的 {int(tp.sum())}/{int(fp.sum())} 不一致", file=sys.stderr)

        per_gt = []
        for gi2, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            bv, bi = 0.0, -1
            for i in range(len(scores_out)):
                if int(labels_out[i]) == int(gl):
                    v = iou_xyxy(gb, boxes_out[i])
                    if v > bv:
                        bv, bi = v, i
            per_gt.append({"gt_class": CN[int(gl)], "best_pred": bi,
                           "best_iou": round(float(bv), 4), "hit@0.5": bool(bv >= 0.5)})

        r = {
            "name": item["name"], "split": item["split"], "code": item["code"],
            "class_cn": item["class_cn"], "image": str(item["img"].relative_to(ROOT)),
            "ann": str(item["ann"].relative_to(ROOT)), "size_wh": [W, H],
            "gt": gt_list, "preds": preds, "per_gt": per_gt,
            "stages": {"windows": int(len(wins)), "above_score_th": n_cand,
                       "after_topk": n_kept, "after_nms": int(len(scores_out))},
            "match": {"tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum())},
            "timing": {"infer_sec": round(t_infer, 2), "total_sec": round(time.time() - t0, 2)},
            "pipeline": {"stride": cfg["stride"], "score_th": cfg["score_th"],
                         "nms_iou": cfg["nms_iou"], "top_k": cfg["top_k"],
                         "max_det": cfg["max_det"], "in_size": cfg["in_size"],
                         "norm": cfg["norm"], "mode": cfg["mode"],
                         "verifier_weights": Path(V_WEIGHTS).name,
                         "refiner_weights": Path(R_WEIGHTS).name},
            "reference": REFERENCE, "from_cache": False,
        }
        cp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        return r


def summarize(r):
    """一句话结论（与全库指标对照）"""
    tp, fp, fn = r["match"]["tp"], r["match"]["fp"], r["match"]["fn"]
    p, rec, _ = compute_prf(tp, fp, fn)
    out = f"该图 TP {tp} / FP {fp} / FN {fn}；本图 P {p:.2f} / R {rec:.2f}"
    ref = r["reference"]
    out += (f"（全 test 集 P {ref['test_p']:.4f} / R {ref['test_r']:.4f}、"
            f"mAP@0.5 {ref['test_map']:.4f}）")
    return out


# --------------------------------------------------------------------------
# 导图 / 导 HTML
# --------------------------------------------------------------------------
def _draw_pil(r, show_gt=True, show_pred=True, show_tag=True):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(ROOT / r["image"]).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("msyh.ttc", 11)
    except Exception:
        font = ImageFont.load_default()

    def box(px, color, text, dashed=False):
        x1, y1, x2, y2 = px
        if dashed:
            step, seg = 6, 4
            for x in range(int(x1), int(x2), step + seg):
                d.line([x, y1, min(x + seg, x2), y1], fill=color, width=1)
                d.line([x, y2, min(x + seg, x2), y2], fill=color, width=1)
            for y in range(int(y1), int(y2), step + seg):
                d.line([x1, y, x1, min(y + seg, y2)], fill=color, width=1)
                d.line([x2, y, x2, min(y + seg, y2)], fill=color, width=1)
        else:
            d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if text:
            d.text((x1 + 1, max(0, y1 - 12)), text, fill=color, font=font)

    if show_gt:
        for i, g in enumerate(r["gt"]):
            box(g["box_px"], COLOR_GT, f"GT{i+1} {g['class']}" if show_tag else "")
    if show_pred:
        for p in r["preds"]:
            box(p["box_px"], COLOR_TP if p["verdict"] == "TP" else COLOR_FP,
                f"#{p['rank']} {p['class']} {p['score']:.2f} {p['verdict']}" if show_tag else "",
                dashed=(p["verdict"] == "FP"))
    return im


def export_png(r, out_path):
    """三联图：真值 / 预测 / 叠加"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    from PIL import Image as PILImage
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    panels = [(True, False, "真值 GT（%d 框）" % len(r["gt"])),
              (False, True, "融合预测（%d 框）" % len(r["preds"])),
              (True, True, "叠加（绿=真值 / 红=命中 / 橙虚线=误检）")]
    for ax, (g, p, title) in zip(axes, panels):
        ax.imshow(_draw_pil(r, g, p, True))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{r['name']}（{r['class_cn']} · {r['split']}） · 滑窗+验证器+精修器融合 · "
                 f"{summarize(r)}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def export_html(r, out_path):
    """单文件 HTML 报告（图片内联，双击即开）"""
    im = _draw_pil(r, True, True, True)
    import io
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    S = r["stages"]

    def rows_pred():
        out = []
        for p in r["preds"]:
            color = "#ff6b8a" if p["verdict"] == "TP" else "#ffb457"
            note = "重复框" if p.get("dup") else ""
            out.append(
                f"<tr><td>{p['rank']}</td><td>{p['class']}</td><td>{p['score']:.4f}</td>"
                f"<td style='color:{color};font-weight:700'>{p['verdict']}</td>"
                f"<td>{p.get('matched_gt') or '-'}</td><td>{p['best_gt'] or '-'}</td>"
                f"<td>{p['best_gt_iou']:.4f}</td><td>{note}</td>"
                f"<td class='m'>{', '.join(f'{v:.4f}' for v in p['box_norm'])}</td>"
                f"<td class='m'>{', '.join(str(round(v)) for v in p['box_px'])}</td></tr>")
        return "\n".join(out)

    def rows_gt():
        return "\n".join(
            f"<tr><td>{i+1}</td><td>{g['class']}</td>"
            f"<td class='m'>{', '.join(f'{v:.4f}' for v in g['box_norm'])}</td>"
            f"<td class='m'>{', '.join(str(round(v)) for v in g['box_px'])}</td>"
            f"<td class='m'>{r['per_gt'][i]['best_iou']:.4f}</td></tr>"
            for i, g in enumerate(r["gt"]))

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>检测结果 · {r['name']}</title><style>
body{{margin:0;background:#11151c;color:#e6edf3;font:14px/1.6 "Microsoft YaHei",sans-serif}}
header{{padding:16px 22px;border-bottom:1px solid #2c3542}}
h1{{margin:0 0 6px;font-size:18px}} h2{{font-size:14px;color:#4da3ff;margin:0 0 8px}}
.wrap{{padding:16px 22px;display:grid;grid-template-columns:minmax(300px,1fr) minmax(420px,1.2fr);gap:18px}}
@media(max-width:1000px){{.wrap{{grid-template-columns:1fr}}}}
.panel{{background:#1a2029;border:1px solid #2c3542;border-radius:10px;padding:14px;margin-bottom:14px}}
img{{width:100%;image-rendering:pixelated;border-radius:6px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th,td{{padding:5px 7px;border-bottom:1px solid #2c3542;text-align:right;white-space:nowrap}}
th{{color:#8b98a5;background:#161c24}} td.m{{font-family:Consolas,monospace;font-size:11.5px}}
.chip{{display:inline-block;background:#1a2029;border:1px solid #2c3542;border-radius:999px;
padding:3px 11px;margin:0 6px 6px 0;font-size:12px;color:#8b98a5}} .chip b{{color:#e6edf3}}
.note{{color:#c8d2dc;font-size:13px;margin:6px 0 0;padding-left:20px}}
code{{background:#0b0e13;border:1px solid #2c3542;border-radius:4px;padding:1px 5px;font-size:11.5px}}
</style></head><body>
<header><h1>缺陷检测结果 · {r['name']}</h1>
<span class="chip">类别：<b>{r['class_cn']}（{r['code']}）</b></span>
<span class="chip">划分：<b>{r['split']}</b></span>
<span class="chip">尺寸：<b>{r['size_wh'][0]}×{r['size_wh'][1]}</b></span>
<span class="chip">真值：<b>{len(r['gt'])} 框</b></span>
<span class="chip">预测：<b>{len(r['preds'])} 框</b></span>
<span class="chip">TP/FP/FN：<b>{r['match']['tp']}/{r['match']['fp']}/{r['match']['fn']}</b></span>
</header>
<div class="wrap">
 <section class="panel"><h2>叠加结果（绿=真值，红=命中，橙虚线=误检）</h2>
 <img src="data:image/jpeg;base64,{b64}" alt="result"></section>
 <section>
  <div class="panel"><h2>模型输出（融合流程，按分数降序）</h2><table>
  <tr><th>#</th><th>类别</th><th>分数</th><th>判定</th><th>匹配GT</th><th>最近GT</th>
      <th>IoU</th><th>说明</th><th>归一化框</th><th>像素框</th></tr>{rows_pred()}</table>
  <p class="note">判定口径与 mAP 一致：按分数降序、同类别、IoU≥0.5、真值只能被占用一次；
  「最近GT」是位置最近的真值（仅作参考），「重复框」表示 IoU 够但该真值已被分数更高的框占走（记为 FP）。</p></div>
  <div class="panel"><h2>真值标注 GT</h2><table>
  <tr><th>#</th><th>类别</th><th>归一化框</th><th>像素框</th><th>最佳预测 IoU</th></tr>
  {rows_gt()}</table></div>
 </section>
</div>
<div class="wrap" style="grid-template-columns:1fr"><section class="panel">
 <h2>候选漏斗 / 匹配结果 / 参数</h2>
 <p>滑窗候选 <b>{S['windows']}</b> → 融合缺陷概率 ≥ {r['pipeline']['score_th']} 的
    <b>{S['above_score_th']}</b> → 截 top-{r['pipeline']['top_k']} 的 <b>{S['after_topk']}</b>
    → NMS({r['pipeline']['nms_iou']}) 后 <b>{S['after_nms']}</b> 框</p>
 <p>{summarize(r)}</p>
 <p class="note">权重 <code>{r['pipeline']['verifier_weights']}</code> +
    <code>{r['pipeline']['refiner_weights']}</code>；输入 {r['pipeline']['in_size']}px /
    norm={r['pipeline']['norm']}；stride {r['pipeline']['stride']}；
    本图推理耗时 {r['timing']['infer_sec']}s</p>
</section></div></body></html>"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------
# 图形界面
# --------------------------------------------------------------------------
def launch_gui(pipe, items):
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk

    class App:
        def __init__(self, root):
            self.root = root
            self.items = items
            self.pipe = pipe
            self.q = queue.Queue()
            self.busy = False
            self.cur = None
            self.zoom = 2
            self.show = {"gt": True, "pred": True, "tag": True}
            self.view = []
            root.title("NEU-CLS 钢材表面缺陷检测 · 融合流程演示（成员 C）")
            root.configure(bg=COLOR_BG)
            root.geometry("1520x940")
            self._build()
            self.apply_filter()
            self.root.after(80, self.poll)

        # ---------- 界面 ----------
        def _build(self):
            top = tk.Frame(self.root, bg=COLOR_BG)
            top.pack(fill="x", padx=10, pady=(10, 6))
            tk.Label(top, text="搜索", fg="#8b98a5", bg=COLOR_BG).pack(side="left")
            self.ent = tk.Entry(top, width=18)
            self.ent.pack(side="left", padx=(4, 10))
            self.ent.bind("<Return>", lambda e: self.apply_filter())
            tk.Label(top, text="划分", fg="#8b98a5", bg=COLOR_BG).pack(side="left")
            self.cb_split = ttk.Combobox(top, width=8, state="readonly",
                                         values=["全部", "train", "val", "test"])
            self.cb_split.set("全部")
            self.cb_split.pack(side="left", padx=(4, 10))
            self.cb_split.bind("<<ComboboxSelected>>", lambda e: self.apply_filter())
            tk.Label(top, text="类别", fg="#8b98a5", bg=COLOR_BG).pack(side="left")
            self.cb_cls = ttk.Combobox(top, width=10, state="readonly",
                                       values=["全部"] + [f"{c} {CN_ZH[c]}" for c in CN])
            self.cb_cls.set("全部")
            self.cb_cls.pack(side="left", padx=(4, 14))
            self.cb_cls.bind("<<ComboboxSelected>>", lambda e: self.apply_filter())
            tk.Label(top, text="排序", fg="#8b98a5", bg=COLOR_BG).pack(side="left")
            self.cb_sort = ttk.Combobox(top, width=11, state="readonly",
                                        values=["类别+编号", "文件名", "划分", "GT框数↓"])
            self.cb_sort.set("类别+编号")
            self.cb_sort.pack(side="left", padx=(4, 14))
            self.cb_sort.bind("<<ComboboxSelected>>", lambda e: self.apply_filter())
            for key, text in (("gt", "真值"), ("pred", "预测"), ("tag", "标签")):
                v = tk.BooleanVar(value=True)
                self.__dict__["var_" + key] = v
                tk.Checkbutton(top, text=text, variable=v, fg="#e6edf3", bg=COLOR_BG,
                               selectcolor="#232c37", activebackground=COLOR_BG,
                               activeforeground="#e6edf3",
                               command=self.render).pack(side="left", padx=2)
            for z in (2, 3, 4):
                tk.Button(top, text=f"{z}×", bg="#232c37", fg="#e6edf3", relief="flat",
                          command=lambda zz=z: self.set_zoom(zz)).pack(side="left", padx=2)
            tk.Button(top, text="导出 HTML", bg="#232c37", fg="#e6edf3", relief="flat",
                      command=self.export_html_click).pack(side="right", padx=3)
            tk.Button(top, text="导出 PNG", bg="#232c37", fg="#e6edf3", relief="flat",
                      command=self.export_png_click).pack(side="right", padx=3)

            body = tk.Frame(self.root, bg=COLOR_BG)
            body.pack(fill="both", expand=True, padx=10, pady=6)

            left = tk.Frame(body, bg=COLOR_BG)
            left.pack(side="left", fill="y")
            self.cnt = tk.Label(left, text="", fg="#8b98a5", bg=COLOR_BG)
            self.cnt.pack(anchor="w")
            self.listbox = tk.Listbox(left, width=34, height=30, bg="#1a2029", fg="#e6edf3",
                                      selectbackground="#4da3ff", selectforeground="#08121f",
                                      relief="flat", activestyle="none")
            self.listbox.pack(side="left", fill="y")
            sb = tk.Scrollbar(left, command=self.listbox.yview)
            sb.pack(side="left", fill="y")
            self.listbox.config(yscrollcommand=sb.set)
            self.listbox.bind("<<ListboxSelect>>", self.on_pick)

            right = tk.Frame(body, bg=COLOR_BG)
            right.pack(side="left", fill="both", expand=True, padx=(12, 0))
            self.canvas = tk.Canvas(right, bg="#0b0e13", highlightthickness=1,
                                    highlightbackground="#2c3542")
            self.canvas.pack(anchor="nw")
            self.info = tk.Label(right, text="", fg="#4da3ff", bg=COLOR_BG,
                                 justify="left", anchor="w", font=("Microsoft YaHei", 10))
            self.info.pack(anchor="w", pady=(8, 4))
            nb = ttk.Notebook(right)
            nb.pack(fill="both", expand=True)
            self.tv_pred = self._table(nb, "模型输出",
                                       ["#", "类别", "分数", "判定", "匹配GT", "最近GT",
                                        "IoU", "说明"],
                                       widths=[36, 52, 66, 46, 62, 62, 56, 60])
            self.tv_gt = self._table(nb, "真值 GT", ["#", "类别", "像素框", "最佳预测IoU"])
            self.tv_note = tk.Text(nb, height=8, bg="#1a2029", fg="#c8d2dc", relief="flat",
                                   wrap="word", font=("Microsoft YaHei", 9))
            nb.add(self.tv_note, text="解读")
            self.status = tk.Label(self.root, text="就绪", fg="#8b98a5", bg="#161c24",
                                   anchor="w", font=("Microsoft YaHei", 9))
            self.status.pack(fill="x", side="bottom")

        def _table(self, parent, title, cols, widths=None):
            f = tk.Frame(parent, bg=COLOR_BG)
            tv = ttk.Treeview(f, columns=cols, show="headings", height=6)
            for i, c in enumerate(cols):
                tv.heading(c, text=c)
                tv.column(c, width=(widths[i] if widths else 90), anchor="center",
                          stretch=False)
            tv.pack(fill="both", expand=True)
            parent.add(f, text=title)
            return tv

        # ---------- 交互 ----------
        def apply_filter(self):
            kw = self.ent.get().strip().lower()
            sp = self.cb_split.get()
            cl = self.cb_cls.get()
            self.view = [it for it in self.items
                         if (not kw or kw in it["name"].lower() or kw in it["class_cn"]
                             or kw in it["code"].lower())
                         and (sp == "全部" or it["split"] == sp)
                         and (cl == "全部" or it["code"] == cl.split()[0])]
            order = self.cb_sort.get()
            if order == "文件名":
                self.view.sort(key=lambda it: it["name"])
            elif order == "划分":
                rank = {"train": 0, "val": 1, "test": 2}
                self.view.sort(key=lambda it: (rank.get(it["split"], 9),
                                               sort_key(it["name"])))
            elif order == "GT框数↓":
                self.view.sort(key=lambda it: (-it["n_gt"], sort_key(it["name"])))
            self.listbox.delete(0, "end")
            for it in self.view:
                self.listbox.insert("end", f"{it['name']:<18} {it['code']:<3} "
                                           f"{it['split']:<6} GT{it['n_gt']}")
            self.cnt.config(text=f"共 {len(self.view)} / {len(self.items)} 张")
            if self.view:
                self.listbox.selection_set(0)
                self.on_pick()

        def on_pick(self, event=None):
            sel = self.listbox.curselection()
            if not sel:
                return
            it = self.view[sel[0]]
            self.status.config(text=f"正在计算 {it['name']} …（首次约 4~6 秒，之后读缓存）")
            if self.busy:
                return
            self.busy = True
            threading.Thread(target=self._work, args=(it,), daemon=True).start()

        def _work(self, item):
            try:
                r = self.pipe.infer(item)
                self.q.put(("ok", r))
            except Exception as e:  # 演示程序：出错也要能看到原因
                self.q.put(("err", f"{type(e).__name__}: {e}"))

        def poll(self):
            try:
                while True:
                    kind, payload = self.q.get_nowait()
                    self.busy = False
                    if kind == "err":
                        self.status.config(text="计算失败：" + payload)
                        continue
                    self.cur = payload
                    self.render()
                    self.fill_tables()
                    t = payload["timing"]
                    cost = "读缓存" if payload.get("from_cache") else f"{t['total_sec']}s"
                    self.status.config(
                        text=f"{payload['name']} 完成：{cost} ｜ 滑窗 "
                             f"{payload['stages']['windows']} → 输出 "
                             f"{payload['stages']['after_nms']} 框 ｜ TP/FP/FN "
                             f"{payload['match']['tp']}/{payload['match']['fp']}/{payload['match']['fn']}")
            except queue.Empty:
                pass
            self.root.after(80, self.poll)

        def set_zoom(self, z):
            self.zoom = z
            self.render()

        def render(self):
            from PIL import Image
            r = self.cur
            if not r:
                return
            im = Image.open(ROOT / r["image"]).convert("RGB")
            W, H = im.size
            z = self.zoom
            self.canvas.config(width=W * z, height=H * z)
            self.canvas.delete("all")
            self.photo = ImageTk.PhotoImage(im.resize((W * z, H * z), Image.NEAREST))
            self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
            if self.var_gt.get():
                for i, g in enumerate(r["gt"]):
                    x1, y1, x2, y2 = [v * z for v in g["box_px"]]
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline=COLOR_GT, width=2)
                    if self.var_tag.get():
                        self.canvas.create_text(x1 + 2, max(2, y1 - 8), anchor="w",
                                                text=f"GT{i+1} {g['class']}",
                                                fill=COLOR_GT, font=("Microsoft YaHei", 9))
            if self.var_pred.get():
                for p in r["preds"]:
                    x1, y1, x2, y2 = [v * z for v in p["box_px"]]
                    col = COLOR_TP if p["verdict"] == "TP" else COLOR_FP
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline=col, width=2,
                                                 dash=() if p["verdict"] == "TP" else (5, 3))
                    if self.var_tag.get():
                        self.canvas.create_text(x1 + 2, max(2, y1 - 8), anchor="w",
                                                text=f"#{p['rank']} {p['class']} {p['score']:.2f} "
                                                     f"{p['verdict']}", fill=col,
                                                font=("Microsoft YaHei", 9))
            ref = r["reference"]
            self.info.config(
                text=f"{r['name']}（{r['class_cn']} · {r['split']}）　{summarize(r)}\n"
                     f"权重 {r['pipeline']['verifier_weights']} + {r['pipeline']['refiner_weights']}"
                     f"　输入 {r['pipeline']['in_size']}px/{r['pipeline']['norm']}"
                     f"　stride {r['pipeline']['stride']}　"
                     f"（全 test 集 mAP@0.5 {ref['test_map']:.4f}）")

        def fill_tables(self):
            r = self.cur
            for tv in (self.tv_pred, self.tv_gt):
                tv.delete(*tv.get_children())
            for p in r["preds"]:
                tv = self.tv_pred
                iid = tv.insert("", "end", values=(
                    p["rank"], p["class"], f"{p['score']:.4f}", p["verdict"],
                    p.get("matched_gt") or "-", p["best_gt"] or "-",
                    f"{p['best_gt_iou']:.4f}", "重复框" if p.get("dup") else ""))
                if p["verdict"] == "FP":
                    tv.item(iid, tags=("fp",))
                else:
                    tv.item(iid, tags=("tp",))
            self.tv_pred.tag_configure("fp", foreground="#ffb457")
            self.tv_pred.tag_configure("tp", foreground="#ff6b8a")
            for i, g in enumerate(r["gt"]):
                self.tv_gt.insert("", "end", values=(
                    i + 1, g["class"],
                    ", ".join(str(round(v)) for v in g["box_px"]),
                    f"{r['per_gt'][i]['best_iou']:.4f}"))
            S = r["stages"]
            txt = [f"• {summarize(r)}", "",
                   f"• 候选漏斗：滑窗 {S['windows']} → 概率≥{r['pipeline']['score_th']} 的 "
                   f"{S['above_score_th']} → 截 top-{r['pipeline']['top_k']} 的 {S['after_topk']}"
                   f" → NMS({r['pipeline']['nms_iou']}) 后 {S['after_nms']} 框",
                   f"• 耗时：本图 {r['timing']['infer_sec']}s（权重加载另计 {self.pipe.load_sec:.1f}s）"]
            miss = [i + 1 for i, g in enumerate(r["per_gt"]) if not g["hit@0.5"]]
            if miss:
                txt.append(f"• 漏检：GT{'、GT'.join(map(str, miss))}"
                           f"（最佳 IoU {min(r['per_gt'][i-1]['best_iou'] for i in miss):.4f}）")
            fps = [p for p in r["preds"] if p["verdict"] == "FP"]
            if fps:
                near = [p for p in fps if p["best_gt_iou"] >= 0.35]
                dups = [p for p in fps if p.get("dup")]
                txt.append(f"• 误检 {len(fps)} 个，"
                           + (f"其中 {len(near)} 个较接近（最高 IoU "
                              f"{max(p['best_gt_iou'] for p in fps):.4f}）" if near else
                              f"最高 IoU {max(p['best_gt_iou'] for p in fps):.4f}"))
                if dups:
                    txt.append(f"• 其中 {len(dups)} 个是「重复框」（IoU≥0.5 但真值已被分数更高的框占走）："
                               + "、".join(f"#{p['rank']}" for p in dups))
            txt.append("• 判定口径与 mAP 完全一致：按分数降序、同类别、IoU≥0.5、真值只能被占用一次"
                       "（所以表格里 TP 的个数 = 状态栏的 TP）")
            self.tv_note.delete("1.0", "end")
            self.tv_note.insert("1.0", "\n".join(txt))

        def export_html_click(self):
            if not self.cur:
                return
            out = BASE / "export" / f"{self.cur['name']}.html"
            p = export_html(self.cur, out)
            self.status.config(text=f"已导出 HTML：{p}")
            webbrowser.open(p.as_uri())

        def export_png_click(self):
            if not self.cur:
                return
            out = BASE / "export" / f"{self.cur['name']}_三联图.png"
            p = export_png(self.cur, out)
            self.status.config(text=f"已导出 PNG：{p}")

    root = tk.Tk()
    app = App(root)
    return root, app


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------
def cli_print(r, pipe=None):
    print("=" * 84)
    print(f"{r['name']}（{r['class_cn']} · {r['split']}）  {r['size_wh'][0]}×{r['size_wh'][1]}"
          f"  GT {len(r['gt'])} 框  预测 {len(r['preds'])} 框")
    print("-" * 84)
    for i, g in enumerate(r["gt"]):
        print(f"  GT{i+1} {g['class']:<3} px {g['box_px']}  最佳预测 IoU "
              f"{r['per_gt'][i]['best_iou']:.4f}  命中={r['per_gt'][i]['hit@0.5']}")
    print(f"{'#':<3}{'类别':<5}{'分数':>8}{'判定':>5}{'匹配GT':>8}{'最近GT':>8}{'IoU':>8}"
          f"{'说明':>9}   像素框")
    for p in r["preds"]:
        note = "重复框" if p.get("dup") else ""
        print(f"{p['rank']:<3}{p['class']:<5}{p['score']:>8.4f}{p['verdict']:>5}"
              f"{str(p.get('matched_gt') or '-'):>8}{str(p['best_gt'] or '-'):>8}"
              f"{p['best_gt_iou']:>8.4f}{note:>9}   {p['box_px']}")
    print("  判定口径：按分数降序、同类别、IoU≥0.5、真值只能被占用一次（与 mAP 计算一致）；")
    print("            “匹配GT”= 该框认定的真值，“最近GT”= 位置最近的真值（仅参考）；")
    print("            “重复框”= IoU 够但真值已被分数更高的框占走（记为 FP）")
    S = r["stages"]
    print("-" * 84)
    print(f"  候选漏斗：滑窗 {S['windows']} → 过阈值 {S['above_score_th']} → "
          f"top-{r['pipeline']['top_k']} {S['after_topk']} → NMS 后 {S['after_nms']}")
    print(f"  {summarize(r)}")
    print(f"  耗时 {r['timing']['total_sec']}s"
          + ("（读缓存，未重新推理）" if r.get("from_cache") else "")
          + (f"（权重加载 {pipe.load_sec:.1f}s）" if pipe else ""))
    print("=" * 84)


def main():
    ap = argparse.ArgumentParser(description="C · 融合检测流程交互式演示",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gui", action="store_true", help="打开图形界面（无参数时默认）")
    ap.add_argument("--list", action="store_true", help="列出数据集图片")
    ap.add_argument("--image", default=None, help="图片名（如 crazing_10）或相对/绝对路径")
    ap.add_argument("--index", type=int, default=None, help="按 --list 的序号选图（从 1 开始）")
    ap.add_argument("--split", default="全部", choices=["全部", "train", "val", "test"])
    ap.add_argument("--class", dest="cls", default="全部", help="按类别过滤，如 Cr / 龟裂")
    ap.add_argument("--limit", type=int, default=0, help="--list 最多打印多少行")
    ap.add_argument("--export-png", default=None, help="导出三联图 PNG 路径")
    ap.add_argument("--export-html", default=None, help="导出单文件 HTML 报告路径")
    ap.add_argument("--export-dir", default=None,
                    help="导出目录：自动命名 <名称>_三联图.png 与 <名称>.html（推荐，中文名由 Python 生成）")
    ap.add_argument("--open", action="store_true", help="导出后自动打开")
    ap.add_argument("--no-cache", action="store_true", help="不使用结果缓存")
    ap.add_argument("--cache-dir", default=None, help="缓存目录（默认 results/logs/demo_cache）")
    ap.add_argument("--selftest", action="store_true", help="无界面自检")
    for k, v in DEFAULT.items():
        ap.add_argument("--" + k.replace("_", "-"), dest=k, default=v,
                        type=(int if isinstance(v, int) else float) if k != "norm" and k != "mode"
                        and k != "device" else str)
    args = ap.parse_args()

    items = index_dataset()
    if not items:
        raise SystemExit("数据集里没找到图片，请先准备 dataset/det/JPEGImages")

    # 过滤 + 清单
    def match(it):
        if args.split != "全部" and it["split"] != args.split:
            return False
        if args.cls != "全部":
            key = args.cls.strip()
            if not (key == it["code"] or key in it["class_cn"] or key.lower() in it["code"].lower()):
                return False
        return True

    view = [it for it in items if match(it)]

    if args.list:
        print(f"数据集共 {len(items)} 张，符合筛选条件 {len(view)} 张"
              f"（顺序：类别 Cr→In→Pa→PS→RS→Sc，类别内按编号数值序）：")
        print(f"{'#':<5}{'名称':<20}{'类别':<6}{'划分':<7}GT框数")
        rows = view[:args.limit] if args.limit else view
        for i, it in enumerate(rows, 1):
            print(f"{i:<5}{it['name']:<20}{it['code']:<6}{it['split']:<7}{it['n_gt']}")
        if args.limit and len(view) > args.limit:
            print(f"...（共 {len(view)} 张，已截断到 {args.limit}）")
        return

    cfg = {k: getattr(args, k) for k in DEFAULT}
    pipe = Pipeline(cfg, cache_dir=args.cache_dir)

    if args.selftest:
        root, app = launch_gui(pipe, items)
        root.update()
        print(f"[自检] 界面构建成功；列表 {len(app.view)} 张；"
              f"canvas 尺寸 {app.canvas.winfo_reqwidth()}x{app.canvas.winfo_reqheight()}")
        if app.view:
            app.listbox.selection_clear(0, "end")
            app.listbox.selection_set(0)
            app.busy = True
            app._work(app.view[0])
            app.poll()
            root.update()
            n_rect = len([i for i in app.canvas.find_all()
                          if app.canvas.type(i) == "rectangle"])
            n_txt = len([i for i in app.canvas.find_all() if app.canvas.type(i) == "text"])
            print(f"[自检] 已渲染 {app.cur['name']}：矩形 {n_rect} 个、标签 {n_txt} 个、"
                  f"表格 {len(app.tv_pred.get_children())} 行")
            out = Path(tempfile.gettempdir()) / f"selftest_{app.cur['name']}.png"
            export_png(app.cur, out)
            print(f"[自检] 三联图已导出：{out}")
        root.destroy()
        print("[自检] 通过")
        return

    target = None
    if args.image:
        p = Path(args.image)
        if p.suffix.lower() in (".jpg", ".png", ".bmp"):
            target = next((it for it in items if it["img"] == p or it["name"] == p.stem), None)
        else:
            target = next((it for it in items if it["name"] == args.image), None)
        if target is None:
            raise SystemExit(f"没找到图片 {args.image}；用 --list 看看有哪些")
    elif args.index is not None:
        if not (1 <= args.index <= len(view)):
            raise SystemExit(f"--index 超出范围 1..{len(view)}")
        target = view[args.index - 1]

    if target is None or args.gui:
        root, app = launch_gui(pipe, items)
        if target is not None:
            app.ent.delete(0, "end")
            app.ent.insert(0, target["name"])
            app.apply_filter()
        root.mainloop()
        return

    r = pipe.infer(target, use_cache=not args.no_cache)
    cli_print(r, pipe)
    if args.export_dir:
        d = Path(args.export_dir)
        d.mkdir(parents=True, exist_ok=True)
        print("PNG :", export_png(r, d / f"{r['name']}_三联图.png"))
        p = export_html(r, d / f"{r['name']}.html")
        print("HTML:", p)
        if args.open:
            webbrowser.open(p.as_uri())
    if args.export_png:
        print("PNG :", export_png(r, args.export_png))
    if args.export_html:
        p = export_html(r, args.export_html)
        print("HTML:", p)
        if args.open:
            webbrowser.open(p.as_uri())


if __name__ == "__main__":
    main()
