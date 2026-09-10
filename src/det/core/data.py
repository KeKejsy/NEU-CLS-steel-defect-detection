"""成员 C · 检测核心：数据集与增强

数据来源（成员 A 的交付，路径与 README 的「C 的数据接口」一致）：
    dataset/det/JPEGImages/<name>.jpg
    dataset/det/Annotations/<name>.xml      PASCAL VOC 格式
    dataset/det/ImageSets/Main/{train,val,test}.txt    每行一个文件名（无后缀）
    dataset/det/label_list.txt              6 行，一行一个类名，顺序固定

坐标约定：所有边框用「归一化 cxcywh」，取值 [0,1]。极坐标（cxcywh）在裁剪时
天然安全——cx/cy 夹到 [w/2, 1-w/2] 就等价于把框贴边，不需要像 xyxy 那样
先夹坐标再修宽高，能少一类越界 bug。

增强策略（训练）：Mosaic -> 随机缩放+padding -> 随机水平翻转
    Mosaic 提升小目标与上下文多样性，随机缩放让网络适应多尺度，
    水平翻转对钢材表面缺陷是合理的不变变换（左右镜像不改变缺陷类别）。
"""

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import paddle
from PIL import Image

# 类别名顺序与 A 的 label_list.txt 严格一致，改这里必须同步改 label_list.txt
CLASS_NAMES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
CLASS_NAMES_CN = {
    "Cr": "龟裂", "In": "夹杂", "Pa": "斑块",
    "PS": "麻点", "RS": "氧化铁皮压入", "Sc": "划痕",
}


def read_label_list(ann_dir: Path, expected=None) -> list:
    """读 label_list.txt 并校验，防止有人改了顺序导致评估结果错位"""
    p = Path(ann_dir).parent / "label_list.txt"
    names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if expected is not None and names != list(expected):
        raise ValueError(
            f"{p} 的类别顺序 {names} 与代码内约定 {list(expected)} 不一致；"
            "类别顺序错位会让评估结果整体失真，必须先对齐再训练。")
    return names


def parse_voc_xml(xml_path: Path, class_names) -> np.ndarray:
    """解析 VOC XML，返回 (N,5) 数组，每行 [cls_id, cx, cy, w, h]（归一化）"""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    img_w = float(size.findtext("width"))
    img_h = float(size.findtext("height"))

    rows = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_names:
            # 与 xml2voc.py 的严格口径一致：类名不认识就报错，不静默丢样本
            raise ValueError(f"{xml_path.name} 出现未知类别名：{name!r}")
        bb = obj.find("bndbox")
        x1 = float(bb.findtext("xmin"))
        y1 = float(bb.findtext("ymin"))
        x2 = float(bb.findtext("xmax"))
        y2 = float(bb.findtext("ymax"))
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append([
            float(class_names.index(name)),
            (x1 + x2) / 2 / img_w,
            (y1 + y2) / 2 / img_h,
            (x2 - x1) / img_w,
            (y2 - y1) / img_h,
        ])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------------------
# 几何变换（全部在归一化 cxcywh 上做）
# --------------------------------------------------------------------------
def clip_boxes_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """把归一化 cxcywh 框夹回 [0,1]，并丢掉退化的框

    中心点夹到 [w/2, 1-w/2] 等价于让框贴住边界，宽高保持不变（不产生形变）。
    """
    if boxes.shape[0] == 0:
        return boxes
    out = boxes.copy()
    w = np.clip(out[:, 3], 1e-6, 1.0)
    h = np.clip(out[:, 4], 1e-6, 1.0)
    out[:, 3] = w
    out[:, 4] = h
    out[:, 1] = np.clip(out[:, 1], w / 2, 1 - w / 2)
    out[:, 2] = np.clip(out[:, 2], h / 2, 1 - h / 2)
    # 丢弃缩放后过小的框（面积占比 < 0.02%）
    keep = (w * h) > 2e-4
    return out[keep]


def hflip(img: np.ndarray, boxes: np.ndarray):
    """水平翻转：图像左右镜像，框中心 x -> 1-x，宽高不变"""
    img = img[:, ::-1, :]
    if boxes.shape[0]:
        boxes = boxes.copy()
        boxes[:, 1] = 1.0 - boxes[:, 1]
    return img, boxes


def random_scale_pad(img: np.ndarray, boxes: np.ndarray, scale: float,
                     im_size: int = None, pad_value: int = 114):
    """等比缩放，然后统一补齐到固定边长 im_size

    **为什么必须统一到固定边长（这是本项目踩过的一个大坑）**：
    最初我让每张图随机缩放到不同尺寸，再让 collate 把同一 batch 补齐成最大尺寸。
    结果是训练时图像被缩到约 224（特征图 7x7），而验证时是 416（13x13）。
    归一化坐标虽然都是相对自身图像，但「一格 stride 代表原图的多大比例」在训练与
    验证之间不一致 —— 同一个归一化偏移量对应的网格位置不同，回归目标自相矛盾。
    症状是 box loss 卡在 0.33 降不下去、验证集定位 IoU 只有 0.06、mAP 恒为 0，
    而 loss 曲线看着一切正常，极难排查。

    所以这里改成：等比缩放到「不超过 im_size 的目标尺寸」，再 padding 到 im_size。
    训练与验证的输入尺度因此完全一致，坐标口径也一致。
    缩放纯粹是为了数据增强（让网络见到不同尺度），不再影响网格对齐。

    坐标是归一化的，(缩放 + padding) 本身是「归一化坐标不变」的变换，
    所以这里只改图像内容，boxes 原样返回。
    """
    if im_size is None:
        raise ValueError("random_scale_pad 必须指定 im_size（统一输入边长）")
    src_h, src_w = img.shape[:2]
    new_h, new_w = int(round(src_h * scale)), int(round(src_w * scale))
    # 先缩放到不超过 im_size，保持宽高比
    ratio = min(im_size / new_w, im_size / new_h)
    new_w, new_h = max(int(round(new_w * ratio)), 8), max(int(round(new_h * ratio)), 8)
    new_w, new_h = min(new_w, im_size), min(new_h, im_size)

    pil = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
    out = np.full((im_size, im_size, 3), pad_value, dtype=np.uint8)
    # 居中放置：让缺陷尽量落在图像中部，减少 padding 边缘的影响
    ox, oy = (im_size - new_w) // 2, (im_size - new_h) // 2
    out[oy:oy + new_h, ox:ox + new_w] = np.asarray(pil)

    if boxes.shape[0]:
        # 居中放置会平移归一化坐标：先换算到画布再归一化
        b = boxes.copy()
        b[:, 1] = (b[:, 1] * new_w + ox) / im_size
        b[:, 2] = (b[:, 2] * new_h + oy) / im_size
        b[:, 3] = b[:, 3] * new_w / im_size
        b[:, 4] = b[:, 4] * new_h / im_size
        boxes = clip_boxes_cxcywh(b)
    return out, boxes


def mosaic_four(images, boxes_list, out_size=None):
    """把 4 张图拼成 2x2 马赛克，返回拼好的图与框

    每张子图先等比缩放进四分之一画布并 padding（保持宽高比），
    再按归一化坐标把框映射到整体画布，最后统一裁剪。
    """
    h0, w0 = images[0].shape[:2]
    out_h = out_size or h0
    out_w = out_size or w0
    half_h, half_w = out_h // 2, out_w // 2

    canvas = np.full((out_h, out_w, 3), 114, dtype=np.uint8)
    all_boxes = []
    offsets = [(0, 0), (half_w, 0), (0, half_h), (half_w, half_h)]

    for img, bx, (ox, oy) in zip(images, boxes_list, offsets):
        sh, sw = img.shape[:2]
        scale = min(half_w / sw, half_h / sh)
        nw, nh = max(int(round(sw * scale)), 1), max(int(round(sh * scale)), 1)
        resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
        canvas[oy:oy + nh, ox:ox + nw] = resized

        if bx.shape[0]:
            b = bx.copy()
            # 归一化 -> 相对该子图的实际像素 -> 平移到画布 -> 再归一化
            b[:, 1] = (b[:, 1] * nw + ox) / out_w
            b[:, 2] = (b[:, 2] * nh + oy) / out_h
            b[:, 3] = b[:, 3] * nw / out_w
            b[:, 4] = b[:, 4] * nh / out_h
            all_boxes.append(b)

    if all_boxes:
        merged = np.concatenate(all_boxes, axis=0)
        merged = clip_boxes_cxcywh(merged)
    else:
        merged = np.zeros((0, 5), dtype=np.float32)
    return canvas, merged


# --------------------------------------------------------------------------
# 数据集
# --------------------------------------------------------------------------
class YoloDataset(paddle.io.Dataset):
    """检测数据集

    split: train / val / test
    im_size: 训练/评估时的短边目标尺寸（长边等比缩放后可更大，最终按批次 padding）
    augment: 是否开启训练增强（val/test 必须为 False，否则评估不可比）
    """

    def __init__(self, root, split="train", im_size=416, augment=None,
                 class_names=None, max_boxes=64, resize_mode="keep_ratio"):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.im_size = im_size
        self.class_names = list(class_names or CLASS_NAMES)
        self.max_boxes = max_boxes

        # resize_mode 决定如何把原图变成网络输入：
        #   keep_ratio : 随机等比缩放 + padding（训练用，保持框的宽高比不失真）
        #   square     : 直接缩放到 im_size x im_size（评估用）
        # 为什么评估必须用 square：坐标是归一化的，等比缩放+padding 虽然不改归一化坐标，
        # 但 padding 出来的黑边会进入网络，不同图的 padding 比例不同，
        # 会让同一张图在不同 batch 里得到不一致的输入，评估结果不可复现。统一缩放到固定
        # 正方形后，输入分布确定，且与训练后期的数据形态一致。
        assert resize_mode in ("keep_ratio", "square"), f"未知 resize_mode: {resize_mode}"
        self.resize_mode = resize_mode

        if augment is None:
            augment = (split == "train")
        if split != "train" and augment:
            raise ValueError("验证集/测试集不允许开启增强，否则评估结果不可复现")
        self.augment = augment

        self.img_dir = self.root / "dataset" / "det" / "JPEGImages"
        self.ann_dir = self.root / "dataset" / "det" / "Annotations"
        list_file = self.root / "dataset" / "det" / "ImageSets" / "Main" / f"{split}.txt"
        if not list_file.exists():
            raise FileNotFoundError(f"找不到划分文件 {list_file}，请先运行 src/data/split_dataset.py")

        names = [ln.strip() for ln in list_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not names:
            raise ValueError(f"{list_file} 是空的")

        self.samples = []
        for n in names:
            img_p = self.img_dir / f"{n}.jpg"
            ann_p = self.ann_dir / f"{n}.xml"
            if not img_p.exists():
                raise FileNotFoundError(f"划分里有图片但文件不存在：{img_p}")
            if not ann_p.exists():
                raise FileNotFoundError(f"图片缺少对应标注：{ann_p}")
            self.samples.append((img_p, ann_p))

        # 增强参数：mosaic 概率、缩放范围、翻转概率
        self.mosaic_p = 0.5 if self.augment else 0.0
        self.scale_range = (0.5, 1.5)
        self.hflip_p = 0.5
        # 由训练脚本按 epoch 调整（前期关 mosaic 让网络先学会单个目标）
        self.epoch = 0
        self.close_mosaic_epoch = 10 ** 9
        # 变尺寸批次需要每个 batch 内 padding 到同一尺寸，这里不需要额外状态

    def __len__(self):
        return len(self.samples)

    def _load(self, idx):
        img_p, ann_p = self.samples[idx]
        img = np.asarray(Image.open(img_p).convert("RGB"))
        boxes = parse_voc_xml(ann_p, self.class_names)
        return img, boxes

    def __getitem__(self, idx):
        img, boxes = self._load(idx)

        if self.augment and self.mosaic_p > 0 and self.epoch < self.close_mosaic_epoch:
            if random.random() < self.mosaic_p:
                idxs = [idx] + [random.randrange(len(self.samples)) for _ in range(3)]
                imgs, bxs = [], []
                for i in idxs:
                    if i == idx:
                        imgs.append(img)
                        bxs.append(boxes)
                    else:
                        im2, bx2 = self._load(i)
                        imgs.append(im2)
                        bxs.append(bx2)
                img, boxes = mosaic_four(imgs, bxs)

        if self.augment:
            scale = random.uniform(*self.scale_range)
            img, boxes = random_scale_pad(img, boxes, scale, im_size=self.im_size)
            if random.random() < self.hflip_p:
                img, boxes = hflip(img, boxes)
        elif self.resize_mode == "square":
            # 评估：直接缩放到固定正方形
            img = np.asarray(Image.fromarray(img).resize((self.im_size, self.im_size),
                                                         Image.BILINEAR))
        else:
            # 评估但保持宽高比：等比缩放后 padding 到固定边长
            img, boxes = random_scale_pad(img, boxes, 1.0, im_size=self.im_size)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW（归一化，paddle 会自动补 batch 维）

        # 框数量封顶，避免单张图（mosaic 后最多 4 张）撑爆显存
        if boxes.shape[0] > self.max_boxes:
            boxes = boxes[:self.max_boxes]

        h, w = img.shape[1], img.shape[2]
        target = np.zeros((self.max_boxes, 5), dtype=np.float32)
        target[:boxes.shape[0]] = boxes
        return {
            "image": img,
            "target": target,
            "num_boxes": np.int64(boxes.shape[0]),
            "im_shape": np.asarray([h, w], dtype=np.float32),
            "image_id": np.int64(idx),
        }


def collate_detection(batch):
    """把变尺寸图像 padding 到同批次最大尺寸

    batch_size=1 时仍是 200x200 或增强后的尺寸，不做任何强制 resize，
    这样评估阶段能完全保持原图分辨率，不引入插值误差。
    """
    max_h = max(int(s["im_shape"][0]) for s in batch)
    max_w = max(int(s["im_shape"][1]) for s in batch)

    images = []
    for s in batch:
        c, h, w = s["image"].shape
        if h == max_h and w == max_w:
            images.append(s["image"])
        else:
            padded = np.zeros((c, max_h, max_w), dtype=np.float32)
            padded[:, :h, :w] = s["image"]
            images.append(padded)

    return {
        "image": paddle.to_tensor(np.stack(images, axis=0)),
        "target": paddle.to_tensor(np.stack([s["target"] for s in batch], axis=0)),
        "num_boxes": paddle.to_tensor(np.stack([s["num_boxes"] for s in batch], axis=0)),
        "im_shape": paddle.to_tensor(np.stack([s["im_shape"] for s in batch], axis=0)),
        "image_id": paddle.to_tensor(np.stack([s["image_id"] for s in batch], axis=0)),
    }


def build_loader(root, split, im_size, batch_size, augment=None, shuffle=None,
                 num_workers=0, class_names=None, resize_mode=None):
    """构造 DataLoader。shuffle 默认仅训练集为 True

    resize_mode 缺省时按 split 自动选：训练用 keep_ratio（配合增强），
    验证/测试用 square（保证评估输入确定、可复现）。
    """
    if resize_mode is None:
        resize_mode = "keep_ratio" if split == "train" else "square"
    ds = YoloDataset(root, split=split, im_size=im_size, augment=augment,
                     class_names=class_names, resize_mode=resize_mode)
    if shuffle is None:
        shuffle = (split == "train")
    loader = paddle.io.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_detection,
        drop_last=False,
        return_list=True,
    )
    return ds, loader


# --------------------------------------------------------------------------
# 模型构建与推理（训练/评估/viz/demo 共用，避免各处各写一份）
# --------------------------------------------------------------------------
def build_model(cfg):
    """按配置里的 model.arch 建网络

    在函数内 import，避免 core 与 nets 两个包在导入期相互牵制。
    """
    from nets import NETWORKS
    m = dict(cfg["model"])
    arch = m.pop("arch")
    if arch not in NETWORKS:
        raise ValueError(f"未知网络 {arch}，可选：{list(NETWORKS)}")
    return NETWORKS[arch](**m)


@paddle.no_grad()
def detect_loader(model, loader, num_classes, score_threshold=0.01,
                  nms_threshold=0.5, top_k=100):
    """在 DataLoader 上跑推理，返回 (predictions, ground_truths)

    predictions[i] = {'boxes': (N,4) 归一化 xyxy, 'scores': (N,), 'labels': (N,)}
    ground_truths[i] = {'boxes': (M,4) 归一化 xyxy, 'labels': (M,)}

    关键：输出框的归一化坐标与输入图像的归一化坐标一致，而数据集里的 GT 也是归一化的，
    所以评估阶段**完全不需要做坐标反变换** —— 这是全程使用归一化坐标带来的直接好处。
    """
    model.eval()
    preds, gts = [], []
    for batch in loader:
        raw = model(batch["image"])
        res = model.postprocess(raw, im_shape=batch["im_shape"],
                                score_threshold=score_threshold,
                                nms_threshold=nms_threshold, top_k=top_k)
        for bi, (boxes, scores, labels) in enumerate(res):
            preds.append({
                "boxes": boxes.numpy(),
                "scores": scores.numpy(),
                "labels": labels.numpy().astype("int64"),
            })
            tgt = batch["target"][bi].numpy()
            nb = int(batch["num_boxes"][bi])
            tgt = tgt[:nb]
            gts.append({
                "boxes": tgt[:, 1:5] if nb else np.zeros((0, 4), dtype="float32"),
                "labels": tgt[:, 0].astype("int64") if nb else np.zeros(0, dtype="int64"),
            })
    model.train()
    return preds, gts
