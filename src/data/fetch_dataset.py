"""成员 A · 步骤0：下载数据集（可重复执行，已下载会跳过）

作用：
    从 GitHub 镜像仓库下载 NEU-DET（1800 张图 + 1800 个 XML 标注）并解压。
    该镜像把数据拆成了两份：IMAGES（1770 张）+ Validation_Images（30 张），
    合并后正好 6 类各 300 张、共 1800 张。脚本会自动合并。

    原官方数据来自东北大学宋克臣老师课题组主页（Google Drive / 百度网盘），
    国内直接下载不便，这里用社区镜像，内容与官方一致。

输出：
    dataset/raw/IMAGES/*.jpg      （1800 张 200x200 灰度图）
    dataset/raw/ANNOTATIONS/*.xml （1800 个 VOC 格式标注）

用法：
    python src/data/fetch_dataset.py
    python src/data/fetch_dataset.py --url <其他镜像 zip 地址>
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_URL = ("https://codeload.github.com/siddhartamukherjee/"
               "NEU-DET-Steel-Surface-Defect-Detection/zip/refs/heads/master")


def download(url: str, dst: Path):
    print(f"开始下载：{url}")
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=900) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  已下载 {done / 1048576:.1f} / {total / 1048576:.1f} MB", end="")
            else:
                print(f"\r  已下载 {done / 1048576:.1f} MB", end="")
    print()
    tmp.rename(dst)
    print(f"下载完成：{dst}  ({dst.stat().st_size / 1048576:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description="下载 NEU-DET 数据集")
    ap.add_argument("--url", default=DEFAULT_URL, help="数据集 zip 地址")
    ap.add_argument("--zip", default="dataset/raw/repo.zip", help="zip 保存位置")
    args = ap.parse_args()

    raw = ROOT / "dataset" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = ROOT / args.zip

    out_img = raw / "IMAGES"
    out_ann = raw / "ANNOTATIONS"
    if out_img.is_dir() and out_ann.is_dir() and len(list(out_img.glob("*.jpg"))) >= 1800:
        print(f"已存在 {out_img} 与 {out_ann}，跳过下载。")
        print("如需重新下载，请先删除这两个目录。")
        return

    if not zip_path.exists():
        download(args.url, zip_path)

    print("解压中（只取图片与标注目录）...")
    tmp_out = raw / "_unzip_tmp"
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if any(
            k in n for k in ("/IMAGES/", "/ANNOTATIONS/",
                             "/Validation_Images/", "/Validation_Annotations/"))]
        for n in names:
            z.extract(n, tmp_out)
    # 根目录名：取解压出的第一层目录
    subs = [p for p in tmp_out.iterdir() if p.is_dir()]
    base = subs[0] if subs else tmp_out

    out_img.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)
    n_img = n_ann = 0
    # IMAGES + Validation_Images 合并后正好 1800 张
    img_dirs = [base / "IMAGES", base / "Validation_Images"]
    ann_dirs = [base / "ANNOTATIONS", base / "Validation_Annotations"]
    for src_dir in img_dirs:
        if not src_dir.is_dir():
            print(f"  警告：压缩包里没有 {src_dir.name}")
            continue
        for p in src_dir.glob("*.jpg"):
            shutil.copy2(p, out_img / p.name)
            n_img += 1
    for src_dir in ann_dirs:
        if not src_dir.is_dir():
            print(f"  警告：压缩包里没有 {src_dir.name}")
            continue
        for p in src_dir.glob("*.xml"):
            shutil.copy2(p, out_ann / p.name)
            n_ann += 1
    try:
        shutil.rmtree(tmp_out, ignore_errors=True)
    except Exception as e:
        print(f"  （临时目录 {tmp_out.name} 未能自动删除，可手动删除：{e}）")
    print(f"完成：图片 {n_img} 张 -> {out_img}")
    print(f"      标注 {n_ann} 个 -> {out_ann}")
    print("\n下一步：python src/data/download_check.py --src dataset/raw")


if __name__ == "__main__":
    main()
