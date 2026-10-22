#!/usr/bin/env python3
"""VisDrone2019-DET helper: download, convert to YOLO format, write the dataset YAML, print stats.

    python VisDrone.py prepare                 # download + convert + yaml
    python VisDrone.py download [--splits train val test challenge]
    python VisDrone.py convert [--force]
    python VisDrone.py yaml
    python VisDrone.py stats [--imgsz 640 960 1280]

Downloads come from the Ultralytics assets release on GitHub, which hosts the
original VisDrone2019-DET zips. Only the standard library, PyYAML and Pillow are needed.
"""
import argparse
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import yaml

BASE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
    "challenge": "VisDrone2019-DET-test-challenge",  # images only, no public labels
}
NAMES = ["pedestrian", "people", "bicycle", "car", "van",
         "truck", "tricycle", "awning-tricycle", "bus", "motor"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def default_root(config_path: str) -> str:
    p = Path(config_path)
    if p.exists():
        cfg = yaml.safe_load(p.read_text()) or {}
        return cfg.get("dataset", {}).get("root", "datasets/VisDrone")
    return "datasets/VisDrone"


# --------------------------------------------------------------------------- download
def download(root: Path, splits, keep_zips=False):
    root.mkdir(parents=True, exist_ok=True)
    for split in splits:
        folder = SPLITS[split]
        if (root / folder / "images").exists():
            print(f"[download] {folder}: already present, skipping")
            continue
        url = f"{BASE_URL}{folder}.zip"
        zip_path = root / f"{folder}.zip"
        print(f"[download] {url}")

        def hook(blocks, block_size, total):
            if total > 0:
                pct = min(100, blocks * block_size * 100 // total)
                print(f"\r  {pct:3d}%", end="", flush=True)

        urllib.request.urlretrieve(url, zip_path, hook)
        print()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
        if not keep_zips:
            zip_path.unlink()


# --------------------------------------------------------------------------- convert
def convert_split(split_dir: Path, force=False) -> int:
    """Convert VisDrone annotation files to YOLO label files. Returns number of boxes written."""
    from PIL import Image

    ann_dir, img_dir, lbl_dir = split_dir / "annotations", split_dir / "images", split_dir / "labels"
    if not ann_dir.exists():
        print(f"[convert] {split_dir.name}: no annotations folder, skipping")
        return 0
    if lbl_dir.exists() and any(lbl_dir.iterdir()) and not force:
        print(f"[convert] {split_dir.name}: labels already exist (use --force to redo)")
        return 0
    lbl_dir.mkdir(exist_ok=True)

    total = 0
    for ann in sorted(ann_dir.glob("*.txt")):
        img_path = next((img_dir / (ann.stem + e) for e in (".jpg", ".png", ".jpeg")
                         if (img_dir / (ann.stem + e)).exists()), None)
        if img_path is None:
            continue
        w, h = Image.open(img_path).size
        lines = []
        # Row format: left, top, width, height, score, category, truncation, occlusion
        for row in ann.read_text().strip().splitlines():
            f = row.split(",")
            if len(f) < 6:
                continue
            left, top, bw, bh = (float(v) for v in f[:4])
            score, cat = int(f[4]), int(f[5])
            if score == 0 or cat < 1 or cat > 10:  # ignored regions (0) and "others" (11)
                continue
            if bw <= 0 or bh <= 0:
                continue
            xc = min(max((left + bw / 2) / w, 0.0), 1.0)
            yc = min(max((top + bh / 2) / h, 0.0), 1.0)
            nw, nh = min(bw / w, 1.0), min(bh / h, 1.0)
            lines.append(f"{cat - 1} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        (lbl_dir / f"{ann.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        total += len(lines)
    print(f"[convert] {split_dir.name}: {total} boxes written")
    return total


# --------------------------------------------------------------------------- yaml
def write_yaml(root: Path) -> Path:
    data = {
        "path": str(root.resolve()),
        "train": f"{SPLITS['train']}/images",
        "val": f"{SPLITS['val']}/images",
        "names": {i: n for i, n in enumerate(NAMES)},
    }
    if (root / SPLITS["test"] / "images").exists():
        data["test"] = f"{SPLITS['test']}/images"
    out = root / "visdrone.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"[yaml] wrote {out}")
    return out


# --------------------------------------------------------------------------- stats
def stats(root: Path, imgsz_list):
    from PIL import Image

    for key in ("train", "val", "test"):
        split_dir = root / SPLITS[key]
        lbl_dir, img_dir = split_dir / "labels", split_dir / "images"
        if not lbl_dir.exists():
            continue
        class_counts = Counter()
        per_image = []
        small = Counter()  # imgsz -> boxes under 32x32 px after resizing
        n_boxes = 0
        for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            w, h = Image.open(img).size
            lf = lbl_dir / f"{img.stem}.txt"
            rows = [r.split() for r in lf.read_text().splitlines() if r.strip()] if lf.exists() else []
            per_image.append(len(rows))
            for r in rows:
                c, bw, bh = int(r[0]), float(r[3]) * w, float(r[4]) * h
                class_counts[c] += 1
                n_boxes += 1
                for s in imgsz_list:
                    scale = s / max(w, h)
                    if (bw * scale) * (bh * scale) < 32 ** 2:
                        small[s] += 1
        if not per_image:
            continue
        print(f"\n=== {key} ({SPLITS[key]}) ===")
        print(f"images: {len(per_image)}   boxes: {n_boxes}   "
              f"boxes/image: mean {n_boxes / len(per_image):.1f}, max {max(per_image)}")
        print("class counts:")
        for i, name in enumerate(NAMES):
            print(f"  {name:<16}{class_counts[i]:>8}  ({100 * class_counts[i] / max(n_boxes, 1):4.1f}%)")
        print("share of boxes smaller than 32x32 px after resizing to:")
        for s in imgsz_list:
            print(f"  imgsz {s:<5} {100 * small[s] / max(n_boxes, 1):5.1f}%")


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--root", default=None, help="dataset folder (default: dataset.root from config.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("download", "prepare"):
        p = sub.add_parser(name)
        p.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=list(SPLITS))
        p.add_argument("--keep-zips", action="store_true")
        p.add_argument("--force", action="store_true", help="redo label conversion")
    c = sub.add_parser("convert")
    c.add_argument("--force", action="store_true")
    sub.add_parser("yaml")
    s = sub.add_parser("stats")
    s.add_argument("--imgsz", nargs="+", type=int, default=[640, 960, 1280])

    args = ap.parse_args()
    root = Path(args.root or default_root(args.config))

    if args.cmd in ("download", "prepare"):
        download(root, args.splits, args.keep_zips)
    if args.cmd in ("convert", "prepare"):
        for key in ("train", "val", "test"):
            if (root / SPLITS[key]).exists():
                convert_split(root / SPLITS[key], getattr(args, "force", False))
    if args.cmd in ("yaml", "prepare"):
        write_yaml(root)
    if args.cmd == "stats":
        stats(root, args.imgsz)


if __name__ == "__main__":
    sys.exit(main())
