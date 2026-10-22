#!/usr/bin/env python3
"""Train and evaluate a YOLO model over a grid of settings on VisDrone.

    python sweep.py --dry-run             # show the runs
    python sweep.py                       # run everything in config.yaml
    python sweep.py --epochs 3 --fraction 0.05    # quick smoke test

Each run is trained, then validated with the best weights. Results are appended to a CSV,
including AP by object size (small < 32^2 px, medium, large) computed with pycocotools,
since overall mAP hides how VisDrone's small objects are handled. Finished runs are
skipped on re-run, so an interrupted sweep can be resumed.

Requires: ultralytics, pyyaml, pillow. Optional: pycocotools (for the size breakdown).
"""
import argparse
import contextlib
import csv
import io
import itertools
import json
import time
from pathlib import Path

import yaml


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text())


def run_name(combo):
    parts = []
    for k, v in combo.items():
        if k == "model":
            v = Path(str(v)).stem
        parts.append(f"{k}-{v}")
    return "_".join(parts)


def coco_size_ap(pred_json, data_yaml, split):
    """AP overall / small / medium / large from Ultralytics' predictions.json. None if unavailable."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("  pycocotools not installed, skipping size breakdown (pip install pycocotools)")
        return None
    from PIL import Image

    d = yaml.safe_load(Path(data_yaml).read_text())
    img_dir = Path(d["path"]) / d[split]
    lbl_dir = img_dir.parent / "labels"
    names = d["names"]

    images, anns, id_map = [], [], {}
    for i, img in enumerate(sorted(img_dir.glob("*.jpg")), start=1):
        w, h = Image.open(img).size
        images.append({"id": i, "width": w, "height": h, "file_name": img.name})
        id_map[img.stem] = i
        if img.stem.isnumeric():
            id_map.setdefault(str(int(img.stem)), i)  # Ultralytics may int() numeric stems
        lf = lbl_dir / f"{img.stem}.txt"
        if not lf.exists():
            continue
        for line in lf.read_text().splitlines():
            if not line.strip():
                continue
            c, x, y, bw, bh = line.split()[:5]
            bw, bh = float(bw) * w, float(bh) * h
            x, y = float(x) * w - bw / 2, float(y) * h - bh / 2
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(c) + 1,
                         "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})

    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": k + 1, "name": v} for k, v in names.items()]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()

    preds = json.loads(Path(pred_json).read_text())
    if not preds:
        return None
    zero_based = min(p["category_id"] for p in preds) == 0
    res = []
    for p in preds:
        iid = id_map.get(str(p["image_id"]))
        if iid is None:
            continue
        res.append({"image_id": iid, "category_id": p["category_id"] + (1 if zero_based else 0),
                    "bbox": [float(v) for v in p["bbox"]], "score": float(p["score"])})
    if not res:
        return None

    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(res)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 10, 1000]  # VisDrone images can hold hundreds of objects
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    return {"coco_ap": s[0], "coco_ap50": s[1], "ap_small": s[3], "ap_medium": s[4], "ap_large": s[5]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    ap.add_argument("--fraction", type=float, default=None, help="override train.fraction")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ds, tr, ev, sw = cfg["dataset"], dict(cfg["train"]), cfg["eval"], cfg["sweep"]
    if args.epochs is not None:
        tr["epochs"] = args.epochs
    if args.fraction is not None:
        tr["fraction"] = args.fraction
    extra = tr.pop("extra", None) or {}
    project = tr.pop("project")
    device = tr.pop("device", None)

    grid = sw["grid"]
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]

    data_yaml = ds["yaml"]
    if not args.dry_run and not Path(data_yaml).exists():
        raise SystemExit(f"{data_yaml} not found. Run: python VisDrone.py prepare")

    csv_path = Path(sw["results_csv"])
    class_names = []
    if Path(data_yaml).exists():
        class_names = list(yaml.safe_load(Path(data_yaml).read_text())["names"].values())
    fields = (["run"] + keys + ["map50_95", "map50", "coco_ap", "coco_ap50",
              "ap_small", "ap_medium", "ap_large", "train_minutes", "weights"]
              + [f"ap_{n}" for n in class_names])

    done = set()
    if csv_path.exists():
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != fields:
                raise SystemExit(f"{csv_path} has different columns from this sweep. "
                                 f"Move it aside or change sweep.results_csv.")
            done = {row["run"] for row in reader}

    print(f"{len(combos)} runs, {len([c for c in combos if run_name(c) in done])} already done")
    if args.dry_run:
        for c in combos:
            tag = "  (done)" if run_name(c) in done else ""
            print(f"  {run_name(c)}{tag}")
        return

    from ultralytics import YOLO

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    for n, combo in enumerate(combos, start=1):
        name = run_name(combo)
        if name in done:
            continue
        print(f"\n[{n}/{len(combos)}] {name}")
        train_args = {**tr, **extra, **{k: v for k, v in combo.items() if k != "model"}}
        if device is not None:
            train_args["device"] = device
        try:
            t0 = time.time()
            model = YOLO(str(combo.get("model", "yolo11n.pt")))
            model.train(data=data_yaml, project=project, name=name, exist_ok=False, **train_args)
            minutes = (time.time() - t0) / 60
            best = Path(model.trainer.best)

            val_args = dict(data=data_yaml, imgsz=train_args["imgsz"], batch=train_args["batch"],
                            conf=ev["conf"], iou=ev["iou"], max_det=ev["max_det"],
                            split=ds["eval_split"], save_json=True, plots=False,
                            project=project, name=f"{name}_eval")
            if device is not None:
                val_args["device"] = device
            metrics = YOLO(str(best)).val(**val_args)

            row = {"run": name, **{k: combo[k] for k in keys},
                   "map50_95": round(float(metrics.box.map), 4),
                   "map50": round(float(metrics.box.map50), 4),
                   "train_minutes": round(minutes, 1), "weights": str(best)}
            for cname, v in zip(class_names, metrics.box.maps):
                row[f"ap_{cname}"] = round(float(v), 4)

            pred_json = Path(metrics.save_dir) / "predictions.json"
            if pred_json.exists():
                size_ap = coco_size_ap(pred_json, data_yaml, ds["eval_split"])
                if size_ap:
                    row.update({k: round(float(v), 4) for k, v in size_ap.items()})
        except Exception as e:  # keep the sweep going, e.g. after an out-of-memory error
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue

        with csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            if new_file:
                w.writeheader()
                new_file = False
            w.writerow(row)
        print(f"  mAP50-95 {row['map50_95']}  mAP50 {row['map50']}  "
              f"small {row.get('ap_small', 'n/a')}  large {row.get('ap_large', 'n/a')}")

    print(f"\nResults: {csv_path}")


if __name__ == "__main__":
    main()
