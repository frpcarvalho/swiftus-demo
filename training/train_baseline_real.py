#!/usr/bin/env python3
"""
SWIFT-US — Baseline YOLO on the real Ackermann dataset (yolo_dataset_v2).

What it does
  1. Reads manifest_qc.csv and builds filtered image lists:
       train / val : all views except qc_flag containing 'blank_white'
       test        : same, and also excludes case wrist 433 (physis/fracture query)
  2. Writes a data.yaml pointing at those lists (the dataset folder is NOT modified).
  3. Fine-tunes a pretrained YOLO model (default yolov8n.pt) on Apple MPS / CUDA / CPU.
  4. Saves everything under 03_paper_PoC/results/baseline_real/<run_name>/

Usage (from the SWIFT-US folder, inside your conda env with ultralytics installed):
  python 02_code/training/train_baseline_real.py --model yolov8n.pt --epochs 100 --imgsz 640
  python 02_code/training/train_baseline_real.py --model yolov8s.pt --epochs 100 --device mps
Then run evaluate_baseline_real.py on the resulting run.
"""
import argparse, os, sys, json, platform
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]              # .../SWIFT-US
DATA = Path(os.environ.get("SWIFTUS_DATA", Path.home() / "swiftus_data" / "yolo_dataset_v2"))  # dataset OUTSIDE iCloud
OUT  = ROOT / "03_paper_PoC" / "results" / "baseline_real"

PHYSIS_QUERY_CASE = ("wrist", 433)   # image 13 of the clinical review

def build_lists(run_dir: Path):
    if not (DATA / "manifest_qc.csv").exists():
        sys.exit(f"Dataset not found at {DATA}. Set SWIFTUS_DATA=/path/to/yolo_dataset_v2 or use --data.")
    m = pd.read_csv(DATA / "manifest_qc.csv")
    m["qc_flag"] = m["qc_flag"].fillna("")
    keep = ~m["qc_flag"].str.contains("blank_white")
    lists = {}
    for split in ("train", "val", "test"):
        sel = m[(m.split == split) & keep]
        if split == "test":
            sel = sel[~((sel.region == PHYSIS_QUERY_CASE[0]) & (sel.case == PHYSIS_QUERY_CASE[1]))]
        paths = [str(DATA / "images" / split / f"{f}.png") for f in sel.file]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            sys.exit(f"{len(missing)} images missing on disk (iCloud not downloaded?). First: {missing[0]}")
        (run_dir / f"{split}.txt").write_text("\n".join(paths) + "\n")
        lists[split] = sel
        print(f"{split:5s}: {len(sel):5d} views, {int(sel.positive_panel.sum()):4d} positive, "
              f"{sel.case.nunique():3d} cases")
    # strata of the test set for the evaluation script
    t = lists["test"].copy()
    t["stratum"] = "clean"
    t.loc[t.qc_flag.str.contains("screen_photo|reviewers_poor_quality"), "stratum"] = "poor"
    t.to_csv(run_dir / "test_manifest.csv", index=False)
    yaml = run_dir / "data.yaml"
    yaml.write_text(
        "# SWIFT-US baseline on real data — filtered lists (blank views excluded; case wrist 433 excluded from test)\n"
        f"path: {run_dir}\n"
        f"train: train.txt\nval: val.txt\ntest: test.txt\n"
        "names:\n  0: fracture\n  1: joint_effusion\n")
    return yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="mps | cuda | cpu (auto if omitted)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None)
    ap.add_argument("--data", default=None, help="path to yolo_dataset_v2 (default ~/swiftus_data/yolo_dataset_v2)")
    ap.add_argument("--workers", type=int, default=0, help="dataloader workers; 0 avoids multiprocessing hangs on macOS")
    a = ap.parse_args()
    global DATA
    if a.data: DATA = Path(a.data).expanduser().resolve()

    from ultralytics import YOLO
    import torch
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    name = a.name or f"{Path(a.model).stem}_img{a.imgsz}_e{a.epochs}_s{a.seed}"
    run_dir = OUT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml = build_lists(run_dir)

    model = YOLO(a.model)
    model.train(data=str(yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=device,
                workers=a.workers, amp=(device == "cuda"),
                seed=a.seed, deterministic=True, project=str(run_dir), name="train", exist_ok=True,
                patience=30, pretrained=True, plots=True,
                # ultrasound: no colour, keep geometry realistic
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, fliplr=0.5, flipud=0.0, degrees=5.0,
                scale=0.3, translate=0.1, mosaic=0.5, mixup=0.0)
    meta = dict(model=a.model, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=device, seed=a.seed,
                platform=platform.platform(), torch=torch.__version__)
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("\nDone. Weights:", run_dir / "train" / "weights" / "best.pt")
    print("Next: python 02_code/training/evaluate_baseline_real.py --run", run_dir)

if __name__ == "__main__":
    main()
