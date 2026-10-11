#!/usr/bin/env python3
"""
SWIFT-US — 5-fold cross-validation BY CASE on the real Ackermann dataset (yolo_dataset_v2).

Every case (region, case id) is tested exactly once. The fold assignment is fixed
(stratified by region x case-positive, random_state=42) and written to folds.csv, so all
models/seeds use the same folds. Within each fold, the remaining cases are split 85/15 into
train/val (val is used for early stopping and for choosing the operating threshold).

Per fold it writes, under <cvdir>/fold<k>/:
  train.txt val.txt test.txt data.yaml, train/ (ultralytics run), detection_metrics.csv
  (test, all + per region), predictions_test.csv, predictions_val.csv
Then run aggregate_cv.py on <cvdir>.

Usage (one SLURM job per fold):
  python 02_code/training/train_cv_real.py --fold 0 --model yolov8n.pt --epochs 100 --seed 42
"""
import argparse, os, sys, json, platform
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("SWIFTUS_DATA", Path.home() / "swiftus_data" / "yolo_dataset_v2"))
OUT  = ROOT / "03_paper_PoC" / "results" / "baseline_real"
PHYSIS_QUERY_CASE = ("wrist", 433)   # kept for training, excluded from evaluation
NFOLDS, FOLD_SEED = 5, 42

def load_manifest():
    m = pd.read_csv(DATA / "manifest_qc.csv"); m["qc_flag"] = m["qc_flag"].fillna("")
    m = m[~m.qc_flag.str.contains("blank_white")].copy()
    m["path"] = [str(DATA / "images" / s / f"{f}.png") for s, f in zip(m.split, m.file)]
    m["stratum"] = np.where(m.qc_flag.str.contains("screen_photo|reviewers_poor_quality"), "poor", "clean")
    m["case_key"] = m.region + "_" + m.case.astype(str)
    m = m.rename(columns={"n_boxes": "n_boxes_gt"})   # manifest column clashes with the prediction column
    return m

def assign_folds(m, cvdir):
    f = cvdir / "folds.csv"
    if f.exists(): return pd.read_csv(f)
    from sklearn.model_selection import StratifiedKFold
    cases = m.groupby(["case_key", "region"]).positive_panel.max().reset_index().rename(columns={"positive_panel": "case_pos"})
    cases["strat"] = cases.region + "_" + cases.case_pos.astype(int).astype(str)
    cases["fold"] = -1
    for k, (_, te) in enumerate(StratifiedKFold(NFOLDS, shuffle=True, random_state=FOLD_SEED).split(cases, cases.strat)):
        cases.loc[cases.index[te], "fold"] = k
    cases.to_csv(f, index=False); return cases

def predict_views(model, paths, imgsz, device):
    out = []
    for res in model.predict(list(paths), imgsz=imgsz, device=device, conf=0.001, verbose=False, stream=True):
        conf = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.array([])
        out.append(dict(path=res.path, max_conf=float(conf.max()) if conf.size else 0.0, n_boxes=int(conf.size)))
    return pd.DataFrame(out)

def det_metrics(model, fdir, sub, name, imgsz, device):
    txt = fdir / f"test_{name}.txt"; txt.write_text("\n".join(sub.path) + "\n")
    tmp = fdir / f"data_{name}.yaml"
    tmp.write_text(f"path: {fdir}\ntrain: train.txt\nval: {txt.name}\nnames:\n  0: fracture\n  1: joint_effusion\n")
    r = model.val(data=str(tmp), imgsz=imgsz, device=device, plots=False, verbose=False,
                  project=str(fdir / "eval"), name=f"val_{name}", exist_ok=True)
    b = r.box
    return dict(subset=name, n_images=len(sub), mAP50=round(float(b.map50), 4), mAP50_95=round(float(b.map), 4),
                precision=round(float(b.mp), 4), recall=round(float(b.mr), 4))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default=None)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    global DATA
    if a.data: DATA = Path(a.data).expanduser().resolve()
    from ultralytics import YOLO
    import torch
    from sklearn.model_selection import train_test_split
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    cvdir = OUT / f"cv{NFOLDS}_{Path(a.model).stem}_img{a.imgsz}_e{a.epochs}_s{a.seed}"
    fdir = cvdir / f"fold{a.fold}"; fdir.mkdir(parents=True, exist_ok=True)
    m = load_manifest(); cases = assign_folds(m, cvdir)
    test_keys = set(cases[cases.fold == a.fold].case_key)
    rest = cases[cases.fold != a.fold]
    tr_keys, va_keys = train_test_split(rest.case_key, test_size=0.15, random_state=FOLD_SEED + a.fold, stratify=rest.strat)
    sets = {"train": m[m.case_key.isin(set(tr_keys))], "val": m[m.case_key.isin(set(va_keys))], "test": m[m.case_key.isin(test_keys)]}
    for s, df in sets.items():
        missing = [p for p in df.path if not os.path.exists(p)]
        if missing: sys.exit(f"{len(missing)} images missing, e.g. {missing[0]}")
        (fdir / f"{s}.txt").write_text("\n".join(df.path) + "\n")
        print(f"fold {a.fold} {s:5s}: {len(df):5d} views, {int(df.positive_panel.sum()):4d} positive, {df.case_key.nunique():3d} cases")
    yaml = fdir / "data.yaml"
    yaml.write_text(f"path: {fdir}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: fracture\n  1: joint_effusion\n")

    best_pt = fdir / "train" / "weights" / "best.pt"
    if best_pt.exists():
        print("best.pt already exists — skipping training, re-running evaluation only")
    else:
        model = YOLO(a.model)
        model.train(data=str(yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=device,
                    workers=a.workers, amp=(device == "cuda"), seed=a.seed, deterministic=True,
                    project=str(fdir), name="train", exist_ok=True, patience=30, pretrained=True, plots=False,
                    hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, fliplr=0.5, flipud=0.0, degrees=5.0,
                    scale=0.3, translate=0.1, mosaic=0.5, mixup=0.0)
    best = YOLO(str(best_pt))

    # evaluation: exclude the physis-query case from the test fold
    te = sets["test"]; te = te[~((te.region == PHYSIS_QUERY_CASE[0]) & (te.case == PHYSIS_QUERY_CASE[1]))]
    rows = [det_metrics(best, fdir, te, "all", a.imgsz, device)]
    for reg in sorted(te.region.unique()):
        rows.append(det_metrics(best, fdir, te[te.region == reg], f"region_{reg}", a.imgsz, device))
    pd.DataFrame(rows).assign(fold=a.fold).to_csv(fdir / "detection_metrics.csv", index=False)
    for s, df in (("val", sets["val"]), ("test", te)):
        P = df.merge(predict_views(best, df.path, a.imgsz, device), on="path")
        P["label"] = P.positive_panel.astype(int); P["fold"] = a.fold
        P[["fold", "region", "case", "case_key", "file", "stratum", "label", "max_conf", "n_boxes"]].to_csv(fdir / f"predictions_{s}.csv", index=False)
    (fdir / "run_meta.json").write_text(json.dumps(dict(vars(a), device=device, platform=platform.platform(), torch=torch.__version__), indent=2))
    print("Done fold", a.fold, "->", fdir)

if __name__ == "__main__":
    main()
