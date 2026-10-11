#!/usr/bin/env python3
"""
SWIFT-US — Evaluate a trained baseline on the held-out test set.

v2 (2026-10-07): the operating threshold is chosen on the VALIDATION set (Youden) and
applied to the test set; AUC on the test set is threshold-free. Previous version chose the
threshold on the test set itself, which is optimistic.

Reports, in <run>/eval/:
  1. detection metrics (mAP50, mAP50-95, precision, recall) on the full test set,
     per quality stratum (clean / poor) and per region (elbow / humerus / wrist);
  2. view-level classification (positive view vs not) from the max box confidence:
     test AUC, and sensitivity/specificity at the validation-derived threshold;
  3. case-level classification (a case is positive if any of its views is positive);
  4. predictions.csv (test) and predictions_val.csv with one row per view.

Usage:
  python 02_code/training/evaluate_baseline_real.py --run 03_paper_PoC/results/baseline_real/<run_name>
"""
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd

def det_metrics(model, subset_txt, name, imgsz, device):
    tmp = subset_txt.parent / f"data_{name}.yaml"
    tmp.write_text(f"path: {subset_txt.parent}\ntrain: train.txt\nval: {subset_txt.name}\n"
                   "names:\n  0: fracture\n  1: joint_effusion\n")
    r = model.val(data=str(tmp), imgsz=imgsz, device=device, plots=False, verbose=False,
                  project=str(subset_txt.parent / "eval"), name=f"val_{name}", exist_ok=True)
    b = r.box
    return dict(subset=name, n_images=len(subset_txt.read_text().split()),
                mAP50=round(float(b.map50), 4), mAP50_95=round(float(b.map), 4),
                precision=round(float(b.mp), 4), recall=round(float(b.mr), 4))

def auc_safe(y, s):
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(y, s)), 3) if len(set(y)) == 2 else np.nan

def youden_thr(y, s):
    from sklearn.metrics import roc_curve
    if len(set(y)) < 2: return np.nan
    fpr, tpr, thr = roc_curve(y, s); k = int(np.argmax(tpr - fpr)); return float(thr[k])

def sens_spec(y, s, thr):
    y = np.asarray(y); p = np.asarray(s) >= thr
    sens = (p & (y == 1)).sum() / max((y == 1).sum(), 1)
    spec = (~p & (y == 0)).sum() / max((y == 0).sum(), 1)
    return round(float(sens), 3), round(float(spec), 3)

def predict_views(model, paths, imgsz, device):
    out = []
    for res in model.predict(list(paths), imgsz=imgsz, device=device, conf=0.001, verbose=False, stream=True):
        conf = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.array([])
        out.append(dict(path=res.path, max_conf=float(conf.max()) if conf.size else 0.0, n_boxes=int(conf.size)))
    return pd.DataFrame(out)

def to_cases(P):
    return P.groupby(["region", "case"]).agg(
        label=("label", "max"), max_conf=("max_conf", "max"),
        stratum=("stratum", lambda s: "poor" if (s == "poor").any() else "clean")).reset_index()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data", default=None, help="path to yolo_dataset_v2 (default ~/swiftus_data/yolo_dataset_v2)")
    a = ap.parse_args()
    from ultralytics import YOLO
    import torch
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    run = Path(a.run).resolve(); ev = run / "eval"; ev.mkdir(exist_ok=True)
    model = YOLO(str(run / "train" / "weights" / "best.pt"))
    DATA = Path(a.data).expanduser().resolve() if a.data else Path(os.environ.get("SWIFTUS_DATA", Path.home() / "swiftus_data" / "yolo_dataset_v2"))

    T = pd.read_csv(run / "test_manifest.csv").rename(columns={"n_boxes": "n_boxes_gt"})
    T["path"] = [str(DATA / "images" / "test" / f"{f}.png") for f in T.file]
    # validation views (same filtering as training lists; stratum derived the same way)
    V = pd.read_csv(DATA / "manifest_qc.csv"); V["qc_flag"] = V["qc_flag"].fillna("")
    V = V[(V.split == "val") & ~V.qc_flag.str.contains("blank_white")].copy().rename(columns={"n_boxes": "n_boxes_gt"})
    V["stratum"] = np.where(V.qc_flag.str.contains("screen_photo|reviewers_poor_quality"), "poor", "clean")
    V["path"] = [str(DATA / "images" / "val" / f"{f}.png") for f in V.file]

    # 1. detection metrics on test subsets
    subsets = {"all": T, "clean": T[T.stratum == "clean"], "poor": T[T.stratum == "poor"]}
    for reg in T.region.unique(): subsets[f"region_{reg}"] = T[T.region == reg]
    rows = []
    for name, sub in subsets.items():
        if len(sub) == 0: continue
        txt = run / f"test_{name}.txt"; txt.write_text("\n".join(sub.path) + "\n")
        rows.append(det_metrics(model, txt, name, a.imgsz, device))
    D = pd.DataFrame(rows); D.to_csv(ev / "detection_metrics.csv", index=False); print(D.to_string(index=False))

    # 2. view-level predictions on val (for thresholds) and test
    PV = V.merge(predict_views(model, V.path, a.imgsz, device), on="path"); PV["label"] = PV.positive_panel.astype(int)
    P  = T.merge(predict_views(model, T.path, a.imgsz, device), on="path"); P["label"] = P.positive_panel.astype(int)
    PV.to_csv(ev / "predictions_val.csv", index=False); P.to_csv(ev / "predictions.csv", index=False)
    CV, C = to_cases(PV), to_cases(P)

    def groups(df):
        g = [("all", df), ("clean", df[df.stratum == "clean"]), ("poor", df[df.stratum == "poor"])]
        return g + [(f"region_{r}", df[df.region == r]) for r in sorted(df.region.unique())]

    cls = []
    for level, (val_df, test_df) in {"view": (PV, P), "case": (CV, C)}.items():
        val_groups = dict(groups(val_df))
        for name, sub in groups(test_df):
            if len(sub) == 0: continue
            vsub = val_groups.get(name, val_df)
            # threshold from validation (fall back to the whole val set if the subgroup is degenerate)
            thr = youden_thr(vsub.label.values, vsub.max_conf.values)
            if np.isnan(thr): thr = youden_thr(val_df.label.values, val_df.max_conf.values)
            se, sp = sens_spec(sub.label.values, sub.max_conf.values, thr)
            cls.append(dict(level=level, subset=name, n=len(sub), n_pos=int(sub.label.sum()),
                            auc_test=auc_safe(sub.label.values, sub.max_conf.values),
                            thr_val=round(thr, 3), sens_test=se, spec_test=sp,
                            n_val=len(vsub), auc_val=auc_safe(vsub.label.values, vsub.max_conf.values)))
    K = pd.DataFrame(cls); K.to_csv(ev / "classification_metrics.csv", index=False); print(K.to_string(index=False))

    summary = dict(run=str(run), test_views=len(T), test_cases=int(C.shape[0]), val_views=len(V), val_cases=int(CV.shape[0]),
                   threshold_policy="Youden on validation set, applied to test",
                   detection=D.to_dict("records"), classification=K.to_dict("records"))
    (ev / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nSaved to", ev)

if __name__ == "__main__":
    main()
