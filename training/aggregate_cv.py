#!/usr/bin/env python3
"""
SWIFT-US — Aggregate a 5-fold cross-validation (train_cv_real.py) into paper-ready tables.

Per level (view / case) and subset (all, clean, poor, region_*):
  - pooled test AUC over all folds, with 95% bootstrap CI (resampling cases);
  - per-fold Youden threshold chosen on that fold's VALIDATION predictions, applied to its
    TEST predictions; sensitivity/specificity pooled over folds (+ bootstrap CI);
  - mAP50 / mAP50-95 mean +/- sd across folds (detection, test folds).
Writes <cvdir>/cv_classification.csv, cv_detection.csv, cv_summary.md

Usage:  python 02_code/training/aggregate_cv.py --cvdir 03_paper_PoC/results/baseline_real/cv5_yolov8n_img640_e100_s42
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

def youden_thr(y, s):
    if len(set(y)) < 2: return np.nan
    fpr, tpr, thr = roc_curve(y, s); return float(thr[int(np.argmax(tpr - fpr))])

def to_cases(P):
    return P.groupby(["fold", "region", "case_key"]).agg(label=("label", "max"), max_conf=("max_conf", "max"),
           stratum=("stratum", lambda s: "poor" if (s == "poor").any() else "clean")).reset_index()

def boot_ci(df, n=1000, seed=0):
    """bootstrap over cases (views of one case stay together); returns 95% CI for auc, sens, spec"""
    rng = np.random.default_rng(seed)
    y, s, p = df.label.values, df.max_conf.values, df.pred.values
    codes, keys = pd.factorize(df.case_key); idx_by_case = [np.flatnonzero(codes == i) for i in range(len(keys))]
    out = {"auc": [], "sens": [], "spec": []}
    for _ in range(n):
        ix = np.concatenate([idx_by_case[i] for i in rng.integers(0, len(keys), len(keys))])
        yy, ss, pp = y[ix], s[ix], p[ix]
        if yy.min() == yy.max(): continue
        out["auc"].append(roc_auc_score(yy, ss))
        out["sens"].append(((pp == 1) & (yy == 1)).sum() / (yy == 1).sum())
        out["spec"].append(((pp == 0) & (yy == 0)).sum() / (yy == 0).sum())
    return {k: (round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)) if v else (np.nan, np.nan) for k, v in out.items()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cvdir", required=True); a = ap.parse_args()
    cv = Path(a.cvdir).resolve()
    folds = sorted(p for p in cv.glob("fold*") if (p / "predictions_test.csv").exists())
    PT = pd.concat([pd.read_csv(f / "predictions_test.csv") for f in folds]); PV = pd.concat([pd.read_csv(f / "predictions_val.csv") for f in folds])
    D = pd.concat([pd.read_csv(f / "detection_metrics.csv") for f in folds])
    print(f"{len(folds)} folds, {len(PT)} test views, {PT.case_key.nunique()} test cases")

    def groups(df):
        g = [("all", df), ("clean", df[df.stratum == "clean"]), ("poor", df[df.stratum == "poor"])]
        return g + [(f"region_{r}", df[df.region == r]) for r in sorted(df.region.unique())]

    rows = []
    for level, (val, test) in {"view": (PV, PT), "case": (to_cases(PV), to_cases(PT))}.items():
        for name, sub in groups(test):
            if len(sub) == 0 or sub.label.nunique() < 2: continue
            vsub = dict(groups(val)).get(name, val)
            # per-fold threshold from that fold's validation subset (fallback: whole val of the fold)
            pred = []
            for k, g in sub.groupby("fold"):
                vk = vsub[vsub.fold == k]; thr = youden_thr(vk.label.values, vk.max_conf.values)
                if np.isnan(thr): vk = val[val.fold == k]; thr = youden_thr(vk.label.values, vk.max_conf.values)
                g = g.copy(); g["pred"] = (g.max_conf >= thr).astype(int); g["thr"] = thr; pred.append(g)
            sub = pd.concat(pred)
            auc = lambda d: roc_auc_score(d.label, d.max_conf) if d.label.nunique() == 2 else np.nan
            sens = lambda d: ((d.pred == 1) & (d.label == 1)).sum() / max((d.label == 1).sum(), 1)
            spec = lambda d: ((d.pred == 0) & (d.label == 0)).sum() / max((d.label == 0).sum(), 1)
            r = dict(level=level, subset=name, n=len(sub), n_pos=int(sub.label.sum()), n_cases=sub.case_key.nunique(),
                     auc=round(auc(sub), 3), sens=round(sens(sub), 3), spec=round(spec(sub), 3),
                     thr_mean=round(float(sub.thr.mean()), 3))
            for nm, (lo, hi) in boot_ci(sub).items():
                r[f"{nm}_ci_lo"], r[f"{nm}_ci_hi"] = lo, hi
            rows.append(r)
    K = pd.DataFrame(rows); K.to_csv(cv / "cv_classification.csv", index=False)

    Dm = D.groupby("subset").agg(n_folds=("fold", "count"), n_images=("n_images", "sum"),
                                  mAP50_mean=("mAP50", "mean"), mAP50_sd=("mAP50", "std"),
                                  mAP50_95_mean=("mAP50_95", "mean"), mAP50_95_sd=("mAP50_95", "std")).round(3).reset_index()
    Dm.to_csv(cv / "cv_detection.csv", index=False)

    md = [f"# {cv.name} — {len(folds)}-fold CV by case ({PT.case_key.nunique()} cases, {len(PT)} views evaluated)\n",
          "Threshold: Youden on each fold's validation set, applied to its test fold. CI: 95% bootstrap over cases.\n",
          "## Detection (test folds, mean ± sd)", "| subset | views | mAP50 | mAP50-95 |", "|---|---|---|---|"]
    md += [f"| {r.subset} | {int(r.n_images)} | {r.mAP50_mean:.3f} ± {r.mAP50_sd:.3f} | {r.mAP50_95_mean:.3f} ± {r.mAP50_95_sd:.3f} |" for r in Dm.itertuples()]
    md += ["\n## Classification (pooled over folds)", "| level | subset | n | pos | AUC [95% CI] | Sens [95% CI] | Spec [95% CI] |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r.level} | {r.subset} | {r.n} | {r.n_pos} | {r.auc:.3f} [{r.auc_ci_lo:.3f}–{r.auc_ci_hi:.3f}] | "
           f"{r.sens:.3f} [{r.sens_ci_lo:.3f}–{r.sens_ci_hi:.3f}] | {r.spec:.3f} [{r.spec_ci_lo:.3f}–{r.spec_ci_hi:.3f}] |" for r in K.itertuples()]
    (cv / "cv_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md)); print("\nSaved to", cv)

if __name__ == "__main__":
    main()
