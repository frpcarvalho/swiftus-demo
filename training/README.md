# SWIFT-US proof of concept — training and evaluation code

Code used for the manuscript *Automated detection of distal forearm fractures on point-of-care
ultrasound with a lightweight deep-learning model* (SWIFT-US consortium, 2026).

| File | Purpose |
|---|---|
| `train_cv_real.py` | 5-fold cross-validation by examination (stratified by region × positive, fixed folds); trains YOLOv8 per fold and writes per-view predictions |
| `aggregate_cv.py` | Pools folds: Youden threshold on each fold's validation set, pooled AUC/sensitivity/specificity, bootstrap CIs over examinations, mAP mean ± SD |
| `train_baseline_real.py` | Single fixed 70/15/15 split (model-size comparison) |
| `evaluate_baseline_real.py` | Evaluation of a fixed-split run (threshold from validation, metrics on test) |
| `train_cv.slurm`, `train_swiftus.slurm` | SLURM scripts used on the Deucalion supercomputer (EuroHPC / FCT, Portugal) |
| `README_deucalion.md` | Environment set-up notes for Deucalion |

Dataset layout expected: `yolo_dataset_v2/` with `images/{train,val,test}/*.png`, YOLO labels and `manifest_qc.csv`
(columns `file, split, region, case, positive_panel, n_boxes, qc_flag`). The images are **not** distributed with
this repository (data-use agreement restricting use to the SWIFT-US project).

Software: Python 3.13, PyTorch 2.14, Ultralytics 8.4.174 (AGPL-3.0), scikit-learn 1.9, pandas 3.0.
