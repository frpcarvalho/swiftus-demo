# SWIFT-US on Deucalion — passo a passo (2026-10-06)

Conta GPU: `f202500005hpcvlabubig` (1000 h, 101 usadas). Projeto: `/projects/F202500005HPCVLABUBI`.
Partições GPU: `dev-a100-40` (máx 4 h, para testes), `normal-a100-40` (máx 2 dias). 4× A100 40 GB por nó; usamos 1.

## 1. No Mac — enviar dados e código (uma vez)

```bash
cd ~/Documents/data_science_projects/SWIFT-US
ssh deucalion "mkdir -p /projects/F202500005HPCVLABUBI/swiftus/{logs,SWIFT-US/02_code,SWIFT-US/03_paper_PoC/results}"
# dataset (468 MB) — comprimido para ir mais depressa
tar -C ~/swiftus_data -czf /tmp/yolo_dataset_v2.tgz yolo_dataset_v2
scp /tmp/yolo_dataset_v2.tgz deucalion:/projects/F202500005HPCVLABUBI/swiftus/
# código de treino e script SLURM
scp -r 02_code/training deucalion:/projects/F202500005HPCVLABUBI/swiftus/SWIFT-US/02_code/
scp 02_code/training/train_swiftus.slurm deucalion:/projects/F202500005HPCVLABUBI/swiftus/
```

## 2. No Deucalion — ambiente (uma vez, no nó de login)

```bash
ssh deucalion
cd /projects/F202500005HPCVLABUBI/swiftus
tar xzf yolo_dataset_v2.tgz && rm yolo_dataset_v2.tgz
sed -i "s|^path:.*|path: /projects/F202500005HPCVLABUBI/swiftus/yolo_dataset_v2|" yolo_dataset_v2/data.yaml

module load Anaconda3/2025.06-1
export PIP_CACHE_DIR=/projects/F202500005HPCVLABUBI/swiftus/.pip_cache
python -m venv /projects/F202500005HPCVLABUBI/swiftus/env
source /projects/F202500005HPCVLABUBI/swiftus/env/bin/activate
pip install --upgrade pip && pip install ultralytics pandas scikit-learn   # espelho PyPI interno (sn02:9191), ~3 GB
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# pesos pré-treinados descarregados no login (os nós de cálculo podem não ter internet)
cd SWIFT-US
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8s.pt'); YOLO('yolov8m.pt')"
ls *.pt
```

Se `pip install` falhar por falta de internet no login, diz-me — usamos então os módulos `PyTorch-bundle` + `pip install --no-deps ultralytics`.

## 3. Correr

```bash
cd /projects/F202500005HPCVLABUBI/swiftus
# teste rápido (partição dev, 3 épocas) para validar o pipeline
sbatch --partition=dev-a100-40 --time=00:30:00 train_swiftus.slurm yolov8n.pt 3 640 42
squeue -u $USER
tail -f logs/swiftus_yolo_<jobid>.out

# corrida definitiva
sbatch train_swiftus.slurm yolov8n.pt 100 640 42
sbatch train_swiftus.slurm yolov8s.pt 100 640 42
sbatch train_swiftus.slurm yolov8m.pt 100 640 42
```

Resultados em `SWIFT-US/03_paper_PoC/results/baseline_real/<run>/` (pesos em `train/weights/best.pt`, métricas em `eval/`).

## 4. Trazer resultados para o Mac

```bash
scp -r deucalion:/projects/F202500005HPCVLABUBI/swiftus/SWIFT-US/03_paper_PoC/results/baseline_real/ \
    ~/Documents/data_science_projects/SWIFT-US/03_paper_PoC/results/baseline_real_deucalion/
```

Nota: as imagens do Dr. Ackermann ficam no espaço de projeto do Deucalion (infraestrutura pública portuguesa, acesso restrito à conta) — compatível com a restrição "uso exclusivo SWIFT-US". Apagar `yolo_dataset_v2` do Deucalion no fim do trabalho.

## 5. Validação cruzada por caso (5 folds) e avaliação v2 (adicionado 07/10/2026 à tarde)

`evaluate_baseline_real.py` v2 escolhe o limiar de Youden na **validação** e aplica-o ao teste (a v1 escolhia no teste — otimista). Re-correr nos runs já feitos:

```bash
cd /projects/F202500005HPCVLABUBI/swiftus/SWIFT-US
export SWIFTUS_DATA=/projects/F202500005HPCVLABUBI/swiftus/yolo_dataset_v2
for r in 03_paper_PoC/results/baseline_real/yolov8*_e100_s*; do python 02_code/training/evaluate_baseline_real.py --run $r --device cuda; done
```
(na partição de login não há GPU: usar `--device cpu`, demora ~2 min por run, ou meter num job).

CV: `train_cv_real.py` fixa os folds em `folds.csv` (estratificado por região × caso positivo, seed 42) e, por fold, treina em 85 % dos restantes casos, valida em 15 % (early stopping + limiar) e testa no fold. Caso 433 fica só em treino/validação.

```bash
cd /projects/F202500005HPCVLABUBI/swiftus
for k in 0 1 2 3 4; do sbatch train_cv.slurm $k yolov8n.pt 100 640 42; done
# depois de terminarem:
cd SWIFT-US && python 02_code/training/aggregate_cv.py --cvdir 03_paper_PoC/results/baseline_real/cv5_yolov8n_img640_e100_s42
```
Saída: `cv_classification.csv`, `cv_detection.csv`, `cv_summary.md` (AUC/sens/espec com IC 95 % por bootstrap sobre casos; mAP média ± DP entre folds).
