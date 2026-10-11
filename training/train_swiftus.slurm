#!/bin/bash
#SBATCH --job-name=swiftus_yolo
#SBATCH --account=f202500005hpcvlabubig
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/F202500005HPCVLABUBI/swiftus/logs/%x_%j.out
#SBATCH --error=/projects/F202500005HPCVLABUBI/swiftus/logs/%x_%j.err

# SWIFT-US — baseline YOLO on the real Ackermann dataset (Deucalion, A100)
# Usage:  sbatch train_swiftus.slurm [model] [epochs] [imgsz] [seed]
MODEL=${1:-yolov8n.pt}
EPOCHS=${2:-100}
IMGSZ=${3:-640}
SEED=${4:-42}

BASE=/projects/F202500005HPCVLABUBI/swiftus
export SWIFTUS_DATA=$BASE/yolo_dataset_v2
export YOLO_CONFIG_DIR=$BASE/.yolo_cfg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load Anaconda3/2025.06-1
source $BASE/env/bin/activate

cd $BASE/SWIFT-US
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch, ultralytics; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ultralytics', ultralytics.__version__)"

python 02_code/training/train_baseline_real.py \
    --model $MODEL --epochs $EPOCHS --imgsz $IMGSZ --batch 32 \
    --device cuda --seed $SEED --workers $SLURM_CPUS_PER_TASK

RUN=03_paper_PoC/results/baseline_real/$(basename ${MODEL%.pt})_img${IMGSZ}_e${EPOCHS}_s${SEED}
python 02_code/training/evaluate_baseline_real.py --run $RUN --imgsz $IMGSZ --device cuda
