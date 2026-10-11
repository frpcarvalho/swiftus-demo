#!/bin/bash
#SBATCH --job-name=swiftus_cv
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

# SWIFT-US — one fold of the 5-fold CV by case (A100)
# Usage:  sbatch train_cv.slurm <fold 0-4> [model] [epochs] [imgsz] [seed]
FOLD=$1
MODEL=${2:-yolov8n.pt}
EPOCHS=${3:-100}
IMGSZ=${4:-640}
SEED=${5:-42}

BASE=/projects/F202500005HPCVLABUBI/swiftus
export SWIFTUS_DATA=$BASE/yolo_dataset_v2
export YOLO_CONFIG_DIR=$BASE/.yolo_cfg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load Anaconda3/2025.06-1
source $BASE/env/bin/activate

cd $BASE/SWIFT-US
python -c "import torch; print('cuda', torch.cuda.is_available())"
python 02_code/training/train_cv_real.py --fold $FOLD --model $MODEL --epochs $EPOCHS --imgsz $IMGSZ \
    --batch 32 --device cuda --seed $SEED --workers $SLURM_CPUS_PER_TASK
