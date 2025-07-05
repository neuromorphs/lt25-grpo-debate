#!/bin/bash
#SBATCH -J grpo-debate
#SBATCH -t 8:00:00
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=s.abreu@rug.nl

source .venv/bin/activate
python main.py --config configs/fullseq.yaml
