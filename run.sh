#!/bin/bash
#SBATCH -J grpo-debate
#SBATCH -t 12:00:00
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=s.abreu@rug.nl

source .venv/bin/activate

CONFIG=$1

# Check if config argument was provided
if [ -z "$CONFIG" ]; then
    echo "Error: Please provide a config file as an argument"
    echo "Usage: sbatch run.sh <config_file>"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file '$CONFIG' not found"
    exit 1
fi

echo "Starting job with config: $CONFIG"
python main.py --config "$CONFIG"
