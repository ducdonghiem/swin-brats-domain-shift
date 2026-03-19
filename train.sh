#!/bin/bash

# Slurm directives (These allow you to run 'sbatch train.sh' directly)
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --output=train_%j.log  # %j inserts the unique Job ID

# 1. Load environment
source /home/dod2/myenv/bin/activate

# 2. Set Memory Management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. Create a unique log filename based on the timestamp (if not using sbatch logs)
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training. Logging to $LOG_FILE"

# 4. Run Python and redirect output
# 2>&1 merges error messages (stderr) into the standard output (stdout)
# 'tee' allows you to see the output in the terminal AND save it to the file
# python src/training/train.py "$@" 2>&1 | tee "$LOG_FILE"
python -m src.training.train 2>&1 | tee "$LOG_FILE"