#!/bin/bash -l

# Number of Nodes
#SBATCH -N 1
#SBATCH --gres=gpu:1

#SBATCH -t 04:20:00

#SBATCH --mail-type=ALL

#SBATCH --mail-user=kevin.trebing@ucdconnect.ie

#SBATCH --job-name=encodeDataset

cd $SLURM_SUBMIT_DIR

# load the conda environment (if you have one)
conda activate gpt-2

# load the required modules
module load cuda
# module load tensorflow

# execute the code
python3 encode.py --model_name 345M ../gpt-2/data/writingprompts_cleaned_fully.txt data/wpdump_cleaned345M.npz
