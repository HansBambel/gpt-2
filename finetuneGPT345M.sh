#!/bin/bash -l

# Number of Nodes
#SBATCH -N 1
# Specify that we need one GPU
#SBATCH --gres=gpu:1

#SBATCH -t 04:20:00

#SBATCH --mail-type=ALL

#SBATCH --mail-user=<YourEmailHere>

#SBATCH --job-name=finetuneGPT2

cd $SLURM_SUBMIT_DIR

# load the conda environment (if you have one)
conda activate gpt-2

# load the required modules
module load cuda
# module load tensorflowgpu

# execute the code
python3 train.py --model_name 345M --run_name writingprompts345M --dataset data/wpdump_cleaned345M.npz --batch_size 1 --top_p 0.9 --save_every 2000 --sample_every 1000
