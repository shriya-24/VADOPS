#!/bin/bash
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=50GB  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 02:00:00  # Job time limit
#SBATCH -o ./logs/slurm-%j.out  # %j = job ID

# load conda
module load conda

# activate vadops env
conda activate /work/pi_adrozdov_umass_edu/$USER/envs/vadops

# Argument 1(checkpoints_out_dir): specify filepath where you want finetuning checkpoint to be saved
python ../models/SNIPS.py finetune $1

# Example
# Using python: python ../models/SNIPS.py finetune /path/to/checkpoint
# Using python: python ../models/SNIPS.py finetune ../checkpoints/snips

# Using bash: sbatch finetune_snips.sh /path/to/checkpoint
# Using bash: sbatch finetune_snips.sh ../checkpoints/snips


