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

# Argument 1(dataset_subset): specify clinc dataset subset you wanna choose. options are ['small', 'imbalanced', 'plus']
# Argument 2(checkpoints_out_dir): specify filepath where you want finetuning checkpoint to be saved
python ../models/CLINC.py finetune $1 $2

# Example
# Using python: python ../models/CLINC.py finetune {clinc_subset} /path/to/checkpoint
# Using python: python ../models/CLINC.py finetune small ../checkpoints/clinc_small

# Using bash: sbatch finetune_clinc.sh {clinc_subset} /path/to/checkpoint
# Using bash: sbatch finetune_clinc.sh small ../checkpoints/clinc_small

