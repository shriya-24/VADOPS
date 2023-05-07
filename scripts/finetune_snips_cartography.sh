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
# Argument 2(cartography_splits): Specify the dataset split from ['train', 'validation'] to log dynamics. You can add multiple values with comma seperated.
# Argument 3(log_dynamics_dir): specify the dir path to store the dynamics
python ../models/SNIPS.py finetune $1 True $2 $3

# Example
# Using python: python ../models/SNIPS.py finetune /path/to/checkpoint True {cartography_splits} {log_dynamics_dir}
# Using python: python ../models/SNIPS.py finetune ../checkpoints/snips True train ../training_dynamics/snips

# Using bash: sbatch finetune_snips_cartography.sh /path/to/checkpoint {cartography_splits} {log_dynamics_dir}
# Using bash: sbatch finetune_snips_cartography.sh ../checkpoints/snips train ../training_dynamics/snips


