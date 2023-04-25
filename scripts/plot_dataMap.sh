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

# Argument 1(log_training_dynamics_dir): specify the training dynamics folder path
# Argument 2(plot_dir): specify the plot_dir path to save the graph
# Argument 3(dataset_name): specify the dataset_name for which the training dynamics are generated
python ../models/main.py plot $1 $2 $3

# Example
# Using python: python ../models/main.py plot /path/to/log_training_dynamics_dir /path/to/plot_dir {dataset_name}
# Using python: python ../models/main.py plot ../training_dynamics/clinc_small /datamap_graphs clinc_small_train

# Using bash: sbatch plot_dataMap.sh /path/to/log_training_dynamics_dir /path/to/plot_dir {dataset_name}
# Using bash: sbatch plot_dataMap.sh ../training_dynamics/clinc_small /datamap_graphs clinc_small_train
