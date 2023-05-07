#!/bin/bash
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=50GB  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 12:00:00  # Job time limit
#SBATCH -o ./logs/slurm-%j.out  # %j = job ID

# load conda
module load conda

# activate vadops env
conda activate /work/pi_adrozdov_umass_edu/$USER/envs/vadops

# Argument 1(config_file_path): specify workflow configuration path
python ../models/workflow.py $1

# Example
# Using python: python ../models/workflow.py /path/to/workflow_config_json_file
# Using python: python ../models/workflow.py ../workflow_configs/config.json

# Using bash: sbatch workflow.sh /path/to/workflow_config_json_file
# Using bash: sbatch workflow.sh ../workflow_configs/config.json

