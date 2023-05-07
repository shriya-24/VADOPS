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

# Argument 1(prompt_llm_name): specify the name of the prompt llm
# Argument 2(prompt_type): specify the prompt type. It is an integer from 1 to 4 (both inclusive)
# Argument 3(num_eg examples): Specify the number of prompt examples to be put in prompt for each class.
# Argument 3(num_gen examples): Specify the number of examples to be generated for each class.

python ../models/prompt.py $1 $2 $3 $4

# Examples
# Using python: python ../models/prompt.py ChatGPT 4 1 50
# Using python: python ../models/prompt.py ChatGPT 4 1 50

# Using bash: sbatch calc_entropy_loss_snips.sh {dataset_type} /path/to/checkpoint /path/to/entropy_analysis_dir/{fileName}.csv
# Using bash: sbatch prompt_plugin.sh ChatGPT 4 1 50

