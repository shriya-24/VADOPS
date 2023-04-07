#!/bin/bash
#SBATCH -c 16  # Number of Cores per Task
#SBATCH --mem=250000  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH --gres=gpu:1  # Number of GPUs
#SBATCH -t 04:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

cd /work/pi_adrozdov_umass_edu/syerawar_umass_edu/
module load conda
conda activate ./envs/vadops
cd ./TextClassification

bash promptLLM.sh
