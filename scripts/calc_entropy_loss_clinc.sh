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

# Note: make sure the dataset subset you chose is the one the model is finetuned. If not, it may give inappropriate predictions
# Argument 1(dataset_subset): specify clinc dataset subset you wanna choose. options are ['small', 'imbalanced', 'plus']
# Argument 2(dataset_type): specify dataset type you wanna calculate entropy loss for. options are ["train", "validation", "test"]
# Argument 3(checkpoints_out_dir): specify finetuning checkpoint filepath you wanna load
# Argument 4(entropy_analysis_path): Specify the filepath to save the entropy loss for each sentence with the .csv extension to ensure OS 
# understandability, as the pd.to_csv() method converts the data into a CSV format but does not save it with the .csv file extension.
python ../models/CLINC.py calc_entropy_loss $1 $2 $3 $4

# Examples
# Using python: python ../models/CLINC.py calc_entropy_loss {clinc_subset} {dataset_type} /path/to/checkpoint /path/to/entropy_analysis_dir/{fileName}.csv
# Using python: python ../models/CLINC.py calc_entropy_loss small test ../checkpoints/clinc_small/checkpoint-15200/ ../predictions/entropy/clinc_small_test.csv

# Using bash: sbatch calc_entropy_loss_clinc.sh {clinc_subset} {dataset_type} /path/to/checkpoint /path/to/entropy_analysis_dir/{fileName}.csv
# Using bash: sbatch calc_entropy_loss_clinc.sh small test ../checkpoints/clinc_small/checkpoint-15200/ ../predictions/entropy/clinc_small_test.csv

