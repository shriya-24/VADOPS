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
# Argument 4(predictions_out_dir): Specify the filepath to save the predictions with the .csv extension to ensure OS 
# understandability, as the pd.to_csv() method converts the data into a CSV format but does not save it with the .csv file extension.
python ../models/CLINC.py eval $1 $2 $3 $4

# Examples
# Using python: python ../models/CLINC.py eval {clinc_subset} {dataset_type} /path/to/checkpoint /path/to/prediction/{fileName}.csv
# Using python: python ../models/CLINC.py eval small test ../checkpoints/clinc_small/checkpoint-15200/ ../predictions/clinc_small_test.csv

# Using bash: sbatch eval_clinc.sh {clinc_subset} {dataset_type} /path/to/checkpoint /path/to/prediction/{fileName}.csv
# Using bash: sbatch eval_clinc.sh small test ../checkpoints/clinc_small/checkpoint-15200/ ../predictions/clinc_small_test.csv

