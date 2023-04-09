# Scripts:

All the scripts load conda and activate vadops environment(`/work/pi_adrozdov_umass_edu/$USER/envs/vadops`) and calls the corresponding python file with arguments passed.  The initial argument is the function name to call of the python file and other arguments are the parameters for that function. Once you hit the command, it creates a job and the job will be thrown into the queue and will start to run once resources become available. All the stdout of your job is written on a file and it is saved at `./logs/slurm-{jobID}.out`.

## Quick commands:

### To check the status of job: 
  ```squeue --me``` #  lists all jobs only running/waiting trigged by you

  ```sacct -j <jobID>``` # single job realted details in more elaborate way

### Reading the slurm output file: 
  ```cat ./logs/slurm-<jobID>``` # to read the file.

  ```tail -f ./logs/slurm-<jobID>``` # to continuously see output of a Slurm job that is running.

# About:
## finetune_clinc.sh
  - calls finetune method in `models/CLINC.py` file.
  - Need 2 command args:
    - dataset_subset: specify clinc dataset subset you wanna choose. options are ['small', 'imbalanced', 'plus']
    - checkpoints_out_dir: specify filepath where you want finetuning checkpoint to be saved
   

    Command: 
    
    ```sbatch finetune_clinc.sh {clinc_subset} /path/to/checkpoint```

    Example: 
    
    ```sbatch finetune_clinc.sh small ../checkpoints/clinc_small```


## finetune_snips.sh
  - calls finetune method in `models/SNIPS.py` file.
  - Need 1 command arg:
    - checkpoints_out_dir: specify filepath where you want finetuning checkpoint to be saved
  
    Command:
    
    ```sbatch finetune_snips.sh /path/to/checkpoint```

    Example:

    ```sbatch finetune_snips.sh ../checkpoints/snips```


## eval_clinc.sh 
  - calls eval method in `models/CLINC.py` file. 
  - Need 3 command args:
    - dataset_subset: specify clinc dataset subset you wanna choose. options are ['small', 'imbalanced', 'plus']
    - checkpoints_out_dir: specify finetuning checkpoint filepath you wanna load
    - predictions_out_dir: Specify the filepath to save the predictions with the .csv extension to ensure OS understandability, as the pd.to_csv() method converts the data into a CSV format but does not save it with the .csv file extension.
    
    Command:
    
    ```sbatch eval_clinc.sh {clinc_subset} /path/to/checkpoint /path/to/prediction/{fileName}.csv```

    Example: 
    
    ```sbatch eval_clinc.sh small ../checkpoints/clinc_small/checkpoint-15200/ ../predictions/clinc_small.csv```

  
## eval_snips.sh 
  - calls eval method in `models/SNIPS.py`file. 
  - Need 2 command args:
    - checkpoints_out_dir: specify finetuning checkpoint filepath you wanna load
    - predictions_out_dir: Specify the filepath to save the predictions with the .csv extension to ensure OS understandability, as the pd.to_csv() method converts the data into a CSV format but does not save it with the .csv file extension.
   
    Command: 
    
    ```sbatch eval_snips.sh /path/to/checkpoint /path/to/prediction/{fileName}.csv```
    
    Example: 
    
    ```sbatch eval_snips.sh ../checkpoints/snips/checkpoint-26170/ ../predictions/snips.csv```


## calc_entropy_loss_clinc.sh
  - calls calc_entropy_loss method in `models/CLINC.py` file. 
  - Need 3 command args:
    - dataset_subset: specify clinc dataset subset you wanna choose. options are ['small', 'imbalanced', 'plus']
    - checkpoints_out_dir: specify finetuning checkpoint filepath you wanna load
    - entropy_analysis_path: Specify the filepath to save the entropy loss for each sentence with the .csv extension to ensure OS understandability, as the pd.to_csv() method converts the data into a CSV format but does not save it with the .csv file extension.
    
    Command:
    
    ```sbatch calc_entropy_loss_clinc.sh {clinc_subset} /path/to/checkpoint /path/to/entropy_analysis_dir/{fileName}.csv```

    Example: 
    
    ```sbatch calc_entropy_loss_clinc.sh small ../checkpoints/clinc_small/checkpoint-15200/ ../predictions/entropy/clinc_small.csv```

  
## calc_entropy_loss_snips.sh 
  - calls calc_entropy_loss method in `models/SNIPS.py`file. 
  - Need 2 command args:
    - checkpoints_out_dir: specify finetuning checkpoint filepath you wanna load
    - entropy_analysis_path: Specify the filepath to save the entropy loss for each sentence with the .csv extension to ensure OS understandability, as the pd.to_csv() method converts the data into a CSV format but does not save it with the .csv file extension.
   
    Command: 
    
    ```sbatch calc_entropy_loss_snips.sh /path/to/checkpoint /path/to/entropy_analysis_dir/{fileName}.csv```
    
    Example: 
    
    ```sbatch calc_entropy_loss_snips.sh ../checkpoints/snips/checkpoint-26170/ ../predictions/entropy/snips.csv```