# 696DS-repo
Repository to contain the code for 696DS project (Spr 2023)

## Folder Structure:
```text
.
├── README.md
├── checkpoints
├── data
│   ├── README.md
│   └── snips
│       ├── test.csv
│       ├── train.csv
│       └── validation.csv
├── models
│   ├── CLINC.py
│   ├── SNIPS.py
│   └── main.py
├── notebooks
│   ├── BERT.ipynb
│   ├── README.md
│   ├── eval_clinc.ipynb
│   └── finetune_clinc.ipynb
├── predictions
└── scripts
    ├── README.md
    ├── eval_clinc.sh
    ├── eval_snips.sh
    ├── finetune_clinc.sh
    ├── finetune_snips.sh
    └── logs
```

## About Folders:
  - data:  contain subfolders where folder_name = dataset_name. Currently, it only contains snips subfolder which has train, validation, test files in csv format.
  - models:contains PyTorch code 
  - notebooks: contains jupyter notebooks 
  - scripts: contains finetuning and evaluation scripts for datasets. Also contains `logs` subfolder(to save the slurm output)
  - checkpoints: to save finetuned model checkpoints
  - predictions: to save prediction results of the checkpoint model passed

**Note:** One can choose different directory path for saving finetuned model checkpoints, prediction and slurm outputs. But, I highly recommend to follow the above folder structure as we gonna work on each workspaces in future.

## Script Commands:
  Before running any of the following command, change your working directory to scripts folder
  - For Finetuning:

    - CLINC:


      ```sbatch finetune_clinc.sh {clinc_subset} /path/to/checkpoint```

    - SNIPS:

      ```sbatch finetune_snips.sh /path/to/checkpoint```


  - For Evaluation:
    
    - CLINC:


      ```sbatch eval_clinc.sh {clinc_subset} /path/to/checkpoint /path/to/prediction/{fileName}.csv```

    - SNIPS:

      ```sbatch eval_snips.sh /path/to/checkpoint /path/to/prediction/{fileName}.csv```


    ***Note:*** For more details about the scripts, checkout the readme file in the scripts folder.


### TODO: Currently this file, is used to understand the folder structure and what are the script commands. Will be updated once everyone got familiar with the above.