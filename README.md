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
│   └── workflow.py
├── notebooks
│   ├── BERT.ipynb
│   ├── README.md
│   ├── eval_clinc.ipynb
│   └── finetune_clinc.ipynb
├── output
├── predictions
├── training_dynamics
├── datamap_graphs
└── workflow_configs
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
  - training_dynamics: to save the training_dynamics during the finetuning
  - datamap_graphs: this is where we save the datamap graphs for a model using the training_dynamics
  - workflow_configs: contains all the workflow configuration
  - output: used to saved the output for the workflows

**Note:** One can choose different directory path for saving finetuned model checkpoints, prediction and slurm outputs. But, I highly recommend to follow the above folder structure as we gonna work on each workspaces in future.

## Script Commands:
  Before running any of the following command, change your working directory to scripts folder
  - For Finetuning:

    - CLINC:

      ```sbatch finetune_clinc.sh {clinc_subset} /path/to/checkpoint```

    - SNIPS:

      ```sbatch finetune_snips.sh /path/to/checkpoint```

  - For Finetuning along with Dataset Cartography:

    - CLINC:
        ```sbatch finetune_clinc_cartography.sh {clinc_subset} /path/to/checkpoint {cartography_split} {log_dynamics_dir}```

    - SNIPS:

      ```sbatch finetune_snips_cartography.sh /path/to/checkpoint {cartography_splits} {log_dynamics_dir}```


  - For Evaluation:
    
    - CLINC:

      ```sbatch eval_clinc.sh {clinc_subset} {dataset_type} /path/to/checkpoint /path/to/prediction/{fileName}.csv```

    - SNIPS:

      ```sbatch eval_snips.sh {dataset_type} /path/to/checkpoint /path/to/prediction/{fileName}.csv```

  - For Calculating Entropy Loss:
    
    - CLINC:

      ```sbatch calc_entropy_loss_clinc.sh {clinc_subset} {dataset_type} /path/to/checkpoint /path/to/entropy_analysis_dir/{fileName}.csv```

    - SNIPS:

      ```sbatch calc_entropy_loss_snips.sh {dataset_type} /path/to/checkpoint /path/to/entropy_analysis_dir/{fileName}.csv```
  
  - For Plotting DataMap for a saved dynamics

     ```sbatch plot_dataMap.sh /path/to/log_dynamics_dir  /path/to/plot_dir {dataset_name}```

  - Run Workflow
  
    ```sbatch workflow.sh /path/to/workflow_config_json_file```

    ***Note:*** For more details about the scripts, checkout the readme file in the scripts folder.


### TODO: Currently this file, is used to understand the folder structure and what are the script commands. Will be updated once everyone got familiar with the above.