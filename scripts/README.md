## Description of scripts

model_script.sh: This script is a wrapper script which connects to the unity compute node, activates the conda environment, and runs the basic model run script

run.sh:This script runs the models present in `models` folder by python3

#### Instructions to use the scripts

model_script.sh
```
cd /path/to/repo
sbatch ./scripts/model_script.sh

squeue --me #to get the status of the batched job
```

