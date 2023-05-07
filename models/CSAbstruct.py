from urllib.request import urlretrieve
from pathlib import Path
import pandas as pd
import json

from datasets import load_dataset, DatasetDict, Dataset
from transformers import TrainingArguments
from main import finetune, eval, calc_entropy_loss
from sys import argv as args
import os

from datasets import DatasetDict, ClassLabel


# variables
L_model = "roberta-base"
dataset_name = 'CSAbstruct'
function_names = ['eval', 'finetune', 'download', 'calc_entropy_loss']
dataset_types = ["train", "validation", "test"]
CSAbstruct_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/CSAbstruct/'))

def download_data():
    """Downloading the CSabstruct dataset from github
    """
    
    os.makedirs(CSAbstruct_data_path)

    #https://github.com/allenai/sequential_sentence_classification/blob/master/data/CSAbstruct/dev.jsonl
    CSABSTRUCT_DATA_BASE_URL = (
        "https://raw.githubusercontent.com/allenai/sequential_sentence_classification/master/data/CSAbstruct/")

    for dataset_type in dataset_types:
        print(f"Downloading {dataset_type} data...")
        file_name = os.path.join(CSAbstruct_data_path, dataset_type + '.csv')

        if dataset_type == 'validation': # github repo has validation data as dev file name
            dataset_type = 'dev'
            
        tempFile, headers = urlretrieve(
            CSABSTRUCT_DATA_BASE_URL + dataset_type + '.jsonl')
        lines = Path(tempFile).read_text("utf-8").strip().splitlines()
        
        for line in lines:
            print(parse_line_json(line))
            break

        l = [parse_json(p) for p in [parse_line_json(line)
                          for line in lines] if p is not None]
        flat_list = []
        for sublist in l:
            for item in sublist:
                flat_list.append(item)
        df = pd.DataFrame(flat_list)
        
        df.to_csv(file_name, index=False)
        print("Saved at:", file_name)


def load_data() -> DatasetDict:
    """Loading CSAbstruct dataset from corresponding csv format

    Returns:
        DatasetDict: it contains train, validation, test datasets
    """
    # dataset_dict - containing the dataset type as key and value is dataset of that type
    dataset_dict = DatasetDict()
    # itreate each dataset type
    for dataset_type in dataset_types:
        file_name = os.path.join(CSAbstruct_data_path, dataset_type + '.csv') # file
        if not Path(file_name).exists():
            print(
                f'{dataset_type} data is not available. Tried to find at:', file_name)
            download_data()

        # load dataset
        ds_dict = load_dataset("csv", data_files = file_name)
        
        ds = ds_dict['train'] # train is the default value when we load the dataset from csv

        # casting label column
        ds = ds.cast_column('label', ClassLabel(names=ds.unique('label')))

        # appending to dataset_dict
        dataset_dict[dataset_type] = ds

    return dataset_dict


def parse_line_json(line: str):
    line_json = json.loads(line)
    return line_json

def parse_json(line:dict):
    result_list = []
    for i in range(len(line['sentences'])):
        result_list.append({"text":line['sentences'][i],"label":line['labels'][i],"confidence":line['confs'][i]})
    return result_list
        
    

if __name__ == "__main__":
    """
    Processing the argument parameters. Currently it requires 
    Args:
    1. function_name: there are two functions. 
      a. finetune - to train the model 
      b. eval - to evaluate the model
      c. calc_entropy_loss - to calculate the entropy loss for each sentence
    2. dataset_type: Options are "train", "validation", "test"
    3. checkpoints_out_dir: 
      a. for finetune function, it is used to save the checkpoint 
      b. for eval function, it is used to pick the model
    4. predictions_out_dir: 
      a. for finetune function - not required
      b. for eval function - file path is required, this is evaluation metrics for each class is saved
    5. entropy_analysis_path:
      a. for calc_entropy_loss - file path is required, this is where the entropy of each sentence is saved
    6. log_dynamics:
      a. for finetune function - Please specify whether you wanna log the dynamics or not. values are either True or False. By default it is False
    7. cartography_splits
      a. for finetune function - Specify the dataset split from ['train', 'validation'] to log dynamics. You can add multiple splits with comma seperated.
        Examples:
         1. train, validation 
         2. train
         3. validation
    8. log_dynamics_dir
      a. for finetune function - specify the dir path to store the dynamics
    """

    if len(args) < 2 or args[1] not in function_names:
        raise Exception("Please provide valid function_name argument")

    # assigning the arg values into variables
    function_name = args[1]

    if function_name == 'finetune':
        if len(args) < 3:
            raise Exception("Please provide checkpoints_out_dir argument")
        
        log_dynamics = False
        cartography_splits = []
        log_dynamics_dir = None
        
        # check for the optional arguments
        if len(args) >= 4:
            if args[3] not in ['False', 'True']:
                raise Exception("Please provide either True or False value for log_dynamics argument")
            
            log_dynamics = bool(args[3])

            # dynamics logging is enabled
            if log_dynamics:
                if len(args) < 5:
                    raise Exception("Please provide proper dataset split([train, validation]) value to log the dynamics")
            
                cartography_splits_text = args[4]
                splits_arr = cartography_splits_text.split(",")

                for split in splits_arr:
                    split = split.strip()
                    if split not in ['train', 'validation']:
                        raise Exception("Please provide proper dataset split([train, validation]) value to log the dynamics")
                    cartography_splits.append(split)
            
                if len(args) < 6:
                    raise Exception("Please specify log_dynamics_dir to save the dynamics")
                
                log_dynamics_dir = args[5]

        checkpoints_out_dir = args[2]

        # load dataset
        dataset = load_data()

        # log statements
        print("Finetuning model: START")
        print("dataset", dataset_name)
        print("checkpoints_out_dir", Path(checkpoints_out_dir).absolute(),
              '-This is where the Finetuning checkpoints will be saved')
        print('log_dynamics', log_dynamics)
        print('cartography_splits', cartography_splits)
        print("log_dynamics_dir", log_dynamics_dir)
        
        train_data, valid_data = dataset['train'], dataset['validation']

        training_args = TrainingArguments(
            output_dir=checkpoints_out_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=5,
            per_device_eval_batch_size=5,
            num_train_epochs=10,
            weight_decay=0.1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=50,
            save_total_limit = 2, # to limit saving checkpoints
            seed = 42 # set seed here
        )

        finetune(L_model, training_args, train_data, valid_data, log_dynamics=log_dynamics, cartography_splits=cartography_splits, log_dynamics_dir=log_dynamics_dir)
        print("Finetuning model: END")

    elif function_name == 'eval':
        if len(args) < 3 or args[2] not in dataset_types:
            raise Exception("Please provide valid dataset_type argument")
            
        if len(args) < 4:
            raise Exception("Please provide checkpoints_out_dir argument")
            
        if len(args) < 5:
            raise Exception("Please provide predictions_out_dir argument")

        dataset_type = args[2]
        checkpoints_out_dir = args[3]
        predictions_out_dir = args[4]

        # load dataset
        dataset = load_data()

        # log statements
        print("Evaluating model: START")
        print("dataset", dataset_name)
        print("checkpoints_out_dir", Path(checkpoints_out_dir).absolute())

        dataset = dataset[dataset_type]
        eval(dataset, checkpoints_out_dir, predictions_out_dir)

        print("evaluating model: END")

    elif function_name == 'download':
        print("Downloading dataset: START")
        download_data()
        print("Downloading dataset: END")
        
    elif function_name == 'calc_entropy_loss':
        if len(args) < 3 or args[2] not in dataset_types:
            raise Exception("Please provide valid dataset_type argument")
        
        if len(args) < 4:
            raise Exception("Please provide checkpoints_out_dir argument")

        if len(args) < 5:
            raise Exception("Please provide entropy_analysis_path argument")

        dataset_type = args[2]
        checkpoints_out_dir = args[3]
        entropy_analysis_path = args[4]

        # load dataset
        dataset = load_data()

        # log statements
        print("Calculating entropy loss: START")
        print("dataset", dataset_name)
        print("checkpoints_out_dir", Path(checkpoints_out_dir).absolute())

        dataset = dataset[dataset_type]
        calc_entropy_loss(dataset, checkpoints_out_dir, entropy_analysis_path)

        print("Calculating entropy loss: END")