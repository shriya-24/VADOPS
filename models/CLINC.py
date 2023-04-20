from datasets import load_dataset
from transformers import TrainingArguments
from main import finetune, eval, preprocess_function, calc_entropy_loss
from sys import argv as args
from pathlib import Path


# variables
dataset_name = 'clinc_oos'
function_names = ['eval', 'finetune', 'calc_entropy_loss']
dataset_types = ["train", "validation", "test"]
dataset_configs = ['small', 'imbalanced', 'plus']


if __name__ == "__main__":
    """
    Processing the argument parameters. Currently it requires 
    Args:
    1. function_name: there are two functions. 
      a. finetune - to train the model 
      b. eval - to evaluate the mode
      c. calc_entropy_loss - to calculate the entropy loss for each sentence
    2. dataset_config: CLINC has 3 subsets. so specify which subset you wanna pick. Options are a. small b. imbalanced c. plus
    3. dataset_type: Options are "train", "validation", "test"
    4. checkpoints_out_dir: 
      a. for finetune function, it is used to save the checkpoint 
      b. for eval function, it is used to pick the model
    5. predictions_out_dir: 
      a. for finetune function - not required
      b. for eval function - file path is required, this is evaluation metrics for each class is saved
    6. entropy_analysis_path:
      a. for calc_entropy_loss - file path is required, this is where the entropy of each sentence is saved
    7. log_training_dynamics:
      a. for finetune function - Please specify whether you wanna log the training dynamics or not. values are either True or False. By default it is False
    8. cartography_split
      a. for finetune function - Specify the dataset split from ['train', 'validation']. By default the value is train
    9. log_training_dynamics_dir
      a. for finetune function - specify the dir path to store the training dynamics
    """

    if len(args) < 2 or args[1] not in function_names:
        raise Exception("Please provide valid function_name argument")

    elif len(args) < 3 or args[2] not in dataset_configs:
        raise Exception("Please provide proper dataset_config argument")

    # assigning the arg values into variables
    function_name = args[1]
    dataset_config = args[2]

    # load dataset
    dataset = load_dataset(dataset_name, dataset_config)
    dataset = dataset.rename_column("intent", "label")

    if function_name == 'finetune':
        if len(args) < 4:
            raise Exception("Please provide checkpoints_out_dir argument")
        
        log_training_dynamics = False
        cartography_split = 'train'
        log_training_dynamics_dir = None
        
        # check for the optional arguments
        if len(args) >= 5:
            if args[4] not in ['False', 'True']:
                raise Exception("Please provide either True or False value for log_training_dynamics argument")
            
            log_training_dynamics = bool(args[4])

            # training dynamics logging is enabled
            if log_training_dynamics:
                if (len(args) < 6 or args[5] not in ['train', 'validation']):
                    raise Exception("Please provide proper dataset split([train, validation]) value to log the training dynamics")
            
                cartography_split = args[5]
            
                if len(args) < 7:
                    raise Exception("Please specify log_training_dynamics_dir to save the training dynamics")
                
                log_training_dynamics_dir = args[6]

        checkpoints_out_dir = args[3]
        # log statements
        print("Finetuning model: START")
        print("dataset", dataset_name)
        print("dataset_config", dataset_config)
        print("checkpoints_out_dir", Path(checkpoints_out_dir).absolute(),
              '-This is where the Finetuning checkpoints will be saved')
        print('log_training_dynamics', log_training_dynamics)
        print('cartography_split', cartography_split)
        print("log_training_dynamics_dir", log_training_dynamics_dir)

        train_data, valid_data = dataset['train'], dataset['validation']

        # initialising the training_args
        training_args = TrainingArguments(
            output_dir=checkpoints_out_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=5,
            per_device_eval_batch_size=5,
            num_train_epochs=10,
            weight_decay=0.1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )

        finetune(training_args, train_data, valid_data, log_training_dynamics=log_training_dynamics, cartography_split=cartography_split, log_training_dynamics_dir=log_training_dynamics_dir)
        print("Training model: END")

    elif function_name == 'eval':
        if len(args) < 4 or args[3] not in dataset_types:
            raise Exception("Please provide valid dataset_type argument")
         
        if len(args) < 5:
            raise Exception("Please provide checkpoints_out_dir argument")

        if len(args) < 6:
            raise Exception("Please provide predictions_out_dir argument")

        dataset_type = args[3]
        checkpoints_out_dir = args[4]
        predictions_out_dir = args[5]

        # log statements
        print("Evaluating model: START")
        print("dataset", dataset_name)
        print("dataset_config", dataset_config)
        print("checkpoints_out_dir", Path(checkpoints_out_dir).absolute())

        dataset = dataset[dataset_type]
        eval(dataset, checkpoints_out_dir, predictions_out_dir)

        print("evaluating model: END")

    elif function_name == 'calc_entropy_loss':
        if len(args) < 4 or args[3] not in dataset_types:
            raise Exception("Please provide valid dataset_type argument")
        
        if len(args) < 5:
            raise Exception("Please provide checkpoints_out_dir argument")

        if len(args) < 6:
            raise Exception("Please provide entropy_analysis_path argument")

        dataset_type = args[3]
        checkpoints_out_dir = args[4]
        entropy_analysis_path = args[5]

        # log statements
        print("Calculating entropy loss: START")
        print("dataset", dataset_name)
        print("dataset_config", dataset_config)
        print("checkpoints_out_dir", Path(checkpoints_out_dir).absolute())

        dataset = dataset[dataset_type]
        calc_entropy_loss(dataset, checkpoints_out_dir, entropy_analysis_path)

        print("Calculating entropy loss: END")   