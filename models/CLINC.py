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
      a. train - to train the model 
      b. eval - to evaluate the mode
    2. dataset_config: CLINC has 3 subsets. so specify which subset you wanna pick. Options are a. small b. imbalanced c. plus
    3. dataset_type: Options are "train", "validation", "test"
    4. checkpoints_out_dir: 
      a. for train function, it is used to save the checkpoint 
      b. for eval function, it is used to pick the model
    5. predictions_out_dir: 
      a. for train function - not required
      b. for eval function - file path is required, this is evaluation metrics for each class is saved
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

        checkpoints_out_dir = args[3]
        # log statements
        print("Finetuning model: START")
        print("dataset", dataset_name)
        print("dataset_config", dataset_config)
        print("checkpoints_out_dir", Path(checkpoints_out_dir).absolute(),
              '-This is where the Finetuning checkpoints will be saved')
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

        finetune(training_args, train_data, valid_data)
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