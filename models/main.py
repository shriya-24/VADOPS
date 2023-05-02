import evaluate
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from transformers import DataCollatorWithPadding
from transformers import pipeline
from datasets import Dataset
import pandas as pd
import os
from sys import argv as args

from sklearn.metrics import classification_report

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cross_entropy

from training_dynamics import compute_train_dy_metrics, read_training_dynamics, plot_data_map

from pathlib import Path

# variables
L_Model = "roberta-base"
device = 'cuda:0'

tokenizer = AutoTokenizer.from_pretrained(L_Model)
accuracy = evaluate.load("accuracy")

function_names = ['plot']


def finetune(training_args: TrainingArguments, train_data: Dataset, eval_data: Dataset, log_training_dynamics = False, cartography_split = 'train', log_training_dynamics_dir = None):
    """Fine tune the model with the training_argument passed and RoBERTa model

    Args:
        training_args (TrainingArguments): pass the training arguments with the parameters required for the dataset
        train_data (Dataset): dataset should have label column specifying the classification
        eval_data (Dataset): dataset should have label column specifying the classification
    """

    # tokenizing the data
    train_data_encodings = train_data.map(preprocess_function, batched=True)
    eval_data_encodings = eval_data.map(preprocess_function, batched=True)
    
    # train_data labels
    labels = train_data_encodings.features["label"].names
    label2id = {labels[i]: i for i in range(len(labels))}
    id2label = {i: labels[i] for i in range(len(labels))}

    model = AutoModelForSequenceClassification.from_pretrained(
        L_Model, num_labels=len(labels), id2label=id2label, label2id=label2id, return_dict=True)

    # making computation to GPU
    if torch.cuda.is_available():
        model = model.to(device)

    # define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data_encodings,
        eval_dataset=eval_data_encodings,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    if log_training_dynamics:
        if not log_training_dynamics_dir:
            raise Exception("Please provide the directory path to save the training dynamics")
        
        if cartography_split not in ['train', 'validation','dev']:
            raise Exception("Please provide proper dataset split to log the training dynamics")
        
        dataset_1 = train_data if cartography_split == 'train' else eval_data
        trainer.add_callback(ExtendedTrainerCallback(model, dataset_1, log_training_dynamics_dir = log_training_dynamics_dir))

    # Train model
    trainer.train()


def eval(dataset: Dataset, checkpoints_out_dir: str, predictions_out_dir: str):
    """Evaluating the model with dataset passed and calculating the evaluation metrics for each class
    and saving it on the filePath passed

    Args:
        dataset (Dataset): dataset should have label column specifying the classification
        checkpoints_out_dir (str): specify the model checkpoint you wanna evaluate
        predictions_out_dir (str): Specify the filepath to save the predictions with the .csv extension to ensure OS understandability, as the pd.to_csv() method converts the data into a CSV format but does not automatically save it with the .csv file extension.
    """
    # variables
    pipeline_task = 'text-classification'

    classifier = pipeline(
        pipeline_task, model=checkpoints_out_dir, device=device)

    # predictions on the dataset
    predictions = classifier(dataset['text'], batch_size=16)

    # Convert predictions to a list of labels
    predicted_labels = [p['label'] for p in predictions]
    true_labels = [classifier.model.config.id2label[label]
                   for label in dataset['label']]

    # generating report
    report = classification_report(
        true_labels, predicted_labels, output_dict=True)

    # report has three root variables 1. accuracy 2. macro avg 3. weighted avg
    macro_avg_f1_score = report['macro avg']['f1-score']
    weighted_avg_f1_score = report['weighted avg']['f1-score']
    accuracy = report['accuracy']

    # logging off overall metrics
    print('Macro Average F1 score: {:.2f}'.format(macro_avg_f1_score))
    print('Weighted Average F1 score: {:.2f}'.format(weighted_avg_f1_score))
    print('Accuracy: {:.2f}%'.format(accuracy * 100))

    # saving the each class intent class evaluation metrics
    df = pd.DataFrame(report)
    df = df.transpose()
    df = df.reset_index().rename(columns={'index': 'label'})

    df = df[:-3]  # removing accuracy, macro avg, weighted avg from the report

    df.insert(df.columns.get_loc('label') + 1, 'label_index',
              [classifier.model.config.label2id[l] for l in df['label']])
    df_sorted = df.sort_values(by='f1-score')
    df_sorted.to_csv(predictions_out_dir, index=False)

    # logging
    print("Report for each class eval metrics is generated and saved at:",
          Path(predictions_out_dir).absolute())
    return


def calc_entropy_loss(dataset: Dataset, checkpoints_out_dir: str, entropy_analysis_path: str):
    """Calcualting the entropy loss for each sentence and save the analysis as csv format at specified
    entropy_analysis_path 

    Args:
        dataset (Dataset): dataset should have label column specifying the classification
        checkpoints_out_dir (str): specify the model checkpoint you wanna evaluate
        predictions_out_dir (str): Specify the filepath to save the cross entropy analysis with the .csv extension to ensure OS understandability, as the pd.to_csv() method converts the data into a CSV format but does not automatically save it with the .csv file extension.
    """
    # Load the tokenizer and the model from saved checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoints_out_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoints_out_dir)

    # set model to device
    model.to(device)

    # set the model to evaluation mode
    model.eval()

    # tokenizing the dataset
    data_encodings = tokenizer(dataset['text'], truncation=True, padding=True, return_tensors='pt')

    # create batches
    batch_size = 16
    tensor_dataset = TensorDataset(data_encodings['input_ids'], data_encodings['attention_mask'], torch.tensor(dataset["label"]))
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size)

    losses = []
    predicted_labels = []
    true_labels = []

    # calculating the cross entropy for each sentence
    for batch in dataloader:
        # Unpack the batch and move it to GPU
        input_ids, attention_mask, batch_true_labels = tuple(t.to(device) for t in batch)
        
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_predicted_labels = torch.argmax(logits, axis = 1)
            
            # calculate the entropy loss
            batch_loss = cross_entropy(logits, batch_true_labels, reduction='none')
            
            losses.extend(batch_loss.tolist())
            predicted_labels.extend(batch_predicted_labels.tolist())
            true_labels.extend(batch_true_labels.tolist())

    # Save calculated entropy loss as csv file
    df = pd.DataFrame([true_labels, predicted_labels, losses])
    df = df.transpose()
    df.columns = ['True_Label_Index', 'Predicted_Label_Index', 'Entropy Loss']
    df = df.reset_index().rename(columns={'index': 'Data_Index'})
    df.insert(df.columns.get_loc('Data_Index') + 1, 'Text', [dataset['text'][i] for i in df['Data_Index']])
    df.insert(df.columns.get_loc('Text') + 1, 'True Label', [model.config.id2label[l] for l in df['True_Label_Index']])
    df.insert(df.columns.get_loc('Predicted_Label_Index') + 1, 'Predicted Label', [model.config.id2label[l] for l in df['Predicted_Label_Index']])
    df.to_csv(entropy_analysis_path, index = False)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def plot(log_training_dynamics_dir, plot_dir, dataset_name):
    "plot data map for the saved training dynamics"
    training_dyn = read_training_dynamics(log_training_dynamics_dir)
    train_dy_metrics, _ = compute_train_dy_metrics(training_dyn)
    plot_data_map(train_dy_metrics, plot_dir, title=dataset_name, show_hist=True)


class ExtendedTrainerCallback(TrainerCallback): 
    "A callback that save the trianing dynamics of the dataset at every epoch"
    def __init__(self, model, dataset, log_training_dynamics_dir = None):
        super().__init__()
        self.log_training_dynamics_dir = log_training_dynamics_dir
        self.model = model
        self.dataset = dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        print(f"Logging Training Dynamics for epoch:{epoch} START")
        
        # tokenizing the dataset
        data_encodings = tokenizer(self.dataset['text'], truncation=True, padding=True, return_tensors='pt')
        
        # create batches
        batch_size = 16
        tensor_dataset = TensorDataset(data_encodings['input_ids'], data_encodings['attention_mask'], torch.tensor(self.dataset["label"]))
        dataloader = DataLoader(tensor_dataset, batch_size=batch_size)

        dataset_logits = []
        gold_labels = []

        # calculating the cross entropy for each sentence
        for batch in dataloader:
            # Unpack the batch and move it to GPU
            input_ids, attention_mask, batch_true_labels = tuple(t.to(device) for t in batch)

            # forward pass
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                dataset_logits.extend(logits.tolist())
                gold_labels.extend(batch_true_labels.tolist())
                
        df = pd.DataFrame([gold_labels, dataset_logits])
        df = df.transpose()
        df.columns = ['gold', f"logits_epoch_{epoch}"]
        df = df.reset_index().rename(columns={'index': 'guid'})
        
        # Create directory for logging training dynamics, if it doesn't already exist.
        if not os.path.exists(self.log_training_dynamics_dir):
            os.makedirs(self.log_training_dynamics_dir)

        epoch_file_name = os.path.join(self.log_training_dynamics_dir, f"dynamics_epoch_{epoch}.jsonl")
        df.to_json(epoch_file_name, lines=True, orient="records")
            # logging
        print("Training dynamics has been saved at:", Path(epoch_file_name).absolute())
        print(f"Logging Training Dynamics for epoch:{epoch} END")


if __name__ == "__main__":
    """
    Processing the argument parameters. Currently it requires 
    Args:
    1. function_name: there are two functions. 
      a. plot - to plot the data map for the specified training dynamics folder 
    
    For plot function:
    2. log_training_dynamics_dir: specify the training dynamics folder path
    3. plot_dir: specify the plot_dir path to save the graph
    3. dataset_name: specify the dataset_name for which the training dynamics are generated
    """

    if len(args) < 2 or args[1] not in function_names:
        raise Exception("Please provide valid function_name argument")
    
    # assigning the arg values into variables
    function_name = args[1]

    if function_name == 'plot':
        if len(args) < 3:
            raise Exception("Please provide log_training_dynamics_dir path for which to plot")
        
        if len(args) < 4:
            raise Exception("Please provide plot_dir path to save the graph")
    
        if len(args) < 5:
            raise Exception("Please provide dataset_name for which training dynamics are generated")
        
        log_training_dynamics_dir = args[2]
        plot_dir = args[3]
        dataset_name = args[4]
        plot(log_training_dynamics_dir, plot_dir, dataset_name)
