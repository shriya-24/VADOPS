import evaluate
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import pipeline
from datasets import Dataset
import pandas as pd
from sklearn.metrics import classification_report

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cross_entropy

from pathlib import Path

# variables
L_Model = "roberta-base"
device = 'cuda:0'

tokenizer = AutoTokenizer.from_pretrained(L_Model)
accuracy = evaluate.load("accuracy")


def finetune(training_args: TrainingArguments, train_data: Dataset, eval_data: Dataset):
    """Fine tune the model with the training_argument passed and RoBERTa model

    Args:
        training_args (TrainingArguments): pass the training arguments with the parameters required for the dataset
        train_data (Dataset): dataset should have label column specifying the classification
        eval_data (Dataset): dataset should have label column specifying the classification
    """

    # train_data labels
    labels = train_data.features["label"].names
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
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()


def eval(test_data: Dataset, checkpoints_out_dir: str, predictions_out_dir: str):
    """Evaluating the model with test data passed and calculating the evaluation metrics for each class
    and saving it on the filePath passed

    Args:
        test_data (Dataset): dataset should have label column specifying the classification
        checkpoints_out_dir (str): specify the model checkpoint you wanna evaluate
        predictions_out_dir (str): Specify the filepath to save the predictions with the .csv extension to ensure OS understandability, as the pd.to_csv() method converts the data into a CSV format but does not automatically save it with the .csv file extension.
    """
    # variables
    pipeline_task = 'text-classification'

    classifier = pipeline(
        pipeline_task, model=checkpoints_out_dir, device=device)

    # predictions on the testing dataset
    predictions = classifier(test_data['text'], batch_size=16)

    # Convert predictions to a list of labels
    predicted_labels = [p['label'] for p in predictions]
    true_labels = [classifier.model.config.id2label[label]
                   for label in test_data['label']]

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


def calc_entropy_loss(test_data: Dataset, checkpoints_out_dir: str, entropy_analysis_path: str):
    """Calcualting the entropy loss for each sentence and save the analysis as csv format at specified
    entropy_analysis_path 

    Args:
        test_data (Dataset): dataset should have label column specifying the classification
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

    # tokenizing the test dataset
    test_encodings = tokenizer(test_data['text'], truncation=True, padding=True, return_tensors='pt')

    # create batches
    batch_size = 16
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_data["label"]))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    losses = []
    predicted_labels = []
    true_labels = []

    # calculating the cross entropy for each sentence
    for batch in test_dataloader:
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
    df = df.reset_index().rename(columns={'index': 'Test_Data_Index'})
    df.insert(df.columns.get_loc('Test_Data_Index') + 1, 'Text', [test_data['text'][i] for i in df['Test_Data_Index']])
    df.insert(df.columns.get_loc('Text') + 1, 'True Label', [model.config.id2label[l] for l in df['True_Label_Index']])
    df.insert(df.columns.get_loc('Predicted_Label_Index') + 1, 'Predicted Label', [model.config.id2label[l] for l in df['Predicted_Label_Index']])
    df.to_csv(entropy_analysis_path, index = False)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
