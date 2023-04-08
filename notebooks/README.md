# Notebooks
We use this directory for jupyter notebooks.

# About:
## BERT_SNIPS.ipynb
- TODO

## BERT.ipynb
- TODO

## finetune_clinc.ipynb

- deals with fine tuning the RoBERTa model with CLINC dataset and saves finetuned model checkpoints to the specified `checkpoints_out_dir` variable value 

- Based on your requirements, make changes to the variables:  `checkpoints_out_dir, dataset_subset`
  - check `dataset_subset` variable - this specifies which dataset subset you wanna use from 'CLINC'. and options are ['small', 'imbalanced', 'plus']
  - check `checkpoints_out_dir`variable - this where the checkpoints are saved.
  - models are trained using GPU. check whether your system is accessible to GPU or not.

## eval_clinc.ipynb
- calculates the evaluation(accuracy and F1 score) of finetuned model on testing dataset.
- calculates the evalutation metrics(accuracy, precision, F1-score) of each class label.

- Based on your requirements, make changes to the variables:  `checkpoints_out_dir, dataset_subset, predictions_out_dir`
  - check `dataset_subset` variable - this specifies which dataset subset you wanna use from 'CLINC'. and options are ['small', 'imbalanced', 'plus']
  - check `checkpoints_out_dir` variable - mention of checkpoint you wanna use to load the model
  - check `predictions_out_dir` variable - this where the evalutation metrics of each class label are saved in csv format.(make sure the directory exists)
- **Note:**  For evaluation, the dataset_subset should be the same with dataset_subset the checkpoint model is finetuned.


## calc_entropy_loss_clinc.ipynb
- calculates entropy loss for each sentence for the test data and saves the predicted label, entropy loss in the specified  `entropy_analysis_path`

- Based on your requirements, make changes to the variables:  `checkpoints_out_dir, dataset_subset, entropy_analysis_path`
  - check `dataset_subset` variable - this specifies which dataset subset you wanna use from 'CLINC'. and options are ['small', 'imbalanced', 'plus']
  - check `checkpoints_out_dir` variable - mention of checkpoint you wanna use to load the model
  - check `entropy_analysis_path` variable - this where the entropy loss, predicted label for each sentence are saved in csv format.(make sure the directory exists)
- **Note:**  For calculation of entropy, the dataset_subset should be the same with dataset_subset the checkpoint model is finetuned.




