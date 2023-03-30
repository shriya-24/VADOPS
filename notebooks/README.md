# What each notebook about

## Datasets we use in notebooks:

### SNIPS dataset:
- TODO


### CLINC dataset:
The dataset is available in Hugging Face and for more details - link
Dataset contains 150 intent classes over 10 domains + one intent class labeled oos(out of scope)

The dataset comes in different subsets:
small : Small, in which there are only 50 training queries per each in-scope intent
imbalanced : Imbalanced, in which intents have either 25, 50, 75, or 100 training queries.
plus: OOS+, in which there are 250 out-of-scope training examples, rather than 100.


## Notebooks:
## BERT_SNIPS.ipynb
- TODO

## BERT.ipynb
- TODO

## CLINC.ipnb
- deals with fine tuning the RoBERTa model with CLINC dataset and saves the training model checkpoints at `model/Roberta_clinc_{dataset_subset}`

Things to do before you train the model using this notebook
- check `dataset_subset` variable - this specifies which data subset you wanna use from 'CLINC'
- models are trained using GPU. check whether your system is accessible to GPU or not.
- check `output_dir`variable - this where the checkpoints are saved.

## eval.ipnb
- calcualtes the evaluation(accuracy and F1 score) of trained model on testing dataset.
- calculates the evalutation metrics(accuracy, precision, F1-score) of each class label.


Things to do before you evaluate a model using this notebook
- check `dataset_subset` variable - this specifies which data subset you wanna use from 'CLINC'
- check `checkpoint` variable - mention of checkpoint you wanna use to load the model
- check `analysis_dir` variable - this where the evalutation metrics of each class label are saved in csv format.(make sure the directory exists)

- **Note:** Use the same subset of CLINC on which the model is trained-on for evaluation.





