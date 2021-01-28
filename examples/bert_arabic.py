# -*- coding: utf-8 -*-
"""mazen.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11yZaN0t_llrdQtr3i_JIm1pw1HwFS-3s
"""

import tensorflow as tf

# Get the GPU device name.
device_name = tf.test.gpu_device_name()

# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')

import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
    !nvidia-smi

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

!pip install optuna==2.3.0
!pip install transformers==4.1.1
!pip install farasapy
!pip install pyarabic
!git clone https://github.com/aub-mind/arabert

from arabert.preprocess import ArabertPreprocessor
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score , recall_score

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer
from transformers.data.processors import SingleSentenceClassificationProcessor
from transformers import Trainer , TrainingArguments
from transformers.trainer_utils import EvaluationStrategy
from transformers.data.processors.utils import InputFeatures
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.utils import resample
import logging
import torch
import optuna

import pandas as pd
def read_files(path):
    file = pd.read_csv(path, sep='\t', header=0)
    print ('The shape of the data: ', file.shape)
    return file

train = read_files('DA_train_labeled.tsv')
dev = read_files('DA_dev_labeled.tsv')
test = read_files('DA_test_unlabeled.tsv')

# Use train_test_split to split our data into train and validation sets for
# training
from sklearn.model_selection import train_test_split

# Use 90% for training and 10% for validation.
train_df, validation_df = train_test_split(train, random_state=2018, test_size=0.1)
len(train_df), len(validation_df)

class Dataset:
    def __init__(
        self,
        name,
        train,
        validation,
        label_list,
    ):
        self.name = name
        self.train = train
        self.validation = validation
        self.label_list = label_list

DATA_COLUMN = "#2_tweet"
LABEL_COLUMN = "#3_country_label"

all_datasets = []

# train data and train_labels
train_df = train_df[["#2_tweet","#3_country_label"]] 
train_df.columns = [DATA_COLUMN, LABEL_COLUMN]
train_LANG = train_df[DATA_COLUMN]
label_train_LANG = train_df[LABEL_COLUMN]

assert len(label_train_LANG.unique())

# val data and val_labels
validation_df = validation_df[["#2_tweet","#3_country_label"]] 
validation_df.columns = [DATA_COLUMN, LABEL_COLUMN]
validation_LANG = validation_df[DATA_COLUMN]
label_validation_LANG = validation_df[LABEL_COLUMN]

assert len(label_validation_LANG.unique())


data_LANG = Dataset("LANG", 
                    train = train_df, 
                    validation = validation_df, 
                    label_list = label_train_LANG)
all_datasets.append(data_LANG)




data_LANG

for x in all_datasets:
  print(x.name)

dataset_name = 'LANG'
model_name = 'aubmindlab/bert-base-arabertv02'
task_name = 'classification'
max_len = 256

for d in all_datasets:
  if d.name==dataset_name:
    selected_dataset = d
    print(selected_dataset)
    print('Dataset found')
    break

arabert_prep = ArabertPreprocessor(model_name.split("/")[-1])

selected_dataset.train[DATA_COLUMN] = selected_dataset.train[DATA_COLUMN].apply(lambda x:   arabert_prep.preprocess(x))
selected_dataset.validation[DATA_COLUMN] = selected_dataset.validation[DATA_COLUMN].apply(lambda x:   arabert_prep.preprocess(x))

class BERTDataset(Dataset):
    def __init__(self, text, target, model_name, max_len, label_map):
      super(BERTDataset).__init__()
      self.text = text
      self.target = target
      self.tokenizer_name = model_name
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.max_len = max_len
      self.label_map = label_map
      
    def __len__(self):
      return len(self.text)

    def __getitem__(self,item):
      text = str(self.text[item])
      text = " ".join(text.split())

      input_ids = self.tokenizer.encode(
          text,
          add_special_tokens=True,
          max_length=self.max_len,
          truncation='longest_first'
      )     
    
      attention_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      padding_length = self.max_len - len(input_ids)
      input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
      # target_item = self.target[item]
      # print(f"self.target[item]: {target_item}")

      # label = self.label_map.get(target_item, "UNK")
      # print(f"label: {label}")  

      attention_mask = attention_mask + ([0] * padding_length) 

      
      return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, 
                           label=self.label_map[self.target[item]])

label_map = { v:index for index, v in enumerate(sorted(set(selected_dataset.label_list))) }
print(label_map)

train_dataset = BERTDataset(text = selected_dataset.train[DATA_COLUMN].to_list(),
                            target = selected_dataset.train[LABEL_COLUMN].to_list(),
                            model_name = model_name,
                            max_len = max_len,
                            label_map = label_map)
validation_dataset = BERTDataset(text = selected_dataset.validation[DATA_COLUMN].to_list(),
                          target = selected_dataset.validation[LABEL_COLUMN].to_list(),
                          model_name = model_name,
                          max_len = max_len,
                          label_map = label_map)

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, return_dict=True, num_labels=len(label_map))

def compute_metrics(p): #p should be of type EvalPrediction
  preds = np.argmax(p.predictions, axis=1)
  assert len(preds) == len(p.label_ids)
  #print(classification_report(p.label_ids,preds))
  #print(confusion_matrix(p.label_ids,preds))

  macro_f1_pos_neg = f1_score(p.label_ids,preds,average='macro',labels=[0,1])
  macro_f1 = f1_score(p.label_ids,preds,average='macro')
  macro_precision = precision_score(p.label_ids,preds,average='macro')
  macro_recall = recall_score(p.label_ids,preds,average='macro')
  acc = accuracy_score(p.label_ids,preds)
  return {
      'macro_f1' : macro_f1,
      'macro_f1_pos_neg' : macro_f1_pos_neg,  
      'macro_precision': macro_precision,
      'macro_recall': macro_recall,
      'accuracy': acc
  }

training_args = TrainingArguments("./train")
training_args.evaluate_during_training = True
training_args.adam_epsilon = 1e-8
training_args.learning_rate = 5e-5
training_args.fp16 = True
training_args.per_device_train_batch_size = 16
training_args.per_device_eval_batch_size = 16
training_args.gradient_accumulation_steps = 2
training_args.num_train_epochs= 8


steps_per_epoch = (len(selected_dataset.train)// (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))
total_steps = steps_per_epoch * training_args.num_train_epochs
print(steps_per_epoch)
print(total_steps)
#Warmup_ratio
warmup_ratio = 0.1
training_args.warmup_steps = total_steps*warmup_ratio # or you can set the warmup steps directly 

training_args.evaluation_strategy = EvaluationStrategy.EPOCH
# training_args.logging_steps = 200
training_args.save_steps = 100000 #don't want to save any model, there is probably a better way to do this :)
training_args.seed = 42
training_args.disable_tqdm = False
training_args.lr_scheduler_type = 'cosine'

training_args.

import numpy as np

# Num steps in epoch = num training samples / batch size
steps_per_epoch = int(np.ceil(len(train_df) / float(training_args.per_device_train_batch_size)))

print('Each epoch will have {:,} steps.'.format(steps_per_epoch))

# Run evaluation periodically during training to monitor progress.
#training_args.evaluate_during_training = True

# "Print results from evaluation during training."
#training_args.evaluate_during_training_verbose = True

# "Perform evaluation at every specified number of steps. A checkpoint model and
#  the evaluation results will be saved."
#training_args.evaluate_during_training_steps = 120

# We only need to tokenize our validation set once, then we can read it from the
# cache.
#training_args.use_cached_eval_features = True

# Turn on early stopping.
training_args.use_early_stopping = True

# "The improvement over best_eval_loss necessary to count as a better checkpoint."
training_args.early_stopping_delta = 0.01

# What metric to use in calculating score for evaluation set (plus whether a low
# vs. high value is better for this metric).

#model.args.early_stopping_metric = "mcc"
#model.args.early_stopping_metric_minimize = False

training_args.early_stopping_metric = "eval_loss"
training_args.early_stopping_metric_minimize = True

# "Terminate training after this many evaluations without an improvement in the
#  evaluation metric greater then early_stopping_delta."
training_args.early_stopping_patience = 2

print('Training on {:,} samples...'.format(len(train_df)))

trainer = Trainer(
    model = model_init(),
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("Arabert_1")

print (model_name)

model_2 = AutoModelForSequenceClassification.from_pretrained("/content/Arabert_1",
                                                              return_dict=True, num_labels=len(label_map))

type(model_2)

s = "هو ده النهاردة"

arabert_prep.preprocess(s)

