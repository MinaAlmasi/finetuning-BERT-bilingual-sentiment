'''
This script comprises several functions necessary for the finetune pipeline, including tokenization.

Fine-tune functions adapted / inspired by  https://huggingface.co/docs/transformers/v4.17.0/en/tasks/sequence_classification#finetune-with-trainer

@MinaAlmasi
'''

# utils
import numpy as np 
from functools import partial 


# transformers tokenizers, models 
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                        Trainer, DataCollatorWithPadding, EarlyStoppingCallback)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def tokenize(example, tokenizer, text_col:str="text"):
    return tokenizer(example[text_col], truncation=True)
    
def tokenize_dataset(dataset, tokenizer, text_col:str="text"): 
    # prepare tokenize func with multiple arguments to be passed to "map"
    tokenize_func = partial(tokenize, tokenizer=tokenizer, text_col=text_col)
    
    # tokenize entire dataset 
    tokenized_dataset = dataset.map(tokenize_func, batched=True)

    return tokenized_dataset

def compute_metrics(pred):
    # get labels 
    labels = pred.label_ids

    # get predictions
    preds = pred.predictions.argmax(-1)

    # calculate metrics 
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)

    # return dict 
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def finetune(dataset, model_name:str, n_labels:int, id2label:dict, label2id:dict, training_args, early_stop_patience:int=3): 
    # import tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # tokenize 
    tokenized_data = tokenize_dataset(dataset, tokenizer, "text")

    # define datacollator 
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # define earlystop
    early_stop = EarlyStoppingCallback(early_stopping_patience = early_stop_patience)

    # initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_labels, id2label=id2label, label2id=label2id
    )

    # initialize trainer 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"], 
        tokenizer=tokenizer, 
        data_collator=data_collator, 
        compute_metrics = compute_metrics, 
        callbacks = [early_stop]
    )

    # train model
    trainer.train()

    return trainer, tokenized_data

