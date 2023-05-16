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
                        Trainer, DataCollatorWithPadding)

# evaluation
import evaluate 


def tokenize(example, tokenizer, text_col:str="text"):
    return tokenizer(example[text_col], truncation=True)
    
def tokenize_dataset(dataset, tokenizer, text_col:str="text"): 
    # prepare tokenize func with multiple arguments to be passed to "map"
    tokenize_func = partial(tokenize, tokenizer=tokenizer, text_col=text_col)
    
    # tokenize entire dataset 
    tokenized_dataset = dataset.map(tokenize_func, batched=True)

    return tokenized_dataset


def compute_metrics(eval_pred):
    # load accuracy metric 
    accuracy = evaluate.load("accuracy")

    # compute predictions 
    predictions, labels = eval_pred

    # choose prediction with highest probability
    predictions = np.argmax(predictions, axis=1)

    # compute accuracy 
    accuracy = accuracy.compute(predictions=predictions, references=labels)

    return accuracy 

def finetune(dataset, model_name:str, n_labels:int, id2label:dict, label2id:dict, training_args): 
    # import tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # tokenize 
    tokenized_data = tokenize_dataset(dataset, tokenizer, "text")

    # define datacollator 
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
        compute_metrics = compute_metrics
    )

    # train model
    trainer.train()

    return trainer 

