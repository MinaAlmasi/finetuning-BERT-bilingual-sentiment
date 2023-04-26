'''
Fine-tuning BERT for bilingual emotion classification in English and Spanish

# https://huggingface.co/docs/transformers/v4.17.0/en/tasks/sequence_classification#finetune-with-trainer
'''

# utils 
import pathlib 
import numpy as np 
from functools import partial 

# dataset 
from datasets import load_dataset 

# transformers tokenizers, models 
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                        TrainingArguments, Trainer, DataCollatorWithPadding)

# evaluation
import evaluate 

# tokenisation functions
def tokenize(example, tokenizer, text_col:str="text"):
    return tokenizer(example[text_col], truncation=True)
    
def tokenize_dataset(dataset, tokenizer, text_col:str="text"): 
    # prepare tokenize func with multiple arguments to be passed to "map"
    tokenize_func = partial(tokenize, tokenizer=tokenizer, text_col=text_col)
    
    # tokenize entire dataset 
    tokenized_dataset = dataset.map(tokenize_func, batched=True)

    return tokenized_dataset

def compute_metrics(eval_pred):
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

    # load accuracy metric 
    accuracy = evaluate.load("accuracy")

    # initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_labels, id2label=id2label, label2id=label2id
    )

    # initialize trainer 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"], 
        tokenizer=tokenizer, 
        data_collator=data_collator, 
        compute_metrics = compute_metrics
    )

    # train model
    trainer.train()

    return trainer 


def main(): 
    # define model outpath
    path = pathlib.Path(__file__)
    modeloutpath = path.parents[1] / "models"
    modeloutpath.mkdir(exist_ok=True)

    # load data 
    data = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "english", "spanish")

    # map labels to ids
    id2label = {0: "negative", 1:"neutral", 2:"positive"}
    label2id = {"negative":0, "neutral":1, "positive":2}

    # define training arguments 
    training_args = TrainingArguments(
        output_dir = modeloutpath, 
        learning_rate=2e-5,
        per_device_train_batch_size = 16, 
        per_device_eval_batch_size = 16, 
        num_train_epochs=2, 
        weight_decay=0.01,
        evaluation_strategy="epoch", 
        save_strategy = "epoch", 
        load_best_model_at_end = True
    )

    finetuned_model = finetune(data, "bert-base-multilingual-cased", 3, id2label, label2id, training_args)


if __name__ == "__main__":
    main()