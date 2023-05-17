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

# for compute_metrics function used during training
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# for plotting train and eval loss 
import matplotlib.pyplot as plt

# for evaluation, getting predictions
from sklearn.metrics import classification_report
import torch 
import pandas as pd 

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
    metrics_dict = {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    return metrics_dict 

def get_loss(trainer_history):
    train_loss = {}
    eval_loss = {}

    for item in trainer_history:
        epoch = item['epoch']
        if "loss" in item:
            train_loss[epoch] = item["loss"]
        if "eval_loss" in item:
            eval_loss[epoch] = item["eval_loss"]
        
    return train_loss, eval_loss

def plot_loss(train_loss, val_loss, epochs, savepath, filename): # adapted from class notebook
    '''
    '''

    # define theme 
    plt.style.use("seaborn-colorblind")

    # define figure size 
    plt.figure(figsize=(8,6))

    # create plot of train and validation loss, defined as two subplots on top of each other ! (but beside the accuracy plot)
    plt.plot(np.arange(1, epochs+1), train_loss.values(), label="Train Loss") # plot train loss 
    plt.plot(np.arange(1, epochs+1), val_loss.values(), label="Val Loss", linestyle=":") # plot val loss
    
    # text description on plot !!
    plt.title("Loss curve") 
    plt.xlabel("Epoch") 
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
   
    plt.savefig(savepath / filename, dpi=300)

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
        callbacks = [early_stop],
    )

    # train model
    trainer.train()

    return trainer, tokenized_data

def get_metrics(trainer, tokenized_ds_split, raw_ds_split, id2label:dict, path, filename):
    # make predictions 
    raw_predictions = trainer.predict(tokenized_ds_split)

    # extract prediction with highest probability (predicted labels)
    y_pred = np.argmax(raw_predictions.predictions, axis=-1)

    # get probabilities
    probabilities = torch.nn.Softmax(dim=-1)(torch.tensor(raw_predictions.predictions))

    # get true labels 
    y_true = raw_predictions.label_ids

    # get summary metrics 
    model_metrics = classification_report(y_true, y_pred, target_names=["neutral", "negative", "positive"])

    # get data 
    data = pd.DataFrame({"prediction":y_pred.flatten(),
                        "true_value":y_true.flatten(),
                        "text":raw_ds_split["text"],
                        "lang":raw_ds_split["lang"],
                        "probabilities": [str(val) for val in probabilities.detach().numpy()],
                        "logits": [str(val) for val in raw_predictions.predictions]
                        })
    
    # add prediction labels
    data["prediction_label"] = data["prediction"].map(id2label)
    data["true_label"] = data["true_value"].map(id2label)

    # save metrics
    metrics_path = path / "metrics"
    metrics_path.mkdir(exist_ok=True, parents=True)
    with open(metrics_path / f"{filename}_metrics.txt", "w") as file: 
        file.write(model_metrics)

    # save data
    predictions_path = path / "predictions"
    predictions_path.mkdir(exist_ok=True, parents=True)
    data.to_csv(predictions_path / f"{filename}_predictions.csv")

    return model_metrics, data


def get_metrics_per_language(trainer, tokenized_ds_split, raw_ds_split, id2label, path, filename, language):
    if language == "ES":
        subset_tokenized = tokenized_ds_split.filter(lambda example: example['lang'] != "ENG") # != to ENG, as the language data has several dialects of Spanish 
        subset_raw = tokenized_ds_split.filter(lambda example: example['lang'] != "ENG")       # (e.g., MX, ES) and we do not want to distinguish here!)

    elif language == "ENG":
         subset_tokenized = tokenized_ds_split.filter(lambda example: example['lang'] == "ENG") 
         subset_raw = tokenized_ds_split.filter(lambda example: example['lang'] == "ENG")     

    metrics, data = get_metrics(trainer, subset_tokenized, subset_raw, id2label, path, filename)

    return metrics, data 
