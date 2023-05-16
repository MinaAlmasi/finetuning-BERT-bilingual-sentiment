import matplotlib.pyplot as plt
import numpy as np
import torch 
import pandas as pd 
from sklearn import metrics

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

def get_metrics(trainer, tokenized_ds_split):
    # make predictions 
    predictions = trainer.predict(tokenized_ds_split)

    # extract prediction with highest probability (predicted labels)
    y_pred = np.argmax(predictions.predictions, axis=-1)

    # calculate probability instead of logits
    probabilities = torch.nn.Softmax(predictions)

    # get true labels 
    y_true = predictions.label_ids

    # get summary metrics 
    model_metrics = metrics.classification_report(y_true, y_pred, target_names=["neutral", "negative", "positive"])

    return model_metrics