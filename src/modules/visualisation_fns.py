'''
Script for self-assigned Assignment 5, Language Analytics, Cultural Data Science, F2023

Contains functions for visualising results from finetuning.

The script features functions which serve three purposes in src/visualise.py
    1. Visualises loss curves for all models in one plot!
    2. create a table of metrics gathering all model metrics 
    3. plot confusion matrix for predictions from finetuning

Note some functions were originally written for a previous project by the same author for another course in cultural data science. 
    https://github.com/MinaAlmasi/CIFAKE-image-classifiers/blob/main/src/modules/visualisation.py
They have been adapted (small modifications) to fit the current project. The functions in question are marked with a comment.

@MinaAlmasi
'''

# utils
import pathlib
import re
import numpy as np
import pickle

# data wrangling
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# custom func for getting loss from model history used in fine-tune pipeline
from modules.finetune_fns import get_loss

# table 
from tabulate import tabulate

def load_model_histories(resultspath:pathlib.Path): # function adapted from previous project by @MinaAlmasi (see script docstring)
    '''
    Loads all history objects from a given path and returns them in a dictionary.

    Args:
        - resultspath: path to directory containing history objects
    
    Returns: 
        - history_objects: dictionary containing all history objects in path 
    '''
    # define empty dictionary where history objects will be saved 
    history_objects = {}

    for file in resultspath.iterdir():
        if "history" in file.name: # open all files which have "history" in their name
            with open(resultspath / file.name , 'rb') as f:
                # load history object 
                history = pickle.load(f)
                # define history object name (e.g., "mBERT_log_history.pkl" -> "mBERT")
                history_name = re.sub("_log_history.pkl", "", file.name)
                # add to history_objects dict ! 
                history_objects[history_name] = history
    
    return history_objects 

def record_best_model(model_history:dict): 
    '''
    Record best model from model history as this is the model that is saved
    and used for predictions as defined with early stopping and in training arguments.

    Args:
        - model_history: history object for the model

    Returns:
        - highest_accuracy: highest accuracy achieved by model
        - highest_epoch: epoch at which highest accuracy was achieved
    '''
    highest_accuracy = 0
    highest_epoch = None

    # loop over model_history, extract the highest validation accuracy and the epoch it was achieved
    for item in model_history:
        if 'eval_accuracy' in item: 
            accuracy = item['eval_accuracy']
            epoch = item['epoch']

            if accuracy > highest_accuracy: # check if the accuracy extracted is higher than the current 'highest_accuracy'. If yes, make it the new accuracy 
                highest_accuracy = accuracy
                highest_epoch = epoch

    # round accuracy val
    highest_accuracy = round(highest_accuracy, 3)
    
    return highest_accuracy, highest_epoch

def plot_model_histories(model_histories:dict, savepath:pathlib.Path):
    '''
    Plot model loss curve from model histories in one plot. Also marks the epoch with highest validation accuracy as a vertical line. 

    Args: 
        - model_histories: dictionary containing model histories
        - savepath: path to save plot to
    '''
    # get loss and total epochs
    loss_dict = {}
    eval_loss = {}
    total_epochs = {}

    for model_name, model_vals in model_histories.items():
        loss_dict[model_name], eval_loss[model_name], total_epochs[model_name] = get_loss(model_vals)

    # load highest accuracy and epoch for each model
    highest_accuracies = {}
    highest_epochs = {}

    for model_name, model_vals in model_histories.items():
        highest_accuracies[model_name], highest_epochs[model_name] = record_best_model(model_vals)

    # define theme
    plt.style.use("seaborn-v0_8-colorblind")

    # plot loss with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)

    # iterate over model names in model_histories, plot train/val loss and mark the epoch with highest validation accuracy
    for i, model_name in enumerate(sorted(model_histories.keys())):
        # get epochs range (from 1 to total epochs)
        epochs_range = np.arange(1, total_epochs[model_name]+1)

        # plot train loss and validation loss 
        axes[i].plot(epochs_range, loss_dict[model_name].values())
        axes[i].plot(epochs_range, eval_loss[model_name].values())

        # add line for highest accuracy
        axes[i].axvline(x = highest_epochs[model_name], linestyle = "--", color = "black")

        # set only the legend for the highest accuracy in the individual subplots (as indicated by the "_", "_")
        axes[i].legend(loc = "upper right", labels=["_", "_", f"VAL ACC: {highest_accuracies[model_name]}, EPOCH: {highest_epochs[model_name]}"], fontsize = 12)

        # set title
        axes[i].set_title(f"{model_name}", fontsize = 14, fontweight = "bold")

    # set fig legend, here only set train and validation loss
    fig.legend(bbox_to_anchor=(0.95, 0.5), loc="center", labels=["TRAIN", "VAL"], prop={'weight': 'bold', 'size': 14})

    # pad plot
    fig.tight_layout(pad=4)

    # set fig axis x-label
    fig.text(0.52, 0.06, 'Epochs', ha='center', va='center', fontsize = 14)

    # save plot
    plt.savefig(savepath, dpi = 300)

def create_data_from_metrics_txt(filepath:pathlib.Path): # function adapted from previous project by @MinaAlmasi (see script docstring)
    '''
    Create a dataframe from a text file containing the classification report from sklearn.metrics.classification_report

    Args:
        - filepath: path to text file

    Returns: 
        - data: dataframe containing the classification report 
    '''

    data = pd.read_csv(filepath)

    # replace macro avg and weighted avg with macro_avg and weighted_avg
    data.iloc[:,0]= data.iloc[:,0].str.replace(r'(macro|weighted)\savg', r'\1_avg', regex=True)

    # split the columns by whitespace
    data = data.iloc[:,0].str.split(expand=True)

    # define new column names 
    new_cols = ['class', 'precision', 'recall', 'f1-score', 'support']
    data.columns = new_cols

    # identify the row with the accuracy score 
    is_accuracy = data['class'] == 'accuracy'

    # move the accuracy row values into the precision and recall columns (they are placed incorrectly when the columns are split)
    data.loc[is_accuracy, ['f1-score', 'support']] = data.loc[is_accuracy, ['precision', 'recall']].values

    # set precision and recall to None for the accuracy row
    data.loc[is_accuracy, ['precision', 'recall']] = None

    return data

def create_metrics_dataframes(resultspath:pathlib.Path, metrics_to_include:str): # function adapted from previous project by @MinaAlmasi (see script docstring)
    '''
    Loads all history objects from a given path and returns them in a dictionary.

    Args:
        - resultspath: path to directory containing txt files with metrics from scikit-learn's classification report
        - metrics_to_include: name of metrics.txt files to includes 
            e.g., "all_metrics" to only include the metrics.txt files with "all_metrics" in their name
            e.g., "eng_metrics" to only include the metrics.txt files with "eng_metrics" in their name
            e.g., "metrics" to include all metrics.txt files within the results path

    Returns: 
        - metrics_dfs: dictionary containing all txt files in path 
    '''

    # empty dictionary where dataframes will be saved
    metrics_dfs = {}

    # sort path in order "_all", "_eng", "_es" if all metrics are to be saved
    if metrics_to_include == "metrics":
        sorted_resultspath = sorted(resultspath.iterdir(), key=lambda x: (1 if '_all' in x.name else 2 if '_eng' in x.name else 3 if '_es' in x.name else 4, x.name))
    else: 
        sorted_resultspath = sorted(resultspath.iterdir())

    for file in sorted_resultspath: 
        if metrics_to_include in file.name: # only work on all files which have "metrics_name" in their name            # create dataframe from txt file 
            metrics_data = create_data_from_metrics_txt(resultspath/file.name)
            # define metrics name with regex (e.g., "REAL_LeNet_18e.txt" -> "REAL_LeNet")
            metrics_name = re.sub("_metrics.txt", "", file.name)
            # add to metrics_dfs dict ! 
            metrics_dfs[metrics_name] = metrics_data

    return metrics_dfs

def create_table(data:dict, header_labels:list, metric:str="f1-score"): # function originally written for previous project by same author (see script docstring)
    '''
    Create table from dictionary with dataframes created from create_metrics_dataframes.

    Args: 
        - data: dictionary of dataframes, one for each model run
        - header_labels: list of header labels
        - metric: "f1-score", "precision" or "recall". Note that the f1-score metric also includes an accuracy column.

    Returns: 
        - table: table in markdown format for github README.
    '''

    # Capitalize header_labels
    header_labels = [header.title() for header in header_labels]

    # define empty list for nicely formatted table data
    tabledata = []

    for key, value in data.items():
        # create name 
        model_name = re.sub("_", " ", key)

        # create table row with model name and chosen metric
        tablerow = [model_name] + [str(value) for value in value[metric]] 

        # append tablerrow to tabledata
        tabledata.append(tablerow)

    # create table 
    table = tabulate(tabledata,
        headers = header_labels, 
        tablefmt="github"
    )

    return table

def plot_confusion_matrix(pred_data, true_col:str, pred_col:str, labels:list, model_name:str, savepath:pathlib.Path):
    '''
    Plot confusion matrix for predictions from finetuning. 

    Args: 
        - pred_data: pandas dataframe containing predictions from finetuning 
        - true_col: name of column containing true labels
        - pred_col: name of column containing predicted labels
        - labels: list of labels 
        - model_name: name of model used for finetuning
        - savepath: path to save plot to

    Returns: 
        - confusion_matrix: pd.Dataframe containing confusion matrix

    Outputs:
        - confusion_matrix: .png plot of confusion matrix
    '''
    # make confusion matrix with percentages, round to 2 decimals
    confusion_matrix = pd.crosstab(pred_data[true_col], pred_data[pred_col], normalize = "index").round(2)

    # make labels more readable
    confusion_matrix.index = labels
    confusion_matrix.columns = labels
    
    # define plot 
    fig = plt.figure(figsize = (10, 8))

    # plot 
    sns.heatmap(confusion_matrix, annot = True, cmap = "Blues", fmt = "g")
    plt.xlabel("Predicted", fontsize = 14)
    plt.ylabel("Actual", fontsize = 14)

    # add title
    plt.title(f"Confusion matrix for {model_name} predictions", fontsize = 14)

    # save plot
    fig.savefig(savepath / f"confusion_matrix_{model_name}.png", dpi=300)

    return confusion_matrix

def plot_confusion_matrices(path:pathlib.Path, savepath:pathlib.Path, preds_to_include:str):
    '''
    Plot confusion matrices for all models in path

    Args: 
        - path: path to directory containing predictions from finetuning
        - savepath: path to save plot to
        - preds_to_include: string to filter predictions to include in plot 
            e.g., "all_predictions" to only include the predictions.csv with "all_predictions" in their name
            e.g., "eng_predictions" to only include the metrics.txt files with "eng_predictions" in their name
            e.g., "predictions" to include all predictions files within the results path

    Returns:
        - matrices: dictionary containing confusion matrices for all predictions in path

    Outputs:
        - confusion_matrix: .png plot of confusion matrix
    '''
    # define dictionary to store confusion matrices
    matrices = {}
    
    # define labels for confusion matrix
    labels = ["Negative", "Neutral", "Positive"]

    # get all prediction files
    pred_files = [file for file in path.iterdir() if preds_to_include in file.name]

    # loop over files and plot confusion matrix
    for file in pred_files:
        # get model name
        model_name = re.sub("_predictions.csv", "", file.name)

        # load predictions
        pred_data = pd.read_csv(file)

        # plot confusion matrix
        matrices[model_name] = plot_confusion_matrix(pred_data, "true_label", "prediction_label", labels, model_name, savepath)

    return matrices