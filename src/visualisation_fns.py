'''
Script for self-assigned Assignment 5, Language Analytics, Cultural Data Science, F2023

Contains functions for visualising results from finetuning.

Concretely, it has functions to 
    - plot confusion matrix for predictions from finetuning
    - create a table of metrics gathering all model metrics 

Note that the metrics functions were originally written for MinaAlmasi's visual analytics project: 
    https://github.com/MinaAlmasi/CIFAKE-image-classifiers/blob/main/src/modules/visualisation.py
They have been adapted (small modifications) to fit the current project.

@MinaAlmasi
'''

# utils
import pathlib
import re

# data wrangling
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# table 
from tabulate import tabulate

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

    # plot 
    sns.heatmap(confusion_matrix, annot = True, cmap = "Blues", fmt = "g")
    plt.xlabel("Predicted", fontsize = 14)
    plt.ylabel("Actual", fontsize = 14)

    # add title
    plt.title(f"Confusion matrix for {model_name} predictions", fontsize = 14)

    # save plot
    plt.savefig(savepath / f"confusion_matrix_{model_name}.png", dpi=300)

    return confusion_matrix

# the following functions are adapted from @MinaAlmasi's own code from visual analytics (https://github.com/MinaAlmasi/CIFAKE-image-classifiers/blob/main/src/modules/visualisation.py)
def create_data_from_metrics_txt(filepath:pathlib.Path):
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

def create_metrics_dataframes(resultspath:pathlib.Path, metrics_files_to_include:str):
    '''
    Loads all history objects from a given path and returns them in a dictionary.

    Args:
        - resultspath: path to directory containing txt files with metrics from scikit-learn's classification report
        - metrics_files_to_include: name of metrics.txt files to includes 
            e.g., "all_metrics" to only include the metrics.txt files with "all_metrics" in their name
            e.g., "eng_metrics" to only include the metrics.txt files with "eng_metrics" in their name
            e.g., "metrics" to include all metrics.txt files within the results path

    Returns: 
        - metrics_dfs: dictionary containing all txt files in path 
    '''

    # empty dictionary where dataframes will be saved
    metrics_dfs = {}

    # sort path in order "_all", "_eng", "_es" if all metrics are to be saved
    if metrics_files_to_include == "metrics":
        sorted_resultspath = sorted(resultspath.iterdir(), key=lambda x: (1 if '_all' in x.name else 2 if '_eng' in x.name else 3 if '_es' in x.name else 4, x.name))
    else: 
        sorted_resultspath = sorted(resultspath.iterdir())

    for file in sorted_resultspath: 
        if metrics_files_to_include in file.name: # only work on all files which have "metrics_name" in their name            # create dataframe from txt file 
            metrics_data = create_data_from_metrics_txt(resultspath/file.name)
            # define metrics name with regex (e.g., "REAL_LeNet_18e.txt" -> "REAL_LeNet")
            metrics_name = re.sub("_metrics.txt", "", file.name)
            # add to metrics_dfs dict ! 
            metrics_dfs[metrics_name] = metrics_data

    return metrics_dfs

def create_table(data:dict, header_labels:list, metric:str="f1-score"): 
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
        modelname = re.sub("_", " ", key)

        # create table row with model name and chosen metric
        tablerow = [modelname] + [str(value) for value in value[metric]] 

        # append tablerrow to tabledata
        tabledata.append(tablerow)

    # create table 
    table = tabulate(tabledata,
        headers = header_labels, 
        tablefmt="github"
    )

    return table