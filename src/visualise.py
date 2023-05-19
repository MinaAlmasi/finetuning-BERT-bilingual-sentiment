'''
Script for self-assigned Assignment 5, Language Analytics, Cultural Data Science, F2023

Visualise results (metrics and predictions) from finetuning.

@MinaAlmasi
'''

# utils
import pathlib

# data load
import pandas as pd

# custom functions
from visualisation_fns import plot_confusion_matrix, create_metrics_dataframes, create_table

def main(): 
    # define paths
    path = pathlib.Path(__file__)
    visualisationspath = path.parents[1] / "visualisations"
    visualisationspath.mkdir(exist_ok = True, parents = True)

    # results path
    results_path = path.parents[1] / "results"

    # load predictions
    predictions_data = pd.read_csv(results_path / "predictions" /  "mBERT_all_predictions.csv")

    # make confusion matrix with percentages, round to 2 decimals
    confusion_matrix = pd.crosstab(predictions_data["true_label"], predictions_data["prediction_label"], normalize = "index").round(2)

    # define labels
    labels = ["Negative", "Neutral", "Positive"]

    # make confusion matrix 
    confusion_matrix = plot_confusion_matrix(predictions_data, "true_label", "prediction_label", labels, "mBERT", visualisationspath)

    # print confusion matrix
    print(confusion_matrix)

    # load metrics dataframes
    metrics_dfs = create_metrics_dataframes(results_path / "metrics", "metrics")

    # headers
    header_labels = metrics_dfs["mBERT_eng"]["class"].tolist()

    print(metrics_dfs.keys())

    # create table
    table = create_table(metrics_dfs, header_labels, metric="f1-score")
    
    #save table
    with open(visualisationspath / "all_models_metrics_table.txt", "w") as file:
        file.write(table)

if __name__ == "__main__":
    main()


