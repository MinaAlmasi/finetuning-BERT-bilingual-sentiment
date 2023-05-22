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
from modules.visualisation_fns import (create_metrics_dataframes, create_table,
                                        load_model_histories, plot_model_histories,
                                        plot_confusion_matrices)

def main(): 
    # define paths
    path = pathlib.Path(__file__)
    visualisationspath = path.parents[1] / "visualisations"
    visualisationspath.mkdir(exist_ok = True, parents = True)

    # results path
    resultspath = path.parents[1] / "results"

    # load model histories
    model_histories = load_model_histories(resultspath)

    # plot model histories
    plot_model_histories(model_histories, visualisationspath / "loss_curves_all_models.png")

    # metrics, create dataframes for each model, include all metrics files ("all", "eng", "es")
    metrics_dfs = create_metrics_dataframes(resultspath / "metrics", metrics_to_include="metrics")

    # define headers for table
    header_labels = metrics_dfs["mBERT_eng"]["class"].tolist()

    # create table
    table = create_table(metrics_dfs, header_labels, metric="f1-score")
    
    # save table
    with open(visualisationspath / "all_models_metrics_table.txt", "w") as file:
        file.write(table)

    # plot confusion matrices for each model
    matrices = plot_confusion_matrices(resultspath / "predictions", visualisationspath, preds_to_include="all_predictions")

if __name__ == "__main__":
    main()


