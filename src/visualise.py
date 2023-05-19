'''
Script for self-assigned Assignment 5, Language Analytics, Cultural Data Science, F2023

Visualise results (metrics and predictions) from finetuning 

@MinaAlmasi
'''

# utils
import pathlib

# data wrangling
import pandas as pd

# plotting
import matplotlib.pyplot as plt

# table 
#from tabulate import tabulate

# transformers
from transformers import AutoModelForSequenceClassification

def main(): 
    # import HF model
    model = AutoModelForSequenceClassification.from_pretrained("MinaAlmasi/ES-ENG-mBERT-sentiment")

    # define paths
    path = pathlib.Path(__file__)
    visualisationspath = path.parents[1] / "visualisations"

    # get log history from model 
    log_history = model.training_summary

    print(log_history)


if __name__ == "__main__":
    main()


