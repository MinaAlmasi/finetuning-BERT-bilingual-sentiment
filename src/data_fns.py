'''
Script for self-assigned Assignment 5, Language Analytics, Cultural Data Science, F2023

Script containing functions for loading and preprocessing data used in the finetune pipeline.
Data is loaded either from the HuggingFace datasets library or from a local path.

@MinaAlmasi
'''

# utils
import pathlib 
import re

# data utils 
import pandas as pd 
import numpy as np
import datasets 

from sklearn.model_selection import train_test_split

def combine_datasets(datasets_list):
    '''
    Take a list of datasets and combine them into one HuggingFace (HF) dataset. 

    Args:   
        - datasets_list: lists of HF datasets 

    Return: 
        - ds: HF dataset
    '''

    data = {}

    # loop over dataset dict
    for split in ["train", "validation", "test"]:
        # concatenate each split with each other, append to list  
        split_datasets = [d[split] for d in datasets_list]
        data[split] = datasets.concatenate_datasets(split_datasets)

    # make into dataset 
    ds = datasets.DatasetDict(data)

    return ds 

def preprocess_TASS(path:pathlib.Path):
    '''
    Load and preprocess the TASS 2020 dataset task 1 train/dev set as Pandas dataframes.
    (LINK: http://tass.sepln.org/2020/?wpdmpro=task-1-train-and-dev-sets)

    Args: 
        - path: subdirectory containing the "dev" and "test" folder from TASS 2020. 

    Returns: 
        - dfs: dictionary containing a dataframe for each split (train and dev)
    '''
    # define paths
    train_path = path / "train"
    dev_path = path / "dev"

    dfs = {}

    # create PANDAs dataframes 
    for path in [train_path, dev_path]:
        temp = []
        for file in path.iterdir():
            if file.is_file():
                df = pd.read_csv(train_path / file, sep = "\t", names=["id", "text", "label_text"])
                # extract name (go from es.tsv -> "ES")
                df["lang"] = file.name.split(".")[0].upper()

                # rm user names with str replace 
                df['text'] = df['text'].apply(lambda x: re.sub(r'@\w+', '@user', x))

                #  add int labels corresponding to other dataset
                int_labels = {"NEU":"negative", "N":"neutral", "P":"positive"}
                df["label"] = df["label_text"].astype(str).map(int_labels)

                # reorder, select only relevant cols 
                df = df[["text", "label", "lang"]]

                # append to temp list 
                temp.append(df)
        
        # concatenate
        dfs[path.name] = pd.concat(temp, ignore_index=True)

    return dfs 

def convert_TASS(dfs):
    '''
    Convert the TASS 2020 Pandas dataframes (dfs["train"] and dfs["dev"]) into a HF dataset dictionary with splits "train", "validation", "test".
    (LINK: http://tass.sepln.org/2020/?wpdmpro=task-1-train-and-dev-sets)

    Args: 
        - dfs: dictionary containing a dataframe for each split (train and dev)

    Returns: 
        - ds: HF dataset dictionary containing splits "train", "validation" and "test". 
    '''
    # split dev into val and test, stratify by language, so that we do not get an overrepresentation of e.g., "Mexican" (MX) Spanish. 
    dfs["validation"], dfs["test"] = train_test_split(dfs["dev"], test_size=0.50, stratify=dfs["dev"]["lang"], random_state=155)
    
    # rm dev
    del dfs["dev"]
    
    # drop indices 
    dfs["validation"] = dfs["validation"].reset_index(drop=True)
    dfs["test"] = dfs["test"].reset_index(drop=True)

    # convert into huggingface datasets dict
    ds = {}

    for split in dfs: 
        ds[split] = datasets.Dataset.from_pandas(dfs[split])
        ds[split]= ds[split].class_encode_column("label") #https://discuss.huggingface.co/t/class-labels-for-custom-datasets/15130

    ds = datasets.DatasetDict(ds)

    return ds

def load_TASS(path:pathlib.Path): 
    '''
    Return a preprocessed, HF dataset of the TASS 2020 dataset task 1 
    train/dev set (http://tass.sepln.org/2020/?wpdmpro=task-1-train-and-dev-sets)

    Args: 
        - path: subdirectory containing the "dev" and "test" folder from TASS 2020. 

    Returns: 
        - ds: HF dataset dictionary containing splits "train", "validation" and "test". 
    '''

    # load dfs 
    dfs = preprocess_TASS(path)
    
    # convert to DS
    ds = convert_TASS(dfs)

    return ds 

# utils for adding lang cols 
def add_eng_lang_col(example): 
    example["lang"] = "ENG"

    return example

def add_es_lang_col(example):
    example["lang"] = "ES"

    return example

def load_cardiff(download_mode=None): # https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual/viewer/arabic/train
    '''
    Load English and Spanish twitter sentiment HF datasets by @cardiffnlp: 
    (LINK: https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual)

    Args:
        - download_mode: whether a cached dataset should be used (=None) or the dataset should be redownloaded (="force_redownload"). Defaults to None.

    Returns: 
        - cardiff_eng: dataset with English text 
        - cardiff_es: dataset with Spanish text 
    '''

    # load datasets 
    cardiff_es = datasets.load_dataset("cardiffnlp/tweet_sentiment_multilingual", "spanish", download_mode=download_mode)
    cardiff_eng = datasets.load_dataset("cardiffnlp/tweet_sentiment_multilingual", "english", download_mode=download_mode)

    # add columns
    cardiff_eng = cardiff_eng.map(add_eng_lang_col)
    cardiff_es = cardiff_es.map(add_es_lang_col)

    return cardiff_eng, cardiff_es


def load_mteb(download_mode): # https://huggingface.co/datasets/mteb/tweet_sentiment_extraction/viewer/mteb--tweet_sentiment_extraction/train?p=0 
    '''
    Load a subset of English Twiter HF dataset from MTEB.
    (LINK: https://huggingface.co/datasets/mteb/tweet_sentiment_extraction)

    Args:
        - download_mode: whether a cached dataset should be used (=None) or the dataset should be redownloaded (="force_redownload"). Defaults to None.

    Returns: 
        - data: subset of MTEB as HF dataset. 
    '''

    data = datasets.load_dataset("mteb/tweet_sentiment_extraction", "english", download_mode=download_mode)

    # subset train to match the amount of ES data in TASS 
    data["train"] = data["train"].shuffle(seed=155).select(range(4802)) # https://huggingface.co/learn/nlp-course/chapter5/3

    # subset test to match the amount of ES data in TASS 
    data["test"] = data["test"].shuffle(seed=155).select(range(2443)) 

    # split test into val and test 
    test_validation = data['test'].train_test_split(test_size=0.5) # solution by https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090

    data = datasets.DatasetDict({
        "train": data["train"],
        "validation": test_validation["train"],
        "test":  test_validation["train"]
        })

    # remove columns 
    data = data.remove_columns(["id", "label"])

    # add lang col
    data = data.map(add_eng_lang_col)

    # encode label_text as label col, rename to 'label' to follow other datasets
    data = data.class_encode_column("label_text")
    data = data.rename_column("label_text", "label")

    return data

def get_ds_overview(datadict):
    '''
    Get overview of lengths of rows in each split train, validation, test per dataset. 

    Args:   
        - datadict: dictionary of the names of datasets (keys) and the datasets objects (values).

    Returns: 
        - lengths_overview: Pandas dataframe with an overview of  n rows in each split train, validation, test per dataset in datadict. 
    '''
    all_lengths = []

    for name, ds in datadict.items():
        lengths = pd.DataFrame() 
        lengths["dataset"] = [name]

        for split in ["train", "validation", "test"]:
            length = len(ds[split])
            lengths[split] = [length]

        all_lengths.append(lengths)
    
    lengths_overview = pd.concat(all_lengths, ignore_index=True)

    return lengths_overview

def load_datasets(TASS_path:pathlib.Path=None, download_mode:str=None):
    '''
    Load all HF datasets as a combined dataset for finetuning. 

    Args: 
        - TASS_path: path to TASS data. If no path is specified, the TASS data will not be loaded. 
        - download_mode: cached HF datasets should be used (=None) or the HF datasets should be redownloaded (="force_redownload"). Defaults to None.

    Returns: 
        - combined_ds: HF dataset containing the TASS 2020 (Spanish), Cardiff (Spanish and English) and MTEB (English) dataset. 
    '''

    if TASS_path: 
        tass_es = load_TASS(TASS_path)
    else: 
        tass_es = None

    # load other datasets 
    cardiff_eng, cardiff_es = load_cardiff(download_mode)
    mteb_eng = load_mteb(download_mode)

    all_ds = {
        "tass_es": tass_es,
        "mteb_eng": mteb_eng,
        "cardiff_es": cardiff_es,
        "cardiff_eng": cardiff_eng,
    }

    if not TASS_path: 
        del all_ds["tass_es"]
    
    # combine datasets 
    combined_ds = combine_datasets(all_ds.values())    
    
    # get overview of lengths in ds 
    ds_lengths = get_ds_overview(all_ds)

    return combined_ds, ds_lengths 


def main():
    path = pathlib.Path(__file__)
    tass_path = path.parents[1] / "data"

    ds, ds_overview = load_datasets(tass_path, download_mode="force_redownload")
    print(ds)
    print(ds_overview)

if __name__ == "__main__":
    main()

