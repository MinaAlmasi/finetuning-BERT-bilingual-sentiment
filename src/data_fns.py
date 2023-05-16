# utils
import pathlib 
import re

# data utils 
import pandas as pd 
import numpy as np
import datasets 

from sklearn.model_selection import train_test_split

def combine_datasets(datasets_list):
    data = {}

    # loop over dataset dict
    for split in ["train", "validation", "test"]:
        # concatenate each split with each other, append to list  
        split_datasets = [d[split] for d in datasets_list]
        data[split] = datasets.concatenate_datasets(split_datasets)

    # make into dataset dict
    data = datasets.DatasetDict(data)

    return data 

def load_TASS(train_path:pathlib.Path, dev_path:pathlib.Path): 
    '''
    Function to load and preprocess the TASS 2020 dataset task 1 
    train/dev set (http://tass.sepln.org/2020/?wpdmpro=task-1-train-and-dev-sets)

    Args: 
        - train_path: subdirectory containing the .tsv files for the train set 
        - dev_path: subdirectory containing the .tsv files for the dev (validation) set 

    Returns: 
        - 
    '''
    dfs = {}

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

def convert_TASS_dataset(dfs):
    # split dev into val and test, stratify by language, so that we do not get an overrepresentation of e.g., "Mexican" (MX) Spanish. 
    dfs["validation"], dfs["test"] = train_test_split(dfs["dev"], test_size=0.5, stratify=dfs["dev"]["lang"], random_state=155)
    
    # rm dev
    del dfs["dev"]
    
    # drop indices 
    dfs["validation"] = dfs["validation"].reset_index(drop=True)
    dfs["test"] = dfs["test"].reset_index(drop=True)

    # convert into huggingface datasets dict
    data = {}

    for split in dfs: 
        data[split] = datasets.Dataset.from_pandas(dfs[split])
        data[split]= data[split].class_encode_column("label") #https://discuss.huggingface.co/t/class-labels-for-custom-datasets/15130

    data = datasets.DatasetDict(data)

    return data

# add lang cols to other dataset, 
def add_eng_lang_col(example): 
    example["lang"] = "ENG"

    return example

def add_es_lang_col(example):
    example["lang"] = "ES"

    return example

def load_cardiff(): # https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual/viewer/arabic/train
    # load datasets 
    cardiff_es = datasets.load_dataset("cardiffnlp/tweet_sentiment_multilingual", "spanish")
    cardiff_eng = datasets.load_dataset("cardiffnlp/tweet_sentiment_multilingual", "english")

    # add columns
    cardiff_eng = cardiff_eng.map(add_eng_lang_col)
    cardiff_es = cardiff_es.map(add_es_lang_col)

    return cardiff_eng, cardiff_es


def load_mteb(): # https://huggingface.co/datasets/mteb/tweet_sentiment_extraction/viewer/mteb--tweet_sentiment_extraction/train?p=0 
    data = datasets.load_dataset("mteb/tweet_sentiment_extraction", "english")

    # subset train
    data["train"] = data["train"].shuffle(seed=155).select(range(4802)) # https://huggingface.co/learn/nlp-course/chapter5/3

    # subset test
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

def get_overview(datadict):
    '''
    Get overview of lengths of rows in each split train, validation, test per combined dataset. 
    '''

    all_lengths = []

    for name, data in datadict.items():
        lengths = pd.DataFrame() 
        lengths["data"] = [name]

        for split in ["train", "validation", "test"]:
            length = len(data[split])
            lengths[split] = [length]

        all_lengths.append(lengths)
    
    final = pd.concat(all_lengths, ignore_index=True)

    return final

def load_datasets(TASS_path=None):
    if TASS_path: 
        train_path = TASS_path / "train"
        dev_path = TASS_path / "dev"

        tass_es = load_TASS(train_path, dev_path)
        tass_es = convert_TASS_dataset(tass_es)
    else: 
        tass_es = None

    # load other datasets 
    cardiff_eng, cardiff_es = load_cardiff()
    mteb_eng = load_mteb()

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
    
    # get overview of lengths 
    ds_lengths = get_overview(all_ds)

    return combined_ds, ds_lengths 


def main():
    path = pathlib.Path(__file__)
    tass_path = path.parents[1] / "data"

    ds, ds_overview = load_datasets(tass_path)
    print(ds)
    print(ds_overview)

if __name__ == "__main__":
    main()

