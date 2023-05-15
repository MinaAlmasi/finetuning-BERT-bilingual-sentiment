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

# add lang cols to other dataset
def add_eng_lang_col(example): # inspired by https://stackoverflow.com/questions/72749730/add-new-column-to-a-huggingface-dataset-inside-a-dictionary
    example["lang"] = "ENG"

    return example

def add_es_lang_col(example): # inspired by https://stackoverflow.com/questions/72749730/add-new-column-to-a-huggingface-dataset-inside-a-dictionary
    example["lang"] = "ES"

    return example

def main():
    path = pathlib.Path(__file__)
    train_path = path.parents[1] / "data" / "train"
    dev_path = path.parents[1] / "data" / "dev"

    # convert TASS 
    dfs = load_TASS(train_path, dev_path)
    tass_es = convert_TASS_dataset(dfs)

    # load datasets 
    cardiff_es = datasets.load_dataset("cardiffnlp/tweet_sentiment_multilingual", "spanish")
    cardiff_eng = datasets.load_dataset("cardiffnlp/tweet_sentiment_multilingual", "english")

    cardiff_eng = cardiff_eng.map(add_eng_lang_col)
    cardiff_es = cardiff_es.map(add_es_lang_col)

    all_data = combine_datasets([cardiff_es, cardiff_eng, tass_es])

    print(all_data)

if __name__ == "__main__":
    main()

