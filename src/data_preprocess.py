# utils
import pathlib 

# dataset 
import pandas as pd 
from datasets import concatenate_datasets, DatasetDict

def combine_datasets(dataset1, dataset2):
    data = {}

    # loop over dataset dict
    for split in ["train", "validation", "test"]:
        # concatenate each split with each other, append to list  
        data[split] = concatenate_datasets([dataset1[split], dataset2[split]])

    # make into dataset dict
    data = DatasetDict(data)

    return data 

def load_TASS(train_path:pathlib.Path, dev_path:pathlib.Path): 
    dfs = {}

    for path in [train_path, dev_path]:
        temp = []
        for file in path.iterdir():
            if file.is_file():
                df = pd.read_csv(train_path / file, sep = "\t", names=["id", "tweet", "sentiment"])
                # extract name (go from es.tsv -> "ES")
                df["lang"] = file.name.split(".")[0].upper()

                # append to temp list 
                temp.append(df)
        
        # concatenate
        dfs[path.name] = pd.concat(temp)

    return dfs 
    

def main():
    pass