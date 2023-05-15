'''
Fine-tuning BERT for bilingual emotion classification in English and Spanish

# https://huggingface.co/docs/transformers/v4.17.0/en/tasks/sequence_classification#finetune-with-trainer
'''

# utils 
import pathlib 

# dataset
import datasets

# to define parameters for model 
from transformers import TrainingArguments

# custom moduels
from finetune_fns import (finetune)
from data_preprocess import (load_TASS, convert_TASS_dataset, add_eng_lang_col, add_es_lang_col, combine_datasets)

def main(): 
    # define model outpath
    path = pathlib.Path(__file__)
    modeloutpath = path.parents[1] / "models"
    modeloutpath.mkdir(exist_ok=True)

    ## TASS DATASET
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

    # map labels to ids
    id2label = {0: "negative", 1:"neutral", 2:"positive"}
    label2id = {"negative":0, "neutral":1, "positive":2}

    # define training arguments 
    training_args = TrainingArguments(
        output_dir = modeloutpath, 
        learning_rate=2e-5,
        per_device_train_batch_size = 16, 
        per_device_eval_batch_size = 16, 
        num_train_epochs=2, 
        weight_decay=0.01,
        evaluation_strategy="epoch", 
        save_strategy = "epoch", 
        load_best_model_at_end = True
    )

    finetuned_model = finetune(
        dataset = all_data, 
        model_name = "bert-base-multilingual-cased", 
        n_labels = 3,
        id2label = id2label,
        label2id = label2id,
        training_args = training_args
        )


if __name__ == "__main__":
    main()