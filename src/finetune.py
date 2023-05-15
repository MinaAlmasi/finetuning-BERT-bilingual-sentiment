'''
Fine-tuning BERT for bilingual emotion classification in English and Spanish

# https://huggingface.co/docs/transformers/v4.17.0/en/tasks/sequence_classification#finetune-with-trainer
'''

# utils 
import pathlib 

# dataset
from datasets import load_dataset

# to define parameters for model 
from transformers import TrainingArguments

# custom moduels
from finetune_fns import (combine_datasets, finetune)

def main(): 
    # define model outpath
    path = pathlib.Path(__file__)
    modeloutpath = path.parents[1] / "models"
    modeloutpath.mkdir(exist_ok=True)

    # load data 
    eng_data = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "english")
    es_data = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "spanish")

    # combine data 
    all_data = combine_datasets(eng_data, es_data)

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