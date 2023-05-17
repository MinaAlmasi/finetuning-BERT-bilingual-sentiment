'''
Fine-tuning BERT for bilingual sentiment classification in English and Spanish

Run in terminal: 
    python src/finetune.py -hub {PUSH_TO_HUB_BOOL}

Where -hub denotes whether to push the model to the huggingface hub. 
Note that this requires a token in a .txt file called "token.txt" in the main repo folder 
'''

# utils 
import pathlib 
import argparse

# to define parameters for model 
from transformers import TrainingArguments 

# custom modules
from finetune_fns import finetune
from data_fns import load_datasets
from evaluate_fns import get_loss, plot_loss, get_metrics

# disable msg datasets 
import datasets 
datasets.utils.logging.set_verbosity_error()

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-hub", "--push_to_hub", help = "Whether to push to huggingface hub or not", type = bool, default = False) #default img defined
    parser.add_argument("-epochs", "--n_epochs", help = "number of epochs the model should run for", type = int, default = 20) #default img defined
    
    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def main(): 
    # intialise args 
    args = input_parse()

    # define paths (for saving model and loss curve)
    path = pathlib.Path(__file__)
    modeloutpath = path.parents[1] / "models"
    resultspath = path.parents[1] / "results"
    
    # ensure that paths exist
    resultspath.mkdir(exist_ok=True, parents=True)
    modeloutpath.mkdir(exist_ok=True, parents=True)

    # push to hub ! 
    if args.push_to_hub == True: 
        from huggingface_hub import login

        # get token from txt
        with open(path.parents[1] / "token.txt") as f:
            hf_token = f.read()

        login(hf_token)

    # load tass_path 
    tass_path = path.parents[1] / "data"

    # load datasets 
    ds, ds_overview = load_datasets(tass_path)

    # map labels to ids
    id2label = {0: "negative", 1:"neutral", 2:"positive"}
    label2id = {"negative":0, "neutral":1, "positive":2}

    # define training arguments 
    training_args = TrainingArguments(
        output_dir = modeloutpath / "ES-ENG-mBERT-sentiment", 
        push_to_hub = args.push_to_hub,
        learning_rate=2e-5,
        per_device_train_batch_size = 16, 
        per_device_eval_batch_size = 16, 
        num_train_epochs=args.n_epochs, #input parse, defaults to 20
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy = "epoch", 
        load_best_model_at_end = True, 
        metric_for_best_model = "accuracy",
    )

    # fine tune 
    trainer, tokenized_data = finetune(
        dataset = ds, 
        model_name = "bert-base-multilingual-cased", 
        n_labels = 3,
        id2label = id2label,
        label2id = label2id,
        training_args = training_args, 
        early_stop_patience=5
        )

    # push model to hub
    trainer.push_to_hub()

    # compute train and val loss, plot loss
    train_loss, val_loss = get_loss(trainer.state.log_history)
    plot_loss(train_loss, val_loss, args.n_epochs, resultspath, "loss_curve.png")

    # evaluate, save summary metrics 
    metrics = get_metrics(trainer, tokenized_data["test"])

    es_test = tokenized_data["test"].filter(lambda example: example['lang'] != "ENG")
    eng_test= tokenized_data["test"].filter(lambda example: example['lang'] == "ENG")

    eng_metrics = get_metrics(trainer, eng_test)
    es_metrics = get_metrics(trainer, es_test)

    for metric_name, metric in {"all":metrics, "eng":eng_metrics, "es":es_metrics}.items():
        with open(resultspath / f"mBERT_{metric_name}_metrics.txt", "w") as file: 
            file.write(metric)


if __name__ == "__main__":
    main()