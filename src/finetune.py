'''
Script for self-assigned Assignment 5, Language Analytics, Cultural Data Science, F2023

Fine-tuning BERT for bilingual sentiment classification in English and Spanish

Run in terminal: 
    python src/finetune.py -hub {PUSH_TO_HUB_BOOL} -epochs {N_EPOCHS} -download {DOWNLOAD_MODE} -mdl {MODEL} -TASS {TASS_DOWNLOAD}

Additonal arguments:
    -epochs (int): number of epochs the model should run for. 
    -download (str): 'force_redownload' to force HF datasets to be redownloaded. Defaults to 'None' for using cached datasets.
    -mdl (str): name of model to use. Defaults to 'mBERT'. Choose between 'mBERT' or 'mDistilBERT'.
    -hub (bool): whether to push to huggingface hub or not. Defaults to False.
    -TASS (bool): whether to include TASS in the finetuning or not. Defaults to True.

NB! Note that pushing to HF hub requires a token in a .txt file called "token.txt" in the main repo folder.

@MinaAlmasi
'''

# utils 
import pathlib 
import argparse

# to define parameters for model 
from transformers import TrainingArguments 

# custom modules
from modules.finetune_fns import finetune, get_loss, plot_loss, get_metrics, get_metrics_per_language
from modules.data_fns import load_datasets

# save log history
import pickle

# disable msg datasets 
import datasets 
datasets.utils.logging.set_verbosity_error()

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-hub", "--push_to_hub", help = "Whether to push to Hugging Face Hub (True) or not (False)", type = bool, default = False) 
    parser.add_argument("-epochs", "--n_epochs", help = "number of epochs the model should run for", type = int, default = 30)
    parser.add_argument("-download", "--download_mode", help = "'force_redownload' to force HF datasets to be redownloaded. None for using cached datasets.", type = str, default = None)
    parser.add_argument("-mdl", "--model", help = "Choose between 'mBERT' or 'mDeBERTa' or 'xlm-roberta'", type = str, default = "xlm-roberta")
    parser.add_argument("-TASS", "--TASS_download", help = "Whether to include to TASS in the finetuning (True) or not (False)", type = bool, default = True)

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def model_picker(chosen_model:str): 
    '''
    Function for picking model to finetune.

    Args:
        chosen_model: name of model to use. Choosen between 'mBERT' or 'mDeBERTa' or 'xlm-roberta'.capitalize()
    
    Returns:
        model_dict: dictionary with name of fine-tune (key) and name of model to be fine-tuned (value).
            - The name of the fine-tune is used in "output_dir" in TrainingArguments i.e., name that will be pushed to the hub or saved as local folder.
    '''

    if chosen_model == "mBERT":
        model_dict = {"ES-ENG-mBERT-sentiment": "bert-base-multilingual-cased"} 

    if chosen_model == "mDeBERTa": 
        model_dict = {"ES-ENG-mDeBERTa-sentiment": "microsoft/mdeberta-v3-base"}

    elif chosen_model == "xlm-roberta":
        model_dict = {"ES-ENG-xlm-roberta-sentiment": "xlm-roberta-base"}

    return model_dict

def main(): 
    # intialise args 
    args = input_parse()

    # define paths (for saving model and loss curve)
    path = pathlib.Path(__file__)
    modeloutpath = path.parents[1] / "models"
    resultspath = path.parents[1] / "results"

    # ensure that modelpath exists
    modeloutpath.mkdir(exist_ok=True, parents=True)

    # pick model
    model_dict = model_picker(args.model)

    # get outputfolder, modelname
    output_folder = list(model_dict.keys())[0]
    model_name = list(model_dict.values())[0]

    # push to hub ! 
    if args.push_to_hub == True: 
        from huggingface_hub import login

        # get token from txt
        with open(path.parents[1] / "token.txt") as f:
            hf_token = f.read()

        login(hf_token)

    # load tass_path 
    if args.TASS_download == True: 
        tass_path = path.parents[1] / "data"
    else:
        tass_path = None

    # load datasets 
    ds, ds_overview = load_datasets(tass_path, args.download_mode)

    # map labels to ids
    id2label = {0: "negative", 1:"neutral", 2:"positive"}
    label2id = {"negative":0, "neutral":1, "positive":2}

    # define batch_size 
    batch_size = 64

    # define training arguments 
    training_args = TrainingArguments(
        output_dir = modeloutpath / output_folder, 
        push_to_hub = args.push_to_hub,
        learning_rate=2e-6,
        per_device_train_batch_size = batch_size, 
        per_device_eval_batch_size = batch_size, 
        num_train_epochs=args.n_epochs, 
        weight_decay=0.1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy = "epoch", 
        load_best_model_at_end = True, 
        metric_for_best_model = "accuracy",
    )

    # fine tune 
    trainer, tokenized_data = finetune(
        dataset = ds, 
        model_name = model_name,
        n_labels = 3,
        id2label = id2label,
        label2id = label2id,
        training_args = training_args, 
        early_stop_patience=3
        )

    # push model to hub
    if args.push_to_hub == True: 
        trainer.push_to_hub()

    # save log history with pickle
    with open (resultspath / f"{args.model}_log_history.pkl", "wb") as file:
        pickle.dump(trainer.state.log_history, file)

    # compute train and val loss, plot loss
    train_loss, val_loss, total_epochs = get_loss(trainer.state.log_history)
    plot_loss(train_loss, val_loss, total_epochs, resultspath, f"{args.model}_loss_curve.png")

    # evaluate, save summary metrics 
    get_metrics(trainer,  tokenized_data["test"], ds["test"], id2label, resultspath, f"{args.model}_all")

    # evaluate per language metrics subset 
    get_metrics_per_language(trainer, tokenized_data["test"], ds["test"], id2label, resultspath, f"{args.model}_es", language="ES")
    get_metrics_per_language(trainer, tokenized_data["test"], ds["test"], id2label, resultspath, f"{args.model}_eng", language="ENG")

if __name__ == "__main__":
    main() 