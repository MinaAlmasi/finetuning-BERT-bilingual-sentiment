'''
Fine-tuning BERT for bilingual emotion classification in English and Spanish

# https://huggingface.co/docs/transformers/v4.17.0/en/tasks/sequence_classification#finetune-with-trainer
'''

# utils 
import pathlib 
import numpy as np 

# dataset 
from datasets import load_dataset 

# transformers tokenizers, models 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# evaluation
import evaluate 

# tokenisation funktions
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True)
    
def tokenize_dataset(dataset): 
    tokenized_dataset = dataset.map(tokenize, batched=True)

def compute_metrics(eval_pred):
    # compute predictions 
    predictions, labels = eval_pred

    # choose prediction with highest probability
    predictions = np.argmax(predictions, axis=1)

    # compute accuracy 
    accuracy = accuracy.compute(predictions=predictions, references=labels)

    return accuracy 

def main(): 
    # define model outpath
    path = pathlib.Path(__file__)
    modeloutpath = path.parents[1] / "models"
    modeloutpath.mkdir(exist_ok=True)

    # load data 
    data = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "english", "spanish")

    # import tokenizer 
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # tokenize 
    tokenize_data = tokenize_dataset(data)

    # define datacollator 
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # load accuracy metric 
    accuracy = evaluate.load("accuracy")

    # map labels to ids
    id2label = {0: "negative", 1:"neutral", 2:"positive"}
    label2id = {"negative":0, "neutral":1, "positive":2}

    # initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=2, id2label=id2label, label2id=label2id
    )
    
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

    # initialize trainer 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"], 
        tokenizer=tokenizer, 
        data_collator=data_collator, 
        compute_metrics = compute_metrics
    )

    # train model
    trainer.train()

if __name__ == "__main__":
    main()