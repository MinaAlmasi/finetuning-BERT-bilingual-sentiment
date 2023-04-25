'''
Fine-tuning BERT for bilingual emotion classification in English and Spanish

# https://huggingface.co/docs/transformers/v4.17.0/en/tasks/sequence_classification#finetune-with-trainer
'''

# utils 
import pathlib 

# dataset 
from datasets import load_dataset 

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True)
    
def tokenize_dataset(dataset): 
    pass

def main(): 
    # load data 
    dataset = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "english", "spanish")

    print(dataset)

if __name__ == "__main__":
    main()