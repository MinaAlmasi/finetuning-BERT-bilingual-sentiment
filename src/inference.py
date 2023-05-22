'''
Script for self-assigned Assignment 5, Language Analytics, Cultural Data Science, F2023

Perform inference on a text using the fine-tuned model "MinaAlmasi/ES-ENG-mDeBERTa-sentiment." Type: 
    python src/inference.py -text {TEXT}

E.g., 
    python src/inference.py -text "I just love exams!"

@MinaAlmasi
'''

# utils
import argparse

# load Hugging Face pipeline
from transformers import pipeline

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-text", "--text_to_classify", help = "Text to perform sentiment classication on", type = str, default = "¡Me encantan los exámenes!") 

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def main(): 
    # define args
    args = input_parse()

    # load pipeline
    classifier = pipeline( "text-classification", 
                        model = "MinaAlmasi/ES-ENG-mDeBERTa-sentiment",
                        return_all_scores=True,
                        top_k=1
                        )

    # predict on txt
    prediction = classifier(args.text_to_classify)

    # print text, predicted label and score
    print(f"Text: '{args.text_to_classify}', Predicted label: '{prediction[0][0]['label']}', Score: '{round(prediction[0][0]['score'], 3)}'")


if __name__ == "__main__":
    main()
