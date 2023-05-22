# Finetuning BERT-models for Bilingual Sentiment Classification in English and Spanish
Repository link: https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment

This repository forms the self-assigned *assignment 5* by Mina Almasi (202005465) in the subject Language Analytics, Cultural Data Science, F2023. 

The repository contains code for finetuning BERT-based models for bilingual sentiment classification in English and Spanish. If you wish to use the models for inference, please refer to the section [*Inference with the Fine-Tunes*](https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment#inference-with-the-fine-tunes). 

## Data 
The data comprises 3 datasets all from Twitter data in either English or Spanish: 
1. [Cardiff NLP (Spanish and English)](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual) (Barbieri et al., 2022)

2. [TASS 2020 (Spanish)](http://tass.sepln.org/2020/?wpdmpro=task-1-train-and-dev-sets)

3. [MTEB (English subset)](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction) (Muennighoff et al., 2023)

All datasets except the ```TASS``` dataset is downloaded with Hugging Face's ```datasets``` package within the scripts. The ```TASS``` dataset can be downloaded on the link above and placed in the ```data``` folder. Note that the use of this dataset has to comply with their license ([TASS Dataset License](http://tass.sepln.org/tass_data/download.php)). For this reason, it is made possible to rerun the code without the use of the ```TASS``` data (see [Pipeline](https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment#pipeline) for instructions). 

### Combining Data
The choice to combine datasets was made to have a greater train, test and validation set. To restrain the variability within the combined dataset, only datasets with Twitter data were considered. Additionally, datasets were processed to resemble each other to the extent that was possible. For instance, the real Twitter usernames in the ```TASS``` dataset were replaced with ```@user``` to match the Cardiff dataset (and also to respect anonymity). However, this was not possible with the MTEB dataset which does not contain any usernames.

The total data amounts to ```20555 tweets```:
| dataset     |   train |   validation |   test |   total |
|:------------|--------:|-------------:|-------:|--------:|
| tass_es     |    4802 |         1221 |   1222 |**7245** |
| mteb_eng    |    4802 |         1221 |   1221 |**7244** |
| cardiff_es  |    1839 |          324 |    870 |**3033** |
| cardiff_eng |    1839 |          324 |    870 |**3033** |
| **total**   |**13282**|     **3090** |**4183**|**20555**|

The MTEB data was subsetted to match the TASS data to ensure a balance of Spanish and English tweets. The splits in the original datasets were respected, resulting in approx. 35 % of the dataset being used for validation (approx. 15%) and test (approx. 20%). 

## Experimental Pipeline and Motivation
As the fourth most spoken language globally, the Spanish language offers the potential to gain insights into the culture and traditions of a considerable portion of the world population. Combined with the English language, the coverage is extended to an even larger segment of global society. 

With this in mind, the current project aims to perform sentiment classifcation but bilingually in Spanish and English. This is achieved by finetuning several BERT-based models on English and Spanish data simultanously using HuggingFace's Trainer from their ```transformers``` package.  The pipeline can be seperated into two parts: 

### ```(P1) FINE-TUNING```
Fine-tune several BERT-based models on labelled Spanish & English Twitter data (3 labels: neutral, negative, positve).

### ```(P2) EVALUATION```
Evaluate the models on test data, extracting overall performance as well as performance stratified by language. 

By seperating the test set into the individual languages and evaluating the model on only these subsets, potential disparities in performance across the two languages is explored.

## Reproducibility 
To reproduce the results, follow the instructions in the [Pipeline](https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment#pipeline) section.

NB! Be aware that finetuning BERT-based models is a computationally heavy task. Cloud computing (e.g., [UCloud](https://cloud.sdu.dk/) with high amounts of ram (or a good GPU) is encouraged.

## Project Structure
The repository is structured as such: 
| <div style="width:120px"></div>| Description |
|---------|:-----------|
| ```results``` | Results from running ```finetune.py```: loss curves, metrics (all + per language), predictions data per example in test set (all + per language). |
| ```visualisations``` | Table and confusion matrices from running ```visualise.py```. |
| ```models``` | Models are saved here after running the fine-tune pipeline. |
| ```src``` | Scripts to run fine-tuning + evaluation and to visualise results. Contains helper functions in the ```modules``` folder.|
| ```requirements.txt``` | Necessary packages to be installed |
| ```setup.sh``` |  Run to install ```requirements.txt``` within newly created ```env```. |
| ```git-lfs-setup.sh``` |  Run to install ```gif-lfs``` necessary for pushing models to Hugging Face hub. |

## Pipeline
The pipeline has been tested on Ubuntu v22.10, Python v3.10.7 ([UCloud](https://cloud.sdu.dk/), Coder Python 1.77.3). Python's [venv](https://docs.python.org/3/library/venv.html) needs to be installed for the pipeline to work.

### Installing TASS 
If you wish to run the pipeline with all data, install the [TASS 2020 (Spanish)](http://tass.sepln.org/2020/?wpdmpro=task-1-train-and-dev-sets) and place it in the ```data``` folder. Ensure that the data follows the structure and naming conventions described in [data/README.md](https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment/tree/main/data). 

### General Setup
To run the fine-tune pipeline, create a virtual environment (```env```) and install necessary requirements by running: 
```
bash setup.sh
```

### Extra Setup (Pushing to the HF Hub)
**NB. OPTIONAL:** Pushing models to the Hugging Face Hub is disabled by default in all scripts, so you can ```SKIP``` this setup if you are not interested in this functionality.

If you wish to push models to the Hugging Face Hub, you need to firstly save a [Hugging Face token](https://huggingface.co/docs/hub/security-tokens) in a .txt file called ```token.txt``` in the main folder. (```token.txt is in .gitignore and will not be pushed!```)

Then, install git-lfs. Note that this wil ```SUDO``` install to your system, ```do at own risk```!
```
bash git-lfs-setup.sh
```

### Running the Experimental Pipeline 
You can run the entire pipeline on ```ALL``` the data (fine-tuning of several models and visualisation) by typing:
```
run.sh
```
If you wish to run the pipeline without ```TASS```, type: 
```
bash run-noTASS.sh
```

### Training with Custom Arguments 
Train models individually by writing in the terminal:
```
python src/finetune.py -mdl {MODEL} -epochs {N_EPOCHS} -hub {PUSH_TO_HUB_BOOL} -download {DOWNLOAD_MODE}
```
NB! Remember to activate the ```env``` first (by running ```source ./env/bin/activate```)

| <div style="width:80px">Arg</div>    | Description                             | <div style="width:120px">Default</div>    |
| :---        |:---                                                                                        |:---             |
|```-mdl```   |  Model to be fine-tuned. Choose between 'mDeBERTa', 'mBERT' or 'xlm-roberta'               | xlm-roberta     |
|```-hub```   | Whether to push to Hugging Face Hub. 'True' if yes, else write 'False'                     | False               |
|```-epochs```| MAX epochs the model should train for (if not stopped after 3 epochs with no improvement)  | 30              |
|```-download```| Write 'force_redownload' to redownload cached datasets. Useful if cache is corrupt.      | None            |

## Inference with the Fine-Tunes
The three fine-tuned models are available on the HuggingFace Hub:
1. [MinaAlmasi/ES-ENG-mBERT-sentiment](https://huggingface.co/MinaAlmasi/ES-ENG-xlm-roberta-sentiment)
2. [MinaAlmasi/ES-ENG-xlm-roberta-sentiment](https://huggingface.co/MinaAlmasi/ES-ENG-xlm-roberta-sentiment)
3. [MinaAlmasi/ES-ENG-mDeBERTa-sentiment](https://huggingface.co/MinaAlmasi/ES-ENG-mDeBERTa-sentiment)


If you want to use the models for inference, click on the links to use the *Hosted inference API* by Hugging Face. If you wish to run inference locally,  the script ```inference.py``` demonstrates the use of the model ```MinaAlmasi/ES-ENG-mDeBERTa-sentiment```: 
```
python src/inference.py -text {TEXT}
```
NB! Remember to activate the ```env``` first (by running ```source ./env/bin/activate```)

## Results 
The following presents three models which have been fine-tuned based on the base version of [mBERT](https://huggingface.co/bert-base-multilingual-cased), [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base), and  [mDeBERTa V3](https://huggingface.co/microsoft/mdeberta-v3-base). The sections include a description of the hyperparameters for the fine-tuning, results during training and an evaluation of the models on the test set. 

### ```(P1)``` Training Hyperparameters
All models are trained with the parameters: 
* Batch size: 64 
* Max epochs: 30 
* Early stopping patience: 3 
* Weight decay: 0.1
* Learning Rate: 2e-6

The early stopping patience stopped model training if validation accuracy did not improve for 3 epochs. The model with the ```BEST``` validation accuracy was saved for inference.  

### ```(P1)``` Loss Curves
The loss curves for each model is displayed below. The dashed vertical lines represent the ```BEST``` model that has been saved for inference. The validation accuracy and final epoch of the ```BEST``` model is in the legend of each plot.


<p align="left">
  <img src="https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment/blob/main/visualisations/loss_curves_all_models.png">
</p>

As evident from the plots above, the models are clearly ```overfitting``` as the training loss continously decreases while validation loss increases. For future projects, one should consider defining the early stopping callback based on the ```validation loss``` rather than the ```validation accuracy```. 

### ```P2``` F1 scores on the Test data
The F1 scores (and a single ```Accuracy``` score) for each model are in the table below. Please see the individual metrics files in ```results/metrics``` for ```precision``` and ```recall``` scores.  

Labels indicate whether the test set includes all examples (```all```), or if it has been stratified by language (```eng = English``` or ```es = Spanish```) . 
|                 |   Neutral |   Negative |   Positive |   Accuracy |   Macro_Avg |   Weighted_Avg |
|-----------------|-----------|------------|------------|------------|-------------|----------------|
| mBERT all       |      0.54 |       0.56 |       0.73 |       0.6  |        0.61 |           0.6  |
| mDeBERTa all    |      0.56 |       0.6  |       0.79 |       0.65 |        0.65 |           0.65 |
| xlm-roberta all |      0.57 |       0.6  |       0.76 |       0.64 |        0.64 |           0.64 |
| mBERT eng       |      0.71 |       0.63 |       0.74 |       0.69 |        0.69 |           0.69 |
| mDeBERTa eng    |      0.77 |       0.65 |       0.79 |       0.73 |        0.74 |           0.73 |
| xlm-roberta eng |      0.77 |       0.65 |       0.77 |       0.73 |        0.73 |           0.72 |
| mBERT es        |      0.34 |       0.5  |       0.71 |       0.52 |        0.52 |           0.51 |
| mDeBERTa es     |      0.3  |       0.56 |       0.79 |       0.57 |        0.55 |           0.55 |
| xlm-roberta es  |      0.32 |       0.56 |       0.76 |       0.56 |        0.55 |           0.54 |

Firstly, when focusing on all examples (```all```), the ```mDeBERTa``` and ```xlm-roberta``` are nearly identical in performance throughout. ```mBERT``` is slightly worse with a ```weighted F1 (Weighted_Avg)``` of ```0.6``` versus ```mDeBERTa = 0.65``` and ```xlm-roberta = 0.64```. 

When seperating the test set into English and Spanish, there is an clear performance gap. ```mDeBERTa``` has a ```weighted F1``` of ```0.73``` for the English test set whereas the weighted F1 is only ```0.55``` for the Spanish set. When looking at the scores per class, the F1 for the neutral class in the Spanish set is at ```0.3``` which is just below the chance level ```33%```. Nevertheless, the higher scores for the English test set suggests that the models are not entirely useless for English sentiment classification. 

### ```(P2)``` Confusion Matrix (mDeBERTa)
As ```mDeBERTa``` and ```xlm-roberta``` were nearly identical in performance, only one of them is highlighted. The plot below shows the confusion matrix for ```mDeBERTa``` on the entire test set: 
<p align="left">
  <img src="https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment/blob/main/visualisations/confusion_matrix_mDeBERTa_all.png">
</p>

The confusion matrix provides an illustated overview of the labels that the model confused by other labels. From this, it is clear that the confusion is greatly located between the ```Neutral``` and ```Negative``` labelled tweets. ```24%``` of ```Neutral``` tweets were predicted as ```Negative``` and ```40%``` of ```Negative``` tweets were predicted as ```Neutral```. Nearly none of the ```Negative``` tweets were predicted as ```Positive``` (```6%```).     

For the other models, the confusion matrices on the entire test set are located in the ```visualisations``` folder. 

### Discussion & Limitations
The three fine-tuned model were close in performance with ```weighted F1``` scores ranging between ```0.60``` and ```65``` for English and Spanish sentiment classification. Importantly, these scores do not show the full picture. Upon assessing the models on the each language individually, a disparity in performance across languages is evident. Model performance is greater on English tweets (```weighted F1 = 0.69 to 0.72```) compared to Spanish (```weighted F1 = 0.51 to 0.55```). In general, this performance also differs greatly across labels with worse performance in ```Negative``` and ```Neutral``` tweets than for ```Positive``` tweets. 

It is complicated to exactly identify why these performance disparities occur. Firstly, it is possible that the models are simply better at English than at Spanish, making it difficult to achieve a bilingual model that performs equally well. Nonetheless, fine-tuning Spanish monolingual models may not fix performance. In a survey of Spanish language models, Agerri and Agirre (2023) found that the multilingual models were generally better performing than Spanish monolingual models in several tasks, highlighting exactly ```mDeBERTA``` and ```xlm-roberta``` for their great performance.

It may also be worth considering whether the Spanish dataset was more varied than the English. For instance, the ```TASS``` set consisted of tweets in several different Spanish dialects (e.g., Mexican, Peru) but were all treated as Spanish for simplicity. 

For the disparities amongst ```labels```, labels ```Positive``` and ```Negative``` were near equally balanced, but there were more ```Neutral``` labels in each split (around 1000 more in train and around 250-300 in test and validation). This should have been balanced by downsampling the ```Neutral``` labels. 

Finally, there may be general problems with the way the datasets were combined. The datasets may have had too great variation, despite efforts to reduce this.

## Author 
This repository was created by Mina Almasi:

* github user: @MinaAlmasi
* student no: 202005465, AUID: au675000
* mail: mina.almasi@post.au.dk

## References
Agerri Gascón, R., & Agirre Bengoa, E. (2023). Lessons learned from the evaluation of Spanish Language Models. Sociedad Española Para El Procesamiento Del Lenguaje Natural. https://doi.org/10.26342/2023-70-13

Barbieri, F., Anke, L. E., & Camacho-Collados, J. (2022). XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond (arXiv:2104.12250). arXiv. https://doi.org/10.48550/arXiv.2104.12250

Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive Text Embedding Benchmark (arXiv:2210.07316). arXiv. https://doi.org/10.48550/arXiv.2210.07316

