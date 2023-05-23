# Finetuning BERT-models for Bilingual Sentiment Classification in English and Spanish
Repository link: https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment

This repository forms the self-assigned *assignment 5* by Mina Almasi (202005465) in the subject Language Analytics, Cultural Data Science, F2023. 

The repository contains code for finetuning BERT-based models for bilingual sentiment classification in English and Spanish. If you wish to use the models for inference, please refer to the section [*Inference with the Finetunes*](https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment#inference-with-the-finetunes). 

## Data 
The data comprises 3 datasets all from Twitter data in either English or Spanish: 
1. [Cardiff NLP (Spanish and English)](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual) (Barbieri et al., 2022)

2. [TASS 2020 (Spanish)](http://tass.sepln.org/2020/?wpdmpro=task-1-train-and-dev-sets)

3. [MTEB (English subset)](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction) (Muennighoff et al., 2023)

All datasets except the ```TASS``` dataset is downloaded with Hugging Face's ```datasets``` package within the scripts. The ```TASS``` dataset can be downloaded on the link above and placed in the ```data``` folder. 

Note that the use of ```TASS``` has to comply with the [TASS Dataset License](http://tass.sepln.org/tass_data/download.php). For this reason, it is possible to exclude the ```TASS``` data when reproducing the code (see [Pipeline](https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment#pipeline) for instructions).

### Combining Data
The choice to combine datasets was made to obtain a larger train, test and validation set for the finetuning.

To restrain the variability within the combined dataset, only datasets with Twitter data were considered. Additionally, datasets were processed to resemble each other to the extent that was possible. For instance, the real Twitter usernames in the ```TASS``` dataset were replaced with ```@user``` to match the Cardiff dataset (and also to respect anonymity). However, this was not possible with the MTEB dataset which does not contain any usernames.

The total data amounts to ```20555 tweets```:

| dataset     |   train |   validation |   test |   total |
|:------------|--------:|-------------:|-------:|--------:|
| tass_es     |    4802 |         1221 |   1222 |**7245** |
| mteb_eng    |    4802 |         1221 |   1221 |**7244** |
| cardiff_es  |    1839 |          324 |    870 |**3033** |
| cardiff_eng |    1839 |          324 |    870 |**3033** |
| **total**   |**13282**|     **3090** |**4183**|**20555**|

The MTEB data was subsetted to match the TASS data to ensure a balance of Spanish and English tweets. The splits in the original datasets were respected, resulting in approx. 35 % of the dataset being used for validation (approx. 15%) and test (approx. 20%). A seed was set for all custom splits to ensure reproducibility.

## Experimental Pipeline and Motivation
As the fourth most spoken language globally, the Spanish language offers the potential to gain insights into the culture and traditions of a considerable portion of the world population. Combined with the English language, the coverage is extended to an even larger segment of global society. 

With this in mind, the current project aims to perform sentiment classifcation but bilingually in Spanish and English. This is achieved by finetuning several BERT-based models on English and Spanish data simultanously using HuggingFace's Trainer from their ```transformers``` package.  The pipeline can be seperated into two parts: 

### ```(P1) FINE-TUNING```
Finetuning several BERT-based models on labelled Spanish & English Twitter data (3 labels: negative, neutral, and positve).

### ```(P2) EVALUATION```
Evaluating the models on test data. 

Overall performance is extracted, but the models are also evaluated on subsets of the test data, seperated into English and Spanish, to uncover potential disparities across languages.
## Reproducibility 
To reproduce the results, follow the instructions in the [Pipeline](https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment#pipeline) section.

NB! Be aware that finetuning BERT-based models is a computationally heavy task. Cloud computing (e.g., [UCloud](https://cloud.sdu.dk/) with high amounts of ram (or a good GPU) is encouraged.

## Project Structure
The repository is structured as such: 
| <div style="width:120px"></div>| Description |
|---------|:-----------|
| ```results``` | Results from running ```finetune.py```: loss curves, metrics (all + per language), predictions data per example in test set (all + per language). |
| ```visualisations``` | Table, loss curves overview and confusion matrices from running ```visualise.py```. |
| ```models``` | Models are saved here after finetuning.|
| ```src``` | Scripts to run finetuning + evaluation and to visualise results. Contains helper functions in the ```modules``` folder.|
| ```requirements.txt``` | Necessary packages to be installed |
| ```setup.sh``` |  Run to install ```requirements.txt``` within newly created ```env```. |
| ```git-lfs-setup.sh``` |  Run to install ```gif-lfs```. Optional setup needed for pushing models to Hugging Face Hub. |

## Pipeline
The pipeline has been tested on Ubuntu v22.10, Python v3.10.7 ([UCloud](https://cloud.sdu.dk/), Coder Python 1.77.3). Python's [venv](https://docs.python.org/3/library/venv.html) needs to be installed for the pipeline to work.

### Installing TASS Data
If you wish to run the pipeline with all data, install the [TASS 2020 (Spanish)](http://tass.sepln.org/2020/?wpdmpro=task-1-train-and-dev-sets) files and place them in the ```data``` folder. Ensure that the data follows the structure and naming conventions described in [data/README.md](https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment/tree/main/data). 

### General Setup
To run the finetune pipeline, create a virtual environment (```env```) and install necessary requirements by firstly running: 
```
bash setup.sh
```

### Extra Setup (Pushing to the HF Hub)
**NB. OPTIONAL:** Pushing models to the Hugging Face Hub is disabled by default in all scripts, so you can ```SKIP``` this setup if you are not interested in this functionality.

If you wish to push models to the Hugging Face Hub, you need to firstly save a [Hugging Face token](https://huggingface.co/docs/hub/security-tokens) in a .txt file called ```token.txt``` in the main folder.

Then, install git-lfs. Note that this wil ```SUDO``` install to your system, ```do at own risk```!
```
bash git-lfs-setup.sh
```

### Running the Experimental Pipeline 
You can run the entire pipeline on ```ALL``` the data (finetuning of several models and visualisation) by typing:
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
python src/finetune.py -mdl {MODEL} -epochs {N_EPOCHS} -download {DOWNLOAD_MODE} -TASS -hub
```
NB! Remember to activate the ```env``` first (by running ```source ./env/bin/activate```)

| <div style="width:80px">Arg</div>    | Description                             | <div style="width:120px">Default</div>    |
| :---        |:---                                                                                        |:---             |
|```-mdl```   |  Model to be finetuned. Choose between 'mDeBERTa', 'mBERT' or 'xlm-roberta'               | xlm-roberta     |
|```-epochs```| MAX epochs the model should train for (if not stopped after 3 epochs with no improvement)  | 30              |
|```-download```| Write 'force_redownload' to redownload cached datasets. Useful if cache is corrupt.      | None            |
|```-hub```   | Write the flag ```-hub``` if you wish to Hugging Face Hub. Leave it out if not.                      |                |
|```-TASS```   | Write the flag ```-TASS``` if you wish to use the TASS dataset. Leave it out if not.                          |                |

## Inference with the Finetunes
The three finetuned models are available on the Hugging Face Hub:
1. [MinaAlmasi/ES-ENG-mBERT-sentiment](https://huggingface.co/MinaAlmasi/ES-ENG-mBERT-sentiment)
2. [MinaAlmasi/ES-ENG-xlm-roberta-sentiment](https://huggingface.co/MinaAlmasi/ES-ENG-xlm-roberta-sentiment)
3. [MinaAlmasi/ES-ENG-mDeBERTa-sentiment](https://huggingface.co/MinaAlmasi/ES-ENG-mDeBERTa-sentiment)


If you want to use the models for inference, click on the links to use the *Hosted inference API* by Hugging Face. 

If you wish to run inference locally,  the script ```inference.py``` demonstrates the use of the model ```MinaAlmasi/ES-ENG-mDeBERTa-sentiment```: 
```
python src/inference.py -text {TEXT}
```
NB! Remember to activate the ```env``` first (by running ```source ./env/bin/activate```)

## Results 
The following presents the results from the three models that were finetuned on the base versions of of [mBERT](https://huggingface.co/bert-base-multilingual-cased), [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base), and  [mDeBERTa V3](https://huggingface.co/microsoft/mdeberta-v3-base). 

### ```(P1)``` Finetuning Hyperparameters
All models are trained with the parameters: 
* Batch size: 64
* Max epochs: 30 
* Early stopping patience: 3 (stops training if validation acc. does not improve for 3 epochs)
* Weight decay: 0.1
* Learning Rate: 2e-6

With the early stopping defined, the model with the ```BEST``` validation accuracy was saved for inference. 

### ```(P1)``` Loss Curves
The loss curves for each model is displayed below. The dashed vertical lines represent the ```BEST model``` with its validation accuracy and final epoch described in the corresponding legend. 


<p align="left">
  <img src="https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment/blob/main/visualisations/loss_curves_all_models.png">
</p>

As evident from the plots above, the models are clearly ```overfitting``` as the training loss continously decreases while validation loss increases. For future projects, one should consider defining the early stopping callback based on the ```validation loss``` rather than the ```validation accuracy```. Nonetheless, from the final validation accuracy, ```mDeBERTa``` is the best performing model out of the three. 

### ```P2``` F1 scores on the Test data
The F1 scores (and a single ```Accuracy``` score) for each model are in the table below. Please see the individual metrics files in ```results/metrics``` for ```precision``` and ```recall``` scores.  

Labels indicate whether the test set includes all examples (```all```), or if it has been stratified by language (```eng = English``` or ```es = Spanish```).

|                 |   Negative |   Neutral |   Positive |   Accuracy |   Macro_Avg |   Weighted_Avg |
|-----------------|------------|-----------|------------|------------|-------------|----------------|
| mBERT all       |       0.52 |      0.55 |       0.74 |       0.6  |        0.6  |           0.6  |
| mDeBERTa all    |       0.58 |      0.59 |       0.79 |       0.65 |        0.65 |           0.65 |
| xlm-roberta all |       0.57 |      0.58 |       0.78 |       0.64 |        0.64 |           0.64 |
| mBERT eng       |       0.7  |      0.59 |       0.77 |       0.68 |        0.68 |           0.68 |
| mDeBERTa eng    |       0.78 |      0.64 |       0.8  |       0.74 |        0.74 |           0.74 |
| xlm-roberta eng |       0.76 |      0.61 |       0.79 |       0.72 |        0.72 |           0.71 |
| mBERT es        |       0.31 |      0.51 |       0.72 |       0.52 |        0.51 |           0.51 |
| mDeBERTa es     |       0.34 |      0.55 |       0.78 |       0.56 |        0.55 |           0.55 |
| xlm-roberta es  |       0.34 |      0.56 |       0.76 |       0.57 |        0.56 |           0.55 |

The table shows that the the three finetuned model are close in performance with ```weighted F1 (Macro_Avg)``` scores ranging between ```0.60``` and ```0.65``` for English and Spanish sentiment classification. However, these scores do not show the full picture. Upon assessing the models on the each language individually, a disparity in performance across languages is evident. Model performance is greater on English tweets (```weighted F1 = 0.68 to 0.74```) compared to Spanish (```weighted F1 = 0.51 to 0.55```). In general, this performance also differs greatly across labels with worse performance in ```Negative``` and ```Neutral``` tweets than for ```Positive``` tweets. On the Spanish data, ```Negative``` labels are especially impacted with scores around chance level (```33%```).

Overall, models ```mDeBERTa``` and ```xlm-roberta``` performed better than ```mBERT```. 

### ```(P2)``` Confusion Matrix (mDeBERTa)
As ```mDeBERTa``` and ```xlm-roberta``` were nearly identical in performance, only one of them is highlighted. The plot below shows the confusion matrix for ```mDeBERTa``` on the entire test set: 
<p align="left">
  <img width=85% height=85% src="https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment/blob/main/visualisations/confusion_matrix_mDeBERTa_all.png">
</p>

The confusion matrix provides an illustated overview of the labels that the model confused by other labels. From this, it is clear that the confusion is greatly located between the ```Neutral``` and ```Negative``` labelled tweets. ```27%``` of ```Neutral``` tweets were predicted as ```Negative``` and ```37%``` of ```Negative``` tweets were predicted as ```Neutral```. Nearly none of the ```Negative``` tweets were predicted as ```Positive``` or vice versa (```7%```).     

For the other models, the confusion matrices on the entire test set are located in the ```visualisations``` folder. 

### Discussion & Limitations
Weighted F1 scores for bilingual sentiment classification was around 0.6, but results also revealed a performance gap between English and Spanish with all models doing better in English. Performance furthermore differed across sentiments with ```Negative``` labels being close to chance level on Spanish data. 

It is complicated to exactly identify why these performance disparities occur. Firstly, it is possible that the models are simply better at English than at Spanish, making it difficult to achieve a bilingual model that performs equally well. Nonetheless, finetuning the currently available Spanish monolingual models may not fix performance. In a survey of Spanish language models, Agerri and Agirre (2023) found that the multilingual models generally surpassed Spanish monolingual models, highlighting exactly ```mDeBERTA``` and ```xlm-roberta``` for their great performance.

It may also be worth considering whether the Spanish dataset was more varied than the English. For instance, the ```TASS``` set consisted of tweets in several different Spanish dialects (e.g., Mexican, Peru) but were all treated as Spanish for simplicity. Moreover, for the disparities amongst ```labels```, labels ```Positive``` and ```Negative``` were near equally balanced, but there were more ```Neutral``` labels in each split (around ```1000``` more in ```train``` and around ```200-250``` in both ```test``` and ```validation```). This could be balanced in future testing. Finally, there may be general problems with the way the datasets were combined. The datasets may have had too great variation, despite efforts to reduce this.

## Author 
This repository was created by Mina Almasi:

* github user: @MinaAlmasi
* student no: 202005465, AUID: au675000
* mail: mina.almasi@post.au.dk

## References
Agerri Gascón, R., & Agirre Bengoa, E. (2023). Lessons learned from the evaluation of Spanish Language Models. Sociedad Española Para El Procesamiento Del Lenguaje Natural. https://doi.org/10.26342/2023-70-13

Barbieri, F., Anke, L. E., & Camacho-Collados, J. (2022). XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond (arXiv:2104.12250). arXiv. https://doi.org/10.48550/arXiv.2104.12250

Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive Text Embedding Benchmark (arXiv:2210.07316). arXiv. https://doi.org/10.48550/arXiv.2210.07316

