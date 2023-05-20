# Finetuning BERT for Bilingual Sentiment Classification in English and Spanish
Repository link: https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment

This repository forms the self-assigned *assignment 5* by Mina Almasi (202005465) in the subject Language Analytics, Cultural Data Science, F2023. 

The repository contains code for finetuning BERT-based models for bilingual sentiment classification in English and Spanish.

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

With this in mind, the current project aims to perform sentiment classifcation but bilingually in Spanish and English. This is achieved by finetuning several BERT-based models on English and Spanish data simultanously using HuggingFace's Trainer from their ```transformers``` package.  The pipeline is a such: 

### ```(E1) FINE-TUNING```
Fine-tune several BERT-based models on labelled Spanish & English Twitter data (3 labels: neutral, negative, positve).

### ```(E2) EVALUATION```
Evaluate the models on test data, extracting overall performance as well as performance stratified by language. 

By seperating the test set into the individual languages, potential disparities in performance across the two languages is explored. 

## Reproducibility 
To reproduce the results, follow the instructions in the [Pipeline](https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment#pipeline) section.

NB! Be aware that finetuning BERT-based models is a computationally heavy task. Cloud computing (e.g., [UCloud](https://cloud.sdu.dk/) with high amounts of ram (or a good GPU) is encouraged.

## Project Structure
The repository is structured as such: 
| <div style="width:120px"></div>| Description |
|---------|:-----------|
| ```results``` | Results from running ```finetune.py```: loss curves, metrics (all + per language), predictions data per example in test set (all + per language). |
| ```visualisations``` | Tables and plots from running ```visualise.py``` |
| ```models``` | Models saved here |
| ```src``` | Scripts to run finetuning + evaluation and to visualise results. Contains helper functions in the ```modules``` folder.|
| ```requirements.txt``` | Necessary packages to be installed |
| ```setup.sh``` |  Run to install ```requirements.txt``` within newly created ```env```. |
| ```git-lfs-setup.sh``` |  Run to install ```gif-lfs``` necessary for pushing models to Hugging Face hub. |

## Pipeline
The pipeline has been tested on Ubuntu v22.10, Python v3.10.7 ([UCloud](https://cloud.sdu.dk/), Coder Python 1.77.3). Python's [venv](https://docs.python.org/3/library/venv.html) needs to be installed for the pipeline to work.

### Installing TASS 
If you wish to run the pipeline with all data, install the [TASS 2020 (Spanish)](http://tass.sepln.org/2020/?wpdmpro=task-1-train-and-dev-sets) and place it in the ```data``` folder. Ensure that the data follows the structure and naming conventions described in [data/README.md](https://github.com/MinaAlmasi/finetuning-BERT-bilingual-sentiment/tree/main/data). 

### General Setup
Secondly, create a virtual environment (```env```) and install necessary requirements by running: 
```
bash setup.sh
```

### Extra Setup (Pushing to the HF Hub)
Pushing models to the Hugging Face Hub is disabled by default in all scripts, so you can skip this setup if you are not interested in this functionality.

If you wish to push models to the Hugging Face Hub, you need firstly save a [Hugging Face token](https://huggingface.co/docs/hub/security-tokens) in a .txt file called ```token.txt``` in the main folder. (token.txt is in ```.gitignore``` and will not be pushed!)

Then, install git-lfs. Note that this wil ```SUDO``` install to your system, do at own risk!
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
python src/finetune.py -mdl {MODEL} -epochs {N_EPOCHS} -hub {PUSH_TO_HUB_BOOL} -download {DOWNLOAD_MODE}
```
NB! Remember to activate the ```env``` first (by running ```source ./env/bin/activate```)

| <div style="width:80px">Arg</div>    | Description                             | <div style="width:120px">Default</div>    |
| :---        |:---                                                                                        |:---             |
|```-mdl```   |  Model to be fine-tuned. Choose between 'mDeBERTa', 'mBERT' or 'xlm-roberta'               | xlm-roberta     |
|```-hub```   | Whether to push to Hugging Face Hub. 'True' if yes, else write 'False'                     | False               |
|```-epochs```| MAX epochs the model should train for (if not stopped after 3 epochs with no improvement)  | 30              |
|```-download```| Write 'force_redownload' to redownload cached datasets. Useful if cache is corrupt.      | None            |

## Results 
Insert results here! (Loss curves, evaluation)

## Author 
This repository was created by Mina Almasi:

* github user: @MinaAlmasi
* student no: 202005465, AUID: au675000
* mail: mina.almasi@post.au.dk

## References
Barbieri, F., Anke, L. E., & Camacho-Collados, J. (2022). XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond (arXiv:2104.12250). arXiv. https://doi.org/10.48550/arXiv.2104.12250

Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive Text Embedding Benchmark (arXiv:2210.07316). arXiv. https://doi.org/10.48550/arXiv.2210.07316

