#!/bin/bash

# activate virtual environment (works only if setup.sh has been run only !)
source ./env/bin/activate

# finetune mBERT 
echo -e "[INFO:] Finetuning mBERT ..." # user msg 
python3 src/finetune.py -mdl mBERT

# run xlm roberta
echo -e "[INFO:] Finetuning xlm-roberta ..." # user msg 
python3 src/finetune.py -mdl xlm-roberta

# finetune mDeBERTa
echo -e "[INFO:] Finetuning mDeBERTa ..." # user msg 
python3 src/finetune.py -mdl mDeBERTa

# deactivate virtual environment
deactivate

# celebratory user msg ! 
echo -e "\n [INFO:] Finetunes complete!"