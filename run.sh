#!/bin/bash

# activate virtual environment (works only if setup.sh has been run only !)
source ./env/bin/activate

# finetune mBERT 
echo -e "[INFO:] Finetuning mBERT ..." # user msg 
python3 src/finetune.py -mdl mBERT -hub True

# finetune mDistilBERT
echo -e "[INFO:] Finetuning mDistilBERT ..." # user msg 
python3 src/finetune.py -mdl mDistilBERT -hub True

# deactivate virtual environment
deactivate

# celebratory user msg ! 
echo -e "\n [INFO:] Finetunes complete!"