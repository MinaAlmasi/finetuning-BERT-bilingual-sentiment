The src folder contains the following scripts that can be run by typing ```python src/XX.py``` with the ```env``` activated:

| <div style="width:120px"></div>| Description |
|---------|:-----------|
| ```finetune.py``` | Script which loads in data, tokenizes, finetunes and evaluates models (choose models with argparse, see ```Pipeline``` in main readme).|
| ```visualise.py```  | Script to visualise the results from model finetuning and evaluation.|
| ```inference.py```  | Script to run the finetuned ```mDeBERTa``` on example text, returning sentiment label and score.|
 
The abovementioned scripts rely on functions defined in the ```modules``` folder.