The src folder contains the following scripts that can be run by typing ```python src/XX.py``` with the ```env``` activated:

| <div style="width:120px"></div>| Description |
|---------|:-----------|
| ```finetune.py``` | Script which loads in data, tokenizes, finetunes and evaluates models (choose models with ```-mdl``` arg, see ```Pipeline``` in main readme).|
| ```inference.py```  | Script to run the finetuned ```mDeBERTa``` on example text, returning sentiment label and score.|
| ```visualise.py```  | Script to visualise the results from model finetuning and evaluation.|
 
The abovementioned scripts rely on functions defined in the ```modules``` folder.