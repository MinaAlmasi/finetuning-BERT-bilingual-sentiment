Place your [TASS 2020](http://tass.sepln.org/2020/?wpdmpro=task-1-train-and-dev-sets,) dataset here!

Please note that the dataset is only allowed to be used in accordance with the ```TASS``` license: 
http://tass.sepln.org/tass_data/download.php 

The data folder should be structured with subdirectories EXACTLY as such:
```
├── README.md
├── dev
│   ├── cr.tsv
│   ├── es.tsv
│   ├── mx.tsv
│   ├── pe.tsv
│   └── uy.tsv
└── train
    ├── cr.tsv
    ├── es.tsv
    ├── mx.tsv
    ├── pe.tsv
    └── uy.tsv
````
The "dev" folder will eventually be seperated into test and validation when loading ```TASS``` in the script.