# WhaleDetection

This repository presents a deep learning approach to the [Whale Detection Challenge](https://www.kaggle.com/c/whale-detection-challenge) hosted on Kaggle :

## Dataset

The dataset is provided within the Kaggle challenge in this [link](https://www.kaggle.com/c/whale-detection-challenge/data), the corresponding file is ```whale_data.zip```

## Requirements :

* pytorch : Deep learning & neaural network framework
* torchaudio : Audio processing library for pytorch
* soundfile : Backend library for torchaudio (used for reading/writing)
* librosa : Audio analysis package
* json : Reading and writing json files (serializing dictionnaries that store networks' results) 
* aifc, matplotlib, pandas .

## Repository structure :
The repository is organized as the following tree
```
├── README.md
├── preprocess.py
├── train.py
├── model_scratch.py
├── train_hp.py
├── models.py
├── eval.py
├── transfer.py
├── models
│── results
└── .gitignore
```

* ```preprocess.py``` is used for data preprocessing
* ```model_scratch.py``` contains the neural network classes trained in the first iteration (baseline)
* ```train.py``` provides code to train and validate the models in ```model_scratch.py```
* ```models.py``` contains the neural network classes trained in the second iteration
* ```train.py``` provides code to train and validate the models in ```models.py```
* ```eval.py``` further evaluates the trained models according to ROC scores
* ```transfer.py``` provides a transfer learning approach

The folders  ```data```, ```processed_data``` and ```img_data``` are used to store datasets and aren't tracked in this git repository (put in ```.gitignore```)
