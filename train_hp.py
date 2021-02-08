import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import load_with_torchaudio
from sklearn.model_selection import train_test_split

from model_scratch import MLP, Conv1D, LSTM
import json

import torchaudio

RANDOM_STATE = 42
torch.manual_seed(42)
np.random.seed(RANDOM_STATE)
device = torch.device("cpu")
    
annotations = pd.read_csv("./data/train.csv")
data_raw_path = "./data/train"
path_to_output = "./results/"
data_processed_path = "./processed_data/"

get_data = False

if get_data:
  
    X ,Y = load_with_torchaudio(data_raw_path, annotations)
    
    
    torch.save(X, data_processed_path + "data.pth")
    torch.save(Y, data_processed_path + "target.pth")

else : 
    X = torch.load(data_processed_path + "data.pth")
    Y = torch.load(data_processed_path + "target.pth")

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.2
                                                              , random_state = RANDOM_STATE
                                                              , stratify = Y)
