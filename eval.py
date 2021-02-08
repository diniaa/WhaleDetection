#TODO: implement ROC evaluation

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from model_scratch import MLP, Conv1D, LSTM


import matplotlib.pyplot as plt

RANDOM_STATE = 42
torch.manual_seed(42)
np.random.seed(RANDOM_STATE)
BATCH_SIZE = 512
device = torch.device("cpu")
    
path_to_output = "./results/"
data_processed_path = "./processed_data/"


valid = torch.load(data_processed_path + "valid.pth", map_location = device)
valid_loader = torch.utils.data.DataLoader(
                 dataset=valid,
                 batch_size= BATCH_SIZE,
                 shuffle=False)
    


model_mlp = MLP(4000)
model_conv = Conv1D()
model_lstm = LSTM()

model_path = "./models/"
state_dict = torch.load(model_path + "MLP_base.pt")
model_mlp.load_state_dict(state_dict)

state_dict = torch.load(model_path + "CNN_base.pt")
model_conv.load_state_dict(state_dict)

state_dict = torch.load(model_path + "LSTM_base.pt")
model_lstm.load_state_dict(state_dict)

def eval(model, valid_loader):
    
    y_true = list()
    y_pred = list()
    model.eval()
    with torch.no_grad():
        for data, target in valid_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data).view_as(target).detach().cpu()
            
            y_true.append(target)
            y_pred.append(output)
    pred_tensor = torch.cat(y_pred)
    true_tensor = torch.cat(y_true)
    
    
    return true_tensor, pred_tensor

true_tensor, pred_tensor = eval(model_mlp, valid_loader)
lr_fpr, lr_tpr, _ = roc_curve(true_tensor, pred_tensor)
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='MLP')


true_tensor, pred_tensor = eval(model_conv, valid_loader)
lr_fpr, lr_tpr, _ = roc_curve(true_tensor, pred_tensor)
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='CNN')
  
   
true_tensor, pred_tensor = eval(model_lstm, valid_loader)
lr_fpr, lr_tpr, _ = roc_curve(true_tensor, pred_tensor)
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='LSTM')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
#save plot
plt.savefig(path_to_output + "Base_ROC.png")
# show the plot
plt.show()




