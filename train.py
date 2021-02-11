import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import load_dataset
from sklearn.model_selection import train_test_split

from model_scratch import MLP, Conv1D, LSTM
import json


# settings

WAVE_LENGTH = 4000 # number of samples in the audio files

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
device = torch.device("cpu")


annotations = pd.read_csv("./data/train.csv")
data_path = "./data/train"
path_to_output = "./results/"
data_output_path = "./processed_data/"

get_data = False

if get_data:
    
    X ,Y = load_dataset(data_path, annotations)
    
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.2
                                                          , random_state = RANDOM_STATE
                                                          , stratify = Y)
    
    X_train = torch.from_numpy(X_train).float()
    X_valid = torch.from_numpy(X_valid).float()
    Y_train = torch.from_numpy(Y_train).float()
    Y_valid = torch.from_numpy(Y_valid).float()
    
    train = torch.utils.data.TensorDataset(X_train, Y_train)
    valid = torch.utils.data.TensorDataset(X_valid, Y_valid)
    
    
    # saving datasets for further training
    torch.save(train, data_output_path + "train.pth")
    torch.save(valid, data_output_path + "valid.pth")
    
else :
    train = torch.load(data_output_path + "train.pth", map_location = device)
    valid = torch.load(data_output_path + "valid.pth", map_location = device)

BATCH_SIZE = 32

train_loader = torch.utils.data.DataLoader(
                 dataset=train,
                 batch_size=BATCH_SIZE,
                 shuffle=True)

valid_loader = torch.utils.data.DataLoader(
                 dataset=valid,
                 batch_size= 8 * BATCH_SIZE,
                 shuffle=False)


# Set the loss function for the classification problem :
# We'll be using the Binary Cross Entropy loss function

criterion = nn.BCELoss()


model_mlp = MLP(WAVE_LENGTH)
model_conv = Conv1D()
model_lstm = LSTM()


results = {"mlp_base" : {}, "cnn_base" : {}, "lstm_base" : {}}

models = {"mlp_base" :{"model": model_mlp, "epochs" : 40},
               "cnn_base" : {"model": model_conv, "epochs" : 20},
               "lstm_base" : {"model": model_lstm, "epochs" : 5}}


# Training loop function that performs stochastic gradient descent on a mini batch

def training_loop(model,train_loader,criterion, n_epochs = 40):
    
    train_loss = 0
    optim = torch.optim.Adam(model.parameters())
    model.train()
    pbar = tqdm(n_epochs)
    for epoch in range(n_epochs):
        
        for batch_idx, (x, target) in enumerate(train_loader):
            optim.zero_grad()

            Y_pred = model(x).view(-1,1)
            loss = criterion(Y_pred, target.view(-1,1))
            
            if epoch == n_epochs -1 :
                train_loss += loss.item() * len(x)
                
            if batch_idx % 100 ==0 :
                print("Current loss at epoch {}, batch {} is : {}".format(epoch + 1, batch_idx, loss.item()))
            
            loss.backward()
            optim.step()
        pbar.update(1)
    
    return train_loss / len(train_loader.dataset)



def eval(model, valid_loader, criterion):
    
    model.train(False)
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(valid_loader):
            out = model(x).view(-1,1)
            loss = criterion(out, target.view(-1,1))
            valid_loss += loss.item() * len(x)
            prediction = out >= 0.5
            correct += prediction.eq(target.view(-1,1)).sum().item()
    accuracy =  correct / len(valid_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)
    return accuracy, valid_loss
    
    


for model_type in models:
    
    model, n_epochs = models[model_type]["model"], models[model_type]["epochs"]
    train_loss = training_loop(model,train_loader,criterion, n_epochs = n_epochs)
    acc, valid_loss = eval(model, valid_loader, criterion)
    results[model_type]['train_loss'] = train_loss
    results[model_type]['validation_loss'] = valid_loss
    results[model_type]['accuracy'] = acc

torch.save(model_mlp.state_dict(), "models/MLP_base.pt")
torch.save(model_conv.state_dict(), "models/CNN_base.pt")
torch.save(model_lstm.state_dict(), "models/LSTM_base.pt")

res_path = "results/baseline.json"
with open(res_path, "w") as fp:
    fp.write(json.dumps(results))
    


