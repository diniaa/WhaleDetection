import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from preprocess import load_dataset
from sklearn.model_selection import train_test_split

from models_baseline import MLP, Conv1D, LSTM
import json


"""
The following script trains our baseline neural network solutions 
for the Whale Detection Challenge .
Each model state dict is then saved to disk for further inference and evaluation
We compute mainly 3 sorts of indicators in this script :
training loss, validation loss, and validation accuracy.

"""


########################################## Settings ##########################################

WAVE_LENGTH = 4000 # number of samples in each audio signal
BATCH_SIZE = 32


# fix random states to have same train/valid partition and same drawn batches from data loader
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


device = torch.device("cpu")


annotations = pd.read_csv("./data/train.csv")
data_raw_path = "./data/train"
data_processed_path = "./processed_data/"


# We load our dataset once with preprocessing then 
# load directly the training and validation tensors for faster training

get_data = False
if get_data:
    
    X ,Y = load_dataset(data_raw_path, annotations)
    
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.2
                                                          , random_state = RANDOM_STATE
                                                          , stratify = Y)
    
    X_train = torch.from_numpy(X_train).float()
    X_valid = torch.from_numpy(X_valid).float()
    Y_train = torch.from_numpy(Y_train).float()
    Y_valid = torch.from_numpy(Y_valid).float()
    
    train = torch.utils.data.TensorDataset(X_train, Y_train)
    valid = torch.utils.data.TensorDataset(X_valid, Y_valid)
    
    
    # save training and validation datasets
    torch.save(train, data_processed_path + "train.pth")
    torch.save(valid, data_processed_path + "valid.pth")
    
else :
    train = torch.load(data_processed_path + "train.pth", map_location = device)
    valid = torch.load(data_processed_path + "valid.pth", map_location = device)



# Initialize the data loaders
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

# initialize the models
model_mlp = MLP(WAVE_LENGTH)
model_conv = Conv1D()
model_lstm = LSTM()


# initialize results dictionnary

results = {"mlp_base" : {}, "cnn_base" : {}, "lstm_base" : {}}

models = {"mlp_base" :{"model": model_mlp, "epochs" : 40},
               "cnn_base" : {"model": model_conv, "epochs" : 20},
               "lstm_base" : {"model": model_lstm, "epochs" : 5}}


# wrap the training process in a function

def training_loop(model,train_loader,criterion, n_epochs = 40):
    
    train_loss = 0
    optim = torch.optim.Adam(model.parameters())
    
    # set the model in training mode 
    model.train(True)
    
    pbar = tqdm(n_epochs)
    for epoch in range(n_epochs):
        
        for batch_idx, (x, target) in enumerate(train_loader):
            # zero the parameter gradients
            optim.zero_grad()
            
            # forward pass
            Y_pred = model(x).view(-1,1)
            loss = criterion(Y_pred, target.view(-1,1))
            
            # compute the training loss over the last epoch 
            # Since the loss is averaged we need to multiply each time by the number of samples
            # in the batch
            
            if epoch == n_epochs -1 :
                train_loss += loss.item() * len(x)
            
            # print loss every 100 batches
            if batch_idx % 100 == 99:
                print("Current loss at epoch {}, batch {} is : {}"
                      .format(epoch + 1, batch_idx + 1, loss.item()))
            
            # backward pass + optimize
            loss.backward()
            optim.step()
        pbar.update(1)
    
    return train_loss / len(train_loader.dataset)


# function for evaluation :
# evaluates loss and accuracy over the validation set

def eval(model, valid_loader, criterion):
    
    # set the model in evaluation mode
    model.train(False)
    
    valid_loss = 0
    correct = 0
    # set the classification threshold to 0.5
    thres = 0.5
    
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(valid_loader):
            out = model(x).view(-1,1)
            loss = criterion(out, target.view(-1,1))
            valid_loss += loss.item() * len(x)
            prediction = out >= thres 
            correct += prediction.eq(target.view(-1,1)).sum().item()
    accuracy =  correct / len(valid_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)
    return accuracy, valid_loss
    
    

# Training + evaluation for each model 

for model_type in models:
    
    model, n_epochs = models[model_type]["model"], models[model_type]["epochs"]
    train_loss = training_loop(model,train_loader,criterion, n_epochs = n_epochs)
    acc, valid_loss = eval(model, valid_loader, criterion)
    results[model_type]['train_loss'] = train_loss
    results[model_type]['validation_loss'] = valid_loss
    results[model_type]['accuracy'] = acc


# save the models for further evaluation

torch.save(model_mlp.state_dict(), "models/MLP_base.pt")
torch.save(model_conv.state_dict(), "models/CNN_base.pt")
torch.save(model_lstm.state_dict(), "models/LSTM_base.pt")


# save baseline results
res_path = "results/baseline.json"
with open(res_path, "w") as fp:
    fp.write(json.dumps(results))
    


