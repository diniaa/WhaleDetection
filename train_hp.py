import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from preprocess import load_with_torchaudio
from sklearn.model_selection import train_test_split

import json

from functools import partial
from itertools import product

from models import MLP, CNN, init_weights

"""
The following script trains different types of neural network architectures 
with different hyperparameter configurations for tuning.
Each trained model is given a run index , each index corresponding to a different combination
of hyperparameters
Each model state dict is then saved to disk for further inference and evaluation
We compute mainly 3 sorts of indicators in this script :
training loss, validation loss, and validation accuracy.

The hyperparameters for a MLP network are : hidden layers size ,batch size, 
and weight decay for L2 regularization

The hyperparameters for a CNN network are : number of feature maps in convolution layers,
and the initialization mode for the network's weihgts

"""

######################################### Settings ##########################################

WAVE_LENGTH = 4000
BATCH_SIZE = 64
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
device = torch.device("cpu")
    
annotations = pd.read_csv("./data/train.csv")
data_raw_path = "./data/train"
path_to_output = "./results/"
data_processed_path = "./processed_data/"


# For faster training we choose to load the dataset once, and save the data tensors.

get_data = False

if get_data:
  
    X ,Y = load_with_torchaudio(data_raw_path, annotations)
    
    # save dataset and target as tensors
    torch.save(X, data_processed_path + "data.pth")
    torch.save(Y, data_processed_path + "target.pth")

else : 
    X = torch.load(data_processed_path + "data.pth")
    Y = torch.load(data_processed_path + "target.pth")

# split into training and validation sets
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.2
                                                             , random_state = RANDOM_STATE
                                                              , stratify = Y)

train = torch.utils.data.TensorDataset(X_train, Y_train)
valid = torch.utils.data.TensorDataset(X_valid, Y_valid)




################################## MLP hyperparameter tuning ##############################

# get the information of the hyperparameters' configuration for a given run

def get_info(combination, model_type = "mlp"):
    
    if model_type == "mlp":
        l1, l2, weight_decay, batch_size = combination
        return {"l1" : l1, "l2" : l2, "weight_decay" : weight_decay,"batch_size": batch_size }
    
    if model_type =="cnn":
        n_featuremap_1, n_featuremap_2, mode = combination
        return {"n_featuremap_1" : n_featuremap_1, "n_featuremap_2" : n_featuremap_2,
                "initialization_mode" : mode }
    

# performs a training + evaluation run for a specific network + hyperparameter configuration
# computes all 3 performance indicators
def training_run_mlp(combination, criterion, train, valid, run):
    
    l1, l2, weight_decay, batch_size = combination
    model_path = "MLP_run_{}.pt".format(run)
    results[model_path] = dict()
    my_net = MLP(WAVE_LENGTH, l1, l2)

    my_net.to(device)

    optimizer = torch.optim.Adam(my_net.parameters(), weight_decay = weight_decay )



    train_loader = torch.utils.data.DataLoader(
                   dataset=train,
                   batch_size=batch_size,
                   shuffle=True)
    
    valid_loader = torch.utils.data.DataLoader(
                 dataset=valid,
                 batch_size= 2 * batch_size,
                 shuffle=False)

    for epoch in range(10):  # loop over the training dataset multiple times
    
        training_loss = .0
        pbar = tqdm(10)
        
        for batch_idx, (x, target) in enumerate(train_loader):
            x, target = x.to(device), target.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = my_net(x).view(-1,1)
            loss = criterion(outputs, target.view(-1,1))
            loss.backward()
            optimizer.step()
            
            if epoch == 9: # update training loss in the last epoch
                training_loss += loss.item() * len(x)
        
            if batch_idx % 100 == 99:  # print every 100 mini-batches
                print("[ Epoch %d,Batch %2d] loss: %.3f" % (epoch + 1, batch_idx + 1,
                                                loss.item()))
                
        
        pbar.update(1)
    # update results
    results[model_path]["training_loss"] = training_loss/ len(train)
    
    print("Finished Training !")
    print("Start Evaluating !")
    
    # Validation loss
    valid_loss = .0
    correct = 0
    thres = 0.5
    with torch.no_grad():
        for batch_idx,(x, target) in enumerate(valid_loader):
            x, target = x.to(device), target.to(device)

            outputs = my_net(x).view(-1,1)
            prediction = outputs >= thres
            correct += prediction.eq(target.view(-1,1)).sum().item()

            loss = criterion(outputs, target.view(-1,1))
            valid_loss += loss.item() * len(x)
                  
    # update results
    results[model_path]["validation_loss"] = valid_loss/ len(valid)
    results[model_path]["accuracy"] = correct/ len(valid)
    
    # save model in disk
    torch.save(my_net.state_dict(), "./models/" + model_path)
        
   
criterion = nn.BCELoss()
params = {
    "l1": [256,  128],
    "l2": [64, 32],
    "weight_decay": np.logspace(-5, -4, 2),
    "batch_size": [64, 128]
}

# initialize results and configuration info dictionnaries
results = dict()
info = dict()


# main function that performs a grid search on the hyperparameters grid and
# calls the training function for every possible combination
def tune_mlp(params):
    
    run = 0
    
    for combination in product(*params.values()):
        
        run += 1
        info["MLP_run_{}.pt".format(run)] = get_info(combination)
        
        print("""Starting training for hyperparameters : l1 = {}, l2 = {},
              weight_decay = {}, batch_size = {}""".format(*combination))
        training_run_mlp(combination, criterion, train, valid, run)
    
    # save model results and configuration infos
    with open("./results/run_info.json", "w") as fp:
        fp.write(json.dumps(info))
    with open("./results/run_results.json", "w") as fp:
        fp.write(json.dumps(results))
    

# start tuning 
tune_mlp(params)



##################################  CNN hyperparameter tuning ##############################

train_loader = torch.utils.data.DataLoader(
                   dataset=train,
                   batch_size=BATCH_SIZE,
                   shuffle=True)
    
valid_loader = torch.utils.data.DataLoader(
                 dataset=valid,
                 batch_size= 2 * BATCH_SIZE,
                 shuffle=False)

def training_run_cnn(combination, criterion, train_loader, valid_loader, run):
    
    n_featuremap_1, n_featuremap_2, mode = combination
    model_path = "CNN_run_{}.pt".format(run)
    results[model_path] = dict()
    
    # initialize the network with the given configuration
    my_net = CNN(n_featuremap_1 = n_featuremap_1, n_featuremap_2 = n_featuremap_2)
    
    # initialize weights with the given mode
    my_net.apply(partial(init_weights, mode = mode))
    my_net.to(device)

    optimizer = torch.optim.Adam(my_net.parameters())

    for epoch in range(10):  # loop over the training dataset multiple times
    
        training_loss = .0
        pbar = tqdm(10)
        
        for batch_idx, (x, target) in enumerate(train_loader):
            x, target = x.to(device), target.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = my_net(x).view(-1,1)
            loss = criterion(outputs, target.view(-1,1))
            loss.backward()
            optimizer.step()
            
            if epoch == 9: # update training loss in the last epoch
                training_loss += loss.item() * len(x)
        
            if batch_idx % 100 == 99:  # print every 100 mini-batches
                print("[ Epoch %d,Batch %2d] loss: %.3f" % (epoch + 1, batch_idx + 1,
                                                loss.item()))
                
        
        pbar.update(1)
    # update results
    results[model_path]["training_loss"] = training_loss/ len(train)
    
    print("Finished Training !")
    print("Start Evaluating !")
    
    # Validation loss
    valid_loss = .0
    correct = 0
    thres = 0.5
    with torch.no_grad():
        for batch_idx,(x, target) in enumerate(valid_loader):
            x, target = x.to(device), target.to(device)

            outputs = my_net(x).view(-1,1)
            prediction = outputs >= thres
            correct += prediction.eq(target.view(-1,1)).sum().item()

            loss = criterion(outputs, target.view(-1,1))
            valid_loss += loss.item() * len(x)
                  
    
    # update results
    results[model_path]["validation_loss"] = valid_loss/ len(valid)
    results[model_path]["accuracy"] = correct/ len(valid)
    
    # save model in disk
    torch.save(my_net.state_dict(), "./models/" + model_path)
        
   

params = {
    "n_featuremap_1": [8,  16],
    "n_featuremap_2": [33, 64],
    "initialization_mode": ["normal", "uniform", "zero"]
}

# main function that performs a grid search on the hyperparameters grid and
# calls the training function for every possible combination of hyperparameters
def tune_cnn(params):
    
    run = 0
    
    for combination in product(*params.values()):
        
        run += 1
        info["CNN_run_{}.pt".format(run)] = get_info(combination, model_type = "cnn")
        
        print("""Starting training for hyperparameters : n_featuremap_1 = {},
              n_featuremap_2 = {}, mode = {}""".format(*combination))
        training_run_cnn(combination, criterion, train_loader, valid_loader, run)
    
    with open("./results/run_info.json", "w") as fp:
        fp.write(json.dumps(info))
    with open("./results/run_results.json", "w") as fp:
        fp.write(json.dumps(results))
        
# start tuning
tune_cnn(params)

