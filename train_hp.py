import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import load_with_torchaudio
from sklearn.model_selection import train_test_split

import json

from itertools import product

from models import MLP

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


# For faster training we choose to load the dataset once, and save the data tensors, 
# rather than implementing a Dataset class with preprocessing transformations
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

train = torch.utils.data.TensorDataset(X_train, Y_train)
valid = torch.utils.data.TensorDataset(X_valid, Y_valid)

################################## MLP hyperparameter tuning ##############################

def get_info(combination):
    l1, l2, weight_decay, batch_size = combination
    return {"l1" : l1, "l2" : l2, "weight_decay" : weight_decay,"batch_size": batch_size }




def training_run(combination, criterion, train, valid, results, run):
    
    l1, l2, weight_decay, batch_size = combination
    model_path = "MLP_run_{}.pt".format(run)
    results[model_path] = dict()
    my_net = MLP(WAVE_LENGTH, l1, l2)

    my_net.to(device)

    optimizer = torch.optim.Adam(my_net.parameters(), weight_decay =weight_decay )



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
            
            if epoch == 9:
                training_loss += loss.item() * len(x)
        
            if batch_idx % 100 == 99:  # print every 100 mini-batches
                print("[ Epoch %d,Batch %2d] loss: %.3f" % (epoch + 1, batch_idx + 1,
                                                loss.item()))
                
        
        pbar.update(1)
    
    results[model_path]["training_loss"] = training_loss/ len(train)
    
    print("Finished Training !")
    print("Start Evaluating !")
    
    # Validation loss
    valid_loss = .0
    correct = 0
    with torch.no_grad():
        for batch_idx,(x, target) in enumerate(valid_loader):
            x, target = x.to(device), target.to(device)

            outputs = my_net(x).view(-1,1)
            prediction = outputs >= 0.5
            correct += prediction.eq(target.view(-1,1)).sum().item()

            loss = criterion(outputs, target.view(-1,1))
            valid_loss += loss.item() * len(x)
                  
    
    results[model_path]["validation_loss"] = valid_loss/ len(valid)
    results[model_path]["accuracy"] = correct/ len(valid)
    
    torch.save(my_net.state_dict(), "./models/" + model_path)
        
   
criterion = nn.BCELoss()
params = {
    "l1": [256,  128],
    "l2": [64, 32],
    "weight_decay": np.logspace(-5, -4, 2),
    "batch_size": [64, 128]
}

def main(params):
    
    results = dict()
    run = 0
    info = dict()
    
    for combination in product(*params.values()):
        
        run += 1
        info["MLP_run_{}.pt".format(run)] = get_info(combination)
        
        print("""Starting training for hyperparameters : l1 = {}, l2 = {},
              weight_decay = {}, batch_size = {}""".format(*combination))
        training_run(combination, criterion, train, valid, results, run)
    
    with open("./results/run_info.json", "w") as fp:
        fp.write(json.dumps(info))
    with open("./results/run_results.json", "w") as fp:
        fp.write(json.dumps(results))
    
    return results


##################################  CNN hyperparameter tuning ##############################