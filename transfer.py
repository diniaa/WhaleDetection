import torch
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import datasets, transforms
from preprocess import get_images
import os
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import json


RANDOM_STATE = 42
INPUT_SIZE = 224, 224
WAVE_LENGTH = 4000 # Number of samples in the audio wave,
# which is the sample rate multiplied by the duration
BATCH_SIZE = 128
device = torch.device("cpu")

data_raw_path = "./data/train/"
data_processed_path = "./processed_data/"
img_dir = "./img_data"

get_data = False
if get_data :
    get_images(data_raw_path, img_dir, input_shape = INPUT_SIZE, wave_length= WAVE_LENGTH)

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

resnet = resnet18(pretrained = True)
class CnnTransferNet(nn.Module):
    def __init__(self):
        super(CnnTransferNet,self).__init__()
        
        self.resnet =  resnet
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x

my_net = CnnTransferNet()
        
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

class CustomDataSet(Dataset):
    
    def __init__(self, main_dir, transform, targets):
        
        self.main_dir = main_dir
        self.transform = transform
        self.imgs = os.listdir(main_dir)
        self.targets = targets
        

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, self.targets[idx]
    



targets = torch.load(data_processed_path + "target.pth")
my_dataset = CustomDataSet(img_dir, data_transforms, targets)

train_idx, valid_idx= train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        random_state= RANDOM_STATE,
        stratify=targets)

# We use SubsetRandomSampler as a strategy for drawing batches 
train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)


train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(my_dataset, batch_size=2*BATCH_SIZE, sampler=valid_sampler)

criterion = nn.BCELoss()

def training_loop(model,train_loader,criterion, n_epochs = 10):
    
    train_loss = 0
    trainable_parameters = model.resnet.fc.parameters()
    optim = torch.optim.SGD(trainable_parameters, lr=0.001, momentum=0.9)
    model.train(True)
    pbar = tqdm(n_epochs)
    for epoch in range(n_epochs):
        
        for batch_idx, (x, target) in enumerate(train_loader):
            optim.zero_grad()

            Y_pred = model(x).view(-1,1)
            loss = criterion(Y_pred, target.view(-1,1))
            
            if epoch == n_epochs -1 :
                train_loss += loss.item() * len(x)
                
            if batch_idx % 10 ==0 :
                print("Current loss at epoch {}, batch {} is : {}".format(epoch, batch_idx, loss.item()))
            
            loss.backward()
            optim.step()
        pbar.update(1)
    
    return train_loss / (0.8 * len(train_loader.dataset))

train_loss = training_loop(my_net,train_loader,criterion, n_epochs = 10)

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
    accuracy =  correct / (0.2 * len(valid_loader.dataset))
    valid_loss = valid_loss / (0.2 * len(valid_loader.dataset))
    return accuracy, valid_loss

acc, valid_loss = eval(my_net, valid_loader, criterion)
results = {"training_loss" : train_loss, "accuracy" : acc, "validation_loss": valid_loss}

with open("./results/CNNTransferLearning.json", "w") as fp:
    fp.write(json.dumps(results))

#save model

torch.save(my_net.state_dict(), "./models/CNNTransferLearning.pth")