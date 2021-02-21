import numpy as np
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset

from model_scratch import MLP, Conv1D, LSTM
from models import CNN

import json
import matplotlib.pyplot as plt

from PIL import Image

"""
The following script loads already trained networks and evaluates them based
on roc score on the validation set

"""

######################################## Settings #############################################
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
BATCH_SIZE = 512
device = torch.device("cpu")
    
output_path = "./results/"
data_processed_path = "./processed_data/"

####################################### First iteration ########################################

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
    pbar = tqdm(len(valid_loader))
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
            pbar.update(1)
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
plt.savefig(output_path + "Base_ROC.png")
# show the plot
plt.show()

####################################### Second iteration ########################################

data = torch.load(data_processed_path + "data.pth")
targets = torch.load(data_processed_path + "target.pth")

X_train, X_valid, Y_train, Y_valid = train_test_split(data, targets, test_size = 0.2
                                                             , random_state = RANDOM_STATE
                                                              , stratify = targets)

valid = torch.utils.data.TensorDataset(X_valid, Y_valid)
valid_loader_2 = torch.utils.data.DataLoader(
                 dataset=valid,
                 batch_size= BATCH_SIZE,
                 shuffle=False)

best_cnn = "CNN_run_10.pt"
# get the configuration info for the selected run
with open(output_path + "run_info.json" , "r") as fp:
    run_info = json.load(fp)

f1, f2 = run_info[best_cnn]['n_featuremap_1'], run_info[best_cnn]['n_featuremap_2']
tuned_cnn = CNN(n_featuremap_1=f1, n_featuremap_2=f2)
state_dict = torch.load(model_path + best_cnn)
tuned_cnn.load_state_dict(state_dict)

true_tensor, pred_tensor = eval(model_conv, valid_loader)
print("base CNN roc score : %.4f" % roc_auc_score(true_tensor, pred_tensor))
lr_fpr, lr_tpr, _ = roc_curve(true_tensor, pred_tensor)
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='base_CNN')

true_tensor, pred_tensor = eval(tuned_cnn, valid_loader_2)
print("tuned CNN roc score : %.4f" % roc_auc_score(true_tensor, pred_tensor))
lr_fpr, lr_tpr, _ = roc_curve(true_tensor, pred_tensor)
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='tuned_CNN')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
#save plot
plt.savefig(output_path + "tuned_vs_base_roc.png")
# show the plot
plt.show()

####################################### Transfer ########################################

# load trained network
resnet = resnet18(pretrained = True)
class CnnTransferNet(nn.Module):
    def __init__(self):
        super(CnnTransferNet,self).__init__()
        
        # freeze all layers except fully connected for which output features are changed
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

state_dict = torch.load(model_path + "CNNTransferLearning.pt" )
my_net.load_state_dict(state_dict)

# we need to reset the same pipeline from the transfer learning script
# for evaluation

INPUT_SIZE = 224,224
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
img_dir = "./img_data/"

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
    

my_dataset = CustomDataSet(img_dir, data_transforms, targets)
train_idx, valid_idx= train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        random_state= RANDOM_STATE,
        stratify=targets)
valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
valid_loader_3 = torch.utils.data.DataLoader(my_dataset, batch_size = 256, sampler = valid_sampler)

true_tensor, pred_tensor = eval(my_net, valid_loader_3)
print("transfer net roc score : %.4f" % roc_auc_score(true_tensor, pred_tensor))
lr_fpr, lr_tpr, _ = roc_curve(true_tensor, pred_tensor)
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='transfer_net')

true_tensor, pred_tensor = eval(tuned_cnn, valid_loader_2)
print("tuned CNN roc score : %.4f" % roc_auc_score(true_tensor, pred_tensor))
lr_fpr, lr_tpr, _ = roc_curve(true_tensor, pred_tensor)
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='tuned_CNN')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
#save plot
plt.savefig(output_path + "transfer_vs_tuned_net_roc.png")
# show the plot
plt.show()