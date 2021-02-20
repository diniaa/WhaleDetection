import numpy as np
import torch

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from model_scratch import MLP, Conv1D, LSTM
from models import CNN
import json
import matplotlib.pyplot as plt


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
