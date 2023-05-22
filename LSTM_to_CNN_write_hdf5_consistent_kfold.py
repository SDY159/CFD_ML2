#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:23:45 2023

@author: user
"""

import numpy as np
import os
import h5py
import torch
import logging
import torch.optim as optim
import random
from typing import Tuple, Dict, List
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
Tensor = torch.Tensor

TensorTuple = Tuple[torch.Tensor]
FloatTuple = Tuple[float]

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
logger = logging.getLogger(__name__)

#Generalization 
config = {
    "model": {
        "device": "cpu", # "cuda" or "cpu"
        "n_features": 1568, # since we are only using 1 feature, close price
        "hidden_dims": 1568,
        "num_layers": 4,
        "batch_size": 64,
        "dropout": 0.0,
        "seed": 12345,
        "path": "/bigdata/wonglab/syang159/CFD2/pytorch_particle_random_order/" + "LSTM_tanh_early_stopping_divide_dataloader/n_layer_4/bidirectional_1/scheduler_type1/lr_0.0001/epoch_100/kfold_4",
        "path_result": "/bigdata/wonglab/syang159/CFD2/pytorch_particle_random_order/LSTM_tanh_early_stopping_divide_dataloader/Result"
    },
    "training": {
        "training_h5_file": "/bigdata/wonglab/syang159/CFD2/Kfold_data/training_data_4.hdf5",
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.0001,
        "scheduler_step_size": 300,
        "block_size": 50,
        "ndata": 2500,
        'Job_number': 4,
        "T_0_for_cosine_annealing": 14,
        "T_mult_for_cosine_annealing": 1,
        "MSE_threshold": 0.0065,
        "es_counter": 3
    },
    "validating": {
        "validating_h5_file": "/bigdata/wonglab/syang159/CFD2/Kfold_data/validating_data_4.hdf5",
        "batch_size": 64,
        "block_size": 50,
        "stride": 3000,
        "ndata": 312,  #312
        "val_steps":10
    },
    "testing": {
        "testing_h5_file": "/bigdata/wonglab/syang159/CFD2/Kfold_data/testing_data.hdf5",
        "batch_size": 64,
        "block_size": 50,
        "stride": 3000,
        "ndata": 313  #313
    }
}

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)

logger.setLevel(logging.DEBUG)

def seed_torch(seed=config['model']['seed']):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
seed_torch()
g= torch.Generator()

#Dataloaders for training and testing
class CylinderDataset(Dataset):
    """Dataset for training flow around a cylinder embedding model

    Args:
        examples (List): list of training/testing example flow fields
        visc (List): list of training/testing example viscosities
    """
    def __init__(self, examples: List, params: List) -> None:
        """Constructor
        """
        self.examples = examples
        self.params = params

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {"data": self.examples[i], "params": self.params[i]}

class CylinderDataCollator:
    """Data collator for flow around a cylinder embedding problem
    """
    # Default collator
    def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Stack examples in mini-batch
        x_data_tensor =  torch.stack([example["data"] for example in examples])
        params_tensor =  torch.stack([example["params"] for example in examples])

        return {"data": x_data_tensor, "params": params_tensor}


# load training data
def createTrainingLoader(
    file_path: str,    
    block_size: int,
    ndata: int = -1,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:

    logging.info('Creating training loader')

    examples=[]
    params=[]
    
    with h5py.File(config["training"]["training_h5_file"],"r") as f:
        # Iterate through stored time-series

        params1 = torch.Tensor(f['params'])
        #print(params1.shape)
        pos_x = torch.Tensor(f['x'])
        pos_y = torch.Tensor(f['y'])
        pos_z = torch.Tensor(f['z'])
        data_series = torch.stack([pos_x, pos_y, pos_z], dim=2) ##torch.Size([1, 3, 196])
        data_series = data_series[:,:,:,torch.randperm(data_series.size(3), generator=g.manual_seed(config['model']['seed']))]
            # Stride over time-series
        for i in range(0, len(pos_x),1):
            for j in range(0, data_series.size(1) - block_size + 1, block_size):
            
                examples.append(data_series[i][j: j + block_size])
                params.append(params1[i])
                break

    data = torch.stack(examples, dim=0) ##torch.Size([37500, 1, 3, 196])
    params = torch.stack(params)
    mu = torch.tensor([torch.mean(data[:,:,0]), torch.mean(data[:,:,1]), torch.mean(data[:,:,2]), 
               torch.mean(params[:,0]), torch.mean(params[:,1]), torch.mean(params[:,2]), torch.mean(params[:,3]), torch.mean(params[:,4])])
    std = torch.tensor([torch.std(data[:,:,0]), torch.std(data[:,:,1]), torch.std(data[:,:,2]), 
               torch.std(params[:,0]), torch.std(params[:,1]), torch.std(params[:,2]), torch.std(params[:,3]), torch.std(params[:,4])])
        
    dataset = CylinderDataset(data, params)
    data_collator = CylinderDataCollator()
    training_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=data_collator, drop_last=False)
    return training_loader , mu, std


#load test data
def createValidatingLoader(
        file_path: str,
        block_size: int,
        ndata: int = -1,
        batch_size: int = 64,
        shuffle: bool =False,
    ) -> DataLoader:

    logging.info('Creating Validating loader')

    examples = []
    params = []
    with h5py.File(file_path, "r") as f:
        # Iterate through stored time-series

        params1 = torch.Tensor(f['params'])
        #print(params1.shape)
        pos_x = torch.Tensor(f['x'])
        pos_y = torch.Tensor(f['y'])
        pos_z = torch.Tensor(f['z'])
        data_series = torch.stack([pos_x, pos_y, pos_z], dim=2) ##torch.Size([1, 3, 196])
        data_series = data_series[:,:,:,torch.randperm(data_series.size(3), generator=g.manual_seed(config['model']['seed']))]
            # Stride over time-series
        for i in range(0, len(pos_x) ,1):
            for j in range(0, data_series.size(1) - block_size + 1, block_size):
            
                examples.append(data_series[i][j: j + block_size])
                params.append(params1[i])
                break
    data = torch.stack(examples, dim=0) ##torch.Size([37500, 1, 3, 196])
    params = torch.stack(params)
    mu = torch.tensor([torch.mean(data[:,:,0]), torch.mean(data[:,:,1]), torch.mean(data[:,:,2]), 
               torch.mean(params[:,0]), torch.mean(params[:,1]), torch.mean(params[:,2]), torch.mean(params[:,3]), torch.mean(params[:,4])])
    std = torch.tensor([torch.std(data[:,:,0]), torch.std(data[:,:,1]), torch.std(data[:,:,2]), 
               torch.std(params[:,0]), torch.std(params[:,1]), torch.std(params[:,2]), torch.std(params[:,3]), torch.std(params[:,4])])
    
    dataset = CylinderDataset(data, params)
    data_collator = CylinderDataCollator()
    validating_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=data_collator, drop_last=False)
    return validating_loader, mu, std

#load test data
def createTestingLoader(
        file_path: str,
        block_size: int,
        ndata: int = -1,
        batch_size: int = 64,
        shuffle: bool =False,
    ) -> DataLoader:

    logging.info('Creating testing loader')

    examples = []
    params = []
    with h5py.File(file_path, "r") as f:

        params1 = torch.Tensor(f['params'])
        #print(params1.shape)
        pos_x = torch.Tensor(f['x'])
        pos_y = torch.Tensor(f['y'])
        pos_z = torch.Tensor(f['z'])
        data_series = torch.stack([pos_x, pos_y, pos_z], dim=2) ##torch.Size([1, 3, 196])
        data_series = data_series[:,:,:,torch.randperm(data_series.size(3), generator=g.manual_seed(config['model']['seed']))]
            # Stride over time-series
        for i in range(0, len(pos_x) ,1):
            for j in range(0, data_series.size(1) - block_size + 1, block_size):
            
                examples.append(data_series[i][j: j + block_size])
                params.append(params1[i])
                break         
              
    data = torch.stack(examples, dim=0) ##torch.Size([37500, 1, 3, 196])
    params = torch.stack(params)
    mu = torch.tensor([torch.mean(data[:,:,0]), torch.mean(data[:,:,1]), torch.mean(data[:,:,2]), 
               torch.mean(params[:,0]), torch.mean(params[:,1]), torch.mean(params[:,2]), torch.mean(params[:,3]), torch.mean(params[:,4])])
    std = torch.tensor([torch.std(data[:,:,0]), torch.std(data[:,:,1]), torch.std(data[:,:,2]), 
               torch.std(params[:,0]), torch.std(params[:,1]), torch.std(params[:,2]), torch.std(params[:,3]), torch.std(params[:,4])])
    
    dataset = CylinderDataset(data, params)
    data_collator = CylinderDataCollator()
    testing_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=data_collator, drop_last=False)
    return testing_loader, mu, std

def _normalize(x: Tensor, mu: Tensor, std: Tensor) -> Tensor:
    x = (x - mu.unsqueeze(0).unsqueeze(-1)) / std.unsqueeze(0).unsqueeze(-1)
    return x
def _unnormalize(x: Tensor, mu: Tensor, std: Tensor) -> Tensor:
    return std[:3].unsqueeze(0).unsqueeze(-1)*x + mu[:3].unsqueeze(0).unsqueeze(-1)

#LSTM model
class LSTM_PT(nn.Module):
    def __init__(self, n_features, hidden_dims, num_layers, dropout=0.0):
        super(LSTM_PT, self).__init__()
        

        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
    
        self.lstm0 = nn.LSTM(    
            input_size = n_features, 
            hidden_size = hidden_dims,
            batch_first = True,
            dropout = dropout,
            num_layers = self.num_layers,
            bidirectional=False)
        self.lstm1 = nn.LSTM(    
            input_size = n_features, 
            hidden_size = hidden_dims,
            batch_first = True,
            dropout = dropout,
            num_layers = self.num_layers,
            bidirectional=True)  
        
        self.linear0 = nn.Linear(self.hidden_dims, self.hidden_dims)         
        self.linear1 = nn.Linear((self.hidden_dims)*2, self.hidden_dims) 
        self.linear2 = nn.Linear(self.hidden_dims, self.hidden_dims)  
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
                
        # LSTM inputs: (input, (h_0, c_0))
        #input of shape (seq_len, batch, input_size)....   input_size = num_features
        #or (batch, seq_len, input_size) if batch_first = True
        
        lstm0_out , (h1_n, c1_n) = self.lstm0(x) #hidden[0] = h_n, hidden[1] = c_n
        y_hat = self.tanh(self.linear0(lstm0_out))
        y_hat = self.tanh(self.linear2(y_hat))
       
 
 
        #Output: output, (h_n, c_n)
        #output is of shape (batch_size, seq_len, hidden_size) with batch_first = True
                       
        #output is shape (N, *, H_out)....this is (batch_size, out_features)
        
        return y_hat

def run_epoch(dataloader, mu, std, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, datasets in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()
        
        params = datasets['params']

        
        xin0 = datasets['data'][:,0]
        yin0 = datasets['data']
        x = torch.cat([xin0, params.unsqueeze(-1) * torch.ones_like(xin0[:,:1])], dim=1) #[B, 8, 196]
        #x = _normalize(x, mu, std)
        x = x.reshape(x.size(0), 1, x.size(1)*x.size(2))
        x = torch.stack([x]*50,axis=1)
        x = x.reshape(x.size(0), 50, x.size(3)) 
        x = x.to(config["model"]["device"])
        y= torch.cat([yin0, params.unsqueeze(1).unsqueeze(-1) * torch.ones_like(yin0[:,:,:1])], dim=2)
        y = y.reshape(y.size(0),50,-1).to(config["model"]["device"])
        
        
        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            
            optimizer.step()
        
        epoch_loss += loss.detach().item() / (len(dataloader)/2)
 
    lr = scheduler1.get_last_lr()[0]    
    return epoch_loss, lr

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, verbose=False, epoch: int = None, mse_threshold: float = None, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.verbose = verbose
        self.mse_threshold = mse_threshold
        self.early_stop = False
        self.path = path
        self.epoch = epoch
        self.counter = 0
        self.prev_epoch = 0
        
    def __call__(self, val_loss, model, epoch):

        if val_loss <= self.mse_threshold:
            if self.counter == 0:
                self.prev_epoch = epoch
            
            self.counter += 1
            
            if self.prev_epoch == (epoch-1):
                self.prev_epoch = epoch
            
        if val_loss > self.mse_threshold:
            self.counter = 0
             
        logger.info('Counter value {}'.format(self.counter))
        logger.info('prev_epoch {}'.format(self.prev_epoch))
        logger.info('epoch {}'.format(epoch))
        
        if self.counter == config["training"]["es_counter"]:
            self.early_stop = True
            self.save_checkpoint(val_loss, model, epoch)
# =============================================================================
#         else:
#             if self.verbose:
#                 logger.info('Early Stopping not activated at epoch {}'.format(epoch))
# =============================================================================
                
    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info('Saving model ...')
        torch.save(model.state_dict(), self.path)
        

model = LSTM_PT(n_features=config["model"]["n_features"], hidden_dims=config["model"]["hidden_dims"], num_layers=config["model"]["num_layers"])
model = model.to(config["model"]["device"])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.9)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config["training"]["T_0_for_cosine_annealing"], config["training"]["T_mult_for_cosine_annealing"], eta_min=1e-9)
early_stopping = EarlyStopping(mse_threshold=config["training"]["MSE_threshold"], verbose=True, path=config["model"]["path"] + "/model/LSTMmodel.pt")
model.load_state_dict(torch.load(config["model"]["path"] + "/model/LSTMmodel.pt"))

#############################
import pandas as pd
from sklearn import preprocessing
Tensor = torch.Tensor
TensorTuple = Tuple[torch.Tensor]
FloatTuple = Tuple[float]

logger = logging.getLogger(__name__)
# read erosion
liy=[]

path = ('/bigdata/wonglab/syang159/CFD1/OP650_2/')

for i in np.arange(1, 3125+1, 1):
    df_1  = pd.read_csv(path + "cell_%i" % i + "/corrosion_pressure")
    lst_1 = range(0,2371)
    df_1 = df_1.drop(lst_1)
    lst_2 = range (40683,42858)
    df_1 = df_1.drop(lst_2)
    erosion = np.array(df_1["dpm-erosion-rate-finnie"])
    liy.append(erosion)

# split into samples

y = np.array(liy)
erosion = preprocessing.normalize(y, norm='max')

training_erosion, testing_erosion = train_test_split(erosion, test_size=0.1, random_state=25, shuffle=True)
train_erosion=[]
validate_erosion=[]

kf = KFold(n_splits=5)
for i, (train_index, val_index) in enumerate(kf.split(training_erosion)):

    train_erosion.append(training_erosion[train_index])
    validate_erosion.append(training_erosion[val_index])

#5params
with h5py.File("/bigdata/wonglab/syang159/CFD2/Kfold_data/X.hdf5", "r") as f:
    params0 = torch.Tensor(f['params']).detach().cpu().numpy()

training_params, testing_params = train_test_split(params0, test_size=0.1, random_state=25, shuffle=True)
train_params=[]
validate_params=[]

kf = KFold(n_splits=5)
for i, (train_index, val_index) in enumerate(kf.split(training_params)):

    train_params.append(training_params[train_index])
    validate_params.append(training_params[val_index])
##########################################################

out_k=[]

in_k=[]  
for i in range(5):
    output=[]
    config["training"]["training_h5_file"] = "/bigdata/wonglab/syang159/CFD2/Kfold_data/training_data_{}.hdf5".format(i)

    training_loader, tr_mu, tr_std = createTrainingLoader(
        file_path=(config["training"]["training_h5_file"]),                     
        block_size=config["training"]["block_size"], 
        ndata=config["training"]["ndata"], 
        batch_size=config["training"]["batch_size"])

    
    for idx, datasets in enumerate(training_loader):
        
        params = datasets['params']
        xin0 = datasets['data'][:,0]
        yin0 = datasets['data']
        x = torch.cat([xin0, params.unsqueeze(-1) * torch.ones_like(xin0[:,:1])], dim=1) #[B, 8, 196]
        #x = _normalize(x, te_mu, te_std)
        x = x.reshape(x.size(0), 1, x.size(1)*x.size(2))
        x = torch.stack([x]*50,axis=1)
        x = x.reshape(x.size(0), 50, x.size(3)) 
        x = x.to(config["model"]["device"])
        y= torch.cat([yin0, params.unsqueeze(1).unsqueeze(-1) * torch.ones_like(yin0[:,:,:1])], dim=2)
        y = y.reshape(y.size(0),50,-1).to(config["model"]["device"])
        
        out = model(x)
        output.append(out) #predicted trajectory
        
    output_cat = torch.cat(output,dim=0)
    output_cat=output_cat.view(output_cat.size(0),output_cat.size(1), 8, 196)
    output_cat=output_cat[:,:,:3]
    out_k.append(output_cat)
    out_k[i]=np.swapaxes(out_k[i], 2,3)
    out_k[i]=out_k[i].detach().cpu()

path = ('/bigdata/wonglab/syang159/CFD2/Kfold_data/')
os.chdir(path)
for i in range(5):
    f1 = h5py.File("training_data_from_LSTM_%s.hdf5" % i ,'w')
    dsetx = f1.create_dataset("xprime", (out_k[i].size(0),50,196), data=out_k[i][:,:,:,0])
    dsety = f1.create_dataset("yprime", (out_k[i].size(0),50,196), data=out_k[i][:,:,:,1])
    dsetz = f1.create_dataset("zprime", (out_k[i].size(0),50,196), data=out_k[i][:,:,:,2])
    params=f1.create_dataset("params", (torch.tensor(train_params[i]).size(0),5), data=train_params[i])
    erosion=f1.create_dataset("erosion", (torch.tensor(train_erosion[i]).size(0),38312), data=train_erosion[i])
    f1.close()

out_k=[]

in_k=[]  
for i in range(5):
    output=[]

    config["validating"]["validating_h5_file"] =  "/bigdata/wonglab/syang159/CFD2/Kfold_data/validating_data_{}.hdf5".format(i)
    validating_loader, vl_mu, vl_std = createValidatingLoader(
        file_path=(config["validating"]["validating_h5_file"]) ,                   
        block_size=config["validating"]["block_size"], 
        ndata=config["validating"]["ndata"], 
        batch_size=config["validating"]["batch_size"])
    
    for idx, datasets in enumerate(validating_loader):
        
        params = datasets['params']
        xin0 = datasets['data'][:,0]
        yin0 = datasets['data']
        x = torch.cat([xin0, params.unsqueeze(-1) * torch.ones_like(xin0[:,:1])], dim=1) #[B, 8, 196]
        #x = _normalize(x, te_mu, te_std)
        x = x.reshape(x.size(0), 1, x.size(1)*x.size(2))
        x = torch.stack([x]*50,axis=1)
        x = x.reshape(x.size(0), 50, x.size(3)) 
        x = x.to(config["model"]["device"])
        y= torch.cat([yin0, params.unsqueeze(1).unsqueeze(-1) * torch.ones_like(yin0[:,:,:1])], dim=2)
        y = y.reshape(y.size(0),50,-1).to(config["model"]["device"])
        
        out = model(x)
        output.append(out) #predicted trajectory
        
    output_cat = torch.cat(output,dim=0)
    output_cat=output_cat.view(output_cat.size(0),output_cat.size(1), 8, 196)
    output_cat=output_cat[:,:,:3]
    out_k.append(output_cat)
    out_k[i]=np.swapaxes(out_k[i], 2,3)
    out_k[i]=out_k[i].detach().cpu()

path = ('/bigdata/wonglab/syang159/CFD2/Kfold_data/')
os.chdir(path)
for i in range(5):

    f2 = h5py.File("validating_data_from_LSTM_%s.hdf5" % i,'w')
    dsetx = f2.create_dataset("xprime", (out_k[i].size(0),50,196), data=out_k[i][:,:,:,0])
    dsety = f2.create_dataset("yprime", (out_k[i].size(0),50,196), data=out_k[i][:,:,:,1])
    dsetz = f2.create_dataset("zprime", (out_k[i].size(0),50,196), data=out_k[i][:,:,:,2])
    params=f2.create_dataset("params", (torch.tensor(validate_params[i]).size(0),5), data=validate_params[i])
    erosion=f2.create_dataset("erosion", (torch.tensor(validate_erosion[i]).size(0),38312), data=validate_erosion[i])
    f2.close()

output=[]
testing_loader, te_mu, te_std = createTestingLoader(
    file_path=(config["testing"]["testing_h5_file"]),                   
    block_size=config["testing"]["block_size"], 
    ndata=config["testing"]["ndata"], 
    batch_size=config["testing"]["batch_size"])

for idx, datasets in enumerate(testing_loader):
    
    params = datasets['params']
    xin0 = datasets['data'][:,0]
    yin0 = datasets['data']
    x = torch.cat([xin0, params.unsqueeze(-1) * torch.ones_like(xin0[:,:1])], dim=1) #[B, 8, 196]
    #x = _normalize(x, te_mu, te_std)
    x = x.reshape(x.size(0), 1, x.size(1)*x.size(2))
    x = torch.stack([x]*50,axis=1)
    x = x.reshape(x.size(0), 50, x.size(3)) 
    x = x.to(config["model"]["device"])
    y= torch.cat([yin0, params.unsqueeze(1).unsqueeze(-1) * torch.ones_like(yin0[:,:,:1])], dim=2)
    y = y.reshape(y.size(0),50,-1).to(config["model"]["device"])
    
    out = model(x)
    output.append(out) #predicted trajectory

output = torch.cat(output,dim=0)
output=output.view(output.size(0),output.size(1), 8, 196)
output=output[:,:,:3]
output=np.swapaxes(output, 2,3)
output=output.detach().cpu()

f3 = h5py.File("testing_data_from_LSTM.hdf5", 'w')
dsetx = f3.create_dataset("xprime", (output.size(0),50,196), data=output[:,:,:,0])
dsety = f3.create_dataset("yprime", (output.size(0),50,196), data=output[:,:,:,1])
dsetz = f3.create_dataset("zprime", (output.size(0),50,196), data=output[:,:,:,2])
params=f3.create_dataset("params", (torch.tensor(testing_params).size(0),5), data=testing_params)
erosion=f3.create_dataset("erosion", (torch.tensor(testing_erosion).size(0),38312), data=testing_erosion)
f3.close()