#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:00:02 2022

@author: user
"""

import numpy as np
import os
import h5py
import torch
import logging
import torch.optim as optim
from typing import Tuple, List, Dict
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cmx
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
Tensor = torch.Tensor
TensorTuple = Tuple[torch.Tensor]
FloatTuple = Tuple[float]
logger = logging.getLogger(__name__)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

#We utilized and modified the code from TrphysX: 
'''@article{geneva2020transformers,
    title={Transformers for Modeling Physical Systems},
    author={Geneva, Nicholas and Zabaras, Nicholas},
    journal={arXiv preprint arXiv:2010.03957},
    year={2020}
}'''
    
#Generalization 
CNN_from = "GPT" #Choose LSTM or GPT

config1 = {
    "model": {
        "batch_size": 64,
        "dropout": 0.0,
        "path": "Your path",
        "path_result": "Your path",
        
    },
    "training": {
        "device": "cuda", # "cuda" or "cpu"
        "batch_size": 64,
        "training_h5_file": "Your path.hdf5",
        "block_size": 50,
        "num_epoch": 300, #100
        "learning_rate": 0.001,
        "scheduler_step_size": 250,
    },
    "validating": {
        "validating_h5_file": "Your path.hdf5",
        "batch_size": 16,
        "block_size": 50,
    },
    "testing": {
        "testing_h5_file": "Your path.hdf5",
        "batch_size": 16,
        "block_size": 50,
    }
}

config2 = {
    "model": {
        "batch_size": 64,
        "dropout": 0.0,
        "path": "Your path",
        "path_result": "Your path",
    },
    "training": {
        "device": "cuda", # "cuda" or "cpu"
        "batch_size": 64,
        "training_h5_file": "Your path.hdf5",
        "block_size": 50,
        "num_epoch": 300, #100
        "learning_rate": 0.001,
        "scheduler_step_size": 250,
    },
    "validating": {
        "validating_h5_file": "Your path.hdf5",
        "batch_size": 16,
        "block_size": 50,

    },
    "testing": {
        "testing_h5_file": "Your path.hdf5",
        "batch_size": 16,
        "block_size": 50,
    }
}


if CNN_from == "GPT":
    config=config1
else:
    config=config2


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)

logger.setLevel(logging.DEBUG)

#Dataloaders
class CylinderDataset(Dataset):

    def __init__(self, examples: List, params: List, erosion: List) -> None:
        self.examples = examples
        self.params = params
        self.erosion = erosion
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {"data": self.examples[i], "params": self.params[i], "erosion": self.erosion[i] }

class CylinderDataCollator:
    # Default collator
    def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Stack examples in mini-batch
        x_data_tensor =  torch.stack([example["data"] for example in examples])
        params_tensor =  torch.stack([example["params"] for example in examples])
        erosion_tensor = torch.stack([example["erosion"] for example in examples])
        return {"data": x_data_tensor, "params": params_tensor, "erosion": erosion_tensor}

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
    erosions=[]
    with h5py.File(file_path,"r") as f:

        params1 = torch.Tensor(f['params'])
        erosion1 = torch.Tensor(f['erosion'])
        #print(params1.shape)
        pos_x = torch.Tensor(f['xprime'])
        pos_y = torch.Tensor(f['yprime'])
        pos_z = torch.Tensor(f['zprime'])
        data_series = torch.stack([pos_x, pos_y, pos_z], dim=2) ##torch.Size([1, 3, 196])
            # Stride over time-series
        for i in range(0, len(pos_x),1):
            for j in range(0, data_series.size(1) - block_size + 1, block_size):
                examples.append(data_series[i][j: j + block_size])
                params.append(params1[i])
                erosions.append(erosion1[i])
                break

    data = torch.stack(examples, dim=0) ##torch.Size([37500, 1, 3, 196])
    params = torch.stack(params)
    erosion =torch.stack(erosions)
    mu = torch.tensor([torch.mean(data[:,:,0]), torch.mean(data[:,:,1]), torch.mean(data[:,:,2]), 
               torch.mean(params[:,0]), torch.mean(params[:,1]), torch.mean(params[:,2]), torch.mean(params[:,3]), torch.mean(params[:,4]) ])
    std = torch.tensor([torch.std(data[:,:,0]), torch.std(data[:,:,1]), torch.std(data[:,:,2]), 
               torch.std(params[:,0]), torch.std(params[:,1]), torch.std(params[:,2]), torch.std(params[:,3]), torch.std(params[:,4]) ])
        
    dataset = CylinderDataset(data, params, erosion)
    data_collator = CylinderDataCollator()
    training_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=data_collator, drop_last=True)
    return training_loader , mu, std

# load validating data
def createValidatingLoader(   
    file_path: str,
    block_size: int,
    ndata: int = -1,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:

    logging.info('Creating validating loader')

    examples=[]
    params=[]
    erosions=[]
    with h5py.File(file_path,"r") as f:

        params1 = torch.Tensor(f['params'])
        erosion1 = torch.Tensor(f['erosion'])
        #print(params1.shape)
        pos_x = torch.Tensor(f['xprime'])
        pos_y = torch.Tensor(f['yprime'])
        pos_z = torch.Tensor(f['zprime'])
        data_series = torch.stack([pos_x, pos_y, pos_z], dim=2)
            # Stride over time-series
        for i in range(0, len(pos_x),1):
            for j in range(0, data_series.size(1) - block_size + 1, block_size):
            
                examples.append(data_series[i][j: j + block_size])
                params.append(params1[i])
                erosions.append(erosion1[i])
                break
            
    data = torch.stack(examples, dim=0) ##torch.Size([37500, 1, 3, 196])
    params = torch.stack(params)
    erosion =torch.stack(erosions)
    mu = torch.tensor([torch.mean(data[:,:,0]), torch.mean(data[:,:,1]), torch.mean(data[:,:,2]), 
               torch.mean(params[:,0]), torch.mean(params[:,1]), torch.mean(params[:,2]), torch.mean(params[:,3]), torch.mean(params[:,4])])
    std = torch.tensor([torch.std(data[:,:,0]), torch.std(data[:,:,1]), torch.std(data[:,:,2]), 
               torch.std(params[:,0]), torch.std(params[:,1]), torch.std(params[:,2]), torch.std(params[:,3]), torch.std(params[:,4]) ])
        
    dataset = CylinderDataset(data, params, erosion)
    data_collator = CylinderDataCollator()
    validating_loader = DataLoader(dataset, batch_size=config["validating"]["batch_size"], shuffle=False, collate_fn=data_collator, drop_last=True)
    return validating_loader , mu, std


# load testing data
def createTestingLoader(   
    file_path: str,
    block_size: int,
    ndata: int = -1,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:

    logging.info('Creating testing loader')

    examples=[]
    params=[]
    erosions=[]
    with h5py.File(file_path,"r") as f:

        params1 = torch.Tensor(f['params'])
        erosion1 = torch.Tensor(f['erosion'])
        pos_x = torch.Tensor(f['xprime'])
        pos_y = torch.Tensor(f['yprime'])
        pos_z = torch.Tensor(f['zprime'])
        data_series = torch.stack([pos_x, pos_y, pos_z], dim=2) 
            # Stride over time-series
        for i in range(0, len(pos_x),1):
            for j in range(0, data_series.size(1) - block_size + 1, block_size):
            
                examples.append(data_series[i][j: j + block_size])
                params.append(params1[i])
                erosions.append(erosion1[i])
                break

    data = torch.stack(examples, dim=0) ##torch.Size([37500, 1, 3, 196])
    params = torch.stack(params)
    erosion =torch.stack(erosions)
    mu = torch.tensor([torch.mean(data[:,:,0]), torch.mean(data[:,:,1]), torch.mean(data[:,:,2]), 
               torch.mean(params[:,0]), torch.mean(params[:,1]), torch.mean(params[:,2]), torch.mean(params[:,3]), torch.mean(params[:,4])])
    std = torch.tensor([torch.std(data[:,:,0]), torch.std(data[:,:,1]), torch.std(data[:,:,2]), 
               torch.std(params[:,0]), torch.std(params[:,1]), torch.std(params[:,2]), torch.std(params[:,3]), torch.std(params[:,4]) ])
        
    dataset = CylinderDataset(data, params, erosion)
    data_collator = CylinderDataCollator()
    testing_loader = DataLoader(dataset, batch_size=config["testing"]["batch_size"], shuffle=False, collate_fn=data_collator, drop_last=True)
    return testing_loader , mu, std


#normalize and unnormalize ftns
def _normalize(x: Tensor, mu: Tensor, std: Tensor) -> Tensor:
    x = (x - mu.unsqueeze(0).unsqueeze(-1)) / std.unsqueeze(0).unsqueeze(-1)
    return x
def _unnormalize(x: Tensor, mu: Tensor, std: Tensor) -> Tensor:
    return std[:3].unsqueeze(0).unsqueeze(-1)*x + mu[:3].unsqueeze(0).unsqueeze(-1)

#Load train and validation sets
training_loader, tr_mu, tr_std = createTrainingLoader(
    file_path=config["training"]["training_h5_file"],                       
    block_size=config["training"]["block_size"], 
    batch_size=config["training"]["batch_size"])

validating_loader, vl_mu, vl_std = createValidatingLoader(
    file_path=config["validating"]["validating_h5_file"], 
    block_size=config["validating"]["block_size"],
    batch_size=config["validating"]["batch_size"])

#Xavier Initiallization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        
#CNN model
class CNN(nn.Module):
    
    def __init__(self, ):
        super(CNN, self).__init__()
        
        self.maxpool = nn.MaxPool3d((2, 2, 2), padding=1)
        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=10, kernel_size=2, padding=1)
        self.conv2 = nn.Conv3d(in_channels=10, out_channels=20, kernel_size=2, padding=1)
        self.conv3 = nn.Conv3d(in_channels=20, out_channels=30, kernel_size=2, padding=1)
        self.conv4 = nn.Conv3d(in_channels=30, out_channels=40, kernel_size=2, padding=1)
        self.linear1 = nn.Linear(9000, 38312)
        self.flat = nn.Flatten()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.lrelu =nn.LeakyReLU()

    def forward(self, x):
        
        out = self.gelu(self.maxpool(self.conv1(x)))
        out = self.gelu(self.maxpool(self.conv2(out)))
        out = self.gelu(self.maxpool(self.conv3(out)))
        out = self.gelu(self.maxpool(self.conv4(out)))
        out = self.flat(out)
        
        out = self.lrelu(self.linear1(out))
        
        return out

#ML process
def run_epoch(dataloader, mu, std, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        with torch.no_grad():
            model.eval()

    for idx, datasets in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()
        
        params = datasets['params'] #torch.Size([64, 5])
        erosion = datasets['erosion'] # torch.Size([64, 38312])    
        xin0 = datasets['data'] #([64, 50, 3, 196])

        
        x = torch.cat([xin0, params.unsqueeze(1).unsqueeze(-1) * torch.ones_like(xin0[:,:,:1])], dim=2) #torch.Size([64, 50, 8, 196])
        x = _normalize(x, mu, std)

        x = x.unsqueeze(1).to(config["training"]["device"])
        y = erosion.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.detach().item() / len(dataloader)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr

#Define model, criterion, optimizer, and scheduler
r2=[]
mse=[]  
model = CNN()
model.apply(init_weights)
model = model.to(config["training"]["device"])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.999), eps=1e-7)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.8)

#Training ML
train_losses = []
val_losses = []
start.record()
for epoch in range(config["training"]["num_epoch"]):
    loss_train, lr_train = run_epoch(training_loader, tr_mu, tr_std, is_training=True)
    train_losses.append(loss_train)
    loss_val, lr_val = run_epoch(validating_loader, vl_mu, vl_std)
    val_losses.append(loss_val)
    scheduler.step()

    print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f} | lr:{:.6f}'
          .format(epoch + 1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))
torch.cuda.synchronize()
end.record()

torch.save(val_losses, config["model"]["path"]+'/val_loss.pt')      
torch.save(train_losses, config["model"]["path"] + '/train_loss.pt') 
print("Elapsed Time = %f hr" % ((start.elapsed_time(end)) / 3600000))
torch.save(model.state_dict(), config["model"]["path"] + "/model/CNN_Pytorch_EEEEEth_fold.pt" )
input1=[]
output=[]
params1=[]

#Loading test set 
testing_loader, te_mu, te_std = createTestingLoader(
    file_path=config["testing"]["testing_h5_file"], 
    block_size=config["testing"]["block_size"],
    batch_size=config["testing"]["batch_size"])

#Evaluate model with test set
for idx, datasets in enumerate(testing_loader):
    
    params = datasets['params'] #torch.Size([64, 5])
    erosion = datasets['erosion'] # torch.Size([64, 38312])
    batch_size = datasets['data'].size(0)    
    xin0 = datasets['data'] #([64, 50, 3, 196])

    x = torch.cat([xin0, params.unsqueeze(1).unsqueeze(-1) * torch.ones_like(xin0[:,:,:1])], dim=2) #torch.Size([64, 50, 8, 196])
    x = _normalize(x, te_mu, te_std)
    x = x.unsqueeze(1)
    x = x.to(config["training"]["device"])
    y = erosion.to(config["training"]["device"])
    
    with torch.no_grad():
        out = model(x)
    output.append(out)
    input1.append(y)
output = torch.cat(output,dim=0)
input1 = torch.cat(input1,dim=0)
output2 = torch.flatten(output)
input2 = torch.flatten(input1)

R2=r2_score(output2.cpu().detach(), input2.cpu().detach())
r2.append(R2)
MSE=mean_squared_error(output2.cpu().detach(), input2.cpu().detach(), squared=True)
mse.append(MSE)
print("r2_score: %.5f" % R2)
print("MSE: %.5f" % MSE)

#Learning curve plotting
val_losses=torch.load(config["model"]["path"]+'/val_loss.pt')
train_losses=torch.load(config["model"]["path"]+'/train_loss.pt')
Epoch=torch.arange(1,config["training"]["num_epoch"] + 1 )
plt.figure(figsize=(10,5))
plt.title("kfold: EEEEE")
plt.plot(Epoch, val_losses,label="validation")
plt.plot(Epoch, train_losses,label="train")
plt.xlabel("Epoch")
plt.ylabel("Loss")
down, up =plt.gca().get_ylim()
plt.text(config["training"]["num_epoch"]/2, (up+down)/2, 'MSE:{:.6f} \n R^2:{:.6f} \n Elapsed_time:{:.6f} hr'.format(MSE,R2,((start.elapsed_time(end)) / 3600000)))
plt.legend()
plt.show()
os.chdir(config["model"]["path_result"])
plt.savefig('learning_curve_%i.png' %config['training']['Job_number']) 


#Visualize the erosion
out = output.cpu().detach()
y_origin = input1.cpu().detach()

index = np.arange(3125)
training_index, testing_index= train_test_split(index, test_size=0.3, random_state=25)
#testing_index[0:36]
def data_for_cylinder_main(center_y,center_z,radius,height_x):
       x = np.linspace(0, height_x, 50)-0.64025
       theta = np.linspace(0, 2*np.pi, 50)
       theta_grid, x_grid=np.meshgrid(theta, x)
       y_grid = radius*np.cos(theta_grid) + center_y
       z_grid = radius*np.sin(theta_grid) + center_z
       return x_grid,y_grid,z_grid
Xm,Ym,Zm = data_for_cylinder_main(0,0,0.269,1.8805)
def data_for_cylinder_inlet(center_x,center_z,radius,height_y):
       y = np.linspace(0, -height_y, 50)-0.269
       theta = np.linspace(0, 2*np.pi, 50)
       theta_grid, y_grid=np.meshgrid(theta, y)
       x_grid = radius*np.cos(theta_grid) + center_x
       z_grid = radius*np.sin(theta_grid) + center_z
       return x_grid,y_grid,z_grid
Xi1,Yi1,Zi1 = data_for_cylinder_inlet(-0.5435,0,0.01875,0.269)
Xi2,Yi2,Zi2 = data_for_cylinder_inlet(0.0,0,0.01875,0.269)
Xi3,Yi3,Zi3 = data_for_cylinder_inlet(0.5435,0,0.01875,0.269)

path = ('/path/')
import pandas as pd
os.chdir(path)
df = pd.read_csv("CFD erosion data")

inlet_1=df.iloc[:2175,:]
inlet_2=df.iloc[2175:2247,:]
inlet_3=df.iloc[2247:2311,:]
inlet_4=df.iloc[2311:2371,:]
main=df.iloc[2371:40683,:]
outlet=df.iloc[40683:,:]

X = main.iloc[:,1].values
Y = main.iloc[:,2].values
Z = main.iloc[:,3].values

os.chdir(path)
df_2 = pd.read_csv("erosion.csv" ,header=None)

for k in range(0,out.size(0), 10):
    df1 = pd.DataFrame(out[k].detach().numpy())
    df_1 = df1 * df_2.to_numpy()[testing_index[k]]
    W = df_1[0].values

    fig = plt.figure()
    
    ax=fig.add_subplot(111,projection='3d')
    fig.set_size_inches(18.5, 10.5)
    
    ax.plot_surface(Xm,Zm,Ym,alpha=0.1,color='blue')
    ax.plot_surface(Xi1,Zi1,Yi1,alpha=0.1,color='blue')
    ax.plot_surface(Xi2,Zi2,Yi2,alpha=0.1,color='blue')
    ax.plot_surface(Xi3,Zi3,Yi3,alpha=0.1,color='blue')
    # ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    
    
    
    cm = plt.get_cmap('rainbow')
    cNorm = matplotlib.colors.Normalize(vmin=min(W), vmax=max(W))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    # ax.set_xlabel('X-coordinate (m)', fontsize=12)
    # ax.set_ylabel('Y-coordinate (m)', fontsize=12)
    # ax.set_zlabel('Z-coordinate (m)', fontsize=12)
    
    
    ax.scatter(X,Z,Y, s = 1000000*W, c=scalarMap.to_rgba(W), label='Erosion profile')
    
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
    
    
    # plt.title("Erosion profile (RNN+CNN) on %ith dataset"%sim_num, fontsize=15)
    scalarMap.set_array(W)
    def fmt(x, pos):
        if x>0:
           a, b = '{:.2e}'.format(x).split('e')
           b = int(b)
           return r'${} \times 10^{{{}}}$'.format(a, b)
    cbar = plt.colorbar(scalarMap, shrink=0.7, pad=0.1, location='bottom', format=ticker.FuncFormatter(fmt), extend="max",  orientation="horizontal")
    cbar.ax.tick_params(labelsize=45, labelrotation=45)
    cbar.ax.invert_xaxis()  
    #cbar.ax.set_title('Erosion rate \n (kg/$m^2$·s)',fontsize=25)
    
    ax.axes.set_xlim3d(left=-0.8, right=0.8)
    ax.axes.set_ylim3d(bottom=-0.8, top=0.8)
    ax.axes.set_zlim3d(bottom=-0.8, top=0.8)
    # plt.grid()
    
    # Hide grid lines
    ax.grid(False)
    
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    
    plt.tight_layout()
    
    
    os.chdir(config["model"]["path"] + '/plot/')
    plt.savefig('new_%i_predicted_ero.png'%k)

    #df3 = pd.DataFrame(y_origin[k].detach().numpy())
    #df_3 = df3 * df_2.to_numpy()[testing_index[k]]
    #W = df_3[0].values

    fig2 = plt.figure()
    
    ax1=fig2.add_subplot(111,projection='3d')
    fig2.set_size_inches(18.5, 10.5)
    
    ax1.plot_surface(Xm,Zm,Ym,alpha=0.1,color='blue')
    ax1.plot_surface(Xi1,Zi1,Yi1,alpha=0.1,color='blue')
    ax1.plot_surface(Xi2,Zi2,Yi2,alpha=0.1,color='blue')
    ax1.plot_surface(Xi3,Zi3,Yi3,alpha=0.1,color='blue')
    # ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    
    
    
    cm = plt.get_cmap('rainbow')
    #cNorm = matplotlib.colors.Normalize(vmin=min(W), vmax=max(W))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    # ax.set_xlabel('X-coordinate (m)', fontsize=12)
    # ax.set_ylabel('Y-coordinate (m)', fontsize=12)
    # ax.set_zlabel('Z-coordinate (m)', fontsize=12)
    
    
    ax1.scatter(X,Z,Y, s = 1000000*W, c=scalarMap.to_rgba(W), label='Erosion profile')
    
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
    
    
    # plt.title("Erosion profile (RNN+CNN) on %ith dataset"%sim_num, fontsize=15)
    scalarMap.set_array(W)
    def fmt(x, pos):
        if x>0:
           a, b = '{:.2e}'.format(x).split('e')
           b = int(b)
           return r'${} \times 10^{{{}}}$'.format(a, b)
    cbar = plt.colorbar(scalarMap, shrink=0.7, pad=0.1, location='bottom', format=ticker.FuncFormatter(fmt), extend="max",  orientation="horizontal")
    cbar.ax.tick_params(labelsize=45, labelrotation=45)
    cbar.ax.invert_xaxis()  
    #cbar.ax.set_title('Erosion rate \n (kg/$m^2$·s)',fontsize=25)
    ax1.axes.set_xlim3d(left=-0.8, right=0.8)
    ax1.axes.set_ylim3d(bottom=-0.8, top=0.8)
    ax1.axes.set_zlim3d(bottom=-0.8, top=0.8)
    # plt.grid()
    
    # Hide grid lines
    ax1.grid(False)
    
    # Hide axes ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    plt.axis('off')
    
    plt.tight_layout()
    
    
    os.chdir(config["model"]["path"] + '/plot/')
    plt.savefig('new_%i_original_ero.png'%k)
    

