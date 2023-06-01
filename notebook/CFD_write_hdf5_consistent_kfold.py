#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:35:19 2022

@author: user
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import h5py
import torch
import logging
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from typing import Dict, Optional
from dataclasses import dataclass
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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

#5 parameters
with h5py.File("/bigdata/wonglab/syang159/CFD2/Kfold_data/X.hdf5", "r") as f:
    params0 = torch.Tensor(f['params']).detach().cpu().numpy()

training_params, testing_params = train_test_split(params0, test_size=0.1, random_state=25, shuffle=True)
train_params=[]
validate_params=[]

kf = KFold(n_splits=5)
for i, (train_index, val_index) in enumerate(kf.split(training_params)):

    train_params.append(training_params[train_index])
    validate_params.append(training_params[val_index])

#Ground Truth trajectory data

#Dataloaders for training and testing
class CategorizeDataset(Dataset):

    def __init__(self, examples: List, params: List) -> None:

        self.examples = examples
        self.params = params

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {"data": self.examples[i], "params": self.params[i]}

class DataCollator:

    # Default collator
    def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Stack examples in mini-batch
        x_data_tensor =  torch.stack([example["data"] for example in examples])
        params_tensor =  torch.stack([example["params"] for example in examples])

        return {"data": x_data_tensor, "params": params_tensor}


#load test data
def createXprimeLoader(
        block_size: int,
        ndata: int = -1,
        batch_size: int = 64,
        shuffle: bool =False,
    ) -> DataLoader:

    logging.info('Creating Dataset loader')

    examples = []
    params = []
    with h5py.File("/bigdata/wonglab/syang159/CFD2/Kfold_data/X.hdf5", "r") as f:
        # Iterate through stored time-series
        samples = 0
    
        for key in f.keys():
            ##print(params0)
            params1 = torch.Tensor(f['params'])
            #print(params1.shape)
            pos_x = torch.Tensor(f['x'])
            pos_y = torch.Tensor(f['y'])
            pos_z = torch.Tensor(f['z'])
            data_series = torch.stack([pos_x, pos_y, pos_z], dim=2) ##torch.Size([1, 3, 196])
    
            # Stride over time-series
        for i in range(0, 3125,1):
            for j in range(0, data_series.size(1) - block_size + 1, block_size):
            
                examples.append(data_series[i][j: j + block_size])
                params.append(params1[i])
                break
                samples = samples + 1
                if (ndata > 0 and samples > ndata):  # If we have enough time-series samples break loop
                    break            
            
    data = torch.stack(examples, dim=0) ##torch.Size([37500, 1, 3, 196])
    params = torch.stack(params)

    dataset = CategorizeDataset(data, params)
    data_collator = DataCollator()
    testing_loader = DataLoader(dataset, batch_size=3125, shuffle=False, collate_fn=data_collator, drop_last=False)
    return testing_loader

testing_loader = createXprimeLoader(
                        block_size=50,
                        ndata=-1, 
                        batch_size=64)
for idx, datasets in enumerate(testing_loader):
    y = datasets['data']
y=np.swapaxes(y, 2,3)

training_y, testing_y = train_test_split(y, test_size=0.1, random_state=25, shuffle=True)
train_y=[]
validate_y=[]

kf = KFold(n_splits=5)
for i, (train_index, val_index) in enumerate(kf.split(training_y)):

    train_y.append(training_y[train_index])
    validate_y.append(training_y[val_index])



path = ('/bigdata/wonglab/syang159/CFD2/Kfold_data/')
os.chdir(path)
for i in range(5):
    f1 = h5py.File("training_data_%s.hdf5" % i ,'w')
    dsetx = f1.create_dataset("x", (train_y[i].size(0),50,196), data=train_y[i][:,:,:,0])
    dsety = f1.create_dataset("y", (train_y[i].size(0),50,196), data=train_y[i][:,:,:,1])
    dsetz = f1.create_dataset("z", (train_y[i].size(0),50,196), data=train_y[i][:,:,:,2])
    params=f1.create_dataset("params", (torch.tensor(train_params[i]).size(0),5), data=train_params[i])
    erosion=f1.create_dataset("erosion", (torch.tensor(train_erosion[i]).size(0),38312), data=train_erosion[i])
    f1.close()


    f2 = h5py.File("validating_data_%s.hdf5" % i,'w')
    dsetx = f2.create_dataset("x", (validate_y[i].size(0),50,196), data=validate_y[i][:,:,:,0])
    dsety = f2.create_dataset("y", (validate_y[i].size(0),50,196), data=validate_y[i][:,:,:,1])
    dsetz = f2.create_dataset("z", (validate_y[i].size(0),50,196), data=validate_y[i][:,:,:,2])
    params=f2.create_dataset("params", (torch.tensor(validate_params[i]).size(0),5), data=validate_params[i])
    erosion=f2.create_dataset("erosion", (torch.tensor(validate_erosion[i]).size(0),38312), data=validate_erosion[i])
    f2.close()

f3 = h5py.File("testing_data.hdf5", 'w')
dsetx = f3.create_dataset("x", (testing_y.size(0),50,196), data=testing_y[:,:,:,0])
dsety = f3.create_dataset("y", (testing_y.size(0),50,196), data=testing_y[:,:,:,1])
dsetz = f3.create_dataset("z", (testing_y.size(0),50,196), data=testing_y[:,:,:,2])
params=f3.create_dataset("params", (torch.tensor(testing_params).size(0),5), data=testing_params)
erosion=f3.create_dataset("erosion", (torch.tensor(testing_erosion).size(0),38312), data=testing_erosion)
f3.close()