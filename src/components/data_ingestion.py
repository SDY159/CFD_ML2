import os
import sys
import h5py
import numpy as np
import torch
# import logging
import random
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataset import Dataset
import torch.nn as nn
from typing import Any, Tuple, Dict, List, Union
import math
import json

from src.exception import CustomException
from src.logger import logging

from src.components.model_trainer import TransformerGPT2, TransformerTrain, EarlyStopping, Trainer


@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts', 'train.hdf5')
    test_data_path=os.path.join('artifacts', 'test.hdf5')
    raw_data_path=os.path.join('artifacts', 'data.hdf5')

class DataIngestion:
    def __init__(self) -> None:
        self.config_path = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)



if __name__=="__main__":
    obj = DataIngestion()
    config = json.loads(os.path.join('.', 'configs.json'))
    train_data, test_data = obj.initiate_data_ingestion()
    


# Shorthanding the notation
Tensor = torch.Tensor
LongTensor = torch.LongTensor


#Load data
class DatasetReader(Dataset):
    def __init__(
        self,
        file_path: str,
        block_size: int,
        stride: int = 1,
        ndata: int = -1,
        eval: bool = False,
        **kwargs
    ):

        self.block_size = block_size
        self.stride = stride
        self.ndata = ndata
        self.eval = eval   
        self.examples = []
        self.states = []
        with h5py.File(file_path, "r") as f:
            self.load_data(f, **kwargs)  

    def load_data(self, h5_file: h5py.File) -> None:

        # Iterate through stored time-series
        with h5_file as f:
            params0 = torch.Tensor(f['params'])
            pos_x = torch.Tensor(f['x'])
            pos_y = torch.Tensor(f['y'])
            pos_z = torch.Tensor(f['z'])
            for (p, x, y, z) in zip(params0, pos_x, pos_y, pos_z):
                data_series = torch.stack([x, y, z], dim=1).to(config["training"]["device"])
                data_series = data_series[:,:,torch.randperm(data_series.size(2), generator=g.manual_seed(config['model']['seed']))]

                p=p.to(config["training"]["device"])
                data_series1 = torch.cat([data_series, p.unsqueeze(-1) * torch.ones_like(data_series[:,:1])], dim=1)
                data_series1 = data_series1.view(data_series1.size(0),data_series1.size(1)*data_series1.size(2))
                    
                # Stride over time-series
                for i in range(0, data_series1.size(0) - self.block_size + 1, self.stride):
                    
                    data_series0 = data_series1[i: i + self.block_size]  # .repeat(1, 4)
                    self.examples.append(data_series0)
    
                    if self.eval:
                        self.states.append(data_series[i: i+ self.block_size].cpu())
        
                
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        if self.eval:
            return {'inputs_x': self.examples[i][:1], 'labels': self.examples[i]}
        else:
            return {'inputs_x': self.examples[i][:-1], 'labels': self.examples[i][1:]}
    
class DataCollator:
    """
    Data collator used for training datasets. 
    Combines examples in a minibatch into one tensor.
    
    Args:
        examples (List[Dict[str, Tensor]]): List of training examples. An example
            should be a dictionary of tensors from the dataset.

        Returns:
            Dict[str, Tensor]: Minibatch dictionary of combined example data tensors
    """
    # Default collator
    def __call__(self, examples:List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        mini_batch = {}
        for key in examples[0].keys():
            mini_batch[key] = self._tensorize_batch([example[key] for example in examples])

        return mini_batch

    def _tensorize_batch(self, examples: List[Tensor]) -> Tensor:
        if not torch.is_tensor(examples[0]):
            return examples

        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)

        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            raise ValueError("Padding not currently supported for physics transformers")
            return



#random seeds
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



#Loading training, validation, and test sets
training_data = DatasetReader(
    config["training"]["training_h5_file"], 
    block_size=config["training"]["n_ctx"], 
    stride=config["training"]["stride"],
    )

validating_data = DatasetReader(
    config["validating"]["validating_h5_file"], 
    block_size=config["validating"]["block_size"], 
    stride=config["validating"]["stride"],
    eval = True,
    )

training_loader = DataLoader(
    training_data,
    batch_size=config["training"]["batch_size"],
    sampler=RandomSampler(training_data),
    collate_fn=DataCollator(),
    drop_last=True,
)

validating_loader = DataLoader(
    validating_data,
    batch_size=config["validating"]["batch_size"],
    sampler=SequentialSampler(validating_data),
    collate_fn=DataCollator(),
    drop_last=True,
)

testing_data = DatasetReader(
    config["testing"]["testing_h5_file"], 
    block_size=config["testing"]["block_size"], 
    stride=config["testing"]["stride"],
    eval = True,
    )

testing_loader = DataLoader(
    testing_data,
    batch_size=config["testing"]["batch_size"],
    sampler=SequentialSampler(testing_data),
    collate_fn=DataCollator(),
    drop_last=True,
)


transformer = TransformerGPT2()
model = TransformerTrain(transformer)


optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config["training"]["learning_rate"], max_lr=config["training"]["max_lr"],step_size_up=5,mode="exp_range",gamma=0.85, cycle_momentum=False)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config["training"]["T_0_for_cosine_annealing"], config["training"]["T_mult_for_cosine_annealing"],  eta_min=1e-9)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.9)
early_stopping = EarlyStopping(mse_threshold=config["training"]["MSE_threshold"], verbose=True, path=config["model"]["path"] + "/model/TransformerGPT.pt")

#Define Trainer
trainer = Trainer(
        model, 
        (optimizer, scheduler), 
        training_loader = training_loader, 
        validating_loader = validating_loader 
        
        )