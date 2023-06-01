import os
import sys

import numpy as np
import pandas as pd
import dill
import torch

# from sklearn.metrics import r2_score
# from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging as logger


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        pass


    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
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
                
    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info('Saving model ...')
        torch.save(model.state_dict(), self.path)