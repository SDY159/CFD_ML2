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
from typing import Any, Tuple, Dict, List, Union
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataset import Dataset
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
import random
import matplotlib.pyplot as plt


Tensor = torch.Tensor
LongTensor = torch.LongTensor

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
logger = logging.getLogger(__name__)

Optimizer = torch.optim.Optimizer
Scheduler = torch.optim.lr_scheduler._LRScheduler

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    )

logger.setLevel(logging.DEBUG)

def gelu(x: Tensor) -> Tensor:
    """Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class Conv1D(nn.Module):
    """1D-convolutional layer (eqv to FCN) as defined by Radford et al. for OpenAI GPT 
    (and also used in GPT-2). Basically works like a linear layer but the weights are transposed.

    Note: 
        Code adopted from: https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_utils.py

    Args:
        nf (int): The number of output features.
        nx (int): The number of input features.
    """
    def __init__(self, nf: int, nx: int) -> None:
        """Constructor
        """
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass

        Args:
            x (Tensor): [..., nx] input features

        Returns:
            Tensor: [..., nf] output features
        """
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x
    
#Generalization 
config = {
    "model": {
        "n_embd": 1568,
        "n_layer": 4,
        "n_head": 2,
        "activation_function": gelu,
        "dropout": 0.0,
        "seed": 12345,
        "path": "/bigdata/wonglab/syang159/CFD2/pytorch_particle_random_order/" + "GPT_CyclicLR_kfold_Early_stopping_k80_stride_2/bs_4_str_2/n_layer_4/n_head_2/lr_1e-7/max_lr_0.002/Batch_size_64/kfold_0",
        "path_result": "/bigdata/wonglab/syang159/CFD2/pytorch_particle_random_order/GPT_CyclicLR_kfold_Early_stopping_k80_stride_2/Result"
    },
    "training": {
        "device": "cuda", # "cuda" or "cpu"
        "batch_size": 64,
        "training_h5_file": "/bigdata/wonglab/syang159/CFD2/Kfold_data/training_data_0.hdf5",
        "n_ctx": 4,
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
        "layer_norm_epsilon": 1e-05,
        "initializer_range": 0.01,
        "stride": 2,
        "ndata": 2500,  #2500
        "num_epoch": 100, #100
        "learning_rate": 1e-7,
        "max_lr": 0.002,
        #"T_0_for_cosine_annealing": EEEEE,
        #"T_mult_for_cosine_annealing": GGGGG,
        "scheduler_step_size": 100,
        "output_hidden_state": False,
        "output_attention": False,
        "use_cache": True,
        "max_length": 50,
        "min_length": 0,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 0.01,
        "Job_number": 4,
        "MSE_threshold": 0.0065,
        "es_counter": 3
    },
    "validating": {
        "validating_h5_file": "/bigdata/wonglab/syang159/CFD2/Kfold_data/validating_data_0.hdf5",
        "batch_size": 64,
        "block_size": 50,
        "stride": 3000,
        "ndata": 312,  #312
        "val_steps":1
    },
    "testing": {
        "testing_h5_file": "/bigdata/wonglab/syang159/CFD2/Kfold_data/testing_data.hdf5",
        "batch_size": 64,
        "block_size": 50,
        "stride": 3000,
        "ndata": 313  #313
    }
}

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
            erosion1 = torch.Tensor(f['erosion'])
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

#Define model
class MLP(nn.Module):
    """Simple fully connected neural network layer.
    Includes activations function and dropout.

    Args:
        n_state (int): dimensionality of input features
        config (PhysConfig): Phys-transformer config object
    """
    def __init__(self, n_state: int) -> None:
        """Constructor 
        """
        super().__init__()
        nx = config["model"]["n_embd"]
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = config["model"]["activation_function"]
        self.dropout = nn.Dropout(config["training"]["resid_pdrop"])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass

        Args:
            x (Tensor): [B, T, n_state] input features

        Returns:
            Tensor: Output features
        """
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)
    
class MaskedAttention(nn.Module):
    """Masked self-attention module based on the Hugging face implementation
    https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_gpt2.py

    Args:
        nx (int): Dimensionality of feature vector
        n_ctx (int): Context length of the attention (TODO: Not needed with config object?)
        config (PhysConfig): Transformer config object
        scale (bool, optional): Scale the attention scores. Defaults to False.
        mask (str, optional): Attention mask type. Defaults to 'tril'.

    Raises:
        ValueError: Invalid mask type
    """
    def __init__(
        self, 
        nx: int, 
        n_ctx: int, 
        scale: bool  = False, 
        mask: str = 'tril'
    ) -> None:
        """Constructor
        """
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        assert n_state % config["model"]["n_head"] == 0
        
        # Create attention mask
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config["model"]["n_head"]
        self.split_size = n_state
        self.scale = scale

        # Conv1D are not PyTorch Conv1D
        # Conv1D(out_features, in_features)
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config["training"]["attn_pdrop"])
        self.resid_dropout = nn.Dropout(config["training"]["resid_pdrop"])

    def _attn(
        self, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor, 
        attention_mask: Tensor = None,
        head_mask: Tensor = None, 
        output_attentions: bool =False
    ) -> List[Tensor]:
        """Dot product attention calculation

        Args:
            q (Tensor): [batch, head, seq_length, head_features] query
            k (Tensor): [batch, head, head_features, seq_length] key
            v (Tensor): [batch, head, seq_length, head_features] value
            attention_mask (Tensor, optional): Optional defined attention mask. Defaults to None.
            head_mask (Tensor, optional): Optional attention value mask. Defaults to None.
            output_attentions (bool, optional): Output attention matrix. Defaults to False.

        Returns:
            List[Tensor]: Output consisting of output feature, attention tensor (if requested)
        """
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)

        nd, ns = w.size(-2), w.size(-1)
        mask = self.bias[:, :, ns - nd : ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask
        
        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x: Tensor) -> Tensor:
        """Merge attention heads

        Args:
            x (Tensor): [batch, head, seq_length, head_features] Input tensor

        Returns:
            Tensor: [batch, seq_length, head * head_features] Concatenated output tensor
        """
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k: bool = False) ->  Tensor:
        """Splits key, query or value tensor into separate heads.
        Dimensionality of output depends if tensor is a key.

        Args:
            x (Tensor): [batch, seq_length, nx] Input tensor
            k (bool): If input tensor is a key tensor

        Returns:
            Tensor: [batch, head, seq_length, head_features] Split features for query
            and value, [batch, head, seq_length, head_features] split feature for key
        """
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self, 
        x: Tensor, 
        layer_past: List[Tensor] = None, 
        attention_mask: Tensor =None, 
        head_mask: Tensor = None,
        use_cache: bool = False, 
        output_attentions: bool = False
    ) -> List[Tensor]:
        """Masked attention forward pass

        Args:
            x (Tensor): [batch, seq_length, nx] Input feature.
            layer_past (Tensor, optional): [2, batch, n_head, seq_length, nx] Precomputed self-attention vectors. Defaults to None.
            attention_mask (Tensor, optional): Optional defined attention mask. Applied before soft mask.
                 Defaults to None.
            head_mask (Tensor, optional): Optional attention value mask. Applied after softmax Defaults to None.
            use_cache (bool, optional): Return calculated key values or faster generation. Defaults to False.
            output_attentions (bool, optional): Return attention matrix. Defaults to False.

        Returns:
            List[Tensor]: Output consisting of output feature, key values (if requested), attention tensor (if requested)
        """
        x = self.c_attn(x) # x -> q, k, v
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True) # k=True for keys which transposes the last two dims
        value = self.split_heads(value)
        # Concat previous key and value tensors 
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)

class Block(nn.Module):
    """Transformer decoder block consisting of layer norm, masked self-attention,
    layer norm and fully connected layer.

    Args:
        n_ctx (int): contex length of block
        config (PhysConfig): Phys-transformer config object
        scale (bool, optional): Scaled self-attention calculation. Defaults to False.
    """
    def __init__(self, n_ctx: int, scale: bool = False) -> None:
        """Constructor
        """
        super().__init__()
        nx = config["model"]["n_embd"]
        self.ln_1 = nn.LayerNorm(nx, eps=config["training"]["layer_norm_epsilon"])
        self.attn = MaskedAttention(nx, n_ctx, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config["training"]["layer_norm_epsilon"])
        self.mlp = MLP(4 * nx)

    def forward(
        self, 
        x: Tensor, 
        layer_past: List[Tensor] = None, 
        attention_mask: LongTensor = None, 
        head_mask: LongTensor = None, 
        use_cache: bool = False, 
        output_attentions: bool = False,
    ) -> List[Tensor]:
        """Forward pass

        Args:
            x (Tensor): [B, T, n_state] input features
            layer_past ([type], optional): Past self-attention calculation. Defaults to None.
            attention_mask (LongTensor, optional): Attention mask. Defaults to None.
            head_mask (LongTensor, optional): Attention value. Defaults to None.
            use_cache (bool, optional): Store attention state (key values). Defaults to False.
            output_attentions (bool, optional): Return attention values. Defaults to False.

        Returns:
            List[Tensor]: List of output tensors
        """
        # Evaluate attention heads
        output_attn = self.attn.forward(
            self.ln_1(x),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions
        )

        a = output_attn[0] 
        # Residual connection 1
        x = x + a
        # FCNN
        m = self.mlp(self.ln_2(x))
        # Residual connection 2
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)
    
class TransformerBase(nn.Module):
    """Parent class for physical transformers
    """
    model_name: str = "transformer_model"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        # Save config in model
        

    def forward(self):
        pass

    def generate(self):
        pass

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=config["training"]["initializer_range"])
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _num_parameters(self) -> int:
        """Gets number of learnable parameters

        Returns:
            int: Number of parameters
        """
        count = 0
        for name, param in self.named_parameters():
            # print(name, param.numel())
            count += param.numel()
        return count
    
    def save_model(self, save_directory: str, epoch: int = 0) -> None:
        """Saves transformer model to the specified directory.

        Args:
            save_directory (str): Folder to save file at
            epoch (int, optional): Epoch number to name model file. Defaults to 0.

        Raises:
            AssertionError: If provided directory is not valid.
        """
        if os.path.isfile(save_directory):
            raise AssertionError("Provided path ({}) should be a directory, not a file".format(save_directory))

        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "{}{:d}.pth".format(self.model_name, epoch))
        # Save pytorch model to file
        torch.save(self.state_dict(prefix='transformer'), output_model_file)


    def load_model(self, file_or_path_directory: str, epoch: int = 0) -> None:
        """Load a transformer model from the specified file or path
        
        Args:
            file_or_path_directory (str): File or folder path to load state dictionary from.
            epoch (int, optional): Epoch of current model for file name, used if folder path is provided. Defaults to 0.
        
        Raises:
            FileNotFoundError: If provided file or directory could not be found.
        """
        if os.path.isfile(file_or_path_directory):
            logger.info('Loading embedding model from file: {}'.format(file_or_path_directory))
            self.load_state_dict(torch.load(file_or_path_directory, map_location=lambda storage, loc: storage))
        elif  os.path.isdir(file_or_path_directory):
            file_path = os.path.join(file_or_path_directory, "{}{:d}.pth".format(self.model_name, epoch))
            logger.info('Loading embedding model from file: {}'.format(file_path))
            self.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))
        else:
            raise FileNotFoundError("Provided path or file ({}) does not exist".format(file_or_path_directory))
            
class TransformerTrain(TransformerBase):
    """Model head for training the physics transformer base.

    Args:
        config (PhysConfig): Phys-transformer config object
        transformer_model (PhysformerBase): Initialized transformer model
    """
    def __init__(self, transformer_model: TransformerBase = None) -> None:
        """Constructor
        """
        super().__init__()
        self.transformer = transformer_model
        self.transformer.apply(self._init_weights)

    def forward(
        self,
        inputs_x: Tensor,
        labels: Tensor,
        **kwargs
    ) -> Tuple[Union[float, Tensor]]:
        """Forward method for this head calculates the MSE between the predicted time-series and target embeddings
        This head allows for easy distribution to multiple GPUs and CPUs. See transformer 

        Args:
            inputs_embeds (Tensor): [B, T, n_embed] Input features
            labels_embeds (Tensor): [B, T, n_embed] Target output features
            **kwargs (optional): Additional tensformer forward pass arguments

        Returns:
            Tuple[Union[float, Tensor]]: mse loss, last hidden state, (present attention state), 
            (all hidden_states), (attention scores)
        """
        outputs = self.transformer.forward(
            inputs_x=inputs_x,
            **kwargs
        )
        # If label embeddings are provided, compute loss
        if labels is not None:
            hidden_states = outputs[0]

            # Flatten the tokens
            loss_fct = nn.MSELoss()
            loss = loss_fct(hidden_states, labels)

            # loss = loss+ loss_fct(shift_hidden[:,:3], shift_labels[:,:3])
            outputs = (loss,) + (hidden_states, labels,) + outputs[1:]

        return outputs # (loss), last hidden state, (presents), (all hidden_states), (attentions)

    def evaluate(
        self,
        inputs_x: Tensor,
        labels: Tensor,
        **kwargs
    ) -> Tuple[Union[float, Tensor]]:
        """Generate a time-series prediction using the transformer and calc MSE error.

        Args:
            inputs_embeds (Tensor): [B, 1, n_embed] Starting input feature(s)
            labels_embeds (Tensor): [B, T, n_embed] Target output features
            **kwargs (optional): Additional tensformer forward pass arguments

        Returns:
            Tuple[Union[float, Tensor]]: mse loss, last hidden state, (present attention state), 
            (all hidden_states), (attention scores)
        """

        max_length = labels.size(1)

        outputs = self.transformer.generate(
            inputs_x=inputs_x,
            max_length = max_length,
            **kwargs
        )
        predicted = outputs[0]

        # Flatten the tokens
        err_fct = nn.MSELoss()
        error = err_fct(predicted, labels)

        outputs = (error,) + (predicted, labels,) + outputs[1:]

        return outputs

    def generate(self, *args, **kwargs):
        """
        Generate call is just the forward call of the transformer
        """
        return self.transformer.generate(*args, **kwargs)

class Time_Series_Generation:
    """Class containing generative functions for transformers
    """
    def prepare_inputs_for_generation(
        self, 
        inputs_x: Tensor,
        position_ids: Tensor = None,
        prop_embeds: Tensor = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """Prepares input features for prediction

        Args:
            inputs_features (Dict[str, Tensor]): Input feature tensors
            that are being generated.

        Returns:
            Dict[str, Tensor]: Dictionary of model inputs
        """
        inputs_features = {
            "inputs_x": inputs_x,
            "position_ids": position_ids,
            "prop_embeds": prop_embeds }
        inputs = {}

        for k, v in inputs_features.items():
            if isinstance(v, torch.Tensor):
                # Make sure all embeddings are of equal and proper length
                inputs[k] = v[:, -config["training"]["n_ctx"]:]

        if "past" in kwargs.keys():
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v[:, -1].unsqueeze(1)

        return {**inputs, **kwargs}

    def _use_cache(self, outputs, use_cache):
        """During generation, decide whether to pass the `past` variable to the next forward pass."""
        if len(outputs) <= 1 or use_cache is False:
            return False
        if hasattr(self.config, "mem_len") and self.config.mem_len == 0:
            return False
        return True

    @torch.no_grad()
    def generate(
        self,
        inputs_x: Tensor,
        position_ids: Tensor = None,
        prop_embeds: Tensor = None,
        max_length: int = None,
        attention_mask: LongTensor = None,
        use_cache: bool = False,
        **model_specific_kwargs
    ) -> Tuple[Tensor]:
        """Generated a predicted sequence of features

        Args:
            inputs_embeds (Tensor): [batch, seq, n_embed] Input feature tensor
            position_ids (Tensor, optional): [seq, n_embed] Position tensor. Defaults to None.
            prop_embeds (Tensor, optional): [batch, seq, n_embed] Property tensor. Defaults to None.
            max_length (int, optional): Length of time series to predict. Defaults to None.
            attention_mask (LongTensor, optional): Manual attention mask. Defaults to None.
            use_cache (bool, optional): Cache past transformer states for faster generation. Defaults to False.

        Returns:
            Tuple[Tensor]: [batch, max_length, n_embed] Predicted feature tensor, additional optional transformer outputs.
        """
        max_length = max_length if max_length is not None else self.config.max_length
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."

        output = self._generate_time_series(
            inputs_x,
            position_ids,
            prop_embeds,
            max_length=max_length,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **model_specific_kwargs,
        )

        return output

    def _generate_time_series(
        self,
        inputs_x: Tensor,
        position_ids: Tensor,
        prop_embeds: Tensor,
        max_length: int,
        use_cache: bool = None,
        **model_specific_kwargs
    ) -> Tuple[Tensor]:
        """Function that calls model forward to predict 

        Args:
            inputs_embeds (Tensor): [batch, seq, n_embed] Input feature tensor
            position_ids (Tensor, optional): [seq, n_embed] Position tensor. Defaults to None.
            prop_embeds (Tensor, optional): [batch, seq, n_embed] Property tensor. Defaults to None.
            max_length (int, optional): Length of time series to predict. Defaults to None.
            attention_mask (LongTensor, optional): Manual attention mask. Defaults to None.
            use_cache (bool, optional): [description]. Defaults to None.

        Returns:
            Tuple[Tensor]: [batch, max_length, n_embed] Predicted feature tensor, additional optional transformer outputs.
        """
        past = None

        cur_len = inputs_x.shape[1]
        assert (
            cur_len < max_length
        ), f"The input context is {cur_len}, but `max_length` is only {max_length}. Please make sure that `max_length` larger than the input"

        while cur_len < max_length:
            # Prepare inputs for transformer
            model_inputs = self.prepare_inputs_for_generation(
                inputs_x, 
                position_ids, 
                prop_embeds, 
                use_cache=use_cache, 
                past = past,
                **model_specific_kwargs,
            )

            outputs = self.forward(**model_inputs)

            next_output = outputs[0][:,-1:]

            if self._use_cache(outputs, use_cache):
                past = [output[:, :, :, -(self.config.n_ctx-1):] for output in outputs[1]]

            # add past output embedding and increase length by one
            inputs_x = torch.cat([inputs_x, next_output], dim=1)
            cur_len = cur_len + 1

            # If number of time-steps has surpassed model capacity, start dropping
            # the earliest time-step from the past states
            # if(cur_len > self.config.n_ctx):
                # Dim [keys/query, batch, heads, tsteps, n_embed]
                # past = tuple(attention_state[:,:,:,1:] for attention_state in past)

        return (inputs_x, ) + outputs[1:]
    
class PhysformerGPT2(Time_Series_Generation, TransformerBase):
    """Transformer decoder model for modeling physics

    Args:
            config (PhysConfig): Phys-transformer config object
            model_name (str, optional): Model name. Defaults to None.
    """
    def __init__(self) -> None:
        """Constructor        
        """
        TransformerBase.__init__(self)
        self.output_hidden_states = config["training"]["output_hidden_state"]

        self.drop = nn.Dropout(config["training"]["embd_pdrop"])
        self.h = nn.ModuleList([Block(config["training"]["n_ctx"], scale=True) for _ in range(config["model"]["n_layer"])])
        self.ln_f = nn.LayerNorm(config["model"]["n_embd"], eps=config["training"]["layer_norm_epsilon"])
        self.mlp_f = nn.Linear(config["model"]["n_embd"], config["model"]["n_embd"])
        #self.wpe = nn.Embedding(config["training"]["n_ctx"], config["model"]["n_embd"])
        self.apply(self._init_weights)

        self.n_embd = config["model"]["n_embd"]

        logger.info('Number of parameters: {}'.format(self._num_parameters()))
            
    def forward(
        self,
        inputs_x: Tensor,
        position_ids: Tensor = None,
        past: List[List[Tensor]] = None,
        attention_mask: LongTensor = None,
        head_mask: LongTensor = None,
        use_cache: bool = True,
        output_attentions: bool = False
    ) -> List[Tensor]:
        """Forward pass

        Note: Attention masks are not properly implemented presently and will likely not work.

        Args:
            inputs_embeds (Tensor): [B, T, n_embed] Input features
            position_ids (Tensor, optional): [T, n_embed] Manually specify position ids. Defaults to None.
            prop_embeds (Tensor, optional): [B, T, n_embed] Optional property feature. Defaults to None.
            past (List[List[Tensor]], optional): Transformer past state. Defaults to None.
            attention_mask (LongTensor, optional): [B, T] Sequence attention mask. Defaults to None.
            head_mask (LongTensor, optional): Attention value mask. Defaults to None.
            use_cache (bool, optional): Return attention states (keys). Defaults to True.
            output_attentions (bool, optional): Return attention scores. Defaults to False.

        Returns:
            List[Tensor]:  Output features, attention state (if requested), 
            hidden states of all layers (if requested), attention tensor (if requested)
        """

        # Input embeddings
        input_shape = inputs_x.size()[:-1]
        batch_size = inputs_x.shape[0]

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
            
        if position_ids is None:
            device = config["training"]["device"]
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.float, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1]).repeat(inputs_x.size(0),1)

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Set mask to 0 for positions we want to attend and -10000 for ones we do not
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Function embeddings proposed in original transformer paper
        # http://papers.nips.cc/paper/7181-attention-is-all-you-need
        position_embeds = torch.zeros_like(inputs_x)
        i = torch.arange(0, config["model"]["n_embd"] // 2, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
        position_embeds[:, :, ::2] = torch.sin(position_ids.unsqueeze(-1) / 10000 ** (2 * i / config["model"]["n_embd"]))
        i = i[:, :, config["model"]["n_embd"] % 2]
        position_embeds[:, :, 1::2] = torch.cos(position_ids.unsqueeze(-1) / 10000 ** (2 * i / config["model"]["n_embd"]))
        
        # Combine input embedding, position embeding and prop embeddings
        hidden_states = inputs_x + position_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        # Loop through transformer self-attention layers
        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.mlp_f(self.ln_f(hidden_states))

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if use_cache is True:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
            
        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)

transformer = PhysformerGPT2()
model = TransformerTrain(transformer)

model.load_state_dict(torch.load(config["model"]["path"] + "/model/TransformerGPT.pt"))
model = model.to(config["training"]["device"])

#############################
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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
    output1=[]
    inputs0=[]
    config["training"]["training_h5_file"] = "/bigdata/wonglab/syang159/CFD2/Kfold_data/training_data_{}.hdf5".format(i)

    training_data = DatasetReader(
          config["training"]["training_h5_file"], 
          block_size=config["validating"]["block_size"], 
          stride=config["validating"]["stride"],
          ndata=config["validating"]["ndata"], 
          eval = True,
          )
    training_loader = DataLoader(
        training_data,
        batch_size=config["training"]["batch_size"],
        sampler=SequentialSampler(training_data),
        collate_fn=DataCollator(),
        drop_last=False,
    )
    
    for mbidx, inputs in enumerate(training_loader):
        model.eval()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(config["training"]["device"])
                
        outputs = model.evaluate(**inputs)
        output1.append(outputs[1])
        
    output1_cat = torch.cat(output1,dim=0)
    out_k.append(output1_cat)
    out_k[i]=out_k[i].view(out_k[i].size(0),out_k[i].size(1), 8, 196) 
    out_k[i]=out_k[i][:,:,:3]
    out_k[i]=np.swapaxes(out_k[i], 2,3)
    out_k[i]=out_k[i].detach().cpu()
    
path = ('/bigdata/wonglab/syang159/CFD2/Kfold_data/')
os.chdir(path)
for i in range(5):
    f1 = h5py.File("training_data_from_GPT_%s.hdf5" % i ,'w')
    dsetx = f1.create_dataset("xprime", (out_k[i].size(0),50,196), data=out_k[i][:,:,:,0])
    dsety = f1.create_dataset("yprime", (out_k[i].size(0),50,196), data=out_k[i][:,:,:,1])
    dsetz = f1.create_dataset("zprime", (out_k[i].size(0),50,196), data=out_k[i][:,:,:,2])
    params=f1.create_dataset("params", (torch.tensor(train_params[i]).size(0),5), data=train_params[i])
    erosion=f1.create_dataset("erosion", (torch.tensor(train_erosion[i]).size(0),38312), data=train_erosion[i])
    f1.close()
   

out_k=[]
in_k=[]  
for i in range(5):
    output1=[]
    inputs0=[]
    config["validating"]["validating_h5_file"] =  "/bigdata/wonglab/syang159/CFD2/Kfold_data/validating_data_{}.hdf5".format(i)
    validating_data = DatasetReader(
          config["validating"]["validating_h5_file"], 
          block_size=config["validating"]["block_size"], 
          stride=config["validating"]["stride"],
          ndata=config["validating"]["ndata"], 
          eval = True,
          )
    
    validating_loader = DataLoader(
        validating_data,
        batch_size=config["validating"]["batch_size"],
        sampler=SequentialSampler(validating_data),
        collate_fn=DataCollator(),
        drop_last=False,
        )
    for mbidx, inputs in enumerate(validating_loader):
        model.eval()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(config["training"]["device"])
                
        outputs = model.evaluate(**inputs)
        output1.append(outputs[1])
        
    output1_cat = torch.cat(output1,dim=0)
    out_k.append(output1_cat)
    out_k[i]=out_k[i].view(out_k[i].size(0),out_k[i].size(1), 8, 196)   
    out_k[i]=out_k[i][:,:,:3]
    out_k[i]=np.swapaxes(out_k[i], 2,3)
    out_k[i]=out_k[i].detach().cpu()
    
path = ('/bigdata/wonglab/syang159/CFD2/Kfold_data/')
os.chdir(path)
for i in range(5):
    f2 = h5py.File("validating_data_from_GPT_%s.hdf5" % i,'w')
    dsetx = f2.create_dataset("xprime", (out_k[i].size(0),50,196), data=out_k[i][:,:,:,0])
    dsety = f2.create_dataset("yprime", (out_k[i].size(0),50,196), data=out_k[i][:,:,:,1])
    dsetz = f2.create_dataset("zprime", (out_k[i].size(0),50,196), data=out_k[i][:,:,:,2])
    params=f2.create_dataset("params", (torch.tensor(validate_params[i]).size(0),5), data=validate_params[i])
    erosion=f2.create_dataset("erosion", (torch.tensor(validate_erosion[i]).size(0),38312), data=validate_erosion[i])
    f2.close()   

output1=[]
inputs0=[]
testing_data = DatasetReader(
      config["testing"]["testing_h5_file"], 
      block_size=config["testing"]["block_size"], 
      stride=config["testing"]["stride"],
      ndata=config["testing"]["ndata"], 
      eval = True,
      )

testing_loader = DataLoader(
    testing_data,
    batch_size=config["testing"]["batch_size"],
    sampler=SequentialSampler(testing_data),
    collate_fn=DataCollator(),
    drop_last=False,
)

torch.no_grad()

for mbidx, inputs in enumerate(testing_loader):
    model.eval()
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(config["training"]["device"])
            
    outputs = model.evaluate(**inputs)
    output1.append(outputs[1])
 
output1 = torch.cat(output1,dim=0)
output1=output1.view(output1.size(0),output1.size(1), 8, 196)
output1=output1[:,:,:3]
output1=np.swapaxes(output1, 2,3)
output1=output1.detach().cpu()

f3 = h5py.File("testing_data_from_GPT.hdf5", 'w')
dsetx = f3.create_dataset("xprime", (output1.size(0),50,196), data=output1[:,:,:,0])
dsety = f3.create_dataset("yprime", (output1.size(0),50,196), data=output1[:,:,:,1])
dsetz = f3.create_dataset("zprime", (output1.size(0),50,196), data=output1[:,:,:,2])
params=f3.create_dataset("params", (torch.tensor(testing_params).size(0),5), data=testing_params)
erosion=f3.create_dataset("erosion", (torch.tensor(testing_erosion).size(0),38312), data=testing_erosion)
f3.close()