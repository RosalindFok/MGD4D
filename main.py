import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Any
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

from config import Train_Config
from models import device, MGD4MD, get_GPU_memory_usage
from dataset import get_major_dataloader_via_fold

def move_to_device(data: Any, device: torch.device) -> Any:  
    """Recursively moves all torch.Tensor objects in a nested structure to the specified device.  

    This function traverses through nested data structures such as dictionaries, lists, and tuples,  
    and moves all `torch.Tensor` objects to the specified device (e.g., 'cuda' or 'cpu'). Non-tensor  
    objects are left unchanged.  

    Args:  
        data: The input data, which can be a nested structure containing dictionaries, lists, tuples,  
              torch.Tensor objects, or other types.  
        device: The target device to which all torch.Tensor objects should be moved.  

    Returns:  
        The input data structure with all torch.Tensor objects moved to the specified device. The  
        structure of the input data (e.g., dict, list, tuple) is preserved.  
    """
    if isinstance(data, dict):  
        return {k: move_to_device(v, device) for k, v in data.items()}  
    elif isinstance(data, list):  
        return [move_to_device(v, device) for v in data]  
    elif isinstance(data, tuple):  
        return tuple(move_to_device(v, device) for v in data)  
    elif isinstance(data, torch.Tensor):  
        return data.to(device)  
    else:
        return data

@dataclass
class Train_Returns:
    train_loss: float
    total_memory: float
    reserved_memory: float

def train(device : torch.device,
          model : nn.Module, 
          loss_fn : nn.modules.loss._Loss, 
          optimizer : torch.optim.Optimizer, 
          dataloader : torch.utils.data.DataLoader) -> Train_Returns:
    model.train()
    train_loss_list = []
    mem_reserved_list = []
    for auxi_info, fc_matrices, vbm_matrices, tag in tqdm(dataloader, desc="Training", leave=True):
        # move to GPU
        auxi_info = move_to_device(auxi_info, device)
        fc_matrices = move_to_device(fc_matrices, device)
        vbm_matrices = move_to_device(vbm_matrices, device)
        tag = move_to_device(tag, device)
        # forward
        prediction =  model(structural_input_dict=vbm_matrices,
                            functional_input_dict=fc_matrices,
                            information_input_dict=auxi_info)
        # loss
        loss = loss_fn(prediction.squeeze(dim=1), tag)
        assert not torch.isnan(loss), f"Loss is NaN: {loss}"
        train_loss_list.append(loss.item())
        # 3 steps of back propagation
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        # monitor GPU memory usage
        total_memory, mem_reserved = get_GPU_memory_usage()
        mem_reserved_list.append(mem_reserved)
    return Train_Returns(train_loss=sum(train_loss_list)/len(train_loss_list), 
                         total_memory=total_memory, reserved_memory=max(mem_reserved_list))

@dataclass
class Test_Returns:
    metrics: dict
    total_memory: float
    reserved_memory: float

def test(device : torch.device,
          model : nn.Module, 
          dataloader : torch.utils.data.DataLoader,
          threshold : float = 0.5,
          is_valid : bool = False) -> Test_Returns:
    model.eval()
    mem_reserved_list = []
    tag_list = []
    prediction_list = []
    with torch.no_grad():
        desc = "Validating" if is_valid else "Testing"
        for auxi_info, fc_matrices, vbm_matrices, tag in tqdm(dataloader, desc=desc, leave=True):
            # move to GPU
            auxi_info = move_to_device(auxi_info, device)
            fc_matrices = move_to_device(fc_matrices, device)
            vbm_matrices = move_to_device(vbm_matrices, device)
            tag = move_to_device(tag, device)
            # forward
            prediction =  model(structural_input_dict=vbm_matrices,
                                functional_input_dict=fc_matrices,
                                information_input_dict=auxi_info)
            # metrices
            tag_list.extend(tag.cpu().numpy())
            prediction_list.extend(prediction.cpu().numpy())
            # monitor GPU memory usage
            total_memory, mem_reserved = get_GPU_memory_usage()
            mem_reserved_list.append(mem_reserved)
    
    tag = np.array(tag_list).astype(int)
    prediction = np.array(prediction_list)
    pred_label = (prediction >= threshold).astype(int)
    metrices = {
        "AUC" : roc_auc_score(tag, prediction),
        "ACC" : accuracy_score(tag, pred_label),
        "PRE" : precision_score(tag, pred_label),
        "SEN" : recall_score(tag, pred_label),
        "F1"  : f1_score(tag, pred_label)
    }
    metrices = {k : float(v) for k, v in metrices.items()}

    return Test_Returns(metrics=metrices, total_memory=total_memory, reserved_memory=max(mem_reserved_list))

def main() -> None:
    test_results = {}
    for fold in Train_Config.n_splits:
        return_dataloaders = get_major_dataloader_via_fold(fold)
        train_dataloader, test_dataloader = return_dataloaders.train, return_dataloaders.test
        
        # Shape
        auxi_info, fc_matrices, vbm_matrices, tag = next(iter(test_dataloader))
        # for k, v in fc_matrices.items():
        #     print(k, v["embedding"].max(), v["embedding"].min())
        # for k, v in vbm_matrices.items():
        #     print(k, v.max(), v.min())
        # exit()

        # Model
        model = MGD4MD(structural_matrices_number=len(vbm_matrices)).to(device)
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"The number of trainable parametes of {model.__class__.__name__} is {trainable_parameters}.")
        with open("model_structure.txt", "w") as f:  
            f.write(str(model))

        # Loss
        loss_fn = nn.BCELoss()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=Train_Config.lr)

        for epoch in Train_Config.epochs:
            print(f"Fold {fold}/{Train_Config.n_splits.stop-1}, Epoch {epoch + 1}/{Train_Config.epochs.stop}")
            # Train
            train_returns = train(device=device, model=model, loss_fn=loss_fn, optimizer=optimizer, dataloader=train_dataloader)
            print(f"Train loss: {train_returns.train_loss}, GPU memory usage: {train_returns.reserved_memory:.2f}GB / {train_returns.total_memory:.2f}GB")
            # Valid
            test_returns = test(device=device, model=model, dataloader=test_dataloader, is_valid=True)
            print(f"Valid GPU memory usage: {test_returns.reserved_memory:.2f}GB / {test_returns.total_memory:.2f}GB")
            print(test_returns.metrics)
    
    # Test
    test_returns = test(device=device, model=model, dataloader=test_dataloader)
    test_results[fold] = test_returns.metrics

if __name__ == "__main__":
    main()