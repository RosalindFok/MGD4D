import gc
import csv
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Any
from pathlib import Path
from itertools import chain

from metrics import Metrics
from config import Train_Config, seed
from plot import plot_confusion_matrix
from dataset import get_major_dataloader_via_fold
from models import device, MGD4MD, get_GPU_memory_usage

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  

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

def write_csv(filename : Path, head : list[Any], data : list[list[Any]]) -> None:
    """  
    Writes data to a CSV file with the specified header.  

    This function takes a file path, a list of column headers, and a 2D list of data,  
    and writes them into a CSV file. Each inner list in the 'data' parameter represents   
    a column, and data is written in column-wise order to produce a correctly formatted  
    CSV file.  

    Args:  
        filename (Path):  
            The path to the CSV file where data will be written.  
        head (list[Any]):  
            A list of column headers, each representing a column in the CSV file.   
            Length must match the number of columns in 'data'.  
        data (list[list[Any]]):  
            A 2D list where each inner list represents a column of data.   
            Each column (inner list) should have the same length.  

    Raises:  
        AssertionError:  
            If the length of 'head' does not match the number of columns in 'data'.   
            This ensures that each header corresponds to one column of data.  
    """
    assert len(head) == len(data), f"Head length {len(head)} does not match data length {len(data)}"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(head)
        for i in range(len(data[0])):
            writer.writerow([data[j][i] for j in range(len(data))])
        
def train(device : torch.device,
          epoch : int,
          model : nn.Module, 
          loss_fn : nn.modules.loss._Loss, 
          optimizer : torch.optim.Optimizer, 
          dataloader : torch.utils.data.DataLoader,
          log : bool = False) -> None:
    model.train()
    subjid_list = []
    train_loss_list = []
    mem_reserved_list = []
    probability_list = []
    prediction_list = []
    target_list = []
    for batches in tqdm(dataloader, desc="Training", leave=True):
        # move to GPU
        auxi_info = move_to_device(batches.info, device)
        fc_matrices = move_to_device(batches.fc, device)
        vbm_matrices = move_to_device(batches.vbm, device)
        target = move_to_device(batches.target, device).flatten()
        target = target.long().flatten()
        # forward
        output =  model(structural_input_dict=vbm_matrices,
                        functional_input_dict=fc_matrices,
                        information_input_dict=auxi_info)
        # loss
        loss = loss_fn(input=output, target=target)
        assert not torch.isnan(loss), f"Loss is NaN: {loss}"
        train_loss_list.append(loss.item())
        # 3 steps of back propagation
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        # monitor GPU memory usage
        total_memory, mem_reserved = get_GPU_memory_usage()
        mem_reserved_list.append(mem_reserved)
        # logging
        if log:
            output = output.softmax(dim=-1)
            probability_list.extend(output[:, 1].detach().cpu().numpy()) 
            prediction_list.extend(output.argmax(dim=-1).detach().cpu().numpy())
            target_list.extend(target.detach().cpu().numpy())
            subjid_list.extend(batches.id)
    print(f"Train loss: {sum(train_loss_list)/len(train_loss_list)}, GPU memory usage: {max(mem_reserved_list):.2f}GB / {total_memory:.2f}GB")
    
    if log:
        probability = np.array(probability_list).flatten()
        target = np.array(target_list).astype(int).flatten()
        prediction = np.array(prediction_list).astype(int).flatten()
        metrics = {
            "AUC" : Metrics.AUC(prob=probability, true=target),
            "ACC" : Metrics.ACC(pred=prediction, true=target),
            "PRE" : Metrics.PRE(pred=prediction, true=target),
            "SEN" : Metrics.SEN(pred=prediction, true=target),
            "F1S" : Metrics.F1S(pred=prediction, true=target)
        }
        metrics = {k : float(v) for k, v in metrics.items()}
        print(f"Train metrics: {metrics}")
        write_csv(filename=Path(f"epoch_{epoch+1}_train_results.csv"), 
                  head=["subj", "true", "prob"], data=[subjid_list, target, probability])

def test(device : torch.device,
         epoch : int,
         model : nn.Module, 
         dataloader : torch.utils.data.DataLoader,
         is_test : int = -1, # <=0 for valid, fold>0 for test
         log : bool = False) -> None:
    model.eval()
    subjid_list = []
    target_list = []
    probability_list = []
    prediction_list = []
    with torch.no_grad():
        desc = "Validating" if is_test <= 0 else "Testing"
        for batches in tqdm(dataloader, desc=desc, leave=True):
            # move to GPU
            auxi_info = move_to_device(batches.info, device)
            fc_matrices = move_to_device(batches.fc, device)
            vbm_matrices = move_to_device(batches.vbm, device)
            # forward
            output =  model(structural_input_dict=vbm_matrices,
                            functional_input_dict=fc_matrices,
                            information_input_dict=auxi_info)
            # add to lists
            output = output.softmax(dim=-1)
            target_list.extend(batches.target.numpy())
            probability_list.extend(output[:, 1].cpu().numpy()) 
            prediction_list.extend(output.argmax(dim=-1).cpu().numpy())
            subjid_list.extend(batches.id)
    
    target = np.array(target_list).astype(int).flatten() 
    probability = np.array(probability_list).astype(float).flatten() 
    prediction = np.array(prediction_list).astype(int).flatten()
    assert target.shape == probability.shape == prediction.shape, f"{target.shape} != {probability.shape} != {prediction.shape}"
    metrics = {
        "AUC" : Metrics.AUC(prob=probability, true=target),
        "ACC" : Metrics.ACC(pred=prediction, true=target),
        "PRE" : Metrics.PRE(pred=prediction, true=target),
        "SEN" : Metrics.SEN(pred=prediction, true=target),
        "F1S" : Metrics.F1S(pred=prediction, true=target)
    }
    metrics = {k : float(v) for k, v in metrics.items()}

    if log and is_test <= 0:
        print(f"Valid metrics: {metrics}")
        plot_confusion_matrix(target, prediction, saved_path=f"epoch_{epoch+1}_valid_confusion_matrix.png")
        write_csv(filename=Path(f"epoch_{epoch+1}_valid_results.csv"),
                  head=["subj", "true", "prob", "pred"], data=[subjid_list, target, probability, prediction])
    
    if is_test > 0: # Test
        print(f"Test metrics: {metrics}")
        plot_confusion_matrix(target, prediction, saved_path=f"fold_{is_test}_confusion_matrix.png")
        with open(f"fold_{is_test}_results.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        write_csv(filename=Path(f"fold_{is_test}_results.csv"),
                  head=["subj", "true", "prob", "pred"], data=[subjid_list, target, probability, prediction])

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str)
    args = parser.parse_args()
    fold = int(args.fold)
    
    return_dataloaders = get_major_dataloader_via_fold(fold)
    train_dataloader, test_dataloader, auxi_info = return_dataloaders.train, return_dataloaders.test, return_dataloaders.info

    # Shape
    fc_matrices = next(iter(test_dataloader)).fc
    vbm_matrices = next(iter(test_dataloader)).vbm
    shapes_dict = {"s" : {k:v.shape[1:] for k,v in vbm_matrices.items()},                   # structural
                   "f" : {k:v.shape[1]*(v.shape[1]-1)//2 for k, v in fc_matrices.items()}}  # functional

    # Model
    model = MGD4MD(info_dict=auxi_info,
                   shapes_dict=shapes_dict,
                   embedding_dim=Train_Config.latent_embedding_dim,
                   use_lgd=Train_Config.use_lgd).to(device)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The number of trainable parametes of {model.__class__.__name__} is {trainable_parameters}.")

    # Loss
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Train_Config.lr)

    for epoch in Train_Config.epochs:
        lr = Train_Config.lr*((1-epoch/Train_Config.epochs.stop)**0.9)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print(f"\nFold {fold}/{Train_Config.n_splits.stop-1}, Epoch {epoch + 1}/{Train_Config.epochs.stop}")
        # Train
        train(device=device, epoch=epoch, model=model, loss_fn=loss_fn, 
              optimizer=optimizer, dataloader=train_dataloader, log=True)
        # Valid
        test(device=device, epoch=epoch, model=model, dataloader=test_dataloader, log=True)
            
    # Test
    print("Test")
    test(device=device, epoch=epoch, model=model, dataloader=test_dataloader, is_test=fold)
    

    del model, optimizer, loss_fn
    del auxi_info, fc_matrices, vbm_matrices
    del train_dataloader, test_dataloader
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()