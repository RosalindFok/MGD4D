import csv
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Any
from dataclasses import dataclass
from collections import defaultdict

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
    for batches in tqdm(dataloader, desc="Training", leave=True):
        # move to GPU
        auxi_info = move_to_device(batches.info, device)
        fc_matrices = move_to_device(batches.fc, device)
        vbm_matrices = move_to_device(batches.vbm, device)
        target = move_to_device(batches.target, device)
        # forward
        output =  model(structural_input_dict=vbm_matrices,
                        functional_input_dict=fc_matrices,
                        information_input_dict=auxi_info)
        target = nn.functional.one_hot(target.long(), num_classes=2).float()
        # loss
        loss = loss_fn(output, target)
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

def test(device : torch.device,
         model : nn.Module, 
         dataloader : torch.utils.data.DataLoader,
         is_valid : bool = False,
         log_epoch : int = -1) -> dict[str, float]:
    model.eval()
    subjid_list = []
    target_list = []
    prediction_list = []
    probability_list = []
    with torch.no_grad():
        desc = "Validating" if is_valid else "Testing"
        for batches in tqdm(dataloader, desc=desc, leave=True):
            # move to GPU
            auxi_info = move_to_device(batches.info, device)
            fc_matrices = move_to_device(batches.fc, device)
            vbm_matrices = move_to_device(batches.vbm, device)
            # forward
            output =  model(structural_input_dict=vbm_matrices,
                            functional_input_dict=fc_matrices,
                            information_input_dict=auxi_info)
            probability = output[:, -1]
            prediction = output.argmax(dim=1)
            target_list.extend(batches.target.numpy())
            prediction_list.extend(prediction.cpu().numpy())
            probability_list.extend(probability.cpu().numpy())
            subjid_list.extend(batches.id)
    
    target = np.array(target_list).astype(int)
    probability = np.array(probability_list).astype(float)
    prediction = np.array(prediction_list).astype(int)
    assert target.shape == probability.shape == prediction.shape, f"{target.shape} != {probability.shape} != {prediction.shape}"
    metrices = {
        "AUC" : Metrics.AUC(prob=probability, true=target),
        "ACC" : Metrics.ACC(pred=prediction, true=target),
        "PRE" : Metrics.PRE(pred=prediction, true=target),
        "SEN" : Metrics.SEN(pred=prediction, true=target),
        "F1S" : Metrics.F1S(pred=prediction, true=target)
    }
    metrices = {k : float(v) for k, v in metrices.items()}

    if log_epoch >= 0:
        plot_confusion_matrix(target, prediction, saved_path=f"epoch_{log_epoch}_confusion_matrix.png")
        with open(f"epoch_{log_epoch}_valid_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Subject", "True Label", "Probability", "Predicted Label"])
            for id, true, prob, pred in zip(subjid_list, target, probability, prediction):
                writer.writerow([id, true, f"{prob:.6f}", pred])
    return metrices

def main() -> None:
    test_results = defaultdict(list)
    for fold in Train_Config.n_splits:
        return_dataloaders = get_major_dataloader_via_fold(fold)
        train_dataloader, test_dataloader = return_dataloaders.train, return_dataloaders.test
        
        # Shape
        auxi_info = next(iter(test_dataloader)).info
        fc_matrices = next(iter(test_dataloader)).fc
        vbm_matrices = next(iter(test_dataloader)).vbm

        # Model
        model = MGD4MD(auxi_info_number=len(auxi_info),
                       functional_embeddings_shape={k:v.shape[1]*(v.shape[1]-1)//2 for k, v in fc_matrices.items()},
                       structural_matrices_shape={k:v.shape[1:] for k,v in vbm_matrices.items()},
                       embedding_dim=Train_Config.latent_embedding_dim,
                       use_lgd=Train_Config.use_lgd).to(device)
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"The number of trainable parametes of {model.__class__.__name__} is {trainable_parameters}.")
        with open("model_structure.txt", "w") as f:  
            f.write(str(model))

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
            train_returns = train(device=device, model=model, loss_fn=loss_fn, optimizer=optimizer, dataloader=train_dataloader)
            print(f"Train loss: {train_returns.train_loss}, GPU memory usage: {train_returns.reserved_memory:.2f}GB / {train_returns.total_memory:.2f}GB")
            # Valid
            metrics = test(device=device, model=model, dataloader=test_dataloader, 
                           is_valid=True, log_epoch=epoch+1)
            print(metrics)
    
        # Test
        print("Test")
        metrics = test(device=device, model=model, dataloader=test_dataloader)
        with open(f"fold_{fold}_test_results.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        
        break

    # Write all results
    results = defaultdict(dict)
    for key, values in test_results.items():
        assert len(values) == Train_Config.n_splits.stop - 1, f"The number of results of {key} = {len(values)} is not equal to the number of folds = {Train_Config.n_splits.stop - 1}."
        results[key] ={"fold" : values, "mean" : np.mean(values)}
    with open("all_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)  

if __name__ == "__main__":
    main()