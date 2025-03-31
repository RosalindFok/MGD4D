import gc
import csv
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Any
from dataclasses import dataclass
from collections import defaultdict

from config import Train_Encoder, seed
from dataset import get_major_dataloader_via_fold
from models import device, Encoder_Structure, get_GPU_memory_usage

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

class SupConLoss(nn.modules.loss._Loss):  
    def __init__(self, temperature=0.07):  
        super().__init__()  
        self.temperature = temperature  

    def forward(self, embeddings, labels):  
        """  
        Args:  
            embeddings: Input features with shape (batch_size, L)  
            labels: Ground truth labels with shape (batch_size,)  
        """  
        batch_size = embeddings.size(0)  
        
        # Normalize embeddings  
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)  
        
        # Compute similarity matrix  
        sim_matrix = torch.matmul(embeddings, embeddings.T)  # (batch_size, batch_size)  
        
        # Create mask for positive samples (same label and not self)  
        labels = labels.view(-1, 1)  
        mask = torch.eq(labels, labels.T).bool()  # (batch_size, batch_size)  
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)  
        mask = mask & ~self_mask  
        
        # Compute logits with temperature scaling  
        logits = sim_matrix / self.temperature  
        
        # Calculate numerator (sum of positive samples)  
        exp_logits = torch.exp(logits)  
        numerator = torch.sum(exp_logits * mask, dim=1)  # (batch_size,)  
        
        # Calculate denominator (sum of all samples except self)  
        denominator = torch.sum(exp_logits, dim=1) - exp_logits.diag()  # (batch_size,)  
        
        # Compute number of positives per sample and handle zero positives  
        num_positives = mask.sum(dim=1).float()  
        valid_indices = num_positives > 0  
        
        # Calculate per-sample loss  
        loss = torch.zeros_like(num_positives)  
        loss[valid_indices] = -torch.log(  
            (numerator[valid_indices] / denominator[valid_indices]) + 1e-8  
        ) / num_positives[valid_indices]  
        
        return loss.mean()  

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
        # fc_matrices = move_to_device(batches.fc, device)
        vbm_matrices = move_to_device(batches.vbm, device)
        target = move_to_device(batches.target, device)
        # forward
        output = model(input_dict=vbm_matrices)
        target = target.flatten()
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
         loss_fn : nn.modules.loss._Loss,
         is_valid : bool = False) -> dict[str, float]:
    model.eval()
    loss_list = []
    with torch.no_grad():
        desc = "Validating" if is_valid else "Testing"
        for batches in tqdm(dataloader, desc=desc, leave=True):
            # move to GPU
            target = move_to_device(batches.target, device)
            # fc_matrices = move_to_device(batches.fc, device)
            vbm_matrices = move_to_device(batches.vbm, device)
            # forward
            output = model(input_dict=vbm_matrices)
            target = target.flatten()
            loss = loss_fn(output, target)
            assert not torch.isnan(loss), f"Loss is NaN: {loss}"
            loss_list.append(loss.cpu().item())
    
    metrices = {
        "MAX" : max(loss_list),
        "MIN" : min(loss_list),
        "MEAN" : sum(loss_list)/len(loss_list),
        "STD" : np.std(loss_list)
    }
    metrices = {k : float(v) for k, v in metrices.items()}

    return metrices

def main() -> None:
    test_results = defaultdict(list)
    for fold in Train_Encoder.n_splits:
        return_dataloaders = get_major_dataloader_via_fold(fold)
        train_dataloader, test_dataloader = return_dataloaders.train, return_dataloaders.test

        # Shape
        fc_matrices = next(iter(test_dataloader)).fc
        vbm_matrices = next(iter(test_dataloader)).vbm
        keys_dict = {"structural" : {k:None for k in vbm_matrices.keys()},
                     "functional" : {k:None for k in fc_matrices.keys() }}

        # Model
        model = Encoder_Structure(matrices_shape={k:v.shape for k, v in vbm_matrices.items()}, 
                                  embedding_dim=Train_Encoder.latent_embedding_dim).to(device)
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"The number of trainable parametes of {model.__class__.__name__} is {trainable_parameters}.")
        with open("model_structure.txt", "w") as f:  
            f.write(str(model))

        # Loss
        loss_fn = SupConLoss()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=Train_Encoder.lr)

        for epoch in Train_Encoder.epochs:
            lr = Train_Encoder.lr*((1-epoch/Train_Encoder.epochs.stop)**0.9)
            lr = max(lr, Train_Encoder.lr*0.1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            print(f"\nFold {fold}/{Train_Encoder.n_splits.stop-1}, Epoch {epoch + 1}/{Train_Encoder.epochs.stop}")
            # Train
            train_returns = train(device=device, model=model, loss_fn=loss_fn, optimizer=optimizer, dataloader=train_dataloader)
            print(f"Train loss: {train_returns.train_loss}, GPU memory usage: {train_returns.reserved_memory:.2f}GB / {train_returns.total_memory:.2f}GB")
            # Valid
            metrics = test(device=device, model=model, dataloader=test_dataloader, loss_fn=loss_fn)
            print(metrics)
            
        # Test
        print("Test")
        metrics = test(device=device, model=model, dataloader=test_dataloader)
        with open(f"fold_{fold}_test_results.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        for key, value in metrics.items():
            test_results[key].append(value)

        del model, optimizer, loss_fn
        del auxi_info, fc_matrices, vbm_matrices
        del train_dataloader, test_dataloader
        torch.cuda.empty_cache()

    # Write all results
    results = defaultdict(dict)
    for key, values in test_results.items():
        assert len(values) == Train_Encoder.n_splits.stop - 1, f"The number of results of {key} = {len(values)} is not equal to the number of folds = {Train_Encoder.n_splits.stop - 1}."
        results[key] ={"fold" : values, "mean" : np.mean(values)}
    with open("all_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)  

if __name__ == "__main__":
    main()