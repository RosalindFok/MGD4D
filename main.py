import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from metrics import Metrics
from config import Configs, seed
from plot import plot_confusion_matrix
from utils import write_csv, move_to_device
from models import device, MGD4D, get_GPU_memory_usage
from dataset import get_major_dataloader_via_fold, get_mild_dataloader_via_fold

torch.manual_seed(seed)
if torch.cuda.is_available():  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
        
def train(device : torch.device,
          model : nn.Module, 
          loss_fn : nn.modules.loss._Loss, 
          optimizer : torch.optim.Optimizer, 
          dataloader : torch.utils.data.DataLoader,
          log : bool = False) -> None:
    model.train()
    subjid_list, train_loss_list, probability_list, prediction_list, target_list = [],[],[],[],[]
    mem_reserved_list = []
    for batches in tqdm(dataloader, desc="Training", leave=False):
        # move to GPU
        auxi_info = move_to_device(batches.info, device)
        f_matrices = move_to_device(batches.func, device)
        s_matrices = move_to_device(batches.anat, device)
        target = move_to_device(batches.target, device).long().flatten()
        # forward
        output =  model(structural_input_dict=s_matrices,
                        functional_input_dict=f_matrices,
                        information_input_dict=auxi_info)
        # loss
        loss = loss_fn(cet_input=output["logits"],    cet_target=target,
                       mse_input=output["mse_input"], mse_target=output["mse_target"])
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
            probability = output["logits"].softmax(dim=-1)
            probability_list.extend(probability[:, 1].detach().cpu().numpy()) 
            prediction_list.extend(probability.argmax(dim=-1).detach().cpu().numpy())
            target_list.extend(target.detach().cpu().numpy())
            subjid_list.extend(batches.id)
    
    print(f"Train Loss: {sum(train_loss_list)/len(train_loss_list)}, GPU memory usage: {max(mem_reserved_list):.2f}GB / {total_memory:.2f}GB")
    
    if log:
        probability = np.array(probability_list).flatten()
        target = np.array(target_list).astype(int).flatten()
        prediction = np.array(prediction_list).astype(int).flatten()
        assert target.shape == probability.shape == prediction.shape, f"{target.shape}!= {probability.shape}!= {prediction.shape}"
        metrics = {
            "AUC" : Metrics.AUC(prob=probability, true=target),
            "ACC" : Metrics.ACC(pred=prediction, true=target),
            "PRE" : Metrics.PRE(pred=prediction, true=target),
            "SEN" : Metrics.SEN(pred=prediction, true=target),
            "F1S" : Metrics.F1S(pred=prediction, true=target)
        }
        metrics = {k : float(v) for k, v in metrics.items()}
        print(f"Train Metrics: {metrics}")

def test(device : torch.device,
         epoch : int,
         fold : int,
         model : nn.Module, 
         dataloader : torch.utils.data.DataLoader,
         loss_fn : nn.modules.loss._Loss, 
         log : bool,
         is_test : bool ) -> bool:
    early_stop = False # TODO if needed
    model.eval()
    subjid_list, probability_list, prediction_list, valid_loss_list, target_list = [],[],[],[],[]
    with torch.no_grad():
        desc = "Testing" if is_test else "Validating"
        for batches in tqdm(dataloader, desc=desc, leave=False):
            # move to GPU
            auxi_info = move_to_device(batches.info, device)
            f_matrices = move_to_device(batches.func, device)
            s_matrices = move_to_device(batches.anat, device)
            target = move_to_device(batches.target, device).long().flatten()
            # forward
            output =  model(structural_input_dict=s_matrices,
                            functional_input_dict=f_matrices,
                            information_input_dict=auxi_info)
            # loss without back propagation
            loss = loss_fn(cet_input=output["logits"],    cet_target=target,
                           mse_input=output["mse_input"], mse_target=output["mse_target"])
            assert not torch.isnan(loss), f"Loss is NaN: {loss}" 
            valid_loss_list.append(loss.cpu().item())
            # add to lists
            probability = output["logits"].softmax(dim=-1)
            target_list.extend(target.cpu().numpy())
            probability_list.extend(probability[:, 1].cpu().numpy()) 
            prediction_list.extend(probability.argmax(dim=-1).cpu().numpy())
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

    # TODO early stop 
    valid_loss = sum(valid_loss_list) / len(valid_loss_list)

    if log and not is_test: # valid
        print(f"Valid Loss: {valid_loss}")
        print(f"Valid Metrics: {metrics}")
        plot_confusion_matrix(target, prediction, saved_path=f"fold_{fold}_epoch_{epoch+1}_valid_confusion_matrix.png")
        write_csv(filename=Path(f"fold_{fold}_epoch_{epoch+1}_valid_results.csv"),
                  head=["subj", "true", "prob", "pred"], data=[subjid_list, target, probability, prediction])
    
    if is_test: # test
        print(f"Test metrics: {metrics}")
        plot_confusion_matrix(target, prediction, saved_path=f"fold_{fold}_test_confusion_matrix.png")
        with open(f"fold_{fold}_test_results.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        write_csv(filename=Path(f"fold_{fold}_test_results.csv"),
                  head=["subj", "true", "prob", "pred"], data=[subjid_list, target, probability, prediction])

    return early_stop

class Combined_Loss(nn.modules.loss._Loss):
    def __init__(self, cet_weight : torch.Tensor, use_mse : bool, 
                 w_1 : float = 1.0, w_2 : float = 1.0) -> None:
        super().__init__()
        self.cet_loss = nn.CrossEntropyLoss(weight=cet_weight)
        self.mse_loss = nn.MSELoss()
        self.use_mse = use_mse
        self.w_1 = w_1
        self.w_2 = w_2
    
    def forward(self, cet_input : torch.Tensor, cet_target : torch.Tensor, 
                      mse_input : torch.Tensor, mse_target : torch.Tensor) -> torch.Tensor:
        assert cet_input is not None and cet_target is not None, "CET input and target must not be None"
        cet_loss = self.cet_loss(input=cet_input, target=cet_target)
        if self.use_mse:
            assert mse_input is not None and mse_target is not None, "MSE input and target must not be None"
            mse_loss = self.mse_loss(input=mse_input, target=mse_target)
        else:
            mse_loss = torch.tensor(0.0, device=cet_input.device)
        return self.w_1*cet_loss + self.w_2*mse_loss
        
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str)
    args = parser.parse_args()
    fold = int(args.fold)
    
    # Dataloaders
    if Configs.dataset.info == "major":
        return_dataloaders = get_major_dataloader_via_fold(fold)
    elif Configs.dataset.info == "mild":
        return_dataloaders = get_mild_dataloader_via_fold(fold)
    else:
        raise ValueError(f"Invalid dataset info: {Configs.dataset.info}")
    train_dataloader, test_dataloader, auxi_info, weight = return_dataloaders.train, return_dataloaders.test, return_dataloaders.info, return_dataloaders.weight
    # Shapes
    f_matrices = next(iter(test_dataloader)).func
    s_matrices = next(iter(test_dataloader)).anat
    shapes_dict = {"s" : {k:v.shape[1:] for k,v in s_matrices.items()},                   # structural
                   "f" : {k:v.shape[1]*(v.shape[1]-1)//2 for k, v in f_matrices.items()}} # functional

    # Model
    model = MGD4D(info_dict=auxi_info,
                   shapes_dict=shapes_dict,
                   embedding_dim=Configs.dataset.latent_embedding_dim,
                   use_batchnorm=Configs.dataset.use_batchnorm,
                   use_lgd=Configs.dataset.use_lgd,
                   use_modal=Configs.dataset.use_modal).to(device)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The number of trainable parametes of {model.__class__.__name__} is {trainable_parameters}.")

    # Loss
    loss_fn = Combined_Loss(cet_weight=weight.to(device), use_mse=Configs.dataset.use_lgd)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Configs.dataset.lr)

    for epoch in Configs.dataset.epochs:
        # learning rate decay
        lr = Configs.dataset.lr*((1-epoch/Configs.dataset.epochs.stop)**0.9)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print(f"\nFold {fold}/{Configs.n_splits.stop-1}, Epoch {epoch + 1}/{Configs.dataset.epochs.stop}")
        # Train
        train(device=device, model=model, loss_fn=loss_fn, optimizer=optimizer, dataloader=train_dataloader, log=True)
        # Valid
        early_stop = test(device=device, epoch=epoch, fold=fold, model=model, loss_fn=loss_fn, dataloader=test_dataloader, log=True, is_test=False)
        if early_stop:
            break

    # Save the trained model
    torch.save(model.state_dict(), f"{fold}.pth")

    # Test
    print("Test")
    test(device=device, epoch=epoch, fold=fold, model=model, loss_fn=loss_fn, dataloader=test_dataloader, log=True, is_test=True)

if __name__ == "__main__":
    main()