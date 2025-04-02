import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from models import device
from metrics import Metrics
from config import Train_Config
from dataset import get_major_dataloader_via_fold


# TODO 考虑各个baseline的License不同，还是不放仓库里了，自己跑一跑算了

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    assert args.model in ["ORCGNN", "DGCN", "GNNMA", "ContrasPool"], f"{args.model} is not supported, please choose from [ORCGNN, DGCN, GNNMA, ContrasPool]"

    if args.model == "GNNMA":
        """GNNMA: https://doi.org/10.1016/j.bspc.2024.107252"""
        from baselines.GNNMA.models import NEResGCN
        model = NEResGCN(layer=5).to(device)
        # default settings, but it is uncertain whether they were adopted by GNNMA
        seed = 127
        epochs = 25
        lr = 0.1
        weight_decay = 5e-4
        # optimizer is the same as GNNMA
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        # loss is the same as GNNMA
        loss = nn.CrossEntropyLoss()
        # batchsize is the same as GNNMA
        batch_size = 6

        # I tried different hyperparameters in an attempt to align with the results in the paper.
        lr = 1e-5
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        test_results = defaultdict(list)
        for fold in Train_Config.n_splits: # it is 5 in GNNMA
            return_dataloaders = get_major_dataloader_via_fold(fold=fold, batch_size=batch_size)
            train_dataloader, test_dataloader = return_dataloaders.train, return_dataloaders.test

            for epoch in range(epochs):
                # train
                model.train()
                loss_list = []
                for _, fc_matrices, _, tag in tqdm(train_dataloader, desc="Train", leave=True):
                    aal_matrix = fc_matrices["AAL"].to(device)
                    tag = nn.functional.one_hot(tag.long(), num_classes=2).to(device)
                    y_pred = model(aal_matrix, aal_matrix)
                    loss_train = loss(y_pred, tag.float())
                    loss_train.backward()
                    optimizer.zero_grad()
                    optimizer.step()
                    loss_list.append(loss_train.item())
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {sum(loss_list) / len(loss_list)}")

                # valid/test
                model.eval()
                prob_list, pred_list, tag_list = [], [], []
                with torch.no_grad():
                    for _, fc_matrices, _, tag in tqdm(test_dataloader, desc="Test", leave=True):
                        aal_matrix = fc_matrices["AAL"].to(device)
                        y_pred = model(aal_matrix, aal_matrix)
                        prob_list.extend(y_pred[:, -1].cpu().numpy())
                        pred_list.extend(y_pred.argmax(dim=1).cpu().numpy())
                        tag_list.extend(tag.cpu().numpy())
                prob = np.array(prob_list).astype(float)
                pred = np.array(pred_list).astype(int)
                tag  = np.array(tag_list).astype(int)

                metrics = {
                    "AUC" : Metrics.AUC(prob=prob, true=tag),
                    "ACC" : Metrics.ACC(pred=pred, true=tag),
                    "PRE" : Metrics.PRE(pred=pred, true=tag),
                    "SEN" : Metrics.SEN(pred=pred, true=tag),
                    "F1S" : Metrics.F1S(pred=pred, true=tag)
                }
                metrics = {k : float(v) for k, v in metrics.items()}
                for key, value in metrics.items():
                    print(f"{key}: {value}")
            for key, value in metrics.items():
                test_results[key].append(value)

        # Write all results
        results = defaultdict(dict)
        for key, values in test_results.items():
            assert len(values) == Train_Config.n_splits.stop - 1, f"The number of results of {key} = {len(values)} is not equal to the number of folds = {Train_Config.n_splits.stop - 1}."
            results[key] ={"fold" : values, "mean" : np.mean(values)}
        with open("test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)  

    
    elif args.model == "ContrasPool":
        """ContrasPool: https://doi.org/10.1109/TMI.2024.3392988"""
        # https://github.com/AngusMonroe/ContrastPool/blob/main/configs/abide_schaefer100/TUs_graph_classification_ContrastPool_abide_schaefer100_100k.json
        