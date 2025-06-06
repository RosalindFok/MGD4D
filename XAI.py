# import json
# import torch
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from pathlib import Path
# from dataclasses import dataclass
# from collections import defaultdict

# from path import Paths
# from config import Configs
# from metrics import Metrics
# from utils import move_to_device
# from models import device, MGD4D
# from dataset import get_mild_dataloader_via_fold

# # Brain network
# @dataclass(frozen=True)
# class Yeo_Network: # for Brainnetome Atlas
#     # in file: subregion_func_network_Yeo_updated.csv
#     #  7: Visual, Somatomotor, Dorsal Attention, Ventral Attention, Limbic, Frontoparietal, Default
#     # 17: Visual peripheral/central, Somato-motor A/B, Dorsal attention A/B, Ventral attention, Salience, Limbic-1/2, Control C/A/B, Default D (Auditory)/C/A/B
#     granularity: str = "fine" # coarse=7 fine=17

# # Networks of Brainnetome Atlas
# def get_yeo_network_of_brainnetome() -> dict[str, np.ndarray]:
#     """  
#     """ 
#     # functional networks, Yeo
#     pd_frame = pd.read_csv(Paths.Atlas.Brainnetome.subregion_func_network_Yeo_updated_csv_path, sep=",", header=1)
   
#     # subregion_func_network_Yeo_updated
#     subregions = pd_frame.iloc[:, :5]
#     fields = ["Label", "subregion_name", "region"]
#     network_name = "Yeo_7network" if Yeo_Network.granularity == "coarse" else "Yeo_17network" if Yeo_Network.granularity == "fine" else None
#     assert network_name is not None, f"Unknown network name: {Yeo_Network.granularity}"
#     subregion_dict = defaultdict(list)
#     for _, row in subregions.iterrows():
#         subregion_dict[row[network_name]].append({field: row[field] for field in fields})
    
#     # Yeo 7 Network  or  Yeo 17 Network, {ID : Network name}
#     yeo_network = pd_frame.iloc[2:9, 10:12] if Yeo_Network.granularity == "coarse" else pd_frame.iloc[12:29, 10:12] if Yeo_Network.granularity == "fine" else None
#     assert yeo_network is not None, f"Unknown network name: {Yeo_Network.granularity}"
#     yeo_dict  = {int(row.iloc[0]) : row.iloc[1] for _, row in yeo_network.iterrows()}

#     label_dict = {} # {Network name : labels from 1 to 246}
#     for index, network in yeo_dict.items():
#         labels = sorted([item["Label"] for item in subregion_dict[index]])
#         labels = np.array(labels)-1 # -1: 0-based indices in functional connectivity matrix
#         label_dict[network] = labels 
#     label_dict.update({"Whole" : np.empty(0, dtype=np.int64)})

#     return label_dict

# def perturb_functional_connectivity(func_dict : dict[str, torch.Tensor], labels : np.ndarray, target : torch.Tensor) -> dict[str, torch.Tensor]:
#     """
#     """
#     perturbed_dict = {k : v.clone() for k, v in func_dict.items()}
#     if not len(labels) == 0:
#         labels = torch.from_numpy(labels)
#         i, j = torch.meshgrid(labels, labels, indexing="ij")
#         i  = i.reshape(-1)
#         j  = j.reshape(-1)
#         for key, tensor in perturbed_dict.items():
#             # selected = torch.where(target == 1)[0] # only positive samples
#             selected = torch.arange(target.size(0))  # all samples
#             num_selected = selected.size(0)
#             batch_indices = selected.unsqueeze(1).expand(-1, i.numel()).reshape(-1)
#             i_rep = i.repeat(num_selected)
#             j_rep = j.repeat(num_selected)
#             tensor[batch_indices, i_rep, j_rep] = 0 # set to 0
#     return perturbed_dict

# def main() -> None:
#     # Dataset: mild depression
#     # Atlas  : Brainnetome

#     all_metrics = defaultdict(lambda: defaultdict(list))
#     for fold in Configs.n_splits:
#         # Dataloaders
#         return_dataloaders = get_mild_dataloader_via_fold(fold=fold, num_workers=4)
#         test_dataloader, auxi_info = return_dataloaders.test, return_dataloaders.info

#         # for network, labels in get_yeo_network_of_brainnetome().items():
#         #     i_indices, j_indices = np.meshgrid(labels, labels)
#         #     i_indices = torch.from_numpy(i_indices.flatten()).long()
#         #     j_indices = torch.from_numpy(j_indices.flatten()).long()
#         #     selected_mean_list, residual_mean_list = [], []
#         #     for batches in tqdm(test_dataloader, desc=f"Inferring {network}", leave=True):
#         #         func_dict=batches.func
#         #         for key, tensor in func_dict.items():
#         #             mask = torch.zeros((tensor.shape[1],tensor.shape[1]), dtype=torch.bool, device=tensor.device)
#         #             mask[i_indices, j_indices] = True
#         #             mask = mask.unsqueeze(0).expand(tensor.shape[0], -1, -1)
#         #             selected_mean = tensor[mask].mean().item()
#         #             residual_mean = tensor[~mask].mean().item()
#         #             selected_mean_list.append(selected_mean)
#         #             residual_mean_list.append(residual_mean)
#         #     selected_mean_list  = np.array(selected_mean_list)
#         #     residual_mean_list = np.array(residual_mean_list)
#         #     print(fold, network, selected_mean_list.mean(), selected_mean_list.max(), selected_mean_list.min())
#         #     print(fold, network, residual_mean_list.mean(), residual_mean_list.max(), residual_mean_list.min())

#         # Shapes
#         f_matrices = next(iter(test_dataloader)).func
#         s_matrices = next(iter(test_dataloader)).anat
#         shapes_dict = {"s" : {k:v.shape[1:] for k,v in s_matrices.items()},                   # structural
#                        "f" : {k:v.shape[1]*(v.shape[1]-1)//2 for k, v in f_matrices.items()}} # functional
#         # Model
#         model = MGD4D(info_dict=auxi_info,
#                        shapes_dict=shapes_dict,
#                        embedding_dim=Configs.dataset.latent_embedding_dim,
#                        use_batchnorm=Configs.dataset.use_batchnorm,
#                        use_lgd=Configs.dataset.use_lgd,
#                        use_modal=Configs.dataset.use_modal).to(device)
#         trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         print(f"Fold: {fold}. The number of trainable parametes of {model.__class__.__name__} is {trainable_parameters}.")
#         pth_path = Path(".") / f"{fold}.pth"
#         assert pth_path.exists(), f"{pth_path} does not exist."
#         model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
#         model.to(device)
#         model.eval()

#         # Infer
#         ## to ensure the reliability of the results, both the data and the model are reloaded in each iteration.
#         for network, labels in get_yeo_network_of_brainnetome().items():
#             subjid_list, probability_list, prediction_list, target_list = [],[],[],[]
#             with torch.no_grad():
#                 for batches in tqdm(test_dataloader, desc=f"Inferring {network}", leave=False):
#                     # move to GPU
#                     auxi_info = move_to_device(batches.info, device)
#                     perturbed_func = perturb_functional_connectivity(func_dict=batches.func, labels=labels, target=batches.target)
#                     f_matrices = move_to_device(perturbed_func, device)
#                     s_matrices = move_to_device(batches.anat, device)
#                     target = move_to_device(batches.target, device).long().flatten()
#                     # forward
#                     output =  model(structural_input_dict=s_matrices,
#                                     functional_input_dict=f_matrices,
#                                     information_input_dict=auxi_info)
#                     # add to lists
#                     probability = output["logits"].softmax(dim=-1)
#                     target_list.extend(target.cpu().numpy())
#                     probability_list.extend(probability[:, 1].cpu().numpy()) 
#                     prediction_list.extend(probability.argmax(dim=-1).cpu().numpy())
#                     subjid_list.extend(batches.id)
#             target = np.array(target_list).astype(int).flatten() 
#             probability = np.array(probability_list).astype(float).flatten() 
#             prediction = np.array(prediction_list).astype(int).flatten()
#             assert target.shape == probability.shape == prediction.shape, f"{target.shape} != {probability.shape} != {prediction.shape}"
#             metrics = {
#                 "AUC" : Metrics.AUC(prob=probability, true=target),
#                 "SEN" : Metrics.SEN(pred=prediction, true=target),
#                 "SPE" : Metrics.SPE(pred=prediction, true=target)
#             }

#             for k, v in metrics.items():
#                 all_metrics[network][k].append(float(v))
            
#     average_metrics = defaultdict(dict)
#     for network, metrics in all_metrics.items():
#         for k, v in metrics.items():
#             average_metrics[network][k] = sum(v) / len(v)

#     with Path("XAI.json").open("w") as f:
#         json.dump({"average" : average_metrics, "raw" : all_metrics}, f, indent=4)
#     print("XAI Done!")

# if __name__ == "__main__":
#     main()







# import umap
# import pacmap
# import trimap
# import warnings
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from typing import Any
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import MinMaxScaler

# from path import Paths
# from config import seed

# warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to", module="sklearn.utils.deprecation")
# warnings.filterwarnings('ignore', message="gradient function is not yet implemented for .* distance metric")

# def plot_points(point_bf : np.ndarray, point_af : np.ndarray, target : np.ndarray, name : Any) -> None:
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     color_map = {0 : "blue",     1 : "red"}
#     label_map = {0 : "Negative", 1 : "Positive"}
#     for t in [0, 1]:
#         idx = (target == t)
#         axes[0].scatter(point_bf[idx, 0], point_bf[idx, 1], 
#                         c=color_map[t], label=label_map[t], alpha=0.7, edgecolors="k", s=30)
#         axes[1].scatter(point_af[idx, 0], point_af[idx, 1], 
#                         c=color_map[t], label=label_map[t], alpha=0.7, edgecolors="k", s=30)
#     axes[0].set_title("Embedding Before LGD")
#     axes[1].set_title("Embedding After  LGD")
#     # axes[0].set_xlabel("UMAP-1")
#     # axes[0].set_ylabel("UMAP-2")
#     # axes[1].set_xlabel("UMAP-1")
#     # axes[1].set_ylabel("UMAP-2")
#     for ax in axes:
#         ax.legend()
#         ax.grid(True, linestyle='--', alpha=0.3)
#     plt.savefig(f"{name}.svg", dpi=600)
#     plt.close(fig)


# def main() -> None:
#     embedding_dir_path = Paths.Run_Files.embedding_dir_path
#     assert embedding_dir_path.exists(), f"{embedding_dir_path} does not exist"
#     bf_embedding_list, af_embedding_list, target_list = [], [], []
#     for npz_path in embedding_dir_path.iterdir():
#         with np.load(npz_path, allow_pickle=True) as data:
#             sub_id = npz_path.stem
#             bf_lgd = data["before"][1024:]
#             af_lgd = data["after"][1024:]
#             target = data["target"]
#             bf_embedding_list.append(bf_lgd)
#             af_embedding_list.append(af_lgd)
#             target_list.append(target)
#     bf_embedding = MinMaxScaler().fit_transform(bf_embedding_list)
#     af_embedding = MinMaxScaler().fit_transform(af_embedding_list)
#     target = np.array(target_list).astype(int).flatten()
    
#     # UMAP 2D
#     metric_list = [
#         "braycurtis","canberra","chebyshev","correlation","cosine","dice","euclidean","hamming","haversine","jaccard","kulsinski",
#         "mahalanobis","manhattan","minkowski","rogerstanimoto","russellrao","seuclidean","sokalmichener","sokalsneath"
#     ]
       
#     for metric in tqdm(metric_list, desc="UMAP 2D", leave=True):
#         if metric in ["haversine", "mahalanobis", "seuclidean"]:
#             continue
#         # n_neighbors = 50 len(target)-1
#         mapper = umap.UMAP(n_neighbors=len(target)-1, n_components=2, min_dist=0.1, metric=metric, n_jobs=-1)
#         # bf_umap_2d = mapper.fit_transform(bf_embedding)
#         # af_umap_2d = mapper.fit_transform(af_embedding)

#         mapper.fit(bf_embedding)
#         bf_umap_2d = mapper.transform(bf_embedding)
#         af_umap_2d = mapper.transform(af_embedding)

#         bf_umap_2d = MinMaxScaler().fit_transform(bf_umap_2d)
#         af_umap_2d = MinMaxScaler().fit_transform(af_umap_2d)

#         # Plot
#         plot_points(bf_umap_2d, af_umap_2d, target, name=f"umap_{metric}")

#     # pacmap
#     reducer = pacmap.PaCMAP(n_components=2)
#     bf_pacmap_2d = reducer.fit_transform(bf_embedding)
#     af_pacmap_2d = reducer.fit_transform(af_embedding)
#     plot_points(bf_pacmap_2d, af_pacmap_2d, target, name="pacmap")

#     # trimap
#     reducer = trimap.TRIMAP(n_dims=2)
#     bf_trimap_2d = reducer.fit_transform(bf_embedding)
#     af_trimap_2d = reducer.fit_transform(af_embedding)
#     plot_points(bf_trimap_2d, af_trimap_2d, target, name="trimap")

#     # t-SNE
#     for lr in tqdm(range(10, 110, 10)):
#         reducer = TSNE(n_components=2, perplexity=50, learning_rate=lr, max_iter=3000, init="pca")
#         bf_tsne_2d = reducer.fit_transform(bf_embedding)
#         af_tsne_2d = reducer.fit_transform(af_embedding)
#         plot_points(bf_tsne_2d, af_tsne_2d, target, name=lr)
        
# if __name__ == "__main__":
#     main()





# import torch
# from pathlib import Path

# from config import Configs

# for fold in Configs.n_splits:
#     pth_path = Path(f"./{fold}.pth")
#     assert pth_path.exists(), f"{pth_path} does not exist"
#     state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
#     layer_name_list = [
#         "encoder_f.projectors.func.0.weight",
#         "encoder_f.projectors.func.0.bias",
#         "encoder_f.projectors.func.3.weight",
#         "encoder_f.projectors.func.3.bias",
#         "encoder_f.projectors.func.6.weight",
#         "encoder_f.projectors.func.6.bias",
#     ]
#     for layer_name in layer_name_list:
#         print(layer_name, state_dict[layer_name].shape)
#     exit()