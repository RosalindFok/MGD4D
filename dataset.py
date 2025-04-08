import json
import torch
import random
import numpy as np
import nibabel as nib
import torch.multiprocessing
from tqdm import tqdm
from typing import Any # tuple, list, dict is correct for Python>=3.9
from pathlib import Path
from scipy.io import loadmat  
from functools import partial
from dataclasses import dataclass
from sklearn.model_selection import KFold
from collections import namedtuple, defaultdict 
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_sharing_strategy("file_system") # to solve the "RuntimeError: unable to open shared memory object </torch_54907_2465325546_996> in read-write mode: Too many open files (24)"

from path import Paths
from plot import draw_atlas
from config import IS_MD, Train_Config, Brain_Network, seed

random.seed(seed)

@dataclass
class Return_Dataloaders:
    train: DataLoader
    test : DataLoader
    info : dict[str, int]
    weight : torch.Tensor

def load_atlas() -> dict[str, list[str]]:
        """
        AAL: Automated Anatomical Labeling, 116=90+26
        HOC: Harvard Oxford Cortical
        HOS: Harvard Oxford Subcortical
        """
        # download DPABI from https://d.rnet.co/DPABI/DPABI_V8.2_240510.zip
        # Unzip the zip file, extract some files from the folder "Templates" and put them under downloaded_atlas_dir_path
        aal_mri = Paths.Atlas.AAL.mri_path
        aal_labels = Paths.Atlas.AAL.labels_path
        hoc_mri = Paths.Atlas.HarvardOxford.cort_mri_path
        hoc_labels = Paths.Atlas.HarvardOxford.cort_labels_path
        hos_mri = Paths.Atlas.HarvardOxford.sub_mri_path
        hos_labels = Paths.Atlas.HarvardOxford.sub_labels_path
        
        atlas_labels_dict = {}
        for name, mri_path, labels_path in zip(["AAL", "HOC", "HOS"], [aal_mri, hoc_mri, hos_mri], [aal_labels, hoc_labels, hos_labels]):
            labels = loadmat(labels_path)["Reference"]
            labels = [str(label[0][0]) for label in labels]
            # the first one is None, which is not used in REST-meta-MDD
            assert labels[0] == "None", f"First label is not None: {labels[0]}"
            labels = labels[1:]
            atlas_labels_dict[name] = {index:label for index, label in enumerate(labels)}
            # plot atlas
            fig_path = Paths.Fig_Dir / f"{name}.png"
            if not fig_path.exists():
                draw_atlas(atlas=nib.load(mri_path), saved_path=fig_path)
        return atlas_labels_dict

to_tensor = partial(torch.tensor, dtype=torch.float32)

# class Dataset_Mild_Depression(Dataset):
#     def __init__(self, md_path_list : list[Path], hc_path_list : list[Path]) -> None:
#         super().__init__()
#         self.path_list = [] # [(dir_path, target), ...]; target, 0: health controls, 1: mild depression
#         max_len = max([len(md_path_list), len(hc_path_list)])
#         for i in range(max_len):
#             if i < len(md_path_list):
#                 self.path_list.append((md_path_list[i], IS_MD.IS))
#             if i < len(hc_path_list):
#                 self.path_list.append((hc_path_list[i], IS_MD.NO))
    
#     def __get_the_only_file_in_dir__(self, dir_path : Path, substr : str) -> Path:
#         path_list = [x for x in dir_path.glob(f"*{substr}*")]
#         assert len(list(path_list)) == 1, f"Multiple or no {substr} file found in {dir_path}"
#         return path_list[0]
    
#     def __getitem__(self, idx) -> tuple[dict, int]:
#         dir_path, target = self.path_list[idx]

#         # denoised_anat
#         denoised_anat_path = self.__get_the_only_file_in_dir__(dir_path, "denoised_anat.nii.gz")

#         # functional connectivity matrix
#         fc_matrix_path = self.__get_the_only_file_in_dir__(dir_path, "fc_matrix.npy")
#         fc_matrix = np.load(fc_matrix_path)

#         # information
#         info_json_path = self.__get_the_only_file_in_dir__(dir_path, "info.json")
        
#         with info_json_path.open("r") as f:
#             info = json.load(f)
#             age = info["age"]
#             gender = info["gender"]

#     def __len__(self):
#         return len(self.path_list)

# class KFold_Mild_Depression:
#     def __init__(self, ds002748_dir_path : Path = Paths.Run_Files.run_files_ds002748_dir_path,
#                        ds003007_dir_path : Path = Paths.Run_Files.run_files_ds003007_dir_path,
#                        n_splits : int = 5, shuffle : bool = False):
#         self.md, md_index = {}, 0 # index : subj_path. 51(ds002748) + 29(ds003007) = 80 subjects
#         self.hc, hc_index = {}, 0 # index : subj_path. + = subjects

#         # ds002748: 72 subjects = 51 subjects with mild depression + 21 healthy controls
#         for dir_path in ds002748_dir_path.iterdir():
#             # information json
#             group = self.__get_group_in_infojson__(dir_path)
#             if group == "depr":
#                 self.md[md_index] = dir_path
#                 md_index += 1
#             elif group == "control":
#                 self.hc[hc_index] = dir_path
#                 hc_index += 1
#             else:
#                 raise ValueError(f"Unknown group: {group}")

#         # ds003007: pre, 29 subjects with mild depression; post, 15 depr_no_treatment + 8 depr_cbt + 6 depr_nfb
#         self.no_treatment, self.cbt, self.nfb = {}, {}, {}
#         nt_index, cbt_index, nfb_index = 0, 0, 0
#         for index, dir_path in enumerate((ds003007_dir_path / "pre").iterdir()):
#             self.md[md_index+index] = dir_path
#         for dir_path in (ds003007_dir_path / "post").iterdir():
#             group = self.__get_group_in_infojson__(dir_path)
#             if group == "depr_no_treatment":
#                 self.no_treatment[nt_index] = dir_path
#                 nt_index += 1
#             elif group == "depr_cbt":
#                 self.cbt[cbt_index] = dir_path
#                 cbt_index += 1
#             elif group == "depr_nfb":
#                 self.nfb[nfb_index] = dir_path
#                 nfb_index += 1
#             else:
#                 raise ValueError(f"Unknown group: {group}")

#         # K Fold
#         self.kf = KFold(n_splits=n_splits, shuffle=shuffle)
               
#     def __get_group_in_infojson__(self, dir_path : Path) -> str:
#         info_path = [x for x in dir_path.glob("*.json")]
#         assert len(list(info_path)) == 1, f"Multiple or no info file found in {dir_path}"
#         info_path = info_path[0]
#         with info_path.open("r") as f:
#             info = json.load(f)
#             group = info["group"]
#         return group
    
#     def __k_fold__(self, group_name : str) -> dict[int, dict[str, list[Path]]]:
#         group_mapping = {"md": self.md, "hc": self.hc}  
#         group = group_mapping.get(group_name)  
#         if group is None:  
#             raise ValueError(f"Unknown group: {group_name}")

#         kfold_dir_paths = {} # {fold : {train_set, test_set}}
#         fold = 1
#         for train_index, test_index in self.kf.split(group):
#             train_set_dir_path_list = [group[i] for i in train_index]
#             test_set_dir_path_list  = [group[i] for i in test_index]
#             kfold_dir_paths[fold] = {"train" : train_set_dir_path_list, "test" : test_set_dir_path_list}
#             fold += 1
#         return kfold_dir_paths

#     def get_dataloader_via_fold(self, fold : int) -> DataLoader:
#         kfold_md_dir_paths = self.__k_fold__(group_name="md")
#         kfold_hc_dir_paths = self.__k_fold__(group_name="hc")
#         assert fold in kfold_md_dir_paths.keys(), f"Unknown fold: {fold}, available folds: {kfold_md_dir_paths.keys()}"
#         Dataset_Mild_Depression(md_path_list=kfold_md_dir_paths[fold]["train"], hc_path_list=kfold_hc_dir_paths[fold]["train"])


# dataclass is not used because dict[tensor] is not supported by PyTorch
MDD_Returns = namedtuple("MDD_Returns", ["id", "info", "fc", "vbm", "target"]) 

class Dataset_Major_Depression(Dataset):
    def __init__(self, path_list : list[dict[str, Any]], 
                       brain_network_name : str) -> None:
        super().__init__()
        self.path_list = path_list
        self.brain_network_name = brain_network_name
    
    def __subgraph__(self, input_dict : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        only for fMRI
        whole graph = whole brain = whole matrix, subgraph = a brain network = sub matrix
        """
        output_dict = {}
        if self.brain_network_name == Brain_Network.DMN:
            pass
        elif self.brain_network_name == Brain_Network.CEN:
            pass
        elif self.brain_network_name == Brain_Network.SN:
            pass
        elif self.brain_network_name is Brain_Network.whole:
            output_dict = input_dict
        else:
            raise ValueError(f"Unknown brain network name: {self.brain_network_name}")
        return output_dict
        
    def __getitem__(self, index) -> tuple[str, dict[str, torch.Tensor], dict[str, torch.Tensor], 
                                          dict[str, torch.Tensor], int]:
        path_dict = self.path_list[index]
        # Auxiliary information
        auxi_info = path_dict["auxi_info"]
        ID = auxi_info["ID"]
        target = IS_MD.IS if "-1-" in ID else IS_MD.NO if "-2-" in ID else None
        assert target == int(auxi_info["depression"])
        auxi_info = {k : to_tensor(v).int() for k, v in auxi_info.items() if k not in ["ID", "depression"]}

        # Functional connectivity
        fc_matrices = {}
        for atlas_name, matrix in np.load(path_dict["fc"], allow_pickle=True).items():
            # AAL shape: (116, 116)
            # HOC shape: (96, 96)
            # HOS shape: (16, 16)
            matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
            fc_matrices[atlas_name] = to_tensor(matrix)
            del matrix

        # VBM
        vbm_matrices = {}
        for group_name, matrix in np.load(path_dict["vbm"], allow_pickle=True).items():
            # matrix shape: (121, 145, 121)
            matrix = np.clip(matrix, 0, 1)
            vbm_matrices[group_name] = to_tensor(matrix)
            del matrix

        # target
        target = to_tensor(target)

        return MDD_Returns(
            id=ID, info=auxi_info, fc=fc_matrices, 
            vbm=vbm_matrices, target=target
        )

    def __len__(self) -> int:
        return len(self.path_list)

class KFold_Major_Depression:
    def __init__(self, is_global_signal_regression : bool, rest_meta_mdd_dir_path : Path = Paths.Run_Files.run_files_rest_meta_mdd_dir_path) -> None:
        self.rest_meta_mdd_dir_path = rest_meta_mdd_dir_path
        dir_name = "ROISignals_FunImgARglobalCWF" if is_global_signal_regression else "ROISignals_FunImgARCWF"
        
        # Auxiliary information
        auxi_info_path = rest_meta_mdd_dir_path / "participants_info.json"
        assert auxi_info_path.exists(), f"Participants info file not found in {rest_meta_mdd_dir_path}"
        # Auxiliary Information: Sex, Age, Education (years)
        # sex: there 3 types in REST-meta-MDD dataset, most of them are female/male, one subject is unspecified
        # age: 0 is unknown, min=12, max=82
        # education years: 0 is unknown, min=3, max=23
        with auxi_info_path.open("r") as f:
            all_info = json.load(f) # {sub_id: {key: value, ...}, ...}
        auxi_info = defaultdict(dict)
        self.max_value = {}
        for sub_id, info_dict in all_info.items():
            for attribute in ["Sex", "Age", "Education (years)", "ID", "depression"]:
                assert attribute in info_dict.keys(), f"Attribute {attribute} not found in auxiliary information"
                value = info_dict[attribute]
                auxi_info[sub_id][attribute] = value
                if attribute in ["Sex", "Age", "Education (years)"]:
                    if attribute not in self.max_value or value > self.max_value[attribute]:  
                        self.max_value[attribute] = int(value)
        # site: 1~25, except 4
        max_val = 0
        for sub_id, info_dict in auxi_info.items():
            site = int(info_dict["ID"].split("-")[0][1:])
            auxi_info[sub_id]["Site"] = site
            max_val = max(max_val, site)
        self.max_value["Site"] = max_val

        # Functional connectivity
        fc_path_dict = {} 
        for sub_dir_path in tqdm(list((rest_meta_mdd_dir_path / dir_name).iterdir()), desc=f"Loading {dir_name}", leave=False):
            fc_matrix_path = [x for x in sub_dir_path.glob("fc_matrix.npz")]
            fc_path_dict[sub_dir_path.name] = fc_matrix_path[0]

        # VBM
        vbm_path_dict = {}
        for path in list((rest_meta_mdd_dir_path / "VBM").iterdir()):
            if not "_augmented" in path.name:
                vbm_path_dict[path.stem.split(".")[0]] = path
        
        assert set(auxi_info.keys()) == set(fc_path_dict.keys()) == set(vbm_path_dict.keys()), f"Keys mismatch!" 
        
        self.paths_dict = defaultdict(dict)
        for sub_id in auxi_info.keys():
            self.paths_dict[sub_id]["auxi_info"] = auxi_info[sub_id]
            self.paths_dict[sub_id]["fc"] = fc_path_dict[sub_id]
            self.paths_dict[sub_id]["vbm"] = vbm_path_dict[sub_id]

        # K Fold
        self.kf = KFold(n_splits=Train_Config.n_splits.stop-1, shuffle=Train_Config.shuffle)
        self.kfold_dir_paths = self.__k_fold__()
        
    def __data_augmentation__(self, old_list : list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        new_dict = {}
        for path_dict in tqdm(old_list, desc="Data augmentation", leave=False):
            # functional connectivity: add noise 
            augmented_fc_path = path_dict["fc"].parent / f"{path_dict['fc'].stem}_augmented.npz"
            if not augmented_fc_path.exists():
                fc_matrix = np.load(path_dict["fc"], allow_pickle=True)
                augmented_fc_dict = {}
                for key,value in fc_matrix.items():
                    noise = np.random.normal(0, 0.01, value.shape)
                    noise = (noise + noise.T) / 2
                    augmented_fc_dict[key] = value + noise
                np.savez(augmented_fc_path, **augmented_fc_dict)
            # VBM: add noise
            augmented_vbm_path = path_dict["vbm"].parent / f"{path_dict['vbm'].stem}_augmented.npz"
            if not augmented_vbm_path.exists():
                vbm_matrix = np.load(path_dict["vbm"], allow_pickle=True)
                augmented_vbm_dict = {}
                for key,value in vbm_matrix.items():
                    noise = np.random.normal(0, 0.01, value.shape)
                    augmented_vbm_dict[key] = value + noise
                    assert value.shape == augmented_vbm_dict[key].shape, f"Shape mismatch: {value.shape}!= {augmented_vbm_dict[key].shape}"
                np.savez(augmented_vbm_path, **augmented_vbm_dict)
            # new path dict
            new_subj_id = path_dict["auxi_info"]["ID"] + "_augmented"
            new_auxi_info = path_dict["auxi_info"].copy()
            new_auxi_info["ID"] = new_subj_id
            new_dict[new_subj_id] = {"auxi_info" : new_auxi_info,
                                     "fc"        : augmented_fc_path,
                                     "vbm"       : augmented_vbm_path}
        assert len(new_dict) == len(old_list), f"Length mismatch: {len(new_dict)}!= {len(old_list)}"
        return new_dict

    def __k_fold__(self) -> dict[int, dict[str, list[dict[str, Any]]]]:
        kfold_dir_paths = defaultdict(dict)
        # REST-meta-MDD: "SiteID-target-SubjectID"
        subj_list = list(self.paths_dict.keys())

        # each sites
        site_dict = defaultdict(lambda: defaultdict(list))
        for subj in subj_list:
            split = subj.split("-")
            site_dict[split[0]][split[1]].append(subj)
        
        # data augmentation: balance positive/negative samples and avoid ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, *])
        augmented_ratio = []
        for site, pn_samples in site_dict.items(): # pn: positive/negative
            shortest_key = min(pn_samples, key=lambda k: len(pn_samples[k])) 
            longest_key  = max(pn_samples, key=lambda k: len(pn_samples[k])) 
            if not len(pn_samples[shortest_key]) == len(pn_samples[longest_key]): # imbalance
                difference = min(len(pn_samples[longest_key]) - len(pn_samples[shortest_key]), len(pn_samples[shortest_key]))
                # randomly select samples from shortest list
                selected_list = random.sample([x for x in pn_samples[shortest_key]], difference)
                augmented_ratio.append(len(selected_list) / (len(pn_samples[longest_key]) + len(pn_samples[shortest_key])))
                selected_dict = self.__data_augmentation__(old_list=[self.paths_dict[x] for x in selected_list])
                # update the new samples
                site_dict[site][shortest_key].extend(selected_list)
                self.paths_dict.update(selected_dict)
            else: # balance 
                augmented_ratio.append(0)
        augmented_ratio = sum(augmented_ratio) / len(augmented_ratio)
        print(f"Augmented samples size accounts for {augmented_ratio*100:.2f}% of the original samples") # 8.80%

        # k-fold
        train_dict, test_dict = defaultdict(list), defaultdict(list)
        for site, pn_samples in site_dict.items(): # pn: positive/negative
            for key, value in pn_samples.items():
                fold = 1
                for train_index, test_index in self.kf.split(value):
                    train_dict[fold].extend(value[i] for i in train_index)
                    test_dict[fold].extend(value[i] for i in test_index)
                    fold += 1
        
        # shuffle and split
        self.weights = {}
        for fold in Train_Config.n_splits:
            for tag, sample_dict in zip(["train", "test"], [train_dict, test_dict]):
                positive_samples = [x for x in sample_dict[fold] if "-1-" in x]
                negative_samples = [x for x in sample_dict[fold] if "-2-" in x]
                random.shuffle(positive_samples)
                random.shuffle(negative_samples)
                min_len = min(len(positive_samples), len(negative_samples))
                merged_list = []
                for i in range(min_len):
                    merged_list.append(positive_samples[i])
                    merged_list.append(negative_samples[i])
                longer_list = positive_samples if len(positive_samples) > len(negative_samples) else negative_samples  
                extra_elements = longer_list[min_len:] 
                for extra in extra_elements:
                    insert_pos = random.randint(0, len(merged_list)) 
                    merged_list.insert(insert_pos, extra) 
                kfold_dir_paths[fold][tag] = [self.paths_dict[x] for x in merged_list]
                if tag == "train": # for CrossEntropyLoss
                    self.weights[fold] = [len(negative_samples) / (len(positive_samples) + len(negative_samples)), 
                                          len(positive_samples) / (len(positive_samples) + len(negative_samples))]
        return kfold_dir_paths

    def get_dataloader_via_fold(self, fold : int, batch_size : int = Train_Config.batch_size) -> Return_Dataloaders:
        train_dataset = Dataset_Major_Depression(path_list=self.kfold_dir_paths[fold]["train"], brain_network_name=Brain_Network.whole)
        test_dataset  = Dataset_Major_Depression(path_list=self.kfold_dir_paths[fold]["test"] , brain_network_name=Brain_Network.whole)
        # set persistent_workers=True to avoid OOM
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=Train_Config.num_workers, shuffle=Train_Config.shuffle, pin_memory=True, persistent_workers=True)
        test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, num_workers=Train_Config.num_workers, shuffle=Train_Config.shuffle, pin_memory=True, persistent_workers=True)
        return_dataloaders = Return_Dataloaders(train=train_dataloader, test=test_dataloader, 
                                                info=self.max_value, weight=to_tensor(self.weights[fold]))
        return return_dataloaders

# global_signal is better than not    
get_major_dataloader_via_fold = KFold_Major_Depression(is_global_signal_regression=True).get_dataloader_via_fold
