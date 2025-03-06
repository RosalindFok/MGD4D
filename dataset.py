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
from collections import defaultdict
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_sharing_strategy("file_system") # to solve the "RuntimeError: unable to open shared memory object </torch_54907_2465325546_996> in read-write mode: Too many open files (24)"

from path import Paths
from plot import draw_atlas
from config import IS_MD, Train_Config, Gender, Brain_Network, seed

random.seed(seed)

@dataclass
class Return_Dataloaders:
    train: DataLoader
    test : DataLoader

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

# class Dataset_Mild_Depression(Dataset):
#     def __init__(self, md_path_list : list[Path], hc_path_list : list[Path]) -> None:
#         super().__init__()
#         self.path_list = [] # [(dir_path, tag), ...]; tag, 0: health controls, 1: mild depression
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
#         dir_path, tag = self.path_list[idx]

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

class Dataset_Major_Depression(Dataset):
    def __init__(self, path_list : list[dict[str, Any]], 
                       auxi_values : dict[str, dict[str, float]],
                       brain_network_name : str) -> None:
        super().__init__()
        self.path_list = path_list
        self.anxi_values = auxi_values
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
        
    def __getitem__(self, index) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], 
                                          dict[str, torch.Tensor], int]:
        path_dict = self.path_list[index]
        # Auxiliary information
        auxi_info = path_dict["auxi_info"]
        ID = auxi_info["ID"]
        tag = IS_MD.IS if "-1-" in ID else IS_MD.NO if "-2-" in ID else None
        assert tag is not None, f"Unknown tag in ID: {ID}"
        to_tensor = partial(torch.tensor, dtype=torch.float32)
        processed_auxi_info = {}
        for attribute in ["Sex", "Age", "Education (years)"]:
            assert attribute in auxi_info.keys(), f"Attribute {attribute} not found in auxiliary information"
            if attribute == "Sex":
                value = Gender.FEMALE if auxi_info["Sex"] == 2 else Gender.MALE if auxi_info["Sex"] == 1 else Gender.UNSPECIFIED
                value /= max(Gender.FEMALE, Gender.MALE, Gender.UNSPECIFIED)
            else:
                value = auxi_info[attribute]
                assert attribute in self.anxi_values.keys(), f"Attribute {attribute} not found in self.anxi_values"
                # value = self.anxi_values[attribute]["mean"] if value == 0 else value
                value /= self.anxi_values[attribute]["max"]
            processed_auxi_info[attribute] = to_tensor([value])
        processed_auxi_info = self.__subgraph__(input_dict=processed_auxi_info)

        # Functional connectivity
        fc_matrices = {}
        for atlas_name, matrix in np.load(path_dict["fc"], allow_pickle=True).items():
            assert np.allclose(matrix, matrix.T), f"{atlas_name} matrix {matrix} is not symmetric"
            # functional connectivity has been restricted to [-1, 1]
            # AAL shape: (116, 116)
            # HOC shape: (96, 96)
            # HOS shape: (16, 16)
            fc_matrices[atlas_name] = to_tensor(matrix)

        # VBM
        vbm_matrices = {}
        for group_name, matrix in np.load(path_dict["vbm"], allow_pickle=True).items():
            # matrix shape: (121, 145, 121)
            matrix = np.clip(matrix, 0, 1)
            vbm_matrices[group_name] = to_tensor(matrix)

        # tag
        tag = to_tensor(tag)

        return processed_auxi_info, fc_matrices, vbm_matrices, tag

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
            auxi_info = json.load(f) # {sub_id: {key: value, ...}, ...}
            age_array = np.array([info_dict["Age"] for _, info_dict in auxi_info.items()])
            education_years_array = np.array([info_dict["Education (years)"] for _, info_dict in auxi_info.items()])
            age_array = age_array[age_array > 0]
            education_years_array = education_years_array[education_years_array > 0]
            self.auxi_values = {"Age" : {"mean": age_array.mean(), "max": age_array.max(), "min": age_array.min()},
                                "Education (years)" : {"mean": education_years_array.mean(), "max": education_years_array.max(), "min": education_years_array.min()}}

        # Functional connectivity
        fc_path_dict = {} 
        for sub_dir_path in tqdm(list((rest_meta_mdd_dir_path / dir_name).iterdir()), desc=f"Loading {dir_name}", leave=True):
            fc_matrix_path = [x for x in sub_dir_path.glob("*.npz")]
            assert len(fc_matrix_path) == 1, f"Multiple or no fc matrix found in {sub_dir_path}"
            fc_path_dict[sub_dir_path.name] = fc_matrix_path[0]

        # VBM
        vbm_path_dict = {path.stem.split(".")[0] : path for path in list((rest_meta_mdd_dir_path / "VBM").iterdir())}
        
        assert len(auxi_info)==len(fc_path_dict)==len(vbm_path_dict), f"Length mismatch: {len(auxi_info)} != {len(fc_path_dict)} != {len(vbm_path_dict)}"
        
        self.paths_dict = {}
        for key, value in auxi_info.items():
            assert key not in self.paths_dict.keys(), f"Duplicate key: {key}"
            self.paths_dict[key] = {"auxi_info" : value}
            assert key in fc_path_dict.keys(), f"Unknown key: {key}"
            self.paths_dict[key]["fc"] = fc_path_dict[key]
            assert key in vbm_path_dict.keys(), f"Unknown key: {key}"
            self.paths_dict[key]["vbm"] = vbm_path_dict[key]

        # K Fold
        self.kf = KFold(n_splits=Train_Config.n_splits.stop-1, shuffle=Train_Config.shuffle)

    def __k_fold__(self) -> dict[int, dict[str, list[dict[str, Any]]]]:
        kfold_dir_paths = {} 
        # REST-meta-MDD: "SiteID-Tag-SubjectID"
        subj_list = list(self.paths_dict.keys())
        random.shuffle(subj_list)
        fold = 1
        for train_index, test_index in self.kf.split(subj_list):
            kfold_dir_paths[fold] = {"train" : [self.paths_dict[subj_list[i]] for i in train_index], 
                                     "test"  : [self.paths_dict[subj_list[i]] for i in test_index]}
            fold += 1
        return kfold_dir_paths

    def get_dataloader_via_fold(self, fold : int, batch_size : int = Train_Config.batch_size) -> tuple[DataLoader, DataLoader]:
        kfold_dir_paths = self.__k_fold__()
        train_dataset = Dataset_Major_Depression(path_list=kfold_dir_paths[fold]["train"], auxi_values=self.auxi_values, brain_network_name=Brain_Network.whole)
        test_dataset  = Dataset_Major_Depression(path_list=kfold_dir_paths[fold]["test"] , auxi_values=self.auxi_values, brain_network_name=Brain_Network.whole)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=Train_Config.num_workers, shuffle=False)
        test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, num_workers=Train_Config.num_workers, shuffle=False)
        return_dataloaders = Return_Dataloaders(train=train_dataloader, test=test_dataloader)
        return return_dataloaders

# global is better than not    
get_major_dataloader_via_fold = KFold_Major_Depression(is_global_signal_regression=True).get_dataloader_via_fold
