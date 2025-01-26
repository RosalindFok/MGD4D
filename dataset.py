import json
import torch
import random
import numpy as np
import nibabel as nib
from tqdm import tqdm
from typing import Any # tuple, list, dict is correct for Python>=3.9
from pathlib import Path
from nilearn import datasets
from dataclasses import dataclass
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

from path import Paths
from plot import draw_atlas
from config import IS_MD, Train_Config

random.seed(66)

@dataclass
class Return_Dataloaders:
    train: DataLoader
    test : DataLoader

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
    def __init__(self, md_path_list : list[tuple[dict[str, Path], int]]) -> None:
        super().__init__()
        self.path_list = md_path_list + hc_path_list
        random.shuffle(self.path_list)
    
    def __getitem__(self, index) -> tuple:
        path_dict, is_md = self.path_list[index]
        sub_id = path_dict.pop("sub_id")
        info_path = path_dict.pop("info")
        with info_path.open("r") as f:
            info = json.load(f)
            auxi_info = [info["Sex"]]
            for key, value in self.auxi_info_max.items():
                auxi_info.append(info[key] / value)
        auxi_info = torch.tensor(auxi_info, dtype=torch.float32) # dim=3
        matrix = np.load(path_dict[self.atlas_name])
        matrix = torch.tensor(matrix, dtype=torch.float32) # dim=[96, 96]
        return auxi_info, matrix, is_md            

    def __len__(self) -> int:
        return len(self.path_list)

class KFold_Major_Depression:
    def __init__(self, is_global_signal_regression : bool, rest_meta_mdd_dir_path : Path = Paths.Run_Files.run_files_rest_meta_mdd_dir_path,
                train_config:Train_Config = Train_Config) -> None:
        self.train_config = train_config
        self.rest_meta_mdd_dir_path = rest_meta_mdd_dir_path
        dir_name = "ROISignals_FunImgARglobalCWF" if is_global_signal_regression else "ROISignals_FunImgARCWF"
        
        self.atlas_labels_dict = self.__download_atlas__()

        # Auxiliary information
        auxi_info_path = rest_meta_mdd_dir_path / "participants_info.json"
        assert auxi_info_path.exists(), f"Participants info file not found in {rest_meta_mdd_dir_path}"
        # Auxiliary Information: Sex, Age, Education (years)
        # sex: there 3 types in REST-meta-MDD dataset, most of them are female/male, one subject is unspecified
        # age: 0 is unknown, min=12, max=82
        # education years: 0 is unknown, min=3, max=23
        with auxi_info_path.open("r") as f:
            auxi_info = json.load(f) # {sub_id: {key: value, ...}, ...}

        # Functional connectivity
        fc_path_dict = {} # {sub_id: {atlas_name: fc_matrix_path, ...}, ...}
        for sub_dir_path in tqdm(list((rest_meta_mdd_dir_path / dir_name).iterdir()), desc=f"Loading {dir_name}", leave=True):
            path_dict = {} # {sub_id:info_path, atalas_name:fc_matrix_path, ...}
            for fc_matrix_path in sub_dir_path.glob("*.npy"):
                atlas_name = fc_matrix_path.stem.split("_")[0]
                path_dict[atlas_name] = fc_matrix_path
            fc_path_dict[sub_dir_path.name] = path_dict
        
        # VBM
        vbm_path_dict = {key:{} for key in fc_path_dict.keys()} # {sub_id: {group_name: vbm_matrix_path, ...}, ...}
        for group_name in ["wc1", "wc2", "mwc1", "mwc2"]:
            for nii_path in tqdm(list((rest_meta_mdd_dir_path / group_name).iterdir()), desc=f"Loading {group_name}", leave=True):
                key = nii_path.stem.split(".")[0]
                assert key in vbm_path_dict.keys(), f"Unknown key: {key}"
                vbm_path_dict[key][group_name] = nii_path
        
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
        self.kf = KFold(n_splits=max(train_config.n_splits), shuffle=train_config.shuffle)

    def __download_atlas__(self) -> dict[Any, list[str]]:
        """
        Download atlas from nilearn
        """
        downloaded_atlas_dir_path = self.rest_meta_mdd_dir_path / "downloaded_atlas"
        method_pair = { # {name of atlas : {atlas, coresponding matrix}}
            # https://nilearn.github.io/stable/modules/description/aal.html
            "AAL" : datasets.fetch_atlas_aal(data_dir=downloaded_atlas_dir_path),
            # https://nilearn.github.io/stable/modules/description/harvard_oxford.html
            "HOC" :  datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr50-1mm", data_dir=downloaded_atlas_dir_path),
            # https://nilearn.github.io/stable/modules/description/harvard_oxford.html
            "HOS" :  datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr50-1mm", data_dir=downloaded_atlas_dir_path),
            # https://nilearn.github.io/stable/modules/description/craddock_2012.html
            # Note: There are no labels for the Craddock atlas, which have been generated from applying spatially constrained clustering on resting-state data.
            # "Craddock" : datasets.fetch_atlas_craddock_2012(data_dir=downloaded_atlas_dir_path),
        }
        atlas_labels_dict = {}
        for key, value in method_pair.items():
            # type of AAL atlas is str, path to nifti file containing the regions.
            # type of Harvard-Oxford atlas is nibabel image.
            atlas, labels = value.maps,value.labels
            atlas_labels_dict[key] = {"atlas" : atlas, "labels" : labels}
            # plot atlas
            fig_path = Paths.Fig_Dir / f"{key}.png"
            if not fig_path.exists():
                if type(atlas) == str: # aal
                    draw_atlas(atlas=nib.load(atlas), saved_path=fig_path)
                elif type(atlas) == nib.nifti1.Nifti1Image: # harvard-oxford
                    draw_atlas(atlas=atlas, saved_path=fig_path)
                else:
                    raise TypeError(f"Unknown type of atlas: {type(atlas)}")
        return atlas_labels_dict

    def __k_fold__(self) -> dict[str, dict[int, dict[str, list[dict[str, Any]]]]]:
        kfold_dir_paths = {} # {{group_name : {fold : {train_set, test_set}}}, ...}
        subj_list = list(self.paths_dict.keys())
        fold = 1
        for train_index, test_index in self.kf.split(subj_list):
            kfold_dir_paths[fold] = {"train" : [self.paths_dict[subj_list[i]] for i in train_index], 
                                     "test"  : [self.paths_dict[subj_list[i]] for i in test_index]}
            fold += 1
        return kfold_dir_paths

    def get_dataloader_via_fold(self, fold : int) -> tuple[DataLoader, DataLoader]:
        kfold_dir_paths = self.__k_fold__()
        train_dataset = Dataset_Major_Depression(md_path_list=kfold_dir_paths[fold]["train"])
        test_dataset  = Dataset_Major_Depression(md_path_list=kfold_dir_paths[fold]["test"])
        train_dataloader = DataLoader(train_dataset, batch_size=self.train_config.batch_size, num_workers=self.train_config.num_workers)
        test_dataloader  = DataLoader(test_dataset,  batch_size=self.train_config.batch_size, num_workers=self.train_config.num_workers)
        return_dataloaders = Return_Dataloaders(train=train_dataloader, test=test_dataloader)
        return return_dataloaders

    
get_major_dataloader_via_fold = KFold_Major_Depression(is_global_signal_regression=True).get_dataloader_via_fold

