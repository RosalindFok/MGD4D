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
from config import IS_MD, Gender, Configs, seed

random.seed(seed)
np.random.seed(seed)

# dataclass is not used because dict[tensor] is not supported by PyTorch
Dataset_Returns = namedtuple("Dataset_Returns", ["id", "info", "anat", "func", "target"])

@dataclass
class Dataloader_Returns:
    train: DataLoader
    test : DataLoader
    info : dict[str, int]
    weight : torch.Tensor

# convert data into PyTorch tensors
to_tensor = partial(torch.tensor, dtype=torch.float32)

########################
### Mild Depression ###
########################

class Dataset_Mild_Depression(Dataset):
    def __init__(self, path_list : list[dict[str, Any]]) -> None:
        super().__init__()
        self.path_list = path_list

    def __getitem__(self, idx) -> tuple[str, dict[str, int], dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
        path_dict = self.path_list[idx]
        # id
        sub_id = path_dict["id"]
        # info
        auxi_info = path_dict["auxi"]
        # structural: anat
        anat_path = path_dict["anat"]
        anat = nib.load(anat_path).get_fdata() # (182, 218, 182)
        # crop the anat to (182, 182, 182)
        y_crop = anat.shape[1] - anat.shape[0]
        y_start = y_crop // 2
        y_end = y_start + anat.shape[0]
        anat = anat[:, y_start:y_end, :]
        anat = (anat - anat.min()) / (anat.max() - anat.min())  # normalize to [0, 1]
        anat = to_tensor(anat)      
        assert anat.shape == (182, 182, 182), f"anat.shape={anat.shape} != (182, 182, 182)"
        # functional: func
        func_path = path_dict["func"]
        func = np.load(func_path, allow_pickle=True) 
        func = 2 * ((func - func.min()) / (func.max() - func.min())) - 1  # normalize to [-1, 1]
        func = to_tensor(func)
        assert func.shape == (246, 246), f"func.shape={func.shape} != (246, 246)"
        # target
        target = path_dict["target"]
        return Dataset_Returns(id=sub_id, info=auxi_info, anat={"anat" : anat}, func={"func" : func}, target=target) 

    def __len__(self) -> int:
        return len(self.path_list)

class KFold_Mild_Depression:
    def __init__(self, ds002748_dir_path : Path = Paths.Run_Files.run_files_ds002748_dir_path,
                       ds003007_dir_path : Path = Paths.Run_Files.run_files_ds003007_dir_path,
                       cambridge_dir_path : Path = Paths.Run_Files.run_files_cambridge_dir_path):
        self.md_dict, self.hc_dict = {}, {} # {{id : {key : value}}, ...}

        # mild depression and healthy controls
        for (name, root_dir) in [
            ("ds002748",      ds002748_dir_path),         # 72 = 51 mild depression + 21 healthy controls
            ("ds003007 pre",  ds003007_dir_path / "pre"), # 29 subjects with mild depression
            # Not used:
            # ("ds003007 post", ds003007_dir_path / "post"), # 15 depr_no_treatment + 8 depr_cbt + 6 depr_nfb
        ]:
            for dir_path in tqdm(root_dir.iterdir(), desc=name, leave=False):
                # information
                info = self.__get_info__(dir_path)
                participant_id = "_".join([name, info["participant_id"]])
                if name == "ds002748":
                    target = IS_MD.IS if info["group"] == "depr" else IS_MD.NO if info["group"] == "control" else None
                elif name == "ds003007 pre":
                    target = IS_MD.IS # all of them are "depr_no_treatment"
                # Not used:
                # elif name == "ds003007 post":
                    # target = IS_MD.IS if info["group"] == "depr_no_treatment" else IS_MD.NO if info["group"] in ["depr_cbt", "depr_nfb"] else None
                else:
                    raise ValueError(f"Unknown dataset name: {name}")
                assert target is not None, f"{dir_path}'s group={info['group']} is not valid."
                auxi_info = self.__get_auxi_info__(info)
                # anatomical
                anat_path = self.__get_anat_path__(dir_path)
                # functional
                func_path = self.__get_func_path__(dir_path)
                # save to dict
                saved_dict = {"auxi" : auxi_info, "anat" : anat_path,
                              "func" : func_path, "target" : target}
                if target == IS_MD.IS:
                    assert participant_id not in self.md_dict, f"{participant_id} is already in self.md_dict"
                    self.md_dict[participant_id] = saved_dict
                elif target == IS_MD.NO:
                    assert participant_id not in self.hc_dict, f"{participant_id} is already in self.hc_dict"
                    self.hc_dict[participant_id] = saved_dict
                else:
                    raise ValueError(f"Unknown target: {target}")
        
        # data augmentation: add more healthy controls
        difference = len(self.md_dict) - len(self.hc_dict) # difference = 80 - 21 = 59;
        selected_list = random.sample(list(cambridge_dir_path.iterdir()), difference)
        
        # cambridge: 198 healthy controls
        name = "cambridge"
        for dir_path in tqdm(selected_list, desc=name, leave=False):
            participant_id = "_".join([name, dir_path.name])
            saved_dict = {"auxi" : self.__get_auxi_info__(self.__get_info__(dir_path)), 
                          "anat" : self.__get_anat_path__(dir_path),
                          "func" : self.__get_func_path__(dir_path), 
                          "target" : IS_MD.NO}
            assert participant_id not in self.hc_dict, f"{participant_id} is already in self.hc_dict"
            self.hc_dict[participant_id] = saved_dict
        
        assert len(self.md_dict) == len(self.hc_dict), f"Length of self.md_dict and self.hc_dict are not equal: {len(self.md_dict)} != {len(self.hc_dict)}"
        
        # max value of auxiliary information
        max_val = defaultdict(list)
        for sub_dict in [self.md_dict, self.hc_dict]:
            for _, dicts in sub_dict.items():
                auxi_info = dicts["auxi"]
                for key, value in auxi_info.items():
                    max_val[key].append(value)
        self.max_val = {key : max(value) for key, value in max_val.items()} 

        # K Fold
        self.kf = KFold(n_splits=Configs.n_splits.stop-1, shuffle=Configs.shuffle)
        self.kfold_dir_paths = self.__k_fold__() 
    
    def __get_info__(self, dir_path : Path) -> dict[str, Any]:
        """
        Read info.json in dir_path and return a dict.
        """
        info_json_path = dir_path / "info.json"
        assert info_json_path.exists(), f"{info_json_path} does not exist."
        with info_json_path.open("r") as f:
            info = json.load(f)
        return info
    
    def __get_auxi_info__(self, info : dict[str, Any]) -> dict[str, int]:
        """
        Get age and gender from info.json
        """
        assert "age" in info.keys() and "gender" in info.keys(), f"{info} does not contain age and gender."
        auxi_info = {"age" : int(info["age"])}
        if isinstance(info["gender"], str):
            auxi_info["gender"] = Gender.FEMALE if info["gender"] == "f" else Gender.MALE if info["gender"] == "m" else Gender.UNSPECIFIED
        elif isinstance(info["gender"], int):
            auxi_info["gender"] = info["gender"]
        else:
            raise ValueError(f"Unknown gender: {info['gender']}")
        assert len(auxi_info) == 2, f"{info} does not contain age and gender."
        return auxi_info
    
    def __get_anat_path__(self, dir_path : Path) -> Path:
        """
        Get denoised_anat.nii.gz in dir_path and return its path.
        """
        anat_path = dir_path / "denoised_anat.nii.gz"
        assert anat_path.exists(), f"{anat_path} does not exist."
        return anat_path

    def __get_func_path__(self, dir_path : Path) -> Path:
        """
        Get fc_matrix.npy in dir_path and return its path.
        """
        func_path = dir_path / "fc_matrix.npy"
        assert func_path.exists(), f"{func_path} does not exist."
        return func_path

    def __k_fold__(self) -> dict[int, dict[str, list[dict[str, Any]]]]:
        md_keys = list(self.md_dict.keys())
        hc_keys = list(self.hc_dict.keys())
        random.shuffle(md_keys) 
        random.shuffle(hc_keys)
        assert len(md_keys) == len(hc_keys), f"Length of md_keys and hc_keys are not equal: {len(md_keys)}!= {len(hc_keys)}"

        kfold_dir_paths = {} # {fold : {train_set, test_set}}
        fold = 1
        for train_index, test_index in self.kf.split(md_keys):
            train_md = [{**self.md_dict[md_keys[i]], "id":md_keys[i]} for i in train_index]
            train_hc = [{**self.hc_dict[hc_keys[i]], "id":hc_keys[i]} for i in train_index]
            test_md  = [{**self.md_dict[md_keys[i]], "id":md_keys[i]} for i in test_index]
            test_hc  = [{**self.hc_dict[hc_keys[i]], "id":hc_keys[i]} for i in test_index]
            assert len(train_md) == len(train_hc), f"Length of train_md and train_hc are not equal: {len(train_md)}!= {len(train_hc)}"
            assert len(test_md)  == len(test_hc) , f"Length of test_md and test_hc are not equal: {len(test_md)}!= {len(test_hc)}"
            kfold_dir_paths[fold] = {
                "train" : [item for pair in zip(train_md, train_hc) for item in pair], 
                "test"  : [item for pair in zip(test_md , test_hc)  for item in pair]  
            }
            fold += 1

        return kfold_dir_paths

    def get_dataloader_via_fold(self, fold : int) -> Dataloader_Returns:
        assert fold in self.kfold_dir_paths.keys(), f"Unknown fold: {fold}"
        train_dataset = Dataset_Mild_Depression(path_list=self.kfold_dir_paths[fold]["train"])
        test_dataset  = Dataset_Mild_Depression(path_list=self.kfold_dir_paths[fold]["test"])
        train_dataloader = DataLoader(train_dataset, batch_size=Configs.dataset.batch_size, num_workers=Configs.num_workers, shuffle=Configs.shuffle, pin_memory=True, persistent_workers=True)
        test_dataloader  = DataLoader(test_dataset,  batch_size=Configs.dataset.batch_size, num_workers=Configs.num_workers, shuffle=Configs.shuffle, pin_memory=True, persistent_workers=True)
        return Dataloader_Returns(train=train_dataloader, test=test_dataloader, info=self.max_val, weight=to_tensor([0.5,0.5]))

get_mild_dataloader_via_fold = KFold_Mild_Depression().get_dataloader_via_fold

########################
### Major Depression ###
########################

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

class Dataset_Major_Depression(Dataset):
    def __init__(self, path_list : list[dict[str, Any]]) -> None:
        super().__init__()
        self.path_list = path_list
    
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
            matrix = 2 * ((matrix - matrix.min()) / (matrix.max() - matrix.min())) - 1  # normalize to [-1, 1]
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

        return Dataset_Returns(
            id=ID, info=auxi_info, anat=vbm_matrices, func=fc_matrices, target=target
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
        self.kf = KFold(n_splits=Configs.n_splits.stop-1, shuffle=Configs.shuffle)
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
        for fold in Configs.n_splits:
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

    def get_dataloader_via_fold(self, fold : int) -> Dataloader_Returns:
        train_dataset = Dataset_Major_Depression(path_list=self.kfold_dir_paths[fold]["train"])
        test_dataset  = Dataset_Major_Depression(path_list=self.kfold_dir_paths[fold]["test"])
        # set persistent_workers=True to avoid OOM
        train_dataloader = DataLoader(train_dataset, batch_size=Configs.dataset.batch_size, num_workers=Configs.num_workers, shuffle=Configs.shuffle, pin_memory=True, persistent_workers=True)
        test_dataloader  = DataLoader(test_dataset,  batch_size=Configs.dataset.batch_size, num_workers=Configs.num_workers, shuffle=Configs.shuffle, pin_memory=True, persistent_workers=True)
        return Dataloader_Returns(train=train_dataloader, test=test_dataloader, 
                                  info=self.max_value, weight=to_tensor(self.weights[fold]))

# global_signal is better than not    
get_major_dataloader_via_fold = KFold_Major_Depression(is_global_signal_regression=True).get_dataloader_via_fold