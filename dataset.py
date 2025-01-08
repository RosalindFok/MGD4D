import json
import torch
import numpy as np
from pathlib import Path
from nilearn import datasets
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

from path import Paths
from config import IS_MD

class Dataset_Mild_Depression(Dataset):
    def __init__(self, md_path_list : list[Path], hc_path_list : list[Path]) -> None:
        super().__init__()
        self.path_list = [] # [(dir_path, tag), ...]; tag, 0: health controls, 1: mild depression
        max_len = max([len(md_path_list), len(hc_path_list)])
        for i in range(max_len):
            if i < len(md_path_list):
                self.path_list.append((md_path_list[i], IS_MD.IS))
            if i < len(hc_path_list):
                self.path_list.append((hc_path_list[i], IS_MD.NO))
    
    def __get_the_only_file_in_dir__(self, dir_path : Path, substr : str) -> Path:
        path_list = [x for x in dir_path.glob(f"*{substr}*")]
        assert len(list(path_list)) == 1, f"Multiple or no {substr} file found in {dir_path}"
        return path_list[0]
    
    def __getitem__(self, idx) -> tuple[dict, int]:
        dir_path, tag = self.path_list[idx]

        # denoised_anat
        denoised_anat_path = self.__get_the_only_file_in_dir__(dir_path, "denoised_anat.nii.gz")

        # functional connectivity matrix
        fc_matrix_path = self.__get_the_only_file_in_dir__(dir_path, "fc_matrix.npy")
        fc_matrix = np.load(fc_matrix_path)

        # information
        info_json_path = self.__get_the_only_file_in_dir__(dir_path, "info.json")
        
        with info_json_path.open("r") as f:
            info = json.load(f)
            age = info["age"]
            gender = info["gender"]

    def __len__(self):
        return len(self.path_list)

class KFold_Mild_Depression:
    def __init__(self, ds002748_dir_path : Path = Paths.Run_Files.run_files_ds002748_dir_path,
                       ds003007_dir_path : Path = Paths.Run_Files.run_files_ds003007_dir_path,
                       n_splits : int = 5, shuffle : bool = False):
        self.md, md_index = {}, 0 # index : subj_path. 51(ds002748) + 29(ds003007) = 80 subjects
        self.hc, hc_index = {}, 0 # index : subj_path. + = subjects

        # ds002748: 72 subjects = 51 subjects with mild depression + 21 healthy controls
        for dir_path in ds002748_dir_path.iterdir():
            # information json
            group = self.__get_group_in_infojson__(dir_path)
            if group == "depr":
                self.md[md_index] = dir_path
                md_index += 1
            elif group == "control":
                self.hc[hc_index] = dir_path
                hc_index += 1
            else:
                raise ValueError(f"Unknown group: {group}")

        # ds003007: pre, 29 subjects with mild depression; post, 15 depr_no_treatment + 8 depr_cbt + 6 depr_nfb
        self.no_treatment, self.cbt, self.nfb = {}, {}, {}
        nt_index, cbt_index, nfb_index = 0, 0, 0
        for index, dir_path in enumerate((ds003007_dir_path / "pre").iterdir()):
            self.md[md_index+index] = dir_path
        for dir_path in (ds003007_dir_path / "post").iterdir():
            group = self.__get_group_in_infojson__(dir_path)
            if group == "depr_no_treatment":
                self.no_treatment[nt_index] = dir_path
                nt_index += 1
            elif group == "depr_cbt":
                self.cbt[cbt_index] = dir_path
                cbt_index += 1
            elif group == "depr_nfb":
                self.nfb[nfb_index] = dir_path
                nfb_index += 1
            else:
                raise ValueError(f"Unknown group: {group}")

        # K Fold
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle)
               
    def __get_group_in_infojson__(self, dir_path : Path) -> str:
        info_path = [x for x in dir_path.glob("*.json")]
        assert len(list(info_path)) == 1, f"Multiple or no info file found in {dir_path}"
        info_path = info_path[0]
        with info_path.open("r") as f:
            info = json.load(f)
            group = info["group"]
        return group
    
    def __k_fold__(self, group_name : str) -> dict[int, dict[str, list[Path]]]:
        group_mapping = {"md": self.md, "hc": self.hc}  
        group = group_mapping.get(group_name)  
        if group is None:  
            raise ValueError(f"Unknown group: {group_name}")

        kfold_dir_paths = {} # fold : {train_set, test_set}
        fold = 1
        for train_index, test_index in self.kf.split(group):
            train_set_dir_path_list = [group[i] for i in train_index]
            test_set_dir_path_list  = [group[i] for i in test_index]
            kfold_dir_paths[fold] = {"train" : train_set_dir_path_list, "test" : test_set_dir_path_list}
            fold += 1
        return kfold_dir_paths

    def get_dataloader_via_fold(self, fold : int) -> DataLoader:
        kfold_md_dir_paths = self.__k_fold__(group_name="md")
        kfold_hc_dir_paths = self.__k_fold__(group_name="hc")
        assert fold in kfold_md_dir_paths.keys(), f"Unknown fold: {fold}, available folds: {kfold_md_dir_paths.keys()}"
        Dataset_Mild_Depression(md_path_list=kfold_md_dir_paths[fold]["train"], hc_path_list=kfold_hc_dir_paths[fold]["train"])



class Dataset_Major_Depression(Dataset):
    def __init__(self, ) -> None:
        super().__init__()

    def __getitem__(self, index) -> None:
        pass

    def __len__(self) -> None:
        pass

class KFold_Major_Depression:
    def __init__(self) -> None:
        self.atlas_labels_dict = self.__download_atlas__()

        self.paths_dict = {} # {group_name : {major_depression, healthy_control}}
        for group_name in ["ROISignals_FunImgARCWF", "ROISignals_FunImgARglobalCWF"]:
            md_list, hc_list = [], [] # [{sub_id:sub_id, info:info_path, atalas_name:fc_matrix_path}]
            for sub_dir_path in (Paths.Run_Files.run_files_rest_meta_mdd_dir_path / group_name).iterdir():
                middle_label = int(sub_dir_path.name.split("-")[1]) # 1-MDD, 2-HCS
                path_dict = {}
                path_dict["sub_id"] = sub_dir_path.name
                info_path = sub_dir_path / "info.json"
                assert info_path.exists(), f"Info file not found in {sub_dir_path}"
                path_dict["info"] = info_path
                for fc_matrix_path in sub_dir_path.glob("*.npy"):
                    assert fc_matrix_path.exists(), f"FC matrix file not found in {sub_dir_path}"
                    atlas_name = fc_matrix_path.stem.split("_")[0]
                    path_dict[atlas_name] = fc_matrix_path
                if middle_label == 1:
                    md_list.append(path_dict)
                elif middle_label == 2:
                    hc_list.append(path_dict)
                else:
                    raise ValueError(f"Unknown middle label: {middle_label}")
            self.paths_dict[group_name] = {"major_depression" : md_list, "healthy_control" : hc_list}

    def __download_atlas__(self) -> dict[any, list[str]]:
        """
        Download atlas from nilearn
        """
        downloaded_atlas_dir_path = Paths.Run_Files.run_files_rest_meta_mdd_dir_path / "downloaded_atlas"
        method_pair = { # {name of atlas : {atlas, coresponding matrix}}
            # https://nilearn.github.io/stable/modules/description/aal.html
            "AAL" : datasets.fetch_atlas_aal(data_dir=downloaded_atlas_dir_path),
            # https://nilearn.github.io/stable/modules/description/harvard_oxford.html
            "HOC" :  datasets.fetch_atlas_harvard_oxford("cort-prob-1mm", data_dir=downloaded_atlas_dir_path),
            # https://nilearn.github.io/stable/modules/description/harvard_oxford.html
            "HOS" :  datasets.fetch_atlas_harvard_oxford("sub-prob-1mm", data_dir=downloaded_atlas_dir_path),
            # https://nilearn.github.io/stable/modules/description/craddock_2012.html
            # Note: There are no labels for the Craddock atlas, which have been generated from applying spatially constrained clustering on resting-state data.
            # "Craddock" : datasets.fetch_atlas_craddock_2012(data_dir=downloaded_atlas_dir_path),
        }
        atlas_labels_dict = {}
        for key, value in method_pair.items():
            # type of AAL atlas is str, path to nifti file containing the regions.
            # type of Harvard-Oxford atlas is nibabel image.
            atlas_labels_dict[key] = {"atlas" : value.maps, "labels" : value.labels}
        return atlas_labels_dict


    def __k_fold__(self, ) -> None:
        pass

    def get_dataloader_via_fold(self, fold : int) -> DataLoader:
        pass

KFold_Major_Depression()