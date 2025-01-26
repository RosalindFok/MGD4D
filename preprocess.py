import ants
import json
import time
import shutil
import scipy.io  
import warnings
import numpy as np
import pandas as pd 
import nibabel as nib 
from tqdm import tqdm
from nilearn import maskers, connectome

from path import Path, Paths
from config import Gender, IS_MD
from plot import plot_heap_map, draw_atlas


def clear_temporary_files_of_ants(temp_dir_path : Path = Path("C:") / "Users" / "26036" / "AppData"/ "Local" / "Temp") -> None:
    # TODO: set your own temp_dir_path. In Linux, it maybe '/tmp'
    temp_dir_path = Path("C:") / "Users" / "26036" / "AppData"/ "Local" / "Temp"
    for suffix in ["*.nii.gz", "*.mat"]:
        for file_path in temp_dir_path.glob(suffix):
            file_path.unlink(missing_ok=True)
       
def register(fixed_path : str, moving_path : str, do_resample : bool = True) -> ants.ANTsImage:
    fixed_image = ants.image_read(fixed_path)
    moving_image = ants.image_read(moving_path)
    if do_resample:
        moving_image = ants.resample_image(moving_image, fixed_image.shape, 1, 0)
    registered_image = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform="SyN", aff_metric="MI")
    return registered_image["warpedmovout"]

def split_4dantsImageFrame_into_3d(image_4d : ants.ANTsImage) -> str:
    assert image_4d.shape[-1] == 1 and image_4d.dimension == 4, f"image_4d.shape={image_4d.shape}, image_4d.dimension={image_4d.dimension}"
    # image_4d.shape[-1] = 1
    temp_path = "temp.nii"
    ants.image_write(image_4d, temp_path)
    # delete it's last dimension via nibabel, there are bugs in ants.
    img = nib.load(temp_path)
    new_img = nib.Nifti1Image(img.get_fdata().squeeze(-1), img.affine, img.header)
    nib.save(new_img, temp_path)
    return temp_path

def adjust_dim_of_antsImage(image_5d_path : Path) -> None:
    start_time = time.time()
    image_5d = nib.load(image_5d_path)
    if image_5d.header['dim'][0] == 5:
        new_data = image_5d.get_fdata().squeeze()
        new_header = image_5d.header
        new_header.set_data_shape(new_data.shape)
        new_img = nib.Nifti1Image(new_data, image_5d.affine, image_5d.header)
        nib.save(new_img, image_5d_path)
        end_time = time.time()
        print(f"It took {end_time - start_time:.2f} seconds to adjust dim of {image_5d_path.absolute()}")
 
def preprocess_anat3d_and_func4d_with_atlas(saved_dir_path : Path, 
                                            participants_info : pd.DataFrame | dict,
                                            anat3d_path : Path, func4d_path : Path, 
                                            atlas_path : Path = Paths.Brainnetome_Atlas.BN_Atlas_246_1mm_nii_path
                                        ) -> None:
    # Save participant's information
    info_json_path = saved_dir_path / "info.json"
    if not info_json_path.exists():
        if isinstance(participants_info, pd.DataFrame): # ds002748, ds003007
            subject_info = participants_info[participants_info['participant_id'] == saved_dir_path.name]  
            subject_info = subject_info.iloc[0].to_dict()
        elif isinstance(participants_info, dict): # Cambridge
            subject_info = participants_info[saved_dir_path.name]
        else:
            raise ValueError(f"participants_info must be pd.DataFrame or dict, but got {type(participants_info)}")
        with info_json_path.open('w') as f:
            json.dump(subject_info, f, indent=4)
    
    # Plot atlas
    fig_path = Paths.Fig_Dir / (atlas_path.stem + ".png")
    if not fig_path.exists():
        draw_atlas(atlas = nib.load(atlas_path), saved_path=fig_path)

    # Step 1: Register
    # anat: anat -> atlas 
    aligned_anat_path = saved_dir_path / "aligned_anat.nii.gz"
    if not aligned_anat_path.exists():
        print(f"Registering {saved_dir_path.name}: {anat3d_path.parent.name}")
        result = register(fixed_path=str(atlas_path), moving_path=str(anat3d_path))
        ants.image_write(result, str(aligned_anat_path))
        del result
    # func: func -> anat
    aligned_func_path = saved_dir_path / "aligned_func.nii.gz"
    if not aligned_func_path.exists():
        moving_image = ants.image_read(str(func4d_path))
        results_list = [] 
        for t in tqdm(range(moving_image.shape[-1]), desc=f"Registering {saved_dir_path.name}: {func4d_path.parent.name}", leave=True):
            # delete the first 5 frames
            if t < 5:
                continue
            # t:t+1 cannot be t, and ":,:,:," cannot be "...,", otherwise the orientation will be wrong
            moving_image_t = moving_image[:, :, :, t:t+1]
            temp_path = split_4dantsImageFrame_into_3d(image_4d=moving_image_t)
            # register
            result = register(fixed_path=str(aligned_anat_path), moving_path=temp_path)
            results_list.append(result)
            # delete temporary file and variables
            Path(temp_path).unlink()
            del result, moving_image_t, temp_path
        ants.image_write(ants.merge_channels(results_list), str(aligned_func_path))
        del moving_image, results_list
    # change the dim[0] from 5 to 4
    adjust_dim_of_antsImage(aligned_func_path)
    # Clear the temporary files of ants
    clear_temporary_files_of_ants()

    # Step 2: Denoise
    # anat 
    denoised_anat_path = saved_dir_path / "denoised_anat.nii.gz"
    if not denoised_anat_path.exists():
        print(f"Denoising {denoised_anat_path.parent.name}: anat")
        result = ants.denoise_image(image=ants.image_read(str(aligned_anat_path)))
        ants.image_write(result, str(denoised_anat_path))
        del result
    # func
    denoised_func_path = saved_dir_path / "denoised_func.nii.gz"
    if not denoised_func_path.exists():
        aligned_func = ants.image_read(str(aligned_func_path))
        results_list = [] 
        for t in tqdm(range(aligned_func.shape[-1]), desc=f"Denoising {denoised_func_path.parent.name}: func", leave=True):
            aligned_func_t = aligned_func[:, :, :, t:t+1]
            temp_path = split_4dantsImageFrame_into_3d(image_4d=aligned_func_t)
            result = ants.denoise_image(ants.image_read(str(temp_path)))
            results_list.append(result)
            # delete temporary file and variables
            Path(temp_path).unlink()
            del result, aligned_func_t, temp_path
        ants.image_write(ants.merge_channels(results_list, channels_first=True), str(denoised_func_path))
        del aligned_func, results_list
    # change the dim[0] from 5 to 4
    adjust_dim_of_antsImage(denoised_func_path)

    # Step 3: Functional connectivity
    fc_matrix_path = saved_dir_path / "fc_matrix.npy"
    if not fc_matrix_path.exists():
        start_time = time.time()
        atlas = nib.load(Paths.Brainnetome_Atlas.BN_Atlas_246_1mm_nii_path)
        denoised_func = nib.load(denoised_func_path)
        masker = maskers.NiftiLabelsMasker(labels_img=atlas, standardize="zscore_sample")
        time_series = masker.fit_transform(denoised_func)
        matrix = connectome.ConnectivityMeasure(kind="correlation", standardize="zscore_sample").fit_transform([time_series])[0]
        np.fill_diagonal(matrix, 0) # set the diagonal to 0 (1 -> 0)
        np.save(fc_matrix_path, matrix) # (-1, 1), only the diagonal is 0
        plot_heap_map(matrix=matrix, saved_dir_path=saved_dir_path)
        end_time = time.time()
        print(f"It took {end_time - start_time:.2f} seconds to calculate the functional connectivity matrix.")


def process_ds002748(dir_path : Path = Paths.Depression.ds002748_dir_path,
                     saved_root_dir_path : Path = Paths.Run_Files.run_files_ds002748_dir_path) -> None:
    """
    fMRI
    72 subjects = 51 subjects with mild depression + 21 healthy controls
    A session consists of 100 dynamic scans with TR = 2.5 s and 25 slices. 
    anat: shape=[288,288,181], voxel size=1mm * 1mm * 1mm
    func: shape=[112,112,25,100], voxel size=2mm * 2mm * 5mm
    """
    assert dir_path.exists(), f"{dir_path.absolute()} does not exist!"
    participants_tsv_path = dir_path / "participants.tsv"
    assert participants_tsv_path.exists(), f"{participants_tsv_path.absolute()} does not exist!"
    participants_info = pd.read_csv(participants_tsv_path, sep='\t')
    sub_dir_path_list = [d for d in dir_path.iterdir() if d.is_dir()]
    assert len(sub_dir_path_list) == len(participants_info), f"len(sub_dir_path_list)={len(sub_dir_path_list)} != len(participants_info)={len(participants_info)}"
    sub_anat_nii_path_dict = {d.name : p for d in sub_dir_path_list for p in (d / "anat").iterdir()}
    sub_func_nii_path_dict = {d.name : p for d in sub_dir_path_list for p in (d / "func").iterdir()}

    for sub_name in sub_anat_nii_path_dict.keys():
        # path of saved dir, each subject has its own dir
        saved_dir_path = saved_root_dir_path / sub_name
        saved_dir_path.mkdir(parents=True, exist_ok=True)

        preprocess_anat3d_and_func4d_with_atlas(saved_dir_path=saved_dir_path, 
                                                participants_info=participants_info,
                                                anat3d_path=sub_anat_nii_path_dict[sub_name], 
                                                func4d_path=sub_func_nii_path_dict[sub_name],
                                                atlas_path=Paths.Brainnetome_Atlas.BN_Atlas_246_1mm_nii_path)        

def process_ds003007(dir_path : Path = Paths.Depression.ds003007_dir_path,
                     saved_root_dir_path : Path = Paths.Run_Files.run_files_ds003007_dir_path) -> None:
    """
    fMRI
    29 subjects = 15 depr_no_treatment + 8 depr_cbt + 6 depr_nfb
    depr_no_treatment: untreated patients with depression
    depr_cbt: patients who undergo cognitive behavioral therapy
    depr_nfb: patients who undergo rt-fmri neurofeedback treatment course, rt = real time
    There are two session of RS with 2-month interval, 3 groups of patients: no treatment, CBT, or fmri-NFB treatment. 
    A session consists of 100 dynamic scans with TR = 2.5 s and 25 slices.
    ses-pre: anat and func
    ses-post: depr_no_treatment only have func, depr_cbt and depr_nfb have both anat and func
    """
    assert dir_path.exists(), f"{dir_path.absolute()} does not exist!"
    participants_tsv_path = dir_path / "participants.tsv"
    assert participants_tsv_path.exists(), f"{participants_tsv_path.absolute()} does not exist!"
    participants_info = pd.read_csv(participants_tsv_path, sep='\t')
    sub_dir_path_list = [d for d in dir_path.iterdir() if d.is_dir()]
    assert len(sub_dir_path_list) == len(participants_info), f"len(sub_dir_path_list)={len(sub_dir_path_list)} != len(participants_info)={len(participants_info)}"
    
    for d in sub_dir_path_list:
        # pre
        saved_pre_dir_path = saved_root_dir_path / "pre" / d.name
        saved_pre_dir_path.mkdir(parents=True, exist_ok=True)
        pre_anat_file_path = [x for x in (d / "ses-pre" / "anat").glob("*")]
        assert len(pre_anat_file_path) == 1, f"len(pre_anat_file_path)={len(pre_anat_file_path)} != 1"
        pre_anat_file_path = pre_anat_file_path[0]
        pre_func_file_path = [x for x in (d / "ses-pre" / "func").glob("*")]
        assert len(pre_func_file_path) == 1, f"len(pre_func_file_path)={len(pre_func_file_path)} != 1"
        pre_func_file_path = pre_func_file_path[0]
        preprocess_anat3d_and_func4d_with_atlas(saved_dir_path=saved_pre_dir_path, 
                                                participants_info=participants_info,
                                                anat3d_path=pre_anat_file_path, 
                                                func4d_path=pre_func_file_path,
                                                atlas_path=Paths.Brainnetome_Atlas.BN_Atlas_246_1mm_nii_path) 

        # post
        saved_post_dir_path = saved_root_dir_path / "post" / d.name
        saved_post_dir_path.mkdir(parents=True, exist_ok=True)
        post_func_file_path = [x for x in (d / "ses-post" / "func").glob("*")]
        assert len(post_func_file_path) == 1, f"len(post_func_file_path)={len(post_func_file_path)} != 1"
        post_func_file_path = post_func_file_path[0]
        if not (d / "ses-post" / "anat").exists():
            for src in saved_pre_dir_path.glob("*anat*"):
                shutil.copy(src=src, dst=saved_post_dir_path / src.name)
            preprocess_anat3d_and_func4d_with_atlas(saved_dir_path=saved_post_dir_path, 
                                                    participants_info=participants_info,
                                                    anat3d_path=None, 
                                                    func4d_path=post_func_file_path,
                                                    atlas_path=Paths.Brainnetome_Atlas.BN_Atlas_246_1mm_nii_path)
        else:
            post_anat_file_path = [x for x in (d / "ses-post" / "anat").glob("*")]
            assert len(post_anat_file_path) == 1, f"len(post_anat_file_path)={len(post_anat_file_path)} != 1"
            post_anat_file_path = post_anat_file_path[0]
            preprocess_anat3d_and_func4d_with_atlas(saved_dir_path=saved_post_dir_path,
                                                    participants_info=participants_info,
                                                    anat3d_path=post_anat_file_path,
                                                    func4d_path=post_func_file_path,
                                                    atlas_path=Paths.Brainnetome_Atlas.BN_Atlas_246_1mm_nii_path)

def process_Cambridge(dir_path : Path = Paths.Functional_Connectomes_1000,
                      saved_root_dir_path : Path  = Paths.Run_Files.run_files_cambridge_dir_path) -> None:
    # read demographics
    participants_info = {} # {sub_name : {age, gender}}
    with dir_path.demographics_txt_path.open('r') as f: 
        content = [line.split("\t") for line in f.read().splitlines()] # [[sub_name, index, age, gender], ...]
        for line in content:
            participants_info[line[0]] = {'age' : int(line[2]), 'gender' : Gender.FEMALE if line[3] == 'f' else Gender.MALE}
    
    for part_dir_path in dir_path.root_dir.iterdir():
        for sub_dir_path in part_dir_path.iterdir():
            if sub_dir_path.is_dir():
                # path of saved dir, each subject has its own dir
                saved_dir_path  = saved_root_dir_path / sub_dir_path.name
                saved_dir_path.mkdir(parents=True, exist_ok=True)

                preprocess_anat3d_and_func4d_with_atlas(saved_dir_path=saved_dir_path, 
                                                        participants_info=participants_info,
                                                        anat3d_path=sub_dir_path / "anat" / "mprage_skullstripped.nii.gz", 
                                                        func4d_path=sub_dir_path / "func" / "rest.nii.gz",
                                                        atlas_path=Paths.Brainnetome_Atlas.BN_Atlas_246_1mm_nii_path)   
                
        # TODO sub01361 有明显异常，需要检查
            

def read_adjacent_matrix_of_brainnetome_atlas(file_path : Path = Paths.Brainnetome_Atlas.BNA_adjacent_matrix_path) -> np.array:
    return pd.read_csv(file_path, header=None).values # shape = (246, 246)

def process_rest_meta_mdd(dir_path : Path = Paths.Depression.REST_meta_MDD_dir_path,
                          saved_root_dir_path : Path  = Paths.Run_Files.run_files_rest_meta_mdd_dir_path) -> None:
    # Auxiliary information
    participants_info = {} # {sub_id : {key : value}}
    participants_info_xlsx_path = dir_path / "REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx"
    assert participants_info_xlsx_path.exists(), f"{participants_info_xlsx_path} does not exist"
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
    # Note: Data of Site 4 (S4) duplicated from Site 14 (S14). Thus please exclude Site 4 (S4) data from your analyses.
    # Note: There is no data from Site 4 (S4) in "REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx"
    # Samples number (in xlsx): 1276 MDD + 1104 controls = 2380 subjects
    # Samples number (in official website): 1300 MDD + 1128 controls = 2428 subjects
    # Samples number (in ROISignals_*): S4, 48 subjects; S14, 96 subjects
    # 2380 + 48 = 2428, It is correct.
    for sheet, pd_frame in pd.read_excel(participants_info_xlsx_path, sheet_name=None, engine="openpyxl").items():
        sheet = IS_MD.NO if sheet == "Controls" else IS_MD.IS if sheet == "MDD" else None
        assert sheet is not None, f"sheet={sheet} is not valid"
        for _, row in pd_frame.iterrows():
            assert row["ID"] not in participants_info, f"{row['ID']} already exists"
            information_dict = row.to_dict()
            information_dict.update({"depression" : sheet})
            sex_map = {1 : Gender.MALE, 2 : Gender.FEMALE, 0 : Gender.UNSPECIFIED}
            information_dict.update({"Sex" : sex_map[information_dict["Sex"]]})
            participants_info[row["ID"]] = information_dict
    info_json_path = saved_root_dir_path / "participants_info.json"
    with info_json_path.open("w") as f:
        json.dump(participants_info, f, indent=4)

    # Functional connectivity
    results_dir_path = dir_path / "REST-meta-MDD-Phase1-Sharing" / "Results"
    # The suffix of each folder means the preprocessing steps:
    #   A - Slice timing correction
    #   R – Realign
    #   C - Covariates Removed (without global signal regression)
    #   globalC - Covariates Removed (with global signal regression)
    #   W – Spatial Normalization
    #   F – Filter (0.01~0.1Hz)
    for group_name in ["ROISignals_FunImgARCWF", "ROISignals_FunImgARglobalCWF"]:
        for mat_path in tqdm(list((results_dir_path / group_name).iterdir()), desc=f"Process on {group_name}", leave=True):
            assert mat_path.suffix == ".mat", f"{mat_path} is not a mat file"
            sub_id = mat_path.stem.split("_")[1]
            saved_dir_path = saved_root_dir_path / group_name / sub_id
            saved_dir_path.mkdir(parents=True, exist_ok=True)

            if sub_id not in participants_info:
                assert 'S4' in sub_id, f"{sub_id} is not in participants_info" # S4 is duplicated from S14
                saved_dir_path.rmdir() # empty directory
                continue
            else:
                ROISignals = scipy.io.loadmat(mat_path)['ROISignals']
                # ROISignals is a 2D array, with shape (n_timepoints, n_regions)
                # n_timepoints = {90, 140, 150, 170, 174, 190, 200, 202, 230, 232, 240}
                # n_regions = {1568, 1833}
                # n_regions and corresponding atlas:
                #   1~116: Automated Anatomical Labeling (AAL) atlas (Tzourio-Mazoyer et al., 2002)
                #   117~212: Harvard-Oxford atlas (Kennedy et al., 1998)– cortical areas
                #   213~228: Harvard-Oxford atlas (Kennedy et al., 1998)– subcortical areas
                #   229~428: Craddock’s clustering 200 ROIs (Craddock et al., 2012)
                #   429~1408: Zalesky’s random parcelations (compact version: 980 ROIs) (Zalesky et al., 2010)
                #   1409~1568: Dosenbach’s 160 functional ROIs (Dosenbach et al., 2010)
                #   1569: Global signal (Since DPARSF V4.2)
                #   1570~1833: Power’s 264 functional ROIs (Power et al., 2011) (Since DPARSF V4.3)
                #   1834~2233: Schaefer's 400 Parcels (Schaefer et al., 2017) (Since DPARSF V4.4)
                time_series_pair = {
                    "AAL" : ROISignals[5:, :116],
                    "HOC" : ROISignals[5:, 116:212],
                    "HOS" : ROISignals[5:, 212:228],
                }           

                for atlas_name, time_series in time_series_pair.items():
                    # atlas = atlas_labels_dict[atlas_name]["atlas"]
                    # labels = atlas_labels_dict[atlas_name]["labels"]
                    fc_matrix_path = saved_dir_path / f"{atlas_name}_fc_matrix.npy"
                    if not fc_matrix_path.exists():
                        matrix = connectome.ConnectivityMeasure(kind='correlation', standardize='zscore_sample').fit_transform([time_series])[0] 
                        np.fill_diagonal(matrix, 0) # set the diagonal to 0 (1 -> 0)
                        np.save(fc_matrix_path, matrix)
                        plot_heap_map(matrix=matrix, saved_dir_path=saved_dir_path, fig_name=f"{atlas_name}_fc_matrix.svg")

    # VBM
    vbm_dir_path = dir_path / "REST-meta-MDD-VBM-Phase1-Sharing" / "VBM"
    # wc1: Gray matter density in MNI space
    # wc2: White matter density in MNI space
    # mwc1: Gray matter volume in MNI space
    # mwc2: White matter volume in MNI space
    for group_name in ["wc1", "wc2", "mwc1", "mwc2"]:
        saved_dir_path = saved_root_dir_path / group_name
        saved_dir_path.mkdir(parents=True, exist_ok=True)
        for nii_path in tqdm(list((vbm_dir_path / group_name).iterdir()), desc=f"Process on {group_name}", leave=True):
            if 'S4' in nii_path.name:
                continue
            denoised_path = saved_dir_path / nii_path.name
            if not denoised_path.exists():
                result = ants.denoise_image(image=ants.image_read(str(nii_path)))
                ants.image_write(result, str(denoised_path))
                del result

def main() -> None:
    process_ds002748()
    process_ds003007()
    # process_Cambridge()
    process_rest_meta_mdd()
    adjacent_matrix = read_adjacent_matrix_of_brainnetome_atlas()

if __name__ == "__main__":
    main()
