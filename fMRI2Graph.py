import ants
import json
import time
import numpy as np
import pandas as pd 
import nibabel as nib 
import matplotlib.pyplot as plt
from tqdm import tqdm
from nilearn import maskers, connectome

from path import Path, Paths

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

def process_ds002748() -> None:
    """
    72 subjects = 51 subjects with mild depression + 21 healthy controls
    A session consists of 100 dynamic scans with TR = 2.5 s and 25 slices. 
    anat: shape=[288,288,181], voxel size=1mm * 1mm * 1mm
    func: shape=[112,112,25,100], voxel size=2mm * 2mm * 5mm
    """
    assert Paths.ds003007_dir_path.exists(), f"{Paths.ds003007_dir_path.absolute()} does not exist!"
    participants_tsv_path = Paths.ds002748_dir_path / "participants.tsv"
    assert participants_tsv_path.exists(), f"{participants_tsv_path.absolute()} does not exist!"
    participants_info = pd.read_csv(participants_tsv_path, sep='\t')
    sub_dir_path_list = [d for d in Paths.ds002748_dir_path.iterdir() if d.is_dir()]
    assert len(sub_dir_path_list) == len(participants_info), f"len(sub_dir_path_list)={len(sub_dir_path_list)} != len(participants_info)={len(participants_info)}"
    sub_anat_nii_path_dict = {d.name : p for d in sub_dir_path_list for p in (d / "anat").iterdir()}
    sub_func_nii_path_dict = {d.name : p for d in sub_dir_path_list for p in (d / "func").iterdir()}


    for sub_name in sub_anat_nii_path_dict.keys():
        # path of saved dir, each subject has its own dir
        saved_dir_path = Paths.run_files_ds002748_dir_path / sub_name
        saved_dir_path.mkdir(parents=True, exist_ok=True)
       
        # Save participant's information
        info_json_path = saved_dir_path / "info.json"
        if not info_json_path.exists():
            subject_info = participants_info[participants_info['participant_id'] == sub_name]  
            subject_info = subject_info.iloc[0].to_dict()
            with open(info_json_path, 'w') as f:
                json.dump(subject_info, f, indent=4)

        # Step 1: Register
        # anat: anat -> atlas 
        aligned_anat_path = saved_dir_path / "aligned_anat.nii.gz"
        if not aligned_anat_path.exists():
            print(f"Registering {sub_name}: anat")
            result = register(fixed_path=str(Paths.BN_Atlas_246_1mm_nii_path), moving_path=str(sub_anat_nii_path_dict[sub_name]))
            ants.image_write(result, str(aligned_anat_path))
            del result
        # func: func -> anat
        aligned_func_path = saved_dir_path / "aligned_func.nii.gz"
        if not aligned_func_path.exists():
            moving_image = ants.image_read(str(sub_func_nii_path_dict[sub_name]))
            results_list = [] 
            for t in tqdm(range(moving_image.shape[-1]), desc=f"Registering {sub_name}: func", leave=True):
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
        # target dim=[4 182 218 182  95   1   1   1]
        adjust_dim_of_antsImage(aligned_func_path)

        # Clear the temporary files of ants
        temp_dir_path = Path("C:") / "Users" / "26036" / "AppData"/ "Local" / "Temp"
        assert temp_dir_path.exists(), f"{temp_dir_path.absolute()} does not exist!"
        [file_path.unlink(missing_ok=True) for suffix in ["*.nii.gz", "*.mat"] for file_path in temp_dir_path.glob(suffix)]
        
        # Step 2: Denoise
        # anat 
        denoised_anat_path = saved_dir_path / "denoised_anat.nii.gz"
        if not denoised_anat_path.exists():
            print(f"Denoising {sub_name}: anat")
            result = ants.denoise_image(image=ants.image_read(str(aligned_anat_path)))
            ants.image_write(result, str(denoised_anat_path))
            del result
        # func
        denoised_func_path = saved_dir_path / "denoised_func.nii.gz"
        if not denoised_func_path.exists():
            aligned_func = ants.image_read(str(aligned_func_path))
            results_list = [] 
            for t in tqdm(range(aligned_func.shape[-1]), desc=f"Denoising {sub_name}: func", leave=True):
                aligned_func_t = aligned_func[:, :, :, t:t+1]
                temp_path = split_4dantsImageFrame_into_3d(image_4d=aligned_func_t)
                result = ants.denoise_image(ants.image_read(str(temp_path)))
                results_list.append(result)
                # delete temporary file and variables
                Path(temp_path).unlink()
                del result, aligned_func_t, temp_path
            ants.image_write(ants.merge_channels(results_list, channels_first=True), str(denoised_func_path))
            del aligned_func, results_list
        # target dim=[4 182 218 182  95   1   1   1]
        adjust_dim_of_antsImage(denoised_func_path)

        # Step 3: Functional connectivity
        fc_matrix_path = saved_dir_path / "fc_matrix.npy"
        if not fc_matrix_path.exists():
            start_time = time.time()
            atlas = nib.load(Paths.BN_Atlas_246_1mm_nii_path)
            denoised_func = nib.load(denoised_func_path)
            masker = maskers.NiftiLabelsMasker(labels_img=atlas, standardize="zscore_sample")
            time_series = masker.fit_transform(denoised_func)
            matrix = connectome.ConnectivityMeasure(kind="correlation", standardize="zscore_sample").fit_transform([time_series])[0]
            np.save(fc_matrix_path, matrix)
            # plot heatmap
            plt.imshow(matrix, cmap="RdBu_r", interpolation="nearest")  
            plt.colorbar()  
            plt.title("Heatmap")  
            plt.xlabel("X-axis")  
            plt.ylabel("Y-axis")  
            plt.savefig(saved_dir_path / "correlation_matrix.svg", format="svg")   
            end_time = time.time()
            print(f"It took {end_time - start_time:.2f} seconds to calculate the functional connectivity matrix.")


# def process_ds003007() -> None:
#     """

#     """
#     assert Paths.ds003007_dir_path.exists(), f"{Paths.ds003007_dir_path.absolute()} does not exist!"
#     participants_tsv_path = Paths.ds003007_dir_path / "participants.tsv"
#     assert participants_tsv_path.exists(), f"{participants_tsv_path.absolute()} does not exist!"
#     participants_info = pd.read_csv(participants_tsv_path, sep='\t')
#     sub_dir_path_list = [d for d in Paths.ds003007_dir_path.iterdir() if d.is_dir()]
#     assert len(sub_dir_path_list) == len(participants_info), f"len(sub_dir_path_list)={len(sub_dir_path_list)} != len(participants_info)={len(participants_info)}"

process_ds002748()