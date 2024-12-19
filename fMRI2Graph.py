import ants
import numpy as np
import pandas as pd 
import nibabel as nib 
import SimpleITK as sitk  
from tqdm import tqdm
from nilearn import maskers, connectome, image

from path import Path, Paths

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

    # atlas_image = nib.load(Paths.BN_Atlas_246_1mm_nii_path)
    # atlas_image = sitk.ReadImage(Paths.BN_Atlas_246_1mm_nii_path, sitk.sitkFloat32)
    # masker = maskers.NiftiLabelsMasker(labels_img=atlas_image, standardize="zscore_sample")

    for sub_name in tqdm(sub_anat_nii_path_dict.keys(), desc="processing ds002748", leave=True):
        anat_nii_path = sub_anat_nii_path_dict[sub_name]
        func_nii_path = sub_func_nii_path_dict[sub_name]

        fixed_image = ants.image_read(str(Paths.BN_Atlas_246_1mm_nii_path))
        moving_image = ants.image_read(str(anat_nii_path))
        moving_image = ants.resample_image(moving_image, fixed_image.shape, 1, 0)
        result = ants.registration(fixed_image, moving_image, type_of_transform = 'SyN')
        ants.image_write(result["warpedmovout"], "aligned_anat_image.nii.gz")
        
        func = ants.image_read(str(func_nii_path)) # TODO 去除头骨
        anat = ants.resample_image(moving_image, func.shape[:-1], 1, 0)
        nib.save(anat, "111_aligned_anat_image.nii.gz")
        # moving_image = ants.image_read(str(func_nii_path))
        # results = [] 
        # for t in range(moving_image.shape[-1]):
        #     if t < 5: # delete the first 5 time frames
        #         continue
        #     print(f"processing time frame {t}/{moving_image.shape[-1]}")
        #     moving_image_t = moving_image[..., t]
        #     moving_image_t = ants.resample_image(moving_image_t, fixed_image.shape, 1, 0)
        #     result = ants.registration(fixed_image, moving_image_t, type_of_transform = 'SyN')
        #     results.append(result["warpedmovout"])
        #     ants.image_write(result["warpedmovout"], f"aligned_func_image_{t}.nii.gz")
        # ants.image_write(ants.merge_channels(results), "aligned_func_image.nii.gz")
        exit()

    #     anat_image = sitk.ReadImage(anat_nii_path, sitk.sitkFloat32)
    #     # anat_image = nib.load(anat_nii_path)
    #     func_image = sitk.ReadImage(func_nii_path, sitk.sitkFloat32)
    #     # func_image = nib.load(func_nii_path)

    #    # 设置配准的初始参数  
    #     initial_transform = sitk.CenteredTransformInitializer(  
    #         atlas_image,  
    #         anat_image,  
    #         sitk.Euler3DTransform(),  # 可以选择其他类型的变换  
    #     )  
    #     # 选择配准方法  
    #     registration_method = sitk.ImageRegistrationMethod()  
    #     # 设定相似性度量  
    #     registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)  
    #     # 设定优化器  
    #     # registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)  
    #     registration_method.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=1000)  
    #     # 执行配准  
    #     registration_method.SetInitialTransform(initial_transform)  
    #     final_transform = registration_method.Execute(atlas_image, anat_image)  

    #     # 应用配准变换到anat图像  
    #     resampled_anat_image = sitk.Resample(anat_image, atlas_image, final_transform, sitk.sitkLinear, 0.0, anat_image.GetPixelID())  
    #     # 保存配准后的图像  
    #     sitk.WriteImage(resampled_anat_image, 'aligned_anat_image.nii.gz')


    #     # TODO 删除掉func_image的前5个时间帧，与anat_image对齐
    #     print()
    #     num_frames = func_image.GetSize()[-1]
    #     print(num_frames)

    #     exit()
    #     registered_func_image = sitk.Resample(trimmed_func_image, atlas_image, final_transform, sitk.sitkLinear, 0.0, trimmed_func_image.GetPixelID())  
    #     # 保存配准后的功能影像  
    #     registered_func_path = f'registered_func_image_{sub_name}.nii.gz'  
        exit()

def process_ds003007() -> None:
    """

    """
    assert Paths.ds003007_dir_path.exists(), f"{Paths.ds003007_dir_path.absolute()} does not exist!"
    participants_tsv_path = Paths.ds003007_dir_path / "participants.tsv"
    assert participants_tsv_path.exists(), f"{participants_tsv_path.absolute()} does not exist!"
    participants_info = pd.read_csv(participants_tsv_path, sep='\t')
    sub_dir_path_list = [d for d in Paths.ds003007_dir_path.iterdir() if d.is_dir()]
    assert len(sub_dir_path_list) == len(participants_info), f"len(sub_dir_path_list)={len(sub_dir_path_list)} != len(participants_info)={len(participants_info)}"

process_ds002748()