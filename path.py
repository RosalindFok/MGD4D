from pathlib import Path
from dataclasses import dataclass

dataset_dir_path = Path("..") / "dataset"
depression_dir_path = dataset_dir_path / "depression"
# Bezmaternykh DD and Melnikov ME and Savelov AA and Petrovskii ED (2021). Resting state with closed eyes for patients with depression and healthy participants. OpenNeuro. [Dataset] doi: 10.18112/openneuro.ds002748.v1.0.5
ds002748_dir_path = depression_dir_path / "ds002748"
# Bezmaternykh DD and Melnikov ME and Savelov AA and Petrovskii ED (2021). Two sessions of resting state with closed eyes for patients with depression in treatment course (NFB, CBT or No treatment groups). OpenNeuro. [Dataset] doi: 10.18112/openneuro.ds003007.v1.0.1
ds003007_dir_path = depression_dir_path / "ds003007"
# Yan CG et al. Reduced default mode network functional connectivity in patients with recurrent major depressive disorder. Proc Natl Acad Sci U S A 2019; 116(18): 9078-83.
# Chen et al. The DIRECT consortium and the REST-meta-MDD project: towards neuroimaging biomarkers of major depressive disorder, Psychoradiology, Volume 2, Issue 1, March 2022, Pages 32–42.
REST_meta_MDD_dir_path = depression_dir_path / "REST-meta-MDD"

# Brainnetome_Atlas
Brainnetome_Atlas_dir_path= Path("Brainnetome_Atlas")
BN_Atlas_246_1mm_nii_path = Brainnetome_Atlas_dir_path / "BN_Atlas_246_1mm.nii.gz"
BN_Atlas_246_2mm_nii_path = Brainnetome_Atlas_dir_path / "BN_Atlas_246_2mm.nii.gz"
BN_Atlas_246_3mm_nii_path = Brainnetome_Atlas_dir_path / "BN_Atlas_246_3mm.nii.gz"

# run_files
run_files_dir_path = Path("..") / "run_files"
run_files_ds002748_dir_path = run_files_dir_path / "ds002748"
run_files_ds003007_dir_path = run_files_dir_path / "ds003007"
run_files_ds002748_dir_path.mkdir(parents=True, exist_ok=True)
run_files_ds003007_dir_path.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class Paths:
    ds002748_dir_path: Path = ds002748_dir_path
    ds003007_dir_path: Path = ds003007_dir_path
    REST_meta_MDD_dir_path: Path = REST_meta_MDD_dir_path
    BN_Atlas_246_1mm_nii_path: Path = BN_Atlas_246_1mm_nii_path
    BN_Atlas_246_2mm_nii_path: Path = BN_Atlas_246_2mm_nii_path
    BN_Atlas_246_3mm_nii_path: Path = BN_Atlas_246_3mm_nii_path
    run_files_ds002748_dir_path: Path = run_files_ds002748_dir_path
    run_files_ds003007_dir_path: Path = run_files_ds003007_dir_path
