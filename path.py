from pathlib import Path
from dataclasses import dataclass

dataset_dir_path = Path("..") / "dataset"

@dataclass(frozen=True)
class Depression_Path:
    depression_dir_path: Path = dataset_dir_path / "depression"
    # Mild depression
    ds002748_dir_path: Path = depression_dir_path / "ds002748"
    ds003007_dir_path: Path = depression_dir_path / "ds003007"
    # Major depression
    REST_meta_MDD_dir_path: Path = depression_dir_path / "REST-meta-MDD"
    assert ds002748_dir_path.exists(), f"{ds002748_dir_path} does not exist"
    assert ds003007_dir_path.exists(), f"{ds003007_dir_path} does not exist"
    assert REST_meta_MDD_dir_path.exists(), f"{REST_meta_MDD_dir_path} does not exist"

@dataclass(frozen=True)
class Functional_Connectomes_1000_Path:
    root_dir: Path = dataset_dir_path / "1000_Functional_Connectomes"
    demographics_txt_path: Path = root_dir / "Cambridge_Buckner_part1" / "Cambridge_Buckner_demographics.txt"
    assert demographics_txt_path.exists(), f"{demographics_txt_path} does not exist"

@dataclass(frozen=True)
class Brainnetome_Atlas_Path:
    Brainnetome_Atlas_dir_path:Path = Path("Brainnetome_Atlas")
    BN_Atlas_246_1mm_nii_path:Path = Brainnetome_Atlas_dir_path / "BN_Atlas_246_1mm.nii.gz"
    BN_Atlas_246_2mm_nii_path:Path = Brainnetome_Atlas_dir_path / "BN_Atlas_246_2mm.nii.gz"
    BN_Atlas_246_3mm_nii_path:Path = Brainnetome_Atlas_dir_path / "BN_Atlas_246_3mm.nii.gz"
    BNA_adjacent_matrix_path: Path = Brainnetome_Atlas_dir_path / "BNA_matrix_binary_246x246.csv"
    assert BN_Atlas_246_1mm_nii_path.exists(), f"{BN_Atlas_246_1mm_nii_path} does not exist"
    assert BN_Atlas_246_2mm_nii_path.exists(), f"{BN_Atlas_246_2mm_nii_path} does not exist"
    assert BN_Atlas_246_3mm_nii_path.exists(), f"{BN_Atlas_246_3mm_nii_path} does not exist"
    assert BNA_adjacent_matrix_path.exists(), f"{BNA_adjacent_matrix_path} does not exist"

@dataclass(frozen=True)
class Run_Files_Path:
    run_files_dir_path = Path("..") / "run_files"
    assert run_files_dir_path.exists(), f"{run_files_dir_path} does not exist"
    run_files_ds002748_dir_path = run_files_dir_path / "ds002748"
    run_files_ds003007_dir_path = run_files_dir_path / "ds003007"
    run_files_rest_meta_mdd_dir_path = run_files_dir_path / "REST_meta_MDD"
    run_files_cambridge_dir_path = run_files_dir_path / "Cambridge"
    run_files_ds002748_dir_path.mkdir(parents=True, exist_ok=True)
    run_files_ds003007_dir_path.mkdir(parents=True, exist_ok=True)
    run_files_rest_meta_mdd_dir_path.mkdir(parents=True, exist_ok=True)
    run_files_cambridge_dir_path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Paths:
    Depression: Depression_Path = Depression_Path  
    Functional_Connectomes_1000: Functional_Connectomes_1000_Path = Functional_Connectomes_1000_Path
    Brainnetome_Atlas: Brainnetome_Atlas_Path = Brainnetome_Atlas_Path
    Run_Files: Run_Files_Path = Run_Files_Path
