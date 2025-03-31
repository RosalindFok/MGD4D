import platform
from dataclasses import dataclass

seed = 0

# Gender
@dataclass(frozen=True)
class Gender:
    FEMALE : int = 1
    MALE   : int = 2
    UNSPECIFIED  : int = 0

# Is mild/major depression
@dataclass(frozen=True)
class IS_MD:
    NO : int = 0
    IS : int = 1

# DataLoader
@dataclass(frozen=True)
class Train_Config:
    n_splits: range = range(1,6) # 5 folds
    shuffle: bool = False
    batch_size: int = 22 # 60 22 # 代码在dataset.py中，不在main中
    num_workers: int = 6 if platform.system() == 'Linux' else 0
    epochs: range = range(30)
    lr: float = 1e-5 # 1 1e-5; 2 3e-5; 3 2e-6
    latent_embedding_dim: int = 768 
    use_lgd: bool = True # True False

# @dataclass(frozen=True)
# class Train_Encoder:
#     n_splits: range = range(1,6) # 5 folds
#     shuffle: bool = False
#     batch_size: int = 128
#     num_workers: int = 6 if platform.system() == 'Linux' else 0
#     epochs: range = range(30)
#     lr: float = 1e-5
#     latent_embedding_dim: int = 768
    
# Brain network
@dataclass(frozen=True)
class Brain_Network:
    whole: str = "whole"
    DMN: str = "DMN"
    CEN: str = "CEN"
    SN:  str = "SN"
