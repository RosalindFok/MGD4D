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
    batch_size: int = 24
    num_workers: int = 6 if platform.system() == 'Linux' else 0
    epochs: range = range(100)
    lr: float = 2e-5 # 1e-4 is not good, 2e-5 is better than 1e-5 and 5e-5
    latent_embedding_dim: int = 512
    
# Brain network
@dataclass(frozen=True)
class Brain_Network:
    whole: str = "whole"
    DMN: str = "DMN"
    CEN: str = "CEN"
    SN:  str = "SN"
