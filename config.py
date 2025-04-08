import platform
from dataclasses import dataclass

seed = 42

# Is mild/major depression
@dataclass(frozen=True)
class IS_MD:
    NO : int = 0
    IS : int = 1

# DataLoader
@dataclass(frozen=True)
class Train_Config:
    # GPU memory usage: 36.82GB / 39.38GB
    n_splits: range = range(1,6) # 5 folds
    shuffle: bool = False
    batch_size: int = 24 
    num_workers: int = 6 if platform.system() == 'Linux' else 0
    epochs: range = range(20)
    lr: float = 5e-5
    latent_embedding_dim: int = 768 
    use_lgd: bool = True # True False  

@dataclass(frozen=True)
class Train_Encoder:
    n_splits: range = range(1,6) # 5 folds
    shuffle: bool = False
    batch_size: int = 128
    num_workers: int = 6 if platform.system() == 'Linux' else 0
    epochs: range = range(30)
    lr: float = 1e-5
    latent_embedding_dim: int = 768
    
# Brain network
@dataclass(frozen=True)
class Brain_Network:
    whole: str = "whole"
    DMN: str = "DMN"
    CEN: str = "CEN"
    SN:  str = "SN"
