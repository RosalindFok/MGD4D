import platform
from dataclasses import dataclass

seed = 42

# Is mild/major depression
@dataclass(frozen=True)
class IS_MD:
    NO : int = 0
    IS : int = 1

@dataclass(frozen=True)
class Gender:
    MALE : int = 1
    FEMALE : int = 2
    UNSPECIFIED : int = 0

# Ablation settings
set_use_modal = "sf"
set_use_lgd = True

@dataclass(frozen=True)
class Basic_Config:
    # for Dataloader
    batch_size: int
    # for training
    epochs: range
    lr: float
    latent_embedding_dim: int
    use_batchnorm: bool
    # for ablation
    use_lgd: bool  # True False
    use_modal: str # s-structural f-functional sf-structural+functional
    info: str # description

# REST_meta_MDD   
Major_Depression_Config = Basic_Config(
    batch_size=24,
    epochs=range(20),
    lr=5e-5,
    latent_embedding_dim=768,
    use_batchnorm=True,
    use_lgd=set_use_lgd, 
    use_modal=set_use_modal.lower(),
    info="major"
)

# Cambridge ds002748 ds003007
Mild_Depression_Config = Basic_Config(
    batch_size=16,
    epochs=range(20),
    lr=5e-5,
    latent_embedding_dim=1024,
    use_batchnorm=False,
    use_lgd=set_use_lgd,
    use_modal=set_use_modal.lower(),
    info="mild"
)

# Hyperparameters
@dataclass(frozen=True)
class Configs:
    # Dataloader
    n_splits: range = range(1,6) # 5 folds
    shuffle: bool = False
    num_workers: int = 6 if platform.system() == 'Linux' else 0
    
    # different datasets have different configs
    # Major Depression: GPU memory usage: 36.82GB / 39.38GB
    # dataset: Basic_Config = Major_Depression_Config

    # Mild Depression: GPU memory usage: 37.17GB / 39.38GB
    dataset: Basic_Config = Mild_Depression_Config