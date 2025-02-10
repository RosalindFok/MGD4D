import torch
import torch.nn as nn  
import nibabel as nib
from scipy.io import loadmat  

import ResNet
from path import Paths
from plot import draw_atlas

def _setup_device_() -> list[torch.device]:
    """
    """
    if torch.cuda.is_available():
        torch.cuda.init()  
        device_count = torch.cuda.device_count()  
        print(f'Number of GPUs available: {device_count}')  
        devices = []  # List to hold all available devices  
        for device_id in range(device_count):  
            device = torch.device(f'cuda:{device_id}')  
            devices.append(device)  
            device_name = torch.cuda.get_device_name(device_id)  
            print(f'Device {device_id}: {device_name}')  
        torch.cuda.set_device(devices[0])
    else:
        devices = [torch.device('cpu')]
        print('Device: CPU, no CUDA device available') 
    return devices  

devices = _setup_device_()
device = devices[0]
devices_num = len(devices)

def get_GPU_memory_usage() -> tuple[float, float]:
    if torch.cuda.is_available():  
        current_device = torch.cuda.current_device()  
        mem_reserved = torch.cuda.memory_reserved(current_device) / (1024 ** 3)    # GB  
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)  # GB  
        return total_memory, mem_reserved
    
class Encoder_Structure(nn.Module):
    """
    ResNet 3D: 3D structural matrices -> 1D embedding
    """
    def __init__(self, matrices_number : int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([  
            ResNet.resnet26() for _ in range(matrices_number)  
        ]) 

    def forward(self, input_dict : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert len(input_dict) == len(self.resnets), f"Input dictionary size ({len(input_dict)}) must match number of ResNet models ({len(self.resnets)})"
        output_dict = {}
        for (key, tensor), resnet in zip(input_dict.items(), self.resnets):
            output_dict[key] = resnet(tensor.unsqueeze(1))
        return output_dict

def load_atlas() -> dict[str, list[str]]:
        """
        """
        # download DPABI from https://d.rnet.co/DPABI/DPABI_V8.2_240510.zip
        # Unzip the zip file, extract some files from the folder "Templates" and put them under downloaded_atlas_dir_path
        aal_mri = Paths.Atlas.AAL.mri_path
        aal_labels = Paths.Atlas.AAL.labels_path
        hoc_mri = Paths.Atlas.HarvardOxford.cort_mri_path
        hoc_labels = Paths.Atlas.HarvardOxford.cort_labels_path
        hos_mri = Paths.Atlas.HarvardOxford.sub_mri_path
        hos_labels = Paths.Atlas.HarvardOxford.sub_labels_path
        
        atlas_labels_dict = {}
        for name, mri_path, labels_path in zip(["AAL", "HOC", "HOS"], [aal_mri, hoc_mri, hos_mri], [aal_labels, hoc_labels, hos_labels]):
            labels = loadmat(labels_path)["Reference"]
            labels = [str(label[0][0]) for label in labels]
            # the first one is None, which is not used in REST-meta-MDD
            assert labels[0] == "None", f"First label is not None: {labels[0]}"
            labels = labels[1:]
            atlas_labels_dict[name] = {index:label for index, label in enumerate(labels)}
            # plot atlas
            fig_path = Paths.Fig_Dir / f"{name}.png"
            if not fig_path.exists():
                draw_atlas(atlas=nib.load(mri_path), saved_path=fig_path)
        return atlas_labels_dict

class Encoder_Functional(nn.Module):
    """
    upper triangle of functional connectivity matrices
    """
    def __init__(self, ) -> None:
        super().__init__()
        self.atlas_labels_dict = load_atlas()
    
    def subgraph(self, brain_network_name : str) -> None:
        return

    def forward(self, input_dict : dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        output_dict = {k:v["embedding"] for k,v in input_dict.items()}
        return output_dict

class Encoder_Information(nn.Module):
    """
    Concatenate: {'feat1+feat2+...': concatenated_tensor}  
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_dict : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {'+'.join(list(input_dict.keys())) : torch.cat(list(input_dict.values()), dim=1)}
    
class LatentGraphDiffusion(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(13629, 2**10),
            nn.Tanh(),
            nn.Linear(2**10, 2**6),
            nn.Tanh(),
            nn.Linear(2**6, 1)
        )

    def forward(self, *embedding_dicts : dict[str, torch.Tensor]) -> torch.Tensor:
        flattened_tensor = []
        for embedding_dict in embedding_dicts:
            for key, tensor in embedding_dict.items():
                flattened_tensor.append(tensor)
        flattened_tensor = torch.cat(flattened_tensor, dim=1)
        return self.mlp(flattened_tensor)
    
class MGD4MD(nn.Module):
    def __init__(self, structural_matrices_number : int,
                ) -> None:
        super().__init__()
        # Encoders
        self.encoder_s = Encoder_Structure(matrices_number=structural_matrices_number)
        self.encoder_f = Encoder_Functional()
        self.encoder_i = Encoder_Information()

        # LatentGraphDiffusion
        self.lgd = LatentGraphDiffusion()

        # prediction
        self.sigmoid = nn.Sigmoid()

    def forward(self, structural_input_dict : dict[str, torch.Tensor],
                      functional_input_dict : dict[str, torch.Tensor],
                      information_input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # encode
        structural_embedding_dict = self.encoder_s(structural_input_dict) 
        functional_embedding_dict = self.encoder_f(functional_input_dict)
        information_embedding_dict = self.encoder_i(information_input_dict) 
        
        # latent graph diffusion
        p = self.lgd(structural_embedding_dict, functional_embedding_dict, information_embedding_dict)
        p = self.sigmoid(p)
        return p