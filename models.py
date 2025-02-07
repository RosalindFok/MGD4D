import torch
import torch.nn as nn  

import ResNet

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

class Encoder_Functional(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, input_dict : dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(list(input_dict.values()), dim=1)

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

    def forward(self, x) -> torch.Tensor:
        return x
    
class MGD4MD(nn.Module):
    def __init__(self, structural_matrices_number : int,
                ) -> None:
        super().__init__()
        # Encoders
        self.encoder_s = Encoder_Structure(matrices_number=structural_matrices_number)
        self.encoder_f = Encoder_Functional()
        self.encoder_i = Encoder_Information()

    def forward(self, structural_input_dict : dict[str, torch.Tensor],
                      functional_input_dict : dict[str, torch.Tensor],
                      information_input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # encode
        structural_embedding_dict = self.encoder_s(structural_input_dict) # 4个key，4个1维的embedding
        functional_embedding_dict = self.encoder_f(functional_input_dict)
        information_embedding_dict = self.encoder_i(information_input_dict) # 1个key，1个1维的embedding
        
        # latent graph diffusion
        return information_embedding_dict