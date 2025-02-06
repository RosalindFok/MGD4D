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
    def __init__(self, matrices_number : int):
        super().__init__()
        self.resnet50s = nn.ModuleList([  
            ResNet.resnet50() for _ in range(matrices_number)  
        ]) 

    def forward(self, input_dict : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert len(input_dict) == len(self.resnet50s), f"Input dictionary size ({len(input_dict)}) must match number of ResNet models ({len(self.resnet50s)})"
        output_dict = {}
        for (key, tensor), resnet50 in zip(input_dict.items(), self.resnet50s):
            output_dict[key] = resnet50(tensor.unsqueeze(1))
        return output_dict