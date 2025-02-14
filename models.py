import torch
import numpy as np
import torch.nn as nn  
import nibabel as nib
from scipy.io import loadmat  
from functools import partial

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
    
    """
    def __init__(self, brain_network_name : str) -> None:
        super().__init__()
        self.atlas_labels_dict = load_atlas()
        self.brain_network_name = brain_network_name
    
    def subgraph(self, input_dict : dict[str, torch.Tensor]) -> None:
        """
        whole graph = whole brain = whole matrix, subgraph = a brain network = sub matrix
        extract the strict upper triangular matrix as embedding
        """
        def __get_triu__(tensor : torch.Tensor):
            assert tensor.dim() == 3, f"Tensor must be 3-dimensional, but got {tensor.dim()}-dimensional tensor"
            _, matrix_size, _ = tensor.shape
            mask = torch.triu(torch.ones(matrix_size, matrix_size, device=tensor.device), diagonal=1).bool()
            return tensor[:, mask]
        
        output_dict = {}
        if self.brain_network_name == "DMN":
            # TODO 先提取子矩阵出来
            pass
        elif self.brain_network_name == "CEN":
            pass
        elif self.brain_network_name == "SN":
            pass
        elif self.brain_network_name is None:
            output_dict = {key:__get_triu__(tensor) for key, tensor in input_dict.items()}
        else:
            raise ValueError(f"Unknown brain network name: {self.brain_network_name}")
        return output_dict

    def forward(self, input_dict : dict[str, torch.Tensor]) -> torch.Tensor:
        output_dict = self.subgraph(input_dict=input_dict)
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
    def __init__(self, num_timesteps : int = 1000,
                       ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        betas = self.__make_beta_schedule__(schedule="linear", num_timesteps=num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        to_torch = partial(torch.tensor, dtype=torch.float32)
        # Cumulative Noise Decay Factor
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # Cumulative Noise Growth Factor
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))

    def __make_beta_schedule__(self, schedule : str, num_timesteps : int, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):  
        """  
        Creates a beta schedule for the diffusion process based on the specified schedule type.  

        In diffusion models, the beta schedule determines the amount of noise added at each timestep  
        during the forward process. The schedule controls how the data transitions from the original  
        distribution to pure noise over the course of `num_timesteps`.  

        Args:  
            schedule (str): The type of beta schedule to use. Supported values are:  
                - "linear": Linear interpolation between `linear_start` and `linear_end`.  
                - "cosine": Non-linear schedule based on the cosine function.  
                - "sqrt_linear": Linearly spaced values between `linear_start` and `linear_end`.  
                - "sqrt": Square root of linearly spaced values between `linear_start` and `linear_end`.  
            linear_start (float, optional): The starting value for the linear schedule. Defaults to 1e-4.  
            linear_end (float, optional): The ending value for the linear schedule. Defaults to 2e-2.  
            cosine_s (float, optional): A small offset added to the cosine schedule to avoid division by zero.  
                Defaults to 8e-3.  

        Returns:  
            numpy.ndarray: A 1D array of beta values, with length equal to `num_timesteps`.  

        Raises:  
            ValueError: If the provided `schedule` is not one of the supported types.  

        Notes:  
            - The beta values are used to define the noise variance at each timestep in the forward process.  
            - Different schedules affect the diffusion dynamics and may impact the quality of the generated samples.  
        """  

        if schedule == "linear":  
            # Linear beta schedule: Interpolates between `linear_start` and `linear_end` in a squared space.  
            # The square root of the start and end values is used for interpolation to ensure a smoother transition.  
            betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2  
        elif schedule == "cosine":  
            # Cosine beta schedule: A non-linear schedule based on the cosine function.  
            # This schedule is designed to improve the signal-to-noise ratio in the diffusion process.  
            # Normalize timesteps to the range [0, 1] and add a small offset `cosine_s` to prevent division by zero.  
            timesteps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps + cosine_s  
            # Compute alpha values (the proportion of signal retained at each timestep) using a cosine function.  
            alphas = timesteps / (1 + cosine_s) * np.pi / 2  
            alphas = torch.cos(alphas).pow(2)  # Square the cosine values to get alpha values.  
            # Normalize alpha values so that the first timestep has an alpha value of 1.  
            alphas = alphas / alphas[0]  
            # Compute beta values as the difference in alpha values between consecutive timesteps.  
            betas = 1 - alphas[1:] / alphas[:-1]  
            # Clip beta values to ensure they remain within a valid range [0, 0.999].  
            betas = np.clip(betas, a_min=0, a_max=0.999)  
        elif schedule == "sqrt_linear":  
            # Square-root linear beta schedule: Linearly interpolate between `linear_start` and `linear_end`.  
            betas = torch.linspace(linear_start, linear_end, num_timesteps, dtype=torch.float32)  
        elif schedule == "sqrt":  
            # Square-root beta schedule: Interpolate linearly and then take the square root of the values.  
            # This results in a slower growth of beta values compared to the linear schedule.  
            betas = torch.linspace(linear_start, linear_end, num_timesteps, dtype=torch.float32) ** 0.5  
        else:  
            raise ValueError(f"schedule '{schedule}' unknown.")  
        return betas.numpy()

    def __extract_into_tensor__(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:  
        """  
        Extracts values from a tensor `a` based on indices `t` and reshapes the result to match the dimensions of `x_shape`.  

        This function is typically used in diffusion models to extract time-step-dependent values (e.g., alpha or beta values)  
        and broadcast them to match the shape of the input tensor for further computations.  

        Args:  
            a (torch.Tensor): A 1D tensor containing values (e.g., time-step-dependent values like alpha or beta).  
                              The size of `a` should be at least as large as the maximum value in `t`.  
            t (torch.Tensor): A 1D tensor of indices specifying which values to extract from `a`.  
                              The size of `t` is typically equal to the batch size.  
            x_shape (torch.Size): The target shape of the output tensor. The number of dimensions in `x_shape` determines  
                                  how the extracted values are reshaped.  

        Returns:  
            torch.Tensor: A tensor of shape `(t.shape[0], 1, 1, ..., 1)` (same number of dimensions as `x_shape`),  
                          where the first dimension corresponds to the batch size, and all other dimensions are singleton  
                          dimensions (1). This allows the tensor to be broadcasted to match `x_shape`.  

        Example:  
            Suppose `a` is a tensor of time-step-dependent values, `t` is a batch of time-step indices, and `x_shape`  
            is the shape of the input tensor. This function extracts the values from `a` corresponding to the indices in `t`  
            and reshapes them to be broadcast-compatible with `x_shape`.  

            a = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Time-step-dependent values  
            t = torch.tensor([0, 2, 3])            # Batch of time-step indices  
            x_shape = torch.Size([3, 64, 64, 3])   # Target shape (e.g., input tensor shape)  

            result = __extract_into_tensor__(a, t, x_shape)  
            # result will have shape (3, 1, 1, 1), allowing it to be broadcasted to (3, 64, 64, 3).  

        Raises:  
            IndexError: If any value in `t` is out of bounds for the size of `a`.  
        """  
        # Gather values from `a` using indices `t` along the last dimension (-1).  
        # This extracts the values from `a` corresponding to the indices in `t`.  
        gathered = a.gather(-1, t)  
        # Reshape the gathered values to match the target shape.  
        # The first dimension is the batch size (t.shape[0]), and all other dimensions are set to 1.  
        # This ensures the output tensor is broadcast-compatible with `x_shape`.  
        reshaped = gathered.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))  
        return reshaped

    def forward_chain(self, x_0 : torch.Tensor, t : torch.Tensor, noise : torch.Tensor) -> torch.Tensor:
        """
        foward chain: perturbs data to noise,      which is called `q_sample` in `DDPM`
        """
        # x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        x_t = self.__extract_into_tensor__(a=self.sqrt_alphas_cumprod, t=t, x_shape=x_0.shape) * x_0 + self.__extract_into_tensor__(a=self.sqrt_one_minus_alphas_cumprod, t=t, x_shape=x_0.shape) * noise
        return x_t
    
    def reverse_chain(self, ) -> None:
        """
        reverse chain: converts noise back to data, which is called `p_sample` in `DDPM`
        """
        return

    def forward(self, *embedding_dicts : dict[str, torch.Tensor]) -> torch.Tensor:
        tensor = next(iter(embedding_dicts[0].values()))  
        # t: time steps in the diffusion process
        t = torch.randint(0, self.num_timesteps, (tensor.shape[0],), device=tensor.device).long()
        for embedding_dict in embedding_dicts:
            for key, tensor in embedding_dict.items():
                x_0 = tensor.clone().detach()
                noise = torch.randn_like(x_0)
                # forward chain
                x_t = self.forward_chain(x_0=x_0, t=t, noise=noise)
        
        # reverse chain
        
        # global pool: max, add, mean

        return None

class Decoder(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        # prediction
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_dict : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return input_dict
    
class MGD4MD(nn.Module):
    def __init__(self,  structural_matrices_number : int,
                        brain_network_name : str,
                ) -> None:
        super().__init__()
        # Encoders
        self.encoder_s = Encoder_Structure(matrices_number=structural_matrices_number)
        self.encoder_f = Encoder_Functional(brain_network_name=brain_network_name)
        self.encoder_i = Encoder_Information()

        # LatentGraphDiffusion
        self.lgd = LatentGraphDiffusion()

        # Decoder
        self.decoder = Decoder()

    def forward(self, structural_input_dict : dict[str, torch.Tensor],
                      functional_input_dict : dict[str, torch.Tensor],
                      information_input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # encode
        structural_embedding_dict = self.encoder_s(structural_input_dict) 
        functional_embedding_dict = self.encoder_f(functional_input_dict)
        information_embedding_dict = self.encoder_i(information_input_dict) 
        
        # latent graph diffusion
        p = self.lgd(structural_embedding_dict, functional_embedding_dict, information_embedding_dict)
        
        # decode
        p = self.decoder(p)
        return p