import torch
import numpy as np
import torch.nn as nn  
from typing import Any
from functools import partial
from collections import defaultdict 

import ResNet
from UNet import UNet

def _setup_device_() -> list[torch.device]:
    """  
    Sets up the available devices (GPUs or CPU) for PyTorch operations.  

    This function checks if CUDA (NVIDIA GPUs) is available on the system. If CUDA is available,   
    it initializes the CUDA environment, retrieves the number of available GPUs, and creates a   
    list of `torch.device` objects for each GPU. If no GPUs are available, it defaults to using   
    the CPU.  

    Returns:  
        list[torch.device]: A list of `torch.device` objects representing the available devices.  
                            If CUDA is not available, the list contains only the CPU device.  
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
        mem_reserved = torch.cuda.memory_reserved(current_device) / (1024 ** 3)                     # GB  
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)  # GB  
        return total_memory, mem_reserved

class Encoder_Structure(nn.Module):
    """
    ResNet 3D: 3D structural matrices -> 1D embedding
    """
    def __init__(self, matrices_shape : dict[str, torch.Size], embedding_dim : int) -> None:
        super().__init__()
        self.resnets = nn.ModuleDict({k : ResNet.ResNet3D(embedding_dim=embedding_dim) for k in matrices_shape}) 

    def forward(self, input_dict : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert len(input_dict) == len(self.resnets), f"Input dictionary size ({len(input_dict)}) must match number of ResNet models ({len(self.resnets)})"
        output_dict = {}
        for key, tensor in input_dict.items():
            output_dict[key] = self.resnets[key](tensor.unsqueeze(1))
        return output_dict


class Encoder_Functional(nn.Module):
    """
    Embedding Layer + Project Layer: 2D functional matrices -> 1D embedding
    """
    def __init__(self, functional_embeddings_shape : dict[str, int], target_dim : int) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict({
            k: self.__make_projector__(input_dim=v, output_dim=target_dim) for k, v in functional_embeddings_shape.items()
        })

    def __get_triu__(self, tensor : torch.Tensor) -> torch.Tensor:
        """
        extract the strict upper triangular matrix as embedding
        """
        assert tensor.dim() == 3, f"Tensor must be 3-dimensional, but got {tensor.dim()}-dimensional tensor"
        _, matrix_size, _ = tensor.shape
        mask = torch.triu(torch.ones(matrix_size, matrix_size, device=tensor.device), diagonal=1).bool()
        return tensor[:, mask]
    
    def __make_projector__(self, input_dim : int, output_dim : int) -> nn.Module:
        """
        project to the same dimension of structural embedding
        """
        activation = nn.Tanh() # Tanh Softsign
        if input_dim > output_dim:
            return nn.Sequential(
                nn.Linear(input_dim, 2048),
                nn.BatchNorm1d(2048),
                activation,
                nn.Linear(2048, output_dim),
                nn.BatchNorm1d(output_dim)
            )
        elif input_dim == output_dim:
            return nn.Sequential(
                nn.Identity(),
                nn.BatchNorm1d(output_dim)
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim)
            )
    
    def forward(self, input_dict : dict[str, torch.Tensor]) -> torch.Tensor:
        output_dict = {k:self.__get_triu__(v) for k, v in input_dict.items()}
        output_dict = {k:self.projectors[k](v) for k, v in output_dict.items()}
        return output_dict


class Encoder_Information(nn.Module):
    """
    Concatenate: {'feat1+feat2+...': concatenated_tensor}  
    """
    def __init__(self, info_dict : dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.keys = list(info_dict.keys())
    def forward(self, input_dict : dict[str, torch.Tensor]) -> torch.Tensor:
        embedding = torch.stack([input_dict[k] for k in self.keys], dim=1)
        return {''.join(self.keys) : embedding}

class LatentGraphDiffusion(nn.Module):
    def __init__(self, embedding_dim : int, idx_modal_key : dict[str, dict[str, Any]], 
                       features_number : int, num_timesteps : int = 100) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.features_number = features_number
        betas = self.__make_beta_schedule__(schedule="linear", num_timesteps=num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        to_torch = partial(torch.tensor, dtype=torch.float32)
        # cumulative Noise Decay Factor
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # Cumulative Noise Growth Factor
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.idx_modal_key = idx_modal_key

        # Q K V
        self.QKV = nn.ModuleDict({
            key : nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first=True, activation="gelu"),
                    num_layers=6
            ) for key in ["Q", "K", "V"]
        })

        # multi-head attention
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=16, batch_first=True, dropout=0.1)
        
        # score
        self.half_dim = embedding_dim // 2
        self.contribution_mlps = nn.ModuleDict({  
            f"{modal}-{key}" : nn.Sequential(
                nn.Linear(in_features=features_number, out_features=1),  
                nn.Sigmoid()  
            ) for _, (modal, key) in idx_modal_key.items()  
        })

        # time
        self.time_mlp = nn.Sequential(  
            nn.Linear(embedding_dim, embedding_dim * 4),  
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  
        )  

        # U-Net
        self.unet = UNet()

    def __make_beta_schedule__(self, schedule : str, num_timesteps : int, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3) -> np.ndarray:  
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
        foward chain: perturbs data to noise, which is called `q_sample` in `DDPM`
        """
        # x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        sqrt_alphas_cumprod_t = self.__extract_into_tensor__(a=self.sqrt_alphas_cumprod, t=t, x_shape=x_0.shape)
        sqrt_one_minus_alphas_cumprod_t  = self.__extract_into_tensor__(a=self.sqrt_one_minus_alphas_cumprod, t=t, x_shape=x_0.shape)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t

    def reverse_chain(self, noisy_embeddings_dict : dict[str, dict[str, torch.Tensor]], t : torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        """
        reverse chain: converts noise back to data, which is called `p_sample` in `DDPM`
        """
        stacked_embeddings_list = [0] * len(self.idx_modal_key)
        for idx, (modal, key) in self.idx_modal_key.items():
            stacked_embeddings_list[idx] = noisy_embeddings_dict[modal][key]

        stacked_features = torch.stack(stacked_embeddings_list, dim=0)
        stacked_features = stacked_features.permute(1, 0, 2) # shape is [batchsize, features number, embedding_dim]

        # Add time embedding  
        t = t.float().unsqueeze(1) # shape: [batchsize, 1]
        t = t * torch.exp(-torch.arange(0, self.half_dim, dtype=torch.float32, device=t.device) * (torch.log(torch.tensor(1e4, dtype=torch.float32, device=t.device)) / (self.half_dim - 1)))
        t = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)  # Shape: [batch_size, embedding_dim] 
        time_embedding = self.time_mlp(t).unsqueeze(1)  # shape: [batchsize, 1, embedding_dim]  
        stacked_features = stacked_features + time_embedding  

        # Q K V
        Q = self.QKV["Q"](stacked_features)
        K = self.QKV["K"](stacked_features)
        V = self.QKV["V"](stacked_features)

        # multi-head attention
        # attended_features.shape is [batchsize, features number, embedding_dim]
        # attention_weights.shape is [batchsize, features number, features number]
        attended_features, attention_weights = self.multihead_attention(query=Q, key=K, value=V) 
        
        # U-Net
        # attended_features = self.unet(stacked_features) # ablation of Transformer+multihead_attention
        attended_features = self.unet(attended_features)

        # attended_embeddings_dict = {}
        attended_embeddings_dict = defaultdict(dict)
        for idx, (modal, key) in self.idx_modal_key.items():
            attended_embeddings_dict[modal][key] = attended_features[:, idx, :]
        
        # Score
        scores_dict = defaultdict(dict)
        for idx, (modal, key) in self.idx_modal_key.items():
            score = self.contribution_mlps[f"{modal}-{key}"](attention_weights[:, idx])
            scores_dict[modal][key] = score
        
        # Weighted
        weighted_embeddings_dict = defaultdict(dict)
        for _, (modal, key) in self.idx_modal_key.items():
            weighted_embeddings_dict[modal][key] = attended_embeddings_dict[modal][key] * scores_dict[modal][key]
        
        return weighted_embeddings_dict # TODO scores_dict and attention_weights

    def forward(self, latent_embeddings_dict : dict[str, dict[str, torch.Tensor]]) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
        tensor = next(iter(next(iter(latent_embeddings_dict.values())).values()))
        # t: time steps in the diffusion process
        # randomly generate a time step t, i.e., directly sample the t-th step of T steps; there is no need to use 'for t in range(T)' to accumulate.
        t = torch.randint(0, self.num_timesteps, (tensor.shape[0],), device=tensor.device).long()
        noisy_embeddings_dict = defaultdict(dict)
        noise_dict = defaultdict(dict)
        for modal, embeddings_dict in latent_embeddings_dict.items():
            for key, x_0 in embeddings_dict.items():
                noise = torch.randn_like(x_0)
                # forward chain
                x_t = self.forward_chain(x_0=x_0, t=t, noise=noise)
                noisy_embeddings_dict[modal][key] = x_t
                noise_dict[modal][key] = noise
        
        # reverse chain
        output_embeddings_dict = self.reverse_chain(noisy_embeddings_dict=noisy_embeddings_dict, t=t)

        return output_embeddings_dict

class Decoder(nn.Module):
    def __init__(self, auxi_info_number : int, features_number : int, embedding_dim : int, idx_modal_key : dict[int, tuple[str, str]]) -> None:
        super().__init__()
        activation = nn.Tanh()

        self.decode_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=features_number*embedding_dim, out_features=embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            activation,
        )
        
        self.decode_2 = nn.Sequential(
            nn.Linear(in_features=embedding_dim+auxi_info_number, out_features=64),
            nn.BatchNorm1d(64),
            activation, 
            nn.Linear(in_features=64, out_features=2),
        )

        self.idx_modal_key = idx_modal_key

    def forward(self, latent_embeddings_dict : dict[str, dict[str, torch.Tensor]], 
                information_embedding_dict : dict[str, torch.Tensor]) -> torch.Tensor:
        concatenated_embeddings_list = []
        information_embedding = list(information_embedding_dict.values())
        assert len(information_embedding) == 1 #
        information_embedding = information_embedding[0]

        for idx, (modal, key) in self.idx_modal_key.items():
            concatenated_embeddings_list.append(latent_embeddings_dict[modal][key])
        embeddings = torch.stack(concatenated_embeddings_list, dim=1) # shape=[batchsize, features_number, embedding_dim]
        embeddings = self.decode_1(embeddings)
        embeddings = torch.cat((embeddings, information_embedding), dim=1)
        output = self.decode_2(embeddings)

        return output
    
class MGD4MD(nn.Module):
    def __init__(self, info_dict : dict[str, torch.Tensor], shapes_dict : dict[str, dict[str, Any]], 
                 embedding_dim : int = 512, use_lgd : bool = True) -> None:
        super().__init__()
        self.use_lgd = use_lgd
        # index of modal-key
        self.idx_modal_key = {}
        idx = 0
        for modal, keys in shapes_dict.items():
            for key in keys:
                self.idx_modal_key[idx] = (modal, key)
                idx += 1

        # Encoders
        self.encoder_s = Encoder_Structure(matrices_shape=shapes_dict["s"], embedding_dim=embedding_dim)
        self.encoder_f = Encoder_Functional(functional_embeddings_shape=shapes_dict["f"], target_dim=embedding_dim)
        self.encoder_i = Encoder_Information(info_dict=info_dict)

        # LatentGraphDiffusion
        if self.use_lgd:
            self.lgd = LatentGraphDiffusion(embedding_dim=embedding_dim, idx_modal_key=self.idx_modal_key,
                                            features_number=len(shapes_dict["s"])+len(shapes_dict["f"]))

        # Decoder
        self.decoder = Decoder(auxi_info_number=len(info_dict), 
                               features_number=len(shapes_dict["s"])+len(shapes_dict["f"]),
                               embedding_dim=embedding_dim, idx_modal_key=self.idx_modal_key)

    def __clone_tensor_dict__(self, d : Any) -> Any:  
        if isinstance(d, torch.Tensor):  
            return d.clone() 
        elif isinstance(d, dict):  
            return {k: self.__clone_tensor_dict__(v) for k, v in d.items()}  
        elif isinstance(d, list):  
            return [self.__clone_tensor_dict__(v) for v in d]  
        else:  
            return d

    def forward(self, structural_input_dict : dict[str, torch.Tensor],
                      functional_input_dict : dict[str, torch.Tensor], 
                      information_input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # encode
        structural_embeddings_dict = self.encoder_s(structural_input_dict) 
        functional_embeddings_dict = self.encoder_f(functional_input_dict)
        information_embedding_dict = self.encoder_i(information_input_dict) 
        
        # structural_embeddings_dict = {k:torch.zeros_like(v, dtype=v.dtype, device=v.device) for k, v in structural_embeddings_dict.items()}
        # functional_embeddings_dict = {k:torch.zeros_like(v, dtype=v.dtype, device=v.device) for k, v in functional_embeddings_dict.items()}

        latent_embeddings_dict = {"s" : structural_embeddings_dict, "f" : functional_embeddings_dict}

        # latent graph diffusion
        if self.use_lgd:
            original_embeddings_dict = self.__clone_tensor_dict__(latent_embeddings_dict)  
            latent_embeddings_dict = self.lgd(latent_embeddings_dict=latent_embeddings_dict)
            for modal, value in latent_embeddings_dict.items():
                for key, tensor in value.items():
                    assert tensor.shape == original_embeddings_dict[modal][key].shape, f"{tensor.shape} != {original_embeddings_dict[modal][key].shape}"
                    latent_embeddings_dict[modal][key] = tensor + original_embeddings_dict[modal][key]
        
        # decode
        output = self.decoder(latent_embeddings_dict, information_embedding_dict)
       
        return output