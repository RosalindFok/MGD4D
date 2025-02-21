import torch
import numpy as np
import torch.nn as nn  
from functools import partial
from collections import defaultdict 

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
    def __init__(self, matrices_number : int, embedding_dim : int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([ResNet.resnet26(embedding_dim=embedding_dim) for _ in range(matrices_number)]) 
        # self.embedding = nn.Sequential(
        #     nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3),
        #     nn.ReLU(),
        #     nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3),
        #     nn.ReLU(),
        #     nn.AdaptiveMaxPool3d(64),
        #     nn.Flatten(),
        #     nn.Linear(64**3, embedding_dim)
        # )
        # self.resnets = nn.ModuleList([self.embedding for _ in range(matrices_number)])

    def forward(self, input_dict : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert len(input_dict) == len(self.resnets), f"Input dictionary size ({len(input_dict)}) must match number of ResNet models ({len(self.resnets)})"
        output_dict = {}
        for (key, tensor), resnet in zip(input_dict.items(), self.resnets):
            output_dict[key] = resnet(tensor.unsqueeze(1))
        return output_dict


class Encoder_Functional(nn.Module):
    """
    
    """
    def __init__(self, functional_embeddings_shape : dict[str, int], target_dim : int) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict({
            k: self.__make_projector__(input_dim=v, output_dim=target_dim) for k, v in functional_embeddings_shape.items()
        })

    def __get_triu__(self, tensor : torch.Tensor) -> torch.Tensor:
        """
        extract the strict upper triangular matrix as embedding
        then project to the same dimension of structural embedding
        """
        assert tensor.dim() == 3, f"Tensor must be 3-dimensional, but got {tensor.dim()}-dimensional tensor"
        _, matrix_size, _ = tensor.shape
        mask = torch.triu(torch.ones(matrix_size, matrix_size, device=tensor.device), diagonal=1).bool()
        return tensor[:, mask]
    
    def __make_projector__(self, input_dim : int, output_dim : int) -> nn.Module:
        if input_dim > output_dim:
            return nn.Sequential(
                nn.Linear(input_dim, 2048),
                # nn.CELU(),
                nn.Tanh(),
                nn.Linear(2048, output_dim) 
            )
        elif input_dim == output_dim:
            return nn.Identity()
        else:
            return nn.Linear(input_dim, output_dim)
    
    def forward(self, input_dict : dict[str, torch.Tensor]) -> torch.Tensor:
        output_dict = {k:self.__get_triu__(v) for k, v in input_dict.items()}
        output_dict = {k:self.projectors[k](v) for k, v in output_dict.items()}
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
    def __init__(self, embedding_dim : int, features_number : int, num_timesteps : int = 1000) -> None:
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

        # cross attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8, batch_first=True)
        
        # score
        self.contribution_mlps  = nn.ModuleList([nn.Sequential( # TODO 一会儿简单点？去噪可能不对，看那个论文试试
                                                 nn.Linear(embedding_dim, 64),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(64, 1),
                                                 nn.Sigmoid()) for _ in range(features_number)])

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
        foward chain: perturbs data to noise,      which is called `q_sample` in `DDPM`
        """
        # x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        x_t = self.__extract_into_tensor__(a=self.sqrt_alphas_cumprod, t=t, x_shape=x_0.shape) * x_0 + self.__extract_into_tensor__(a=self.sqrt_one_minus_alphas_cumprod, t=t, x_shape=x_0.shape) * noise
        return x_t
    
    def reverse_chain(self, noisy_embeddings_dict : dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
        """
        reverse chain: converts noise back to data, which is called `p_sample` in `DDPM`
        """
        stacked_keys_list, stacked_embeddings_list = [], []
        for group in ["functional", "structural"]:
            for key, embedding in noisy_embeddings_dict[group].items():
                stacked_keys_list.append((group, key))
                stacked_embeddings_list.append(embedding)

        stacked_features = torch.stack(stacked_embeddings_list, dim=0)
        stacked_features = stacked_features.permute(1, 0, 2) # shape is [batchsize, features number, embedding_dim]

        # Cross Attention
        # attended_features.shape is [batchsize, features number, embedding_dim]
        # attention_weights.shape is [batchsize, features number, features number]
        attended_features, attention_weights = self.cross_attention(query=stacked_features, 
                                                                    key=stacked_features, 
                                                                    value=stacked_features) 
        attended_embeddings_dict = {}
        for i, combined_key in enumerate(stacked_keys_list): # combined_key = (group, key)
            attended_embeddings_dict[combined_key] = attended_features[:, i, :]
        assert len(attended_embeddings_dict) == self.features_number, f"number of attended embeddings = {len(attended_embeddings_dict)} is not equal to features number = {self.features_number}"
        
        # Score
        scores_dict = defaultdict(dict)
        for i, (group, key) in enumerate(attended_embeddings_dict.keys()):
            score = self.contribution_mlps[i](attended_embeddings_dict[(group, key)])
            scores_dict[group][key] = score
        
        # Weighted
        weighted_embeddings_dict = defaultdict(dict)
        for (group, key), embedding in attended_embeddings_dict.items():
            weighted_embeddings_dict[group][key] = embedding * scores_dict[group][key]
        
        return weighted_embeddings_dict # TODO scores_dict and attention_weights

    def forward(self, latent_embeddings_dict : dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
        tensor = next(iter(next(iter(latent_embeddings_dict.values())).values()))
        # t: time steps in the diffusion process
        t = torch.randint(0, self.num_timesteps, (tensor.shape[0],), device=tensor.device).long()
        noisy_embeddings_dict = {}
        for modal, embeddings_dict in latent_embeddings_dict.items():
            noisy_embeddings_dict[modal] = {}
            for key, x_0 in embeddings_dict.items():
                noise = torch.randn_like(x_0)
                # forward chaimn
                x_t = self.forward_chain(x_0=x_0, t=t, noise=noise)
                noisy_embeddings_dict[modal][key] = x_t
        
        # reverse chain
        output_embeddings_dict = self.reverse_chain(noisy_embeddings_dict=noisy_embeddings_dict)

        return output_embeddings_dict

class Decoder(nn.Module):
    def __init__(self, features_number : int) -> None:
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv1d(in_channels=features_number, out_channels=features_number, kernel_size=3**3),
            nn.AdaptiveAvgPool1d(256),
            nn.Flatten(),
            nn.Linear(in_features=features_number*256, out_features=512),
            # nn.CELU(),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=64),
            # nn.CELU(),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, latent_embeddings_dict : dict[str, dict[str, torch.Tensor]], 
                information_embedding_dict : dict[str, torch.Tensor]) -> torch.Tensor:
        concatenated_embeddings_list = []
        information_embedding = list(information_embedding_dict.values())
        assert len(information_embedding) == 1 #
        information_embedding = information_embedding[0]
        for group, features in latent_embeddings_dict.items():
            for key, embedding in features.items():
                concatenated_embeddings_list.append(torch.cat((embedding, information_embedding), dim=1))
        embeddings = torch.stack(concatenated_embeddings_list, dim=1) # shape=[batchsize, features_number, embedding_dim+information_dim]
        prediction = self.decode(embeddings)
        return prediction
    
class MGD4MD(nn.Module):
    def __init__(self, structural_matrices_number : int,
                 functional_embeddings_shape : dict[str, int], 
                 embedding_dim : int = 512) -> None:
        super().__init__()
        # Encoders
        self.encoder_s = Encoder_Structure(matrices_number=structural_matrices_number, embedding_dim=embedding_dim)
        self.encoder_f = Encoder_Functional(functional_embeddings_shape=functional_embeddings_shape, target_dim=embedding_dim)
        self.encoder_i = Encoder_Information()

        # LatentGraphDiffusion
        self.lgd = LatentGraphDiffusion(embedding_dim=embedding_dim, features_number=structural_matrices_number+len(functional_embeddings_shape))

        # Decoder
        self.decoder = Decoder(features_number=structural_matrices_number+len(functional_embeddings_shape))

    def forward(self, structural_input_dict : dict[str, torch.Tensor],
                      functional_input_dict : dict[str, torch.Tensor], 
                      information_input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # encode
        structural_embeddings_dict = self.encoder_s(structural_input_dict) 
        functional_embeddings_dict = self.encoder_f(functional_input_dict)
        information_embedding_dict = self.encoder_i(information_input_dict) 
        
        # latent graph diffusion
        latent_embeddings_dict = {"structural"  : structural_embeddings_dict, 
                                 "functional"   : functional_embeddings_dict}
        latent_embeddings_dict = self.lgd(latent_embeddings_dict=latent_embeddings_dict)
      
        # decode
        prediction = self.decoder(latent_embeddings_dict, information_embedding_dict)

        return prediction

class CombinedLoss(nn.modules.loss._Loss):
    def __init__(self, w_mse : float = 1.0, w_bce : float = 1.0) -> None:
        super().__init__()
        self.w_mse = w_mse
        self.w_bce = w_bce
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
    
    def forward(self, pred : torch.Tensor, true : torch.Tensor) -> torch.Tensor:
        assert pred.shape == true.shape, f"Prediction shape = {pred.shape} and true shape = {true.shape} must be equal"
        return self.w_mse * self.mse(pred, true) + self.w_bce * self.bce(pred, true)
