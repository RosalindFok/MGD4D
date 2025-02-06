import torch

from config import Train_Config
from models import device, Encoder_Structure
from dataset import get_major_dataloader_via_fold

def move_to_device(tensor_dict : dict[str, torch.Tensor], device : torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in tensor_dict.items()}

def main() -> None:
    # 
    for fold in Train_Config.n_splits:
        return_dataloaders = get_major_dataloader_via_fold(fold)
        train_dataloader, test_dataloader = return_dataloaders.train, return_dataloaders.test
        
        # Shape
        auxi_info, fc_matrices, vbm_matrices, tag = next(iter(test_dataloader))

        # Model
        encoder_structure = Encoder_Structure(matrices_number=len(vbm_matrices)).to(device)

        # Train
        for auxi_info, fc_matrices, vbm_matrices, tag in train_dataloader:
            # Move to GPU
            auxi_info = move_to_device(auxi_info, device)
            fc_matrices = move_to_device(fc_matrices, device)
            vbm_matrices = move_to_device(vbm_matrices, device)
            tag = tag.to(device)

            ## Structural Encoder
            # Input: wc1,wc2,mwc1,mwc2
            output_dict = encoder_structure(input_dict=vbm_matrices)
            for key, value in output_dict.items():
                print(key, value.shape)
            exit()

            ## Functional Encoder
            fc_matrices
            vbm_matrices    

            ## Auxiliary Information Encoder
            # Input: Sex, Age, Education (years)
            for k, v in auxi_info.items():
                print(k, v) 

            # Ground Truth
            print(tag)

if __name__ == "__main__":
    main()