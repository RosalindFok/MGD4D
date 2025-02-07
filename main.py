import torch

from config import Train_Config
from models import device, MGD4MD
from dataset import get_major_dataloader_via_fold

def main() -> None:
    # 
    for fold in Train_Config.n_splits:
        return_dataloaders = get_major_dataloader_via_fold(fold)
        train_dataloader, test_dataloader = return_dataloaders.train, return_dataloaders.test
        
        # Shape
        auxi_info, fc_matrices, vbm_matrices, tag = next(iter(test_dataloader))

        # Model
        model = MGD4MD(structural_matrices_number=len(vbm_matrices)).to(device)

        # Train
        move_to_device = lambda tensor_dict, device: {k: v.to(device) for k, v in tensor_dict.items()}
        for auxi_info, fc_matrices, vbm_matrices, tag in train_dataloader:
            # Move to GPU
            auxi_info = move_to_device(auxi_info, device)
            fc_matrices = move_to_device(fc_matrices, device)
            vbm_matrices = move_to_device(vbm_matrices, device)
            tag = tag.to(device)

            output_dict = model(structural_input_dict=vbm_matrices,
                                functional_input_dict=fc_matrices,
                                information_input_dict=auxi_info)
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