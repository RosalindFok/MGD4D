from config import Train_Config
from dataset import get_major_dataloader_via_fold

for fold in Train_Config.n_splits:
    return_dataloaders = get_major_dataloader_via_fold(fold)
    train_dataloader, test_dataloader = return_dataloaders.train, return_dataloaders
    # Train
    
