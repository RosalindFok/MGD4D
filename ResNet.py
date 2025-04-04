import torch
import torch.nn as nn

class Conv3DBlock(nn.Module):  
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int = 3, stride : int = 1, padding : int = 1) -> None:  
        super().__init__()  
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)  
        self.bn = nn.BatchNorm3d(out_channels)  
        self.ReLU = nn.ReLU()
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:  
        return self.ReLU(self.bn(self.conv(x)))  

class ResBlock3D(nn.Module):  
    def __init__(self, in_channels : int, out_channels : int, stride : int = 1) -> None:  
        super().__init__()  
        self.conv1 = Conv3DBlock(in_channels, out_channels, stride=stride)  
        self.conv2 = Conv3DBlock(out_channels, out_channels)  
        
        # Shortcut connection  
        self.shortcut = nn.Sequential()  
        if stride != 1 or in_channels != out_channels:  
            self.shortcut = nn.Sequential(  
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),  
                nn.BatchNorm3d(out_channels)  
            )  
        self.ReLU = nn.ReLU()
            
    def forward(self, x : torch.Tensor) -> torch.Tensor:  
        residual = self.shortcut(x)  
        out = self.conv1(x)  
        out = self.conv2(out)  
        out = torch.add(out, residual)
        return self.ReLU(out)  

class ResNet3D(nn.Module):  
    def __init__(self, in_channels : int = 1, embedding_dim : int = 512) -> None:  
        super().__init__()  
        
        out_channels = 64

        # Initial convolution  
        self.conv1 = Conv3DBlock(in_channels, out_channels, kernel_size=7, stride=2, padding=3)  
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)  
        
        # Residual blocks  
        self.layer1 = self.make_layer(in_channels=out_channels  , out_channels=out_channels  , num_blocks=2)  
        self.layer2 = self.make_layer(in_channels=out_channels  , out_channels=out_channels*2, num_blocks=2, stride=2)  
        self.layer3 = self.make_layer(in_channels=out_channels*2, out_channels=out_channels*4, num_blocks=2, stride=2)  
        self.layer4 = self.make_layer(in_channels=out_channels*4, out_channels=out_channels*8, num_blocks=2, stride=2)  
        
        # Global pooling and embedding layer  
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))  
        self.fc = nn.Sequential(
            nn.Linear(out_channels*8, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def make_layer(self, in_channels : int, out_channels : int, num_blocks : int, stride : int = 1) -> nn.Sequential:  
        layers = []  
        layers.append(ResBlock3D(in_channels, out_channels, stride))  
        for _ in range(1, num_blocks):  
            layers.append(ResBlock3D(out_channels, out_channels))  
        return nn.Sequential(*layers)  
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:  
        # Input shape: (batch_size, 1, D, H, W)  
        x = self.conv1(x)  
        x = self.maxpool(x)  
        
        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x)  
        x = self.layer4(x)  
        
        x = self.global_pool(x)  
        x = x.view(x.size(0), -1)  
        x = self.fc(x)  
        
        return x  