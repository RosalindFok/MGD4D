import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class DoubleConv(nn.Module):  
    """(Conv2D => BatchNorm => Tanh) × 2"""  
    def __init__(self, in_channels, out_channels):  
        super().__init__()  
        self.double_conv = nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),  
            nn.BatchNorm2d(out_channels),  
            nn.Tanh(),  
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),  
            nn.BatchNorm2d(out_channels),  
            nn.Tanh()  
        )  

    def forward(self, x):  
        return self.double_conv(x)  

class Down(nn.Module):  
    """downsampling: only downsample in the second dimension"""  
    def __init__(self, in_channels, out_channels):  
        super().__init__()  
        self.maxpool_conv = nn.Sequential(  
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  
            DoubleConv(in_channels, out_channels)  
        )  

    def forward(self, x):  
        return self.maxpool_conv(x)  

class Up(nn.Module):  
    """upsampling: only upsample in the second dimension"""  
    def __init__(self, in_channels, out_channels):  
        super().__init__()  
        self.up = nn.ConvTranspose2d(  
            in_channels, in_channels // 2,   
            kernel_size=(1, 2), stride=(1, 2)  
        )  
        # correct channel number calculation, considering the concatenation operation of skip connections  
        self.conv = DoubleConv(in_channels, out_channels)  

    def forward(self, x1, x2):  
        x1 = self.up(x1)  
        
        diffY = 0  
        diffX = x2.size()[3] - x1.size()[3]  
        
        if diffX > 0:  
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, 0, 0])  
        
        # concatenate feature maps  
        x = torch.cat([x2, x1], dim=1)  
        return self.conv(x)  

class Bottleneck(nn.Module):  
    """bottleneck layer"""  
    def __init__(self, in_channels, out_channels):  
        super().__init__()  
        self.bottleneck = nn.Sequential(  
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=(1, 3), padding=(0, 1)),  
            nn.BatchNorm2d(out_channels * 2),  
            nn.Tanh(),  
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=(1, 3), padding=(0, 1)),  
            nn.BatchNorm2d(out_channels),  
            nn.Tanh()  
        )  
    
    def forward(self, x):  
        return self.bottleneck(x)  

class UNet(nn.Module):  
    def __init__(self):  
        super().__init__()  
        
        # initial convolution layer  
        self.inc = DoubleConv(1, 64)  
        
        # downsampling path  
        self.down1 = Down(64, 128)  
        self.down2 = Down(128, 256)  
        self.down3 = Down(256, 512)  
        self.down4 = Down(512, 1024)  
        
        # bottleneck layer  
        self.bottleneck = Bottleneck(1024, 1024)  
        
        # upsampling path - note the channel number calculation  
        self.up1 = Up(1024, 512)  # 1024 -> 512 + 512 -> 512  
        self.up2 = Up(512, 256)   # 512 -> 256 + 256 -> 256  
        self.up3 = Up(256, 128)   # 256 -> 128 + 128 -> 128  
        self.up4 = Up(128, 64)    # 128 -> 64 + 64 -> 64  
        
        # final convolution layer  
        self.outc = nn.Conv2d(64, 1, kernel_size=1)  
    
    def forward(self, x):  
        x = x.unsqueeze(1)  
        
        # encoder path  
        x1 = self.inc(x)  
        x2 = self.down1(x1)  
        x3 = self.down2(x2)  
        x4 = self.down3(x3)  
        x5 = self.down4(x4)  
        
        # bottleneck layer  
        x5 = self.bottleneck(x5)  
        
        # decoder path with skip connections  
        x = self.up1(x5, x4)  
        x = self.up2(x, x3)  
        x = self.up3(x, x2)  
        x = self.up4(x, x1)  
        
        # final convolution  
        out = self.outc(x)  
        
        out = out.squeeze(1)  
        
        return out  