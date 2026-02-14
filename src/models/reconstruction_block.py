import torch
import torch.nn as nn

class ReconstructionBlock(nn.Module):
    def __init__(self, in_channels=48, hidden_channels=32, num_classes=4, original_depth=155):
        super(ReconstructionBlock, self).__init__()
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        
        # 1. Upsampling layers (2D) to reach 240x240 spatial resolution
        # These remain 2D because your input 'x' is 2D features (B, C, H, W)
        self.tconv_3 = nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=0)
        self.tconv_5 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=5, stride=1, padding=0)
        self.tconv_11 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=11, stride=1, padding=0)
        
        self.bn_3 = nn.BatchNorm2d(hidden_channels)
        self.bn_5 = nn.BatchNorm2d(hidden_channels)
        self.bn_11 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2. The Final Transition to 3D Volume
        # We use a 1x1 convolution to expand the channels to match our Depth (155) 
        # multiplied by our Classes (4). 
        # Output will be: (B, 4 * 155, 240, 240)
        self.final_logits = nn.Conv2d(hidden_channels, num_classes * original_depth, kernel_size=1)

    def forward(self, x):
        # Upsample spatially to 240x240
        x = self.relu(self.bn_3(self.tconv_3(x)))
        x = self.relu(self.bn_5(self.tconv_5(x)))
        x = self.relu(self.bn_11(self.tconv_11(x)))
        
        # x is now (B, 32, 240, 240)
        
        # Project to (B, 4 * 155, 240, 240)
        x = self.final_logits(x)
        
        # Reshape to 5D Tensor: (Batch, Classes, Depth, Height, Width)
        # This gives exactly (B, 4, 155, 240, 240)
        b, _, h, w = x.shape
        x = x.view(b, self.num_classes, -1, h, w)
        
        return x