import torch
import torch.nn as nn

class ReconstructionBlock(nn.Module):
    def __init__(self, in_channels=48, hidden_channels=32, num_classes=4, original_depth=155):
        super(ReconstructionBlock, self).__init__()
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        
        # Transposed convs to upsample spatially: 224 -> 240
        self.tconv_3  = nn.ConvTranspose2d(in_channels,     hidden_channels, kernel_size=3,  stride=1, padding=0)
        self.tconv_5  = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=5,  stride=1, padding=0)
        self.tconv_11 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=11, stride=1, padding=0)
        
        # GroupNorm: batch-size independent, consistent with transformer norms upstream
        self.gn_3  = nn.GroupNorm(num_groups=1, num_channels=hidden_channels)
        self.gn_5  = nn.GroupNorm(num_groups=1, num_channels=hidden_channels)
        self.gn_11 = nn.GroupNorm(num_groups=1, num_channels=hidden_channels)
        self.gelu = nn.GELU()

        # Project to (num_classes * depth) channels, then view as 3D volume
        self.final_logits = nn.Conv2d(hidden_channels, num_classes * original_depth, kernel_size=1)

    def forward(self, x):
        x = self.gelu(self.gn_3(self.tconv_3(x)))    # (B, 32, 226, 226)
        x = self.gelu(self.gn_5(self.tconv_5(x)))    # (B, 32, 230, 230)
        x = self.gelu(self.gn_11(self.tconv_11(x)))  # (B, 32, 240, 240)
        x = self.final_logits(x)                      # (B, 4*155, 240, 240)
        b, _, h, w = x.shape
        x = x.view(b, self.num_classes, -1, h, w)     # (B, 4, 155, 240, 240)
        return x