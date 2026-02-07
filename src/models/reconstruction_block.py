import torch
import torch.nn as nn


class ReconstructionBlock(nn.Module):
    def __init__(self, in_channels=48, hidden_channels=32, num_classes=4):
        super(ReconstructionBlock, self).__init__()
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        
        # Upsampling with transposed convolutions (coarse-to-fine reversal)
        self.tconv_3 = nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=0, bias=True)
        self.tconv_5 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=5, stride=1, padding=0, bias=True)
        self.tconv_11 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=11, stride=1, padding=0, bias=True)
        
        # Batch norm and activation for upsampling
        self.bn_3 = nn.BatchNorm2d(hidden_channels)
        self.bn_5 = nn.BatchNorm2d(hidden_channels)
        self.bn_11 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Final layer: map hidden features to class logits
        self.class_conv = nn.Conv2d(hidden_channels, num_classes, kernel_size=1, padding=0, bias=True)
    
    def forward(self, x):
        # Upsample back to 240×240 resolution
        
        # T-Conv 3×3: 224×224 → 226×226
        x = self.tconv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        
        # T-Conv 5×5: 226×226 → 230×230
        x = self.tconv_5(x)
        x = self.bn_5(x)
        x = self.relu(x)
        
        # T-Conv 11×11: 230×230 → 240×240
        x = self.tconv_11(x)
        x = self.bn_11(x)
        x = self.relu(x)
        
        # x is now (B, 32, 240, 240)
        # Map to class logits: (B, 32, 240, 240) → (B, 4, 240, 240)
        class_logits = self.class_conv(x)  # (B, 4, 240, 240)
        
        # Split into 4 class-specific logits
        logits = torch.split(class_logits, 1, dim=1)  # 4 tensors of (B, 1, 240, 240)
        logits = [logit.squeeze(1) for logit in logits]  # Remove channel dim: 4 × (B, 240, 240)
        
        return logits  # List of 4 tensors, each (B, 155, 240, 240)
