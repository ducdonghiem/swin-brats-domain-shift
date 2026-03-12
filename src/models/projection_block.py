import torch
import torch.nn as nn


class ModalityCNN(nn.Module):
    def __init__(self, in_channels=155, hidden_channels=8):
        super(ModalityCNN, self).__init__()
        
        self.conv_11 = nn.Conv2d(in_channels, hidden_channels, kernel_size=11, padding=0, bias=True)
        self.conv_5 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=0, bias=True)
        self.conv_3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=0, bias=True)
        
        self.bn_11 = nn.BatchNorm2d(hidden_channels)
        self.bn_5 = nn.BatchNorm2d(hidden_channels)
        self.bn_3 = nn.BatchNorm2d(hidden_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Conv 11×11
        x = self.conv_11(x)
        x = self.bn_11(x)
        x = self.relu(x)
        
        # Conv 5×5
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.relu(x)
        
        # Conv 3×3
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        
        return x  # (B, hidden_channels, 224, 224)


class ProjectionBlock(nn.Module):
    def __init__(self, in_channels=155, hidden_channels=8, out_channels=3, num_modalities=4):
        super(ProjectionBlock, self).__init__()
        self.num_modalities = num_modalities
        
        # Separate CNN instance for each modality
        self.modality_cnns = nn.ModuleList([
            ModalityCNN(in_channels=in_channels, hidden_channels=hidden_channels)
            for _ in range(num_modalities)
        ])
        
        # Fusion: Concatenate all modalities 
        self.fusion_conv = nn.Conv2d(
            num_modalities * hidden_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=True
        )
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, modalities):
        # Process each modality with its own CNN
        processed = []
        for modality_idx, mod in enumerate(modalities):
            feat = self.modality_cnns[modality_idx](mod)
            processed.append(feat)
        
        # Concatenate all modality features
        concatenated = torch.cat(processed, dim=1)
        
        fused = self.fusion_conv(concatenated)
        fused = self.fusion_bn(fused)
        fused = self.relu(fused)
        
        return fused  # (B, out_channels, 224, 224)
