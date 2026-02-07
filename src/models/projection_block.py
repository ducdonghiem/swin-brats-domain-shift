import torch
import torch.nn as nn


class ProjectionBlock(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=8, out_channels=3, num_modalities=4):
        super(ProjectionBlock, self).__init__()
        self.num_modalities = num_modalities
        
        # Separate CNN layers for each modality
        self.conv_11_list = nn.ModuleList([nn.Conv2d(in_channels, hidden_channels, kernel_size=11, padding=0, bias=True) for _ in range(num_modalities)])
        self.conv_5_list = nn.ModuleList([nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=0, bias=True) for _ in range(num_modalities)])
        self.conv_3_list = nn.ModuleList([nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=0, bias=True) for _ in range(num_modalities)])
        
        # Batch normalization for stability (separate for each modality)
        self.bn_11_list = nn.ModuleList([nn.BatchNorm2d(hidden_channels) for _ in range(num_modalities)])
        self.bn_5_list = nn.ModuleList([nn.BatchNorm2d(hidden_channels) for _ in range(num_modalities)])
        self.bn_3_list = nn.ModuleList([nn.BatchNorm2d(hidden_channels) for _ in range(num_modalities)])
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
        # Fusion: Concatenate all modalities (num_modalities * hidden_channels) → out_channels
        self.fusion_conv = nn.Conv2d(
            num_modalities * hidden_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=True
        )
        self.fusion_bn = nn.BatchNorm2d(out_channels)
    
    def forward_single_modality(self, x, modality_idx):
        # Conv 11×11: 240×240 → 230×230 (captures large-scale features)
        x = self.conv_11_list[modality_idx](x)
        x = self.bn_11_list[modality_idx](x)
        x = self.relu(x)
        
        # Conv 5×5: 230×230 → 226×226 (captures medium-scale features)
        x = self.conv_5_list[modality_idx](x)
        x = self.bn_5_list[modality_idx](x)
        x = self.relu(x)
        
        # Conv 3×3: 226×226 → 224×224 (captures fine-scale features, edges)
        x = self.conv_3_list[modality_idx](x)
        x = self.bn_3_list[modality_idx](x)
        x = self.relu(x)
        
        return x  # (B, hidden_channels, 224, 224)
    
    def forward(self, modalities):
        # Process each modality independently with its own CNN layers
        processed = []
        for modality_idx, mod in enumerate(modalities):
            feat = self.forward_single_modality(mod, modality_idx)
            processed.append(feat)
        
        # Concatenate all modality features: (B, hidden_channels, 224, 224) × 4 → (B, 4*hidden_channels, 224, 224)
        concatenated = torch.cat(processed, dim=1)
        
        # Reduces from (4 × 8 = 32) channels → 3 channels (RGB-like representation)
        fused = self.fusion_conv(concatenated)
        fused = self.fusion_bn(fused)
        fused = self.relu(fused)
        
        return fused  # (B, out_channels, 224, 224)
