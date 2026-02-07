import torch
import torch.nn as nn


class ModalityDecoder(nn.Module):  
    def __init__(self, in_channels=8, num_classes=4):
        super(ModalityDecoder, self).__init__()
        self.logits_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, padding=0, bias=True)
    
    def forward(self, x):
        logits = self.logits_conv(x)
        return logits  # (B, 4, 240, 240)


class ReconstructionBlock(nn.Module):
    def __init__(self, in_channels=48, hidden_channels=32, num_classes=4, num_modalities=4):
        super(ReconstructionBlock, self).__init__()
        self.num_modalities = num_modalities
        self.hidden_channels = hidden_channels
        
        # Upsampling with transposed convolutions
        self.tconv_3 = nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=0, bias=True)
        self.tconv_5 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=5, stride=1, padding=0, bias=True)
        self.tconv_11 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=11, stride=1, padding=0, bias=True)
        
        # Batch norm and activation for upsampling
        self.bn_3 = nn.BatchNorm2d(hidden_channels)
        self.bn_5 = nn.BatchNorm2d(hidden_channels)
        self.bn_11 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Separate decoder for each modality pathway
        self.decoders = nn.ModuleList([
            ModalityDecoder(in_channels=hidden_channels // num_modalities, num_classes=num_classes)
            for _ in range(num_modalities)
        ])
    
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
        # Split into 4 blocks of (B, 8, 240, 240)
        block_size = self.hidden_channels // self.num_modalities
        blocks = torch.split(x, block_size, dim=1)  # 4 tensors of (B, 8, 240, 240)
        
        # Process each block through its own decoder
        logits = []
        for block_idx, block in enumerate(blocks):
            logit = self.decoders[block_idx](block)  # (B, 4, 240, 240)
            logits.append(logit)
        
        return logits  # List of 4 tensors, each (B, 4, 240, 240)
