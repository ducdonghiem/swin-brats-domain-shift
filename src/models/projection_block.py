import torch
import torch.nn as nn


class ModalityCNN(nn.Module):
    '''
    Adapted CNN for the `ProjectionBlock` that downsamples each modality separately before concatenation.

    Args:
        in_channels (int): Number of input channels. Default: 155.
        hidden_channels (int): Number of output channels. Default: 8.
    '''
    
    def __init__(self, in_channels=155, hidden_channels=8):
        super(ModalityCNN, self).__init__()
        
        self.conv_11 = nn.Conv2d(in_channels, hidden_channels, kernel_size=11, padding=0, bias=False)
        self.conv_5 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=0, bias=False)
        self.conv_3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=0, bias=False)
        
        # GroupNorm instead of BatchNorm: works correctly with batch_size=2,
        # and is batch-size independent (groups=1 is equivalent to LayerNorm over channels).
        self.gn_11 = nn.GroupNorm(num_groups=1, num_channels=hidden_channels)
        self.gn_5  = nn.GroupNorm(num_groups=1, num_channels=hidden_channels)
        self.gn_3  = nn.GroupNorm(num_groups=1, num_channels=hidden_channels)
        
        self.gelu = nn.GELU()
    
    def forward(self, x):
        # Conv 11×11: 240×240 → 230×230
        x = self.gelu(self.gn_11(self.conv_11(x)))
        # Conv 5×5: 230×230 → 226×226
        x = self.gelu(self.gn_5(self.conv_5(x)))
        # Conv 3×3: 226×226 → 224×224
        x = self.gelu(self.gn_3(self.conv_3(x)))
        return x  # (B, hidden_channels, 224, 224)


class ProjectionBlock(nn.Module):
    '''
    Projection block adaptation for BraTS dataset.
    
    Data is stored as four 3D volumes but must be learned as a single input.
    The projection block downsamples each modality volume and fuses them with a 1x1 convolution,
    producing a 2D feature map with the same spatial dimensions as the input image and
    a reduced number of channels (C=8). This is then fed into the SwinUNet encoder as a single input.
    '''

    def __init__(self, in_channels=155, hidden_channels=8, out_channels=3, num_modalities=4):
        super(ProjectionBlock, self).__init__()
        self.num_modalities = num_modalities
        
        # Separate CNN instance for each modality (no weight sharing)
        self.modality_cnns = nn.ModuleList([
            ModalityCNN(in_channels=in_channels, hidden_channels=hidden_channels)
            for _ in range(num_modalities)
        ])
        
        # Fusion: Concatenate all modalities
        # No activation after fusion - the Swin PatchPartition expects
        # unrestricted values. clipping negatives here would destroy signal.
        self.fusion_conv = nn.Conv2d(
            num_modalities * hidden_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=True
        )
        self.fusion_norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
    
    def forward(self, modalities):
        processed = [self.modality_cnns[i](mod) for i, mod in enumerate(modalities)]
        concatenated = torch.cat(processed, dim=1)  # (B, 4*hidden, 224, 224)
        fused = self.fusion_norm(self.fusion_conv(concatenated))  # (B, out_channels, 224, 224)
        # No ReLU - preserve full signed range for UNet input
        return fused